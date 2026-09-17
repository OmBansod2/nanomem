"""v3 byte format, crash recovery, integrity and atomic rewrite."""

import os
import struct

import numpy as np
import pytest

from conftest import D, unit_rows
from nanomem import container as C
from nanomem.engine import VaultEngine
from nanomem.errors import CorruptContainerError, IntegrityError


def _fill(path, n=120, seed=2, **kw):
    e = VaultEngine(filepath=path, embed_dim=D, **kw)
    V = unit_rows(n, seed=seed)
    for i in range(n):
        e.add_fact(f"record {i}", V[i], metadata={"i": i})
    e.flush()
    return e, V


def test_header_roundtrip():
    h = C.FileHeader(flags=C.FLAG_VEC_FP16, embed_dim=D, vault_uuid=os.urandom(16),
                     created_unix=1.5)
    blob = C.pack_file_header(h)
    assert len(blob) == 256 and blob[:8] == b"NANOMEM3"
    back = C.unpack_file_header(blob)
    assert back.embed_dim == D and back.vector_dtype == "float16"
    bad = bytearray(blob); bad[20] ^= 0xFF
    with pytest.raises(CorruptContainerError):
        C.unpack_file_header(bytes(bad))


def test_v2_magic_is_not_v3(tmp_path):
    p = tmp_path / "old.dat"
    p.write_bytes(struct.pack("<8sIIII40s", b"NANOMEM\x00", 2, D, 64, 50, b"\x00" * 40))
    assert C.detect_version(str(p)) == 2
    junk = tmp_path / "junk.dat"
    junk.write_bytes(b"NOTAVAULT" + b"\x00" * 100)
    with pytest.raises(CorruptContainerError):
        C.detect_version(str(junk))


def test_block_alignment_and_layout(tmp_path):
    p = str(tmp_path / "v.dat")
    e, _ = _fill(p, n=120)
    for b in e._cont.blocks:
        assert b.offset % 64 == 0
        assert b.n <= 50 and b.kind == C.KIND_DATA
    assert [b.n for b in e._cont.blocks] == [50, 50, 20]
    e.close()


def test_truncation_recovery(tmp_path):
    p = str(tmp_path / "v.dat")
    e, _ = _fill(p, n=200)
    full = open(p, "rb").read()
    e.close()
    boundaries = {b.offset + b.total_len for b in VaultEngine(p, embed_dim=D)._cont.blocks}
    rng = np.random.default_rng(0)
    for cut in rng.integers(300, len(full), size=40):
        cut = int(cut)
        q = str(tmp_path / "cut.dat")
        open(q, "wb").write(full[:cut])
        e2 = VaultEngine(filepath=q, embed_dim=D)
        complete = sum(1 for b in e2._cont.blocks)
        assert complete == sum(1 for b in boundaries if b <= cut)
        if cut not in boundaries:
            assert e2._cont.truncated_tail_bytes > 0
        n_before = e2.count()
        e2.add_fact("after recovery", unit_rows(1)[0]); e2.flush()
        e2.close()
        e3 = VaultEngine(filepath=q, embed_dim=D)
        assert e3.count() == n_before + 1
        e3.close()
        os.remove(q)


@pytest.mark.parametrize("password", [None, "correct horse battery staple"])
def test_bit_flip_detection(tmp_path, password):
    p = str(tmp_path / "v.dat")
    e, _ = _fill(p, n=120, password=password)
    target = e._cont.blocks[0]
    e.close()
    raw = bytearray(open(p, "rb").read())
    raw[target.offset + C.BLOCK_HEADER_SIZE + 10] ^= 0x01        # payload byte
    open(p, "wb").write(bytes(raw))
    with pytest.raises(IntegrityError) as exc:
        VaultEngine(filepath=p, embed_dim=D, password=password)
    assert exc.value.block_index == 0
    e2 = VaultEngine(filepath=p, embed_dim=D, password=password,
                     on_integrity_error="skip")
    assert e2.stats()["integrity_errors"] == [0]
    assert e2.count() == 70
    e2.close()


def test_atomic_replace_leaves_original(tmp_path, monkeypatch):
    p = str(tmp_path / "v.dat")
    e, _ = _fill(p, n=60)
    before = open(p, "rb").read()
    recs = list(e.iter_records())
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        e.replace_all(recs)
    monkeypatch.undo()
    assert open(p, "rb").read() == before
    assert not [f for f in os.listdir(tmp_path) if ".tmp-" in f]
    e.close()


def test_stale_temp_cleanup(tmp_path):
    p = str(tmp_path / "v.dat")
    e, _ = _fill(p, n=10)
    e.close()
    stale = p + ".tmp-1-deadbeef"
    open(stale, "wb").write(b"junk")
    os.utime(stale, (0, 0))
    VaultEngine(filepath=p, embed_dim=D).close()
    assert not os.path.exists(stale)


def test_append_after_other_instance(tmp_path):
    p = str(tmp_path / "v.dat")
    a = VaultEngine(filepath=p, embed_dim=D)
    b = VaultEngine(filepath=p, embed_dim=D)
    V = unit_rows(120, seed=4)
    header_before = open(p, "rb").read(256)
    for i in range(60):
        b.add_fact(f"b{i}", V[i], metadata={"who": "b"})
    b.flush()
    for i in range(60, 120):
        a.add_fact(f"a{i}", V[i], metadata={"who": "a"})
    a.flush()                                    # must re-scan, not truncate b's blocks
    c = VaultEngine(filepath=p, embed_dim=D)
    assert c.count() == 120
    whos = {r["metadata"]["who"] for r in c.iter_records()}
    assert whos == {"a", "b"}
    assert open(p, "rb").read(256) == header_before         # header never rewritten
    a.close(); b.close(); c.close()


def test_compact_coalesces_blocks(tmp_path):
    p = str(tmp_path / "v.dat")
    e = VaultEngine(filepath=p, embed_dim=D)
    V = unit_rows(200, seed=6)
    for i in range(200):
        e.add_fact(f"r{i}", V[i])
        e.flush()                                            # 200 one-record blocks
    assert e.stats()["compacted_blocks"] == 200
    sizes = [b.total_len for b in e._cont.blocks]
    assert max(sizes) < 4096                                 # a 1-doc flush stays small
    info = e.compact()
    assert info["blocks_after"] == 4 and e.count() == 200
    e.close()


def test_vector_dtype_default_and_fp32(tmp_path):
    p16, p32 = str(tmp_path / "a.dat"), str(tmp_path / "b.dat")
    e16, V = _fill(p16, n=200, seed=8)
    e32, _ = _fill(p32, n=200, seed=8, vector_dtype="float32")
    assert e16.stats()["vector_dtype"] == "float16"
    assert os.path.getsize(p16) < os.path.getsize(p32)
    cos16 = np.sum(np.vstack([r["embedding"] for r in e16.iter_records()]) * V, axis=1)
    assert float(cos16.min()) > 0.9999
    e16.close(); e32.close()
