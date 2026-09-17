"""
nanomem.legacy_v2
~~~~~~~~~~~~~~~~~
Read-only reader for v2 ``.dat`` files plus the one-time migration to v3.

The v2 container obfuscated its payloads with a SHA-256 chain keyed only by the
block id and a salt compiled into the library: no user secret, no nonce, no
integrity tag. It was labelled "256-bit encryption"; it was not encryption. The
constant below therefore survives in exactly one place -- here -- so that old
files can still be read, and nothing in the v3 write path can reach it.

Migration copies the original to ``<path>.v2.bak`` FIRST (``shutil.copyfile``,
not a rename), writes a complete new v3 file to a temp path, and only then
``os.replace``s it over the original. The vault path therefore exists at every
instant, which matters because the constructor creates a vault eagerly and
callers test ``os.path.exists`` right after.
"""

import hashlib
import json
import os
import shutil
import struct
import sys
import warnings
import time

import numpy as np

from . import container as C
from . import crypto
from .entities import derive_entity_for_migrated, make_group_key, normalize_entity

V2_FILE_MAGIC = b"NANOMEM\x00"
V2_BLOCK_MAGIC = b"BLK\x00"
V2_FILE_HEADER_FMT = "<8sIIII40s"
V2_FILE_HEADER_SIZE = 64
V2_BLOCK_HEADER_FMT = "<4s32sIIfI"
V2_BLOCK_HEADER_SIZE = struct.calcsize(V2_BLOCK_HEADER_FMT)   # 52
V2_COORD_DIM = 64
# Legacy constant, read path only. Never used to write anything.
V2_CIPHER_SALT = b"NANOMEM_VAULT_AES256_PROJECTED_LATTICE_2026"
_V2_CHUNK = 32


def _v2_keystream(key_seed: bytes, nbytes: int) -> np.ndarray:
    """Verbatim port of the v2 SHA-256 chain: h0 = SHA256(seed||SALT), h_{i+1} = SHA256(h_i||seed)."""
    n_chunks = (nbytes + _V2_CHUNK - 1) // _V2_CHUNK
    digest = hashlib.sha256(key_seed + V2_CIPHER_SALT).digest()
    chunks = bytearray(n_chunks * _V2_CHUNK)
    for i in range(n_chunks):
        off = i * _V2_CHUNK
        chunks[off:off + _V2_CHUNK] = digest
        digest = hashlib.sha256(digest + key_seed).digest()
    return np.frombuffer(bytes(chunks), dtype=np.uint8)[:nbytes]


def _v2_transform(data: bytes, key_seed: bytes) -> bytes:
    if not data:
        return b""
    buf = np.frombuffer(data, dtype=np.uint8)
    return (buf ^ _v2_keystream(key_seed, buf.size)).tobytes()


def read_v2_header(filepath: str):
    """``(embed_dim, coord_dim, block_capacity)`` or raise on a non-v2 file."""
    with open(filepath, "rb") as f:
        raw = f.read(V2_FILE_HEADER_SIZE)
    if len(raw) < V2_FILE_HEADER_SIZE:
        raise C.CorruptContainerError(f"{filepath}: v2 header truncated")
    magic, version, embed_dim, coord_dim, cap, _ = struct.unpack(V2_FILE_HEADER_FMT, raw)
    if magic != V2_FILE_MAGIC:
        raise C.CorruptContainerError(f"{filepath}: not a v2 container ({magic!r})")
    return int(embed_dim), int(coord_dim), int(cap)


def iter_v2_records(filepath: str):
    """Yield every record of a v2 file in stored order.

    Each record is ``{id, text, source, metadata, timestamp, revision, embedding}``
    with the embedding re-normalised. ``keys``/``basis``/``mean`` are discarded:
    nothing in the library, the benchmarks or the docs ever read them.
    Scanning stops at the first bad magic / short centroid / incomplete payload,
    exactly as the v2 reader did.
    """
    embed_dim, coord_dim, _cap = read_v2_header(filepath)
    size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        pos = V2_FILE_HEADER_SIZE
        while pos + V2_BLOCK_HEADER_SIZE + embed_dim * 4 <= size:
            f.seek(pos)
            hdr = f.read(V2_BLOCK_HEADER_SIZE)
            magic, bid_b, n_docs, _ts, _rad, payload_sz = struct.unpack(V2_BLOCK_HEADER_FMT, hdr)
            if magic != V2_BLOCK_MAGIC:
                break
            centroid = f.read(embed_dim * 4)
            if len(centroid) < embed_dim * 4:
                break
            data_off = pos + V2_BLOCK_HEADER_SIZE + embed_dim * 4
            if data_off + payload_sz > size:
                break
            f.seek(data_off)
            buf = _v2_transform(f.read(payload_sz), key_seed=bid_b)
            try:
                meta_len, r_actual = struct.unpack("<II", buf[:8])
                meta = json.loads(buf[8:8 + meta_len].decode("utf-8", errors="replace"))
            except Exception:
                break
            cur = 8 + meta_len
            cur += n_docs * coord_dim * 4                      # keys (discarded)
            v_sz = n_docs * embed_dim * 4
            values = np.frombuffer(buf[cur:cur + v_sz], dtype=np.float32).reshape(n_docs, embed_dim)
            cur += v_sz
            cur += r_actual * embed_dim * 4                    # basis (discarded)
            cur += embed_dim * 4                               # mean  (discarded)
            ts = np.frombuffer(buf[cur:cur + n_docs * 8], dtype=np.float64)
            cur += n_docs * 8
            rev = np.frombuffer(buf[cur:cur + n_docs * 4], dtype=np.int32)
            texts = meta.get("texts", [])
            sources = meta.get("sources", [])
            metas = meta.get("metadatas", [{}] * n_docs)
            for i in range(min(n_docs, len(texts))):
                m_i = dict(metas[i]) if i < len(metas) and isinstance(metas[i], dict) else {}
                t_i = texts[i]
                ts_i = ts[i]
                doc_id = str(m_i.get("id") or
                             f"doc_{hashlib.md5(f'{t_i}_{ts_i}'.encode()).hexdigest()[:10]}")
                m_i["id"] = doc_id
                v = np.array(values[i], dtype=np.float32)
                v /= (np.linalg.norm(v) + 1e-8)
                yield {"id": doc_id, "text": t_i,
                       "source": sources[i] if i < len(sources) else "unknown",
                       "metadata": m_i, "timestamp": float(ts_i),
                       "revision": int(rev[i]) if i < len(rev) else 1,
                       "embedding": v}
            pos = data_off + payload_sz


def repair_entity(record: dict) -> dict:
    """Rewrite a junk v2 entity with the generic tagger, preserving the original.

    v2 fell back to "first capitalised phrase", which produced revision groups
    such as ``Update``. Anything that does not already look like a class tag
    (``^[a-z0-9_]+$``) is re-derived from the text; the original is kept as
    ``metadata['entity_v2']`` and, of course, in ``<path>.v2.bak``.
    """
    meta = record.get("metadata") or {}
    original = meta.get("entity")
    repaired = derive_entity_for_migrated(meta, record.get("text", ""))
    if original is not None and normalize_entity(original) == (repaired or ""):
        return record
    if repaired:
        meta["entity"] = repaired
    elif original is not None:
        meta.pop("entity", None)
    if original is not None:
        meta["entity_v2"] = original
    record["metadata"] = meta
    return record


def migrate_v2_to_v3(filepath: str, *, password=None, backup="auto",
                     vector_dtype: str = "float16",
                     block_capacity: int = C.BLOCK_CAPACITY,
                     landmarks_per_block: int = C.LANDMARKS_PER_BLOCK,
                     landmark_fn=None, quiet: bool = False) -> int:
    """Rewrite a v2 file in place as v3. Returns the record count.

    Order of operations (the path never stops existing): optional
    ``copyfile -> .v2.bak``, write ``<path>.tmp-*``, fsync, ``os.replace``. The
    backup is a convenience rollback, not the crash guard -- the original file is
    intact until the ``os.replace``, whether or not a copy was taken.

    ``backup``:

    * ``"auto"`` (default) -- write ``<path>.v2.bak`` when the new vault is
      PLAINTEXT, and do NOT write one when a ``password`` was given. A v2 file is
      readable by anyone who has this library (its "cipher" is keyed only by the
      block id and a compiled-in constant), so keeping one beside an encrypted
      vault hands back every record the password was meant to protect. 3.0.0
      wrote it unconditionally, mode 0644, and nothing ever removed it.
    * ``True`` -- always write it. With a password this is an explicit decision to
      keep a readable plaintext copy; it is created mode 0600 and warned about on
      stderr and through :mod:`warnings`.
    * ``False`` -- never write it.

    ``NANOMEM_KEEP_V2_BACKUP=0`` still forces it off.
    """
    filepath = os.path.abspath(filepath)
    embed_dim, _coord, _cap = read_v2_header(filepath)
    want = (bool(password) is False) if backup == "auto" else bool(backup)
    if os.environ.get("NANOMEM_KEEP_V2_BACKUP", "1") == "0":
        want = False
    bak = None
    if want:
        bak = filepath + ".v2.bak"
        if os.path.exists(bak):
            bak = f"{bak}.{int(time.time())}"
        shutil.copyfile(filepath, bak)
        try:
            os.chmod(bak, 0o600)
        except OSError:
            pass
        if password:
            msg = (f"nanomem: {bak} is a PLAINTEXT copy of your v2 vault -- every "
                   f"record in it can be read without the password. Delete it once "
                   f"you have checked the migration.")
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            sys.stderr.write(msg + "\n")
    elif password and not quiet:
        sys.stderr.write(
            f"nanomem: migrating {filepath} into a password-protected vault; no "
            f"plaintext .v2.bak was kept (pass migrate_backup=True to keep one).\n")

    header = C.Container._new_header(embed_dim, vector_dtype, block_capacity,
                                     landmarks_per_block, password)
    keys = None
    if password:
        keys = crypto.derive_keys(password, header.kdf_salt, header.scrypt_log2_n,
                                  header.scrypt_r, header.scrypt_p)

    count = 0

    def blobs():
        nonlocal count
        batch, seq = [], 0
        for rec in iter_v2_records(filepath):
            batch.append(repair_entity(rec))
            if len(batch) >= block_capacity:
                yield _blob_for(header, keys, seq, batch, landmarks_per_block, landmark_fn)
                count += len(batch)
                seq += 1
                batch = []
        if batch:
            yield _blob_for(header, keys, seq, batch, landmarks_per_block, landmark_fn)
            count += len(batch)

    tmp = C.temp_path_for(filepath)
    try:
        cont = object.__new__(C.Container)
        cont.filepath = filepath
        cont._closed = False        # `write_new_file` refuses to write a closed one
        cont.write_new_file(tmp, header, keys, blobs())
        os.replace(tmp, filepath)
        C.fsync_dir(filepath)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        if keys is not None:
            # Every other derivation site wipes; 3.0.2 left this one live until
            # GC (`engine.replace_all` and `container.write_new_file` both call
            # `KeyMaterial.wipe()` in a finally block, and this function did not).
            keys.wipe()
    if not quiet:
        tail = (f"; original kept at {bak} (PLAINTEXT, readable without the password)"
                if bak and password else
                f"; original kept at {bak}" if bak else "")
        sys.stderr.write(f"nanomem: migrated {filepath} to v3 ({count} docs){tail}\n")
    return count


def _blob_for(header, keys, seq, batch, L, landmark_fn):
    V = np.ascontiguousarray([r["embedding"] for r in batch], dtype=np.float32)
    lm = landmark_fn(V, L, seq) if landmark_fn is not None else None
    docs, ids, groups, ts, rev = [], [], [], [], []
    for r in batch:
        meta = r.get("metadata") or {}
        ids.append(r["id"])
        groups.append(make_group_key(meta.get("user_id"), meta.get("project"),
                                     meta.get("entity")))
        docs.append(json.dumps({"text": r["text"], "source": r.get("source", "unknown"),
                                "metadata": meta}, ensure_ascii=False,
                               separators=(",", ":"), default=str).encode("utf-8"))
        ts.append(float(r["timestamp"]))
        rev.append(int(r["revision"]))
    return C.build_block_blob(header, keys, seq, V, lm, ts, rev, ids, groups, docs)


__all__ = ["V2_FILE_MAGIC", "V2_BLOCK_MAGIC", "iter_v2_records", "migrate_v2_to_v3",
           "read_v2_header", "repair_entity"]
