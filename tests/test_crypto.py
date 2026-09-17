"""Optional password mode: KDF, header binding, confidentiality, tamper detection."""

import hashlib
import os

import numpy as np
import pytest

from conftest import D, unit_rows
from nanomem import crypto
from nanomem.engine import VaultEngine
from nanomem.errors import (CorruptContainerError, IntegrityError, NotEncryptedError,
                           PasswordRequiredError, WrongPasswordError)

PW = "correct horse battery staple"
FAST = dict(log2_n=14, r=8, p=1)          # keeps the unit tests quick


def test_derive_keys_deterministic():
    salt = b"\x01" * 16
    a = crypto.derive_keys(PW, salt, **FAST)
    b = crypto.derive_keys(PW, salt, **FAST)
    assert bytes(a.k_enc) == bytes(b.k_enc) and bytes(a.k_mac) == bytes(b.k_mac)
    assert bytes(a.k_enc) != bytes(a.k_mac) != bytes(a.k_chk)
    c = crypto.derive_keys(PW + "!", salt, **FAST)
    assert bytes(c.k_enc) != bytes(a.k_enc)
    nfkc = crypto.derive_keys("café", salt, **FAST)
    same = crypto.derive_keys("café", salt, **FAST)
    assert bytes(nfkc.k_enc) == bytes(same.k_enc)
    a.wipe()
    assert set(bytes(a.k_enc)) == {0} and a.wiped


def test_keystream_unique_per_block():
    km = crypto.derive_keys(PW, b"\x02" * 16, **FAST)
    uuid = os.urandom(16)
    n1 = os.urandom(16)
    s1 = crypto.keystream(km, uuid, n1, 0, 64)
    assert s1 != crypto.keystream(km, uuid, n1, 1, 64)          # seq
    assert s1 != crypto.keystream(km, uuid, os.urandom(16), 0, 64)   # nonce
    assert s1 != crypto.keystream(km, os.urandom(16), n1, 0, 64)     # file generation


def test_wrong_password_before_any_block_read(tmp_path, monkeypatch):
    p = str(tmp_path / "v.dat")
    e = VaultEngine(filepath=p, embed_dim=D, password=PW)
    V = unit_rows(60, seed=3)
    for i in range(60):
        e.add_fact(f"secret record {i}", V[i], metadata={"i": i})
    e.flush(); e.close()

    reads = {"n": 0}
    real_scan = type(e._cont).scan

    def counting_scan(self, sink, from_offset=None):
        reads["n"] += 1
        return real_scan(self, sink, from_offset)

    monkeypatch.setattr(type(e._cont), "scan", counting_scan)
    with pytest.raises(WrongPasswordError):
        VaultEngine(filepath=p, embed_dim=D, password="wrong password entirely")
    assert reads["n"] == 0
    monkeypatch.undo()

    with pytest.raises(PasswordRequiredError):
        VaultEngine(filepath=p, embed_dim=D)


def test_password_on_plaintext_vault(tmp_path):
    """A NanomemError subclass, not a bare ValueError.

    ``errors`` promises one ``except NanomemError`` catches everything this
    library raises; 3.0.1 raised ``ValueError`` here, and the CLI printed a raw
    traceback (and exited 0) for ``--password-stdin`` against a plaintext vault.
    """
    p = str(tmp_path / "plain.dat")
    VaultEngine(filepath=p, embed_dim=D).close()
    with pytest.raises(NotEncryptedError):
        VaultEngine(filepath=p, embed_dim=D, password=PW)
    from nanomem.errors import NanomemError
    with pytest.raises(NanomemError):                 # the documented single guard
        VaultEngine(filepath=p, embed_dim=D, password=PW)


def test_encrypt_roundtrip_and_confidentiality(tmp_path):
    p = str(tmp_path / "v.dat")
    e = VaultEngine(filepath=p, embed_dim=D, password=PW, vector_dtype="float32")
    V = unit_rows(150, seed=5)
    texts = [f"confidential subject line {i}" for i in range(150)]
    ids = [e.add_fact(t, V[i], metadata={"tag": f"tag-{i}"}) for i, t in enumerate(texts)]
    e.flush(); e.close()

    raw = open(p, "rb").read()
    for t in texts[:50]:
        assert t.encode() not in raw
    for i in ids[:50]:
        assert i.encode() not in raw
    for i in range(20):
        assert V[i].tobytes() not in raw

    e2 = VaultEngine(filepath=p, embed_dim=D, password=PW)
    recs = list(e2.iter_records())
    assert [r["text"] for r in recs] == texts
    assert float(np.min(np.sum(np.vstack([r["embedding"] for r in recs]) * V, axis=1))) > 0.99999
    nonces = {bytes(b.nonce) for b in e2._cont.blocks}
    assert len(nonces) == len(e2._cont.blocks)
    st = e2.stats()
    assert st["encrypted_at_rest"] is True
    assert st["cipher"] == crypto.CIPHER_LABEL_SHAKE
    assert "not AES" in st["cipher"] and "256-bit" not in st["cipher"]
    assert st["kdf"].startswith("scrypt n=")
    e2.close()


def test_block_cannot_be_spliced_across_generations(tmp_path):
    p = str(tmp_path / "v.dat")
    e = VaultEngine(filepath=p, embed_dim=D, password=PW)
    V = unit_rows(60, seed=7)
    for i in range(60):
        e.add_fact(f"gen one {i}", V[i])
    e.flush()
    old_bytes = open(p, "rb").read()
    old_uuid = bytes(e.header.vault_uuid)
    old_block = e._cont.blocks[0]
    e.replace_all(list(e.iter_records()))
    assert bytes(e.header.vault_uuid) != old_uuid          # uuid rotates on rewrite
    new_block = e._cont.blocks[0]
    e.close()

    spliced = bytearray(open(p, "rb").read())
    chunk = old_bytes[old_block.offset:old_block.offset + old_block.total_len]
    if len(chunk) == new_block.total_len:
        spliced[new_block.offset:new_block.offset + len(chunk)] = chunk
        open(p, "wb").write(bytes(spliced))
        with pytest.raises(IntegrityError):
            VaultEngine(filepath=p, embed_dim=D, password=PW)


def test_rekey_roundtrip(tmp_path):
    p = str(tmp_path / "v.dat")
    e = VaultEngine(filepath=p, embed_dim=D)
    V = unit_rows(30, seed=11)
    for i in range(30):
        e.add_fact(f"row {i}", V[i])
    e.flush()
    e.rekey(PW)
    assert e.stats()["encrypted_at_rest"] is True
    e.close()
    with pytest.raises(PasswordRequiredError):
        VaultEngine(filepath=p, embed_dim=D)
    e2 = VaultEngine(filepath=p, embed_dim=D, password=PW)
    assert e2.count() == 30
    e2.rekey(None)
    assert e2.stats()["encrypted_at_rest"] is False
    e2.close()
    e3 = VaultEngine(filepath=p, embed_dim=D)
    assert [r["text"] for r in e3.iter_records()] == [f"row {i}" for i in range(30)]
    e3.close()


def test_no_crypto_on_the_hot_path(tmp_path, monkeypatch):
    p = str(tmp_path / "v.dat")
    e = VaultEngine(filepath=p, embed_dim=D, password=PW)
    V = unit_rows(100, seed=13)
    for i in range(100):
        e.add_fact(f"row {i}", V[i])
    e.flush()
    calls = {"n": 0}
    real = crypto.keystream
    monkeypatch.setattr(crypto, "keystream",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1])
    for i in range(50):
        e.search("row", V[i % 100], top_k=4)
    assert calls["n"] == 0
    e.close()


# --------------------------------------------------------------------------
# round-2 regressions
# --------------------------------------------------------------------------
def test_no_crypto_on_the_hot_path_after_reopen(tmp_path, monkeypatch):
    """The real deployment shape: another process opens the vault and searches.

    3.0.0 passed the writer-instance version of this test while a re-opened
    encrypted vault re-derived the scrypt key AND re-decrypted every block on
    every single search -- measured 142 ms per query against 0.31 ms.
    """
    p = str(tmp_path / "v.dat")
    e = VaultEngine(filepath=p, embed_dim=D, password=PW)
    V = unit_rows(150, seed=29)
    for i in range(150):
        e.add_fact(f"row {i}", V[i])
    e.flush()
    e.close()

    reader = VaultEngine(filepath=p, embed_dim=D, password=PW)
    calls = {"ks": 0, "kdf": 0}
    real_ks, real_kdf = crypto.keystream, crypto.derive_keys
    monkeypatch.setattr(crypto, "keystream",
                        lambda *a, **k: (calls.__setitem__("ks", calls["ks"] + 1),
                                         real_ks(*a, **k))[1])
    monkeypatch.setattr(crypto, "derive_keys",
                        lambda *a, **k: (calls.__setitem__("kdf", calls["kdf"] + 1),
                                         real_kdf(*a, **k))[1])
    for i in range(50):
        reader.search("row", V[i % 150], top_k=4)
        reader.get(reader.arena.ids[i])
    assert calls == {"ks": 0, "kdf": 0}, calls
    assert reader._cont.modified() == "same"
    reader.close()


def test_block_header_reserved_bytes_are_authenticated(tmp_path):
    """The tag covers all 96 header bytes, not the first 72.

    3.0.0 left bytes [72:96] of every block header malleable inside a header the
    docs described as authenticated.
    """
    p = str(tmp_path / "v.dat")
    e = VaultEngine(filepath=p, embed_dim=D, password=PW)
    V = unit_rows(60, seed=31)
    for i in range(60):
        e.add_fact(f"row {i}", V[i])
    e.flush()
    off = e._cont.blocks[0].offset
    e.close()
    with open(p, "r+b") as f:
        f.seek(off + 80)
        b = f.read(1)
        f.seek(off + 80)
        f.write(bytes([b[0] ^ 0x01]))
    with pytest.raises(IntegrityError):
        VaultEngine(filepath=p, embed_dim=D, password=PW)


def test_clearing_the_encrypted_flag_is_a_tamper_error(tmp_path):
    """Not "your vault is not encrypted": the header is authenticated first.

    3.0.0 branched on the flag before checking the authenticator, so a tampered
    header told a user with the correct password that their vault had never been
    encrypted.
    """
    import zlib
    p = str(tmp_path / "v.dat")
    e = VaultEngine(filepath=p, embed_dim=D, password=PW)
    e.add_fact("secret", unit_rows(1, seed=3)[0])
    e.flush()
    e.close()
    raw = bytearray(open(p, "rb").read())
    flags = int.from_bytes(raw[20:24], "little")
    raw[20:24] = (flags & ~1).to_bytes(4, "little")
    raw[120:124] = (zlib.crc32(bytes(raw[:120])) & 0xFFFFFFFF).to_bytes(4, "little")
    open(p, "wb").write(bytes(raw))
    with pytest.raises(CorruptContainerError):
        VaultEngine(filepath=p, embed_dim=D, password=PW)


def test_scrypt_cost_is_never_silently_downgraded(tmp_path, monkeypatch):
    """A KDF that cannot be computed is an error, not a weaker KDF."""
    from nanomem.errors import NanomemError
    p = str(tmp_path / "v.dat")

    def boom(*a, **k):
        raise MemoryError("not enough memory for scrypt")

    monkeypatch.setattr(crypto, "derive_keys", boom)
    with pytest.raises(NanomemError):
        VaultEngine(filepath=p, embed_dim=D, password=PW)


def test_threat_model_states_what_it_does_not_protect():
    """The documented limits must be in the package, not only in a review."""
    # Whitespace-normalised: the text is wrapped at 79 columns, so a phrase can
    # legitimately straddle a newline.
    tm = " ".join(crypto.THREAT_MODEL.lower().split())
    for phrase in ("truncate", "roll it back", "does not hide metadata",
                   "not aes", "not a nist-approved aead", "plaintext mode",
                   "losing the passphrase",
                   # round 3: the one undetected case that was NOT documented --
                   # a damaged final block header is indistinguishable from a
                   # crashed write, and the next append makes the loss permanent.
                   "cannot be told apart from a crashed write",
                   "off by default"):
        assert phrase in tm, phrase
    assert "256-bit" not in tm
    # Every number quoted here must come from a results file in the repo.
    assert "crypto_overhead_v3r3.json" in crypto.THREAT_MODEL
