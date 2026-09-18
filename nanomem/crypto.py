"""
nanomem.crypto
~~~~~~~~~~~~~~
Optional password protection for a v3 vault. Pure standard library (``hashlib``,
``hmac``) plus numpy for the XOR; the ``cryptography`` package is deliberately
NOT a dependency.

Construction (this is the whole thing, stated plainly):

* key derivation  -- ``hashlib.scrypt(password, salt, n=2**16, r=8, p=1)`` -> 32 byte
  master key, then three HMAC-SHA256 sub-keys (``enc``, ``mac``, ``chk``).
* confidentiality -- a keyed SHAKE256 XOF keystream, XORed with the block payload.
  The XOF input is ``b"NM3K" || K_enc || vault_uuid || nonce || seq`` so the stream
  is unique per block, per file generation and per vault.
* integrity       -- HMAC-SHA256 over ``b"NM3T" || vault_uuid || block_header[:96] ||
  ciphertext`` (encrypt-then-MAC). The tag therefore authenticates the nonce, the
  sequence number, the kind, the lengths, the reserved bytes and the file
  generation. The sequence number is checked against the block's position in the
  file during the scan, which is what rejects a replayed or reordered block.
* header binding  -- ``HMAC-SHA256(K_chk, b"NM3H" || header[0:88])``, checked before
  any block is read, which is what turns a wrong password into an immediate
  :class:`~nanomem.errors.WrongPasswordError`.

This is NOT AES and NOT a NIST-approved AEAD mode. It is a keyed sponge stream
with a separate HMAC tag; see ``CIPHER_LABEL_SHAKE`` and :data:`THREAT_MODEL`
below, which states in full what this does and does not protect. Nothing here
claims FIPS validation or an independent audit.

Why SHAKE256 and not HMAC-SHA256 counter mode: a pure-Python HMAC-CTR yields 32
bytes per call (measured ~27-33 MB/s), while one ``shake_256().digest(n)`` call
produces the entire block keystream (measured ~1 GB/s on this machine). The
header reserves ``cipher_id`` bits for an HMAC-CTR suite so a reviewer can have
one without a format bump.
"""

import hashlib
import hmac
import unicodedata

import numpy as np

# --- labels -----------------------------------------------------------------
CIPHER_LABEL_NONE = "none (plaintext)"
CIPHER_LABEL_SHAKE = (
    "SHAKE256-XOF stream + HMAC-SHA256 tag (encrypt-then-MAC), scrypt-derived keys; "
    "stdlib construction, not AES, not a NIST AEAD"
)

THREAT_MODEL = """\
nanomem's password mode is OFF BY DEFAULT. A vault is a plaintext file unless you
give it a passphrase; ``stats()['encrypted_at_rest']`` tells you which one you
have.

With a passphrase, nanomem builds the whole scheme from the Python standard
library: scrypt (n=2^16, r=8, p=1) over an NFKC-normalised passphrase and a
16-byte random per-vault salt derives a 32-byte master key, from which three
independent HMAC-SHA256 sub-keys are taken for encryption, authentication and the
header key-check. Each block's payload is XORed with a SHAKE256 keystream derived
from K_enc, the vault's uuid, a fresh 16-byte random nonce and the block sequence
number, and is then authenticated with HMAC-SHA256 over the whole 96-byte block
header and the ciphertext, bound to the vault uuid. Tags are compared with
hmac.compare_digest before anything is decrypted. This is not AES and not a
NIST-approved AEAD: it is a standard-library keyed-sponge stream with a separate
HMAC tag. It has not been independently audited or FIPS validated.

It protects: the confidentiality of your document text, ids, metadata and
embedding vectors against anyone who obtains the .dat file without the
passphrase; and the integrity of any COMPLETE block against modification,
against substitution from another vault, against re-use of a block from an
earlier generation of the same vault (the uuid is rotated on every rewrite), and
-- because the authenticated sequence number is checked against the block's
position during the scan -- against replay or reordering of blocks within one
generation.

It does not protect against an attacker who can write to the file. They can
TRUNCATE the vault or ROLL IT BACK to an earlier byte-for-byte state with no
error, because the format is append-only with an immutable header and carries no
authenticated commit counter. For the same reason, DAMAGE TO THE LAST BLOCK'S
HEADER CANNOT BE TOLD APART FROM A CRASHED WRITE: those records do not load, and
the next append removes them for good. nanomem no longer does that silently --
the scan raises a RuntimeWarning naming the byte count, the append raises a
second one before it truncates, and ``Vault(..., on_torn_tail="raise")`` refuses
both -- but the ambiguity itself is inherent and is not fixed. Check
``stats()['truncated_tail_bytes']`` after any unexpected shutdown.

``on_torn_tail="raise"`` covers ONLY an unreadable trailing FRAGMENT. It does not
catch a truncation that lands exactly on a block boundary and it does not catch a
rollback: both of those produce a shorter, perfectly well-formed vault that opens
with no warning, no ``integrity_errors`` entry and ``truncated_tail_bytes`` 0.
Likewise, each block's tag binds the vault UUID and not the file path, so two of
your own vaults can be swapped on disk undetectably. And ``on_integrity_error=
"skip"`` is an integrity DOWNGRADE, not a repair: after one unreadable block
header the scan resyncs on the next block's own authenticated sequence number, so
an attacker who can write to the file can excise a block cleanly and every later
block still loads. It now warns; the default ``"raise"`` refuses the file.

It does not hide metadata. The file header (vault size, embedding dimension,
block capacity, KDF parameters, salt, vault uuid, creation time) and every block
header (record count, payload and section lengths, sequence number, nonce, and
the wall-clock time the block was written) are stored in the clear, and the
keystream is LENGTH-PRESERVING: an encrypted vault is byte-for-byte the same size
as the plaintext one (measured: file_bytes_plain == file_bytes_password at 1,190,
5,000 and 40,000 records, evidence/crypto_overhead_v3r3.json). The leak is
therefore EXACT, not approximate. An observer learns exactly how many memories you
hold, exactly how many records and exactly how many bytes of text are in each
batch, and the wall-clock time each batch was written.

It does not protect a running process: the passphrase and the derived keys are in
memory for the lifetime of the vault object and are not locked against swap or
core dumps. ``close()`` wipes the keys and drops the passphrase reference.

It does not defend a weak passphrase beyond the cost of one scrypt guess --
measured 95.13 ms per derivation on this machine, about 10.5 offline guesses per
second per core. That cost is paid ONCE PER OPEN, not per query: measured open
overhead +99.16 ms at 1,190 records, +104.86 ms at 5,000 and +163.37 ms at
40,000 (it grows with the file, because every block's MAC is verified), against a
per-search overhead of -0.006 / +0.005 / +0.001 ms, i.e. inside the noise. All of
these: evidence/crypto_overhead_v3r3.json.

In the default plaintext mode there is NO protection at all: the block trailer is
an unkeyed SHA-256 that detects accidental corruption only and can be recomputed
by anyone. Adding or removing a passphrase rewrites the whole file; the previous
copy is unlinked, not overwritten.

Losing the passphrase means losing the vault. There is no recovery key and no
escrow.
"""

# --- domain separation labels ----------------------------------------------
LABEL_KEYSTREAM = b"NM3K"
LABEL_TRAILER = b"NM3T"
LABEL_HEADER = b"NM3H"

KDF_NONE = 0
KDF_SCRYPT = 1

DEFAULT_SCRYPT_LOG2_N = 16
DEFAULT_SCRYPT_R = 8
DEFAULT_SCRYPT_P = 1
SCRYPT_MAXMEM = 512 * 1024 * 1024


class KeyMaterial:
    """Three 32-byte sub-keys held in mutable buffers so they can be wiped."""

    __slots__ = ("k_enc", "k_mac", "k_chk", "_wiped")

    def __init__(self, k_enc: bytes, k_mac: bytes, k_chk: bytes):
        self.k_enc = bytearray(k_enc)
        self.k_mac = bytearray(k_mac)
        self.k_chk = bytearray(k_chk)
        self._wiped = False

    def wipe(self) -> None:
        """Overwrite the key bytes in place. Idempotent."""
        for buf in (self.k_enc, self.k_mac, self.k_chk):
            for i in range(len(buf)):
                buf[i] = 0
        self._wiped = True

    @property
    def wiped(self) -> bool:
        return self._wiped


def check_password(password):
    """Validate a caller-supplied passphrase. Returns it unchanged, or ``None``.

    Two silent failures this closes, both measured on 3.0.2:

    * ``password=""`` produced a PLAINTEXT vault with no warning, because every
      gate in the stack was ``if password:``. An empty prompt, an empty
      environment variable or an empty config field failed OPEN, to no
      encryption at all, while ``stats()`` truthfully reported
      ``encrypted_at_rest: False`` that nobody was looking at.
    * ``password=b"secret"`` was coerced with ``str()``, so the vault was keyed
      to the literal 9-character text ``b'secret'`` -- a structurally
      predictable passphrase -- rather than to the caller's key material.
    """
    if password is None:
        return None
    if isinstance(password, (bytes, bytearray)):
        raise TypeError(
            "password must be str, not bytes: bytes were coerced with str() and "
            "keyed the vault to the literal text \"b'...'\"; decode it yourself "
            "(password.decode('utf-8')) so the encoding is your choice")
    if not isinstance(password, str):
        raise TypeError(f"password must be str or None, not {type(password).__name__}")
    if not password:
        raise ValueError(
            "password must not be empty: an empty string used to create a "
            "PLAINTEXT vault silently; pass None to mean 'no encryption'")
    return password


def normalize_password(password: str) -> bytes:
    """NFKC-normalise then UTF-8 encode, so visually identical passphrases match."""
    check_password(password)
    return unicodedata.normalize("NFKC", password).encode("utf-8")


def derive_keys(password: str, salt: bytes, log2_n: int = DEFAULT_SCRYPT_LOG2_N,
                r: int = DEFAULT_SCRYPT_R, p: int = DEFAULT_SCRYPT_P) -> KeyMaterial:
    """Derive (K_enc, K_mac, K_chk) from a passphrase. Deterministic."""
    master = bytearray(hashlib.scrypt(
        normalize_password(password),
        salt=bytes(salt),
        n=1 << int(log2_n),
        r=int(r),
        p=int(p),
        maxmem=SCRYPT_MAXMEM,
        dklen=32,
    ))
    try:
        return KeyMaterial(
            hmac.new(master, b"nanomem3:enc", hashlib.sha256).digest(),
            hmac.new(master, b"nanomem3:mac", hashlib.sha256).digest(),
            hmac.new(master, b"nanomem3:chk", hashlib.sha256).digest(),
        )
    finally:
        for i in range(len(master)):        # the master key is not kept anywhere
            master[i] = 0


def kdf_label(kdf_id: int, log2_n: int, r: int, p: int):
    """Human string for ``stats()['kdf']`` (None when the vault is plaintext)."""
    if kdf_id != KDF_SCRYPT:
        return None
    return f"scrypt n={1 << int(log2_n)} r={int(r)} p={int(p)}"


def header_auth(km, header_prefix88: bytes) -> bytes:
    """32 bytes binding the immutable header fields.

    Encrypted: ``HMAC-SHA256(K_chk, b"NM3H" || header[0:88])``.
    Plaintext: ``SHA-256(b"NM3H" || header[0:88])`` (corruption detection only).
    """
    data = LABEL_HEADER + bytes(header_prefix88)
    if km is None:
        return hashlib.sha256(data).digest()
    return hmac.new(km.k_chk, data, hashlib.sha256).digest()


def keystream(km: KeyMaterial, vault_uuid: bytes, nonce: bytes, seq: int, n: int) -> bytes:
    """``n`` bytes of keystream for one block. Unique per (key, uuid, nonce, seq)."""
    if n <= 0:
        return b""
    xof = hashlib.shake_256()
    xof.update(LABEL_KEYSTREAM)
    xof.update(km.k_enc)                    # the bytearray itself: no extra copy
    xof.update(bytes(vault_uuid))
    xof.update(bytes(nonce))
    xof.update(int(seq).to_bytes(8, "little"))
    return xof.digest(n)


def _xor(data: bytes, ks: bytes) -> bytes:
    a = np.frombuffer(data, dtype=np.uint8)
    b = np.frombuffer(ks, dtype=np.uint8)
    return (a ^ b).tobytes()


def block_tag(km, vault_uuid: bytes, header_bytes: bytes, body: bytes) -> bytes:
    """The 32-byte trailer for a block body (ciphertext when encrypted).

    ``header_bytes`` is the complete 96-byte block header.
    """
    if km is None:
        h = hashlib.sha256()
        h.update(LABEL_TRAILER)
        h.update(bytes(vault_uuid))
        h.update(bytes(header_bytes))
        h.update(body)
        return h.digest()
    mac = hmac.new(km.k_mac, None, hashlib.sha256)
    mac.update(LABEL_TRAILER)
    mac.update(bytes(vault_uuid))
    mac.update(bytes(header_bytes))
    mac.update(body)
    return mac.digest()


def plaintext_trailer(vault_uuid: bytes, header_bytes: bytes, payload: bytes) -> bytes:
    """SHA-256 trailer used when the vault is not encrypted."""
    return block_tag(None, vault_uuid, header_bytes, payload)


def seal_block(km: KeyMaterial, vault_uuid: bytes, header_bytes: bytes, seq: int,
               payload: bytes):
    """Encrypt-then-MAC one block payload. Returns ``(ciphertext, tag)``."""
    ks = keystream(km, vault_uuid, bytes(header_bytes)[40:56], seq, len(payload))
    ct = _xor(payload, ks)
    return ct, block_tag(km, vault_uuid, header_bytes, ct)


def open_block(km: KeyMaterial, vault_uuid: bytes, header_bytes: bytes, seq: int,
               ciphertext: bytes, tag: bytes, block_index: int = -1, offset: int = -1,
               verified: bool = False) -> bytes:
    """Verify the tag FIRST, then decrypt. Never decrypts unauthenticated bytes.

    ``verified=True`` says the caller has ALREADY compared this exact tag over
    these exact bytes with ``hmac.compare_digest`` and may skip the second
    computation. Only ``Container._scan_loop`` passes it, and only on the branch
    that just did the comparison itself -- 3.0.1 recomputed the HMAC for every
    block on every open and every reload (measured 2.0 ``block_tag`` calls per
    block), doubling the authentication cost of a password-protected open for
    nothing. The default stays fail-closed.
    """
    from .errors import IntegrityError

    if not verified:
        expected = block_tag(km, vault_uuid, header_bytes, ciphertext)
        if not hmac.compare_digest(expected, bytes(tag)):
            raise IntegrityError(block_index, offset, "block MAC mismatch")
    ks = keystream(km, vault_uuid, bytes(header_bytes)[40:56], seq, len(ciphertext))
    return _xor(ciphertext, ks)


__all__ = [
    "CIPHER_LABEL_NONE", "CIPHER_LABEL_SHAKE", "KDF_NONE", "KDF_SCRYPT",
    "DEFAULT_SCRYPT_LOG2_N", "DEFAULT_SCRYPT_R", "DEFAULT_SCRYPT_P",
    "KeyMaterial", "derive_keys", "kdf_label", "header_auth", "keystream",
    "block_tag", "plaintext_trailer", "seal_block", "open_block",
    "normalize_password", "check_password", "THREAT_MODEL",
]
