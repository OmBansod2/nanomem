"""
nanomem.container
~~~~~~~~~~~~~~~~~
The v3 single-file container: byte layout, scanning, the append protocol and
atomic rewrite.

Layout (all integers little-endian):

* 256-byte file header at offset 0, written once at create/rewrite and NEVER
  rewritten by an append. Magic ``b"NANOMEM3"``. It carries the dimensions, the
  block geometry, the vault uuid (rotated on every rewrite), the KDF parameters
  and salt, a 32-byte header authenticator and a CRC32.
* blocks back to back, each starting at a 64-byte aligned offset: a 96-byte
  plaintext block header, then ``payload_len`` bytes of payload, then a 32-byte
  trailer. The payload is ``[landmarks][vectors][records]``; the trailer is a
  SHA-256 over ``b"NM3T" || vault_uuid || block_header[:96] || payload`` for a
  plaintext vault and an HMAC-SHA256 over the ciphertext for an encrypted one
  (encrypt-then-MAC). The tag covers the WHOLE 96-byte header, reserved bytes
  included -- 3.0.0 authenticated only the first 72, leaving 24 malleable bytes
  inside a header the docs called authenticated.

  COMPATIBILITY: because the tag input changed, a v3 file written by engine
  3.0.0 fails to open under 3.0.1 with an IntegrityError on block 0. 3.0.0 was
  never released, and v2 files migrate normally, so the only affected files are
  vaults created during that development round; re-create them.

Two offsets are derived by scanning and never stored on disk:

* ``valid_end`` -- the exact, UNALIGNED end of the last block this reader has
  validated. The next block starts at ``align_up(valid_end)``; that is the only
  place alignment is applied. (In 3.0.0 ``scan()`` stored the *aligned* value
  here while ``append_block()`` stored the raw one. The reader's ``valid_end``
  was then up to 63 bytes past EOF, so ``modified()`` answered ``"replaced"``
  for every re-opened vault -- a full reload per search -- and a second process
  re-scanned from inside the alignment pad and ``ftruncate``d away another
  process's committed blocks. Both are regression-tested now:
  ``tests/test_container.py::test_reopened_container_is_not_modified`` and
  ``::test_two_processes_appending_lose_nothing``.)
* ``scanned_end`` -- how many bytes of the file have been examined. ``modified()``
  compares ``st_size`` against THIS, so a torn tail that has already been looked
  at does not report "appended" forever.

A partial trailing block is a crash remnant: it is reported through
``truncated_tail_bytes`` and cut off by the next appender, under the exclusive
lock, only after a fresh tail re-scan has proved those bytes are not somebody
else's valid append. A block header that does not parse is treated as the end of
the file ONLY when nothing parseable follows it; if a later block is still
readable the file is corrupt in the middle and that is raised, never silently
truncated.

This class holds NO file descriptor and NO mmap between calls. Every operation
opens the file, works under the appropriate ``flock``, and closes it. That is
what makes ``close()`` trivially idempotent and ``os.remove`` / ``os.replace``
safe on every platform.

THREAT MODEL for the optional password mode: see :data:`nanomem.crypto.THREAT_MODEL`.
In particular this format is append-only with an immutable header and carries no
authenticated commit counter, so an attacker with write access can truncate the
vault or roll it back to an earlier byte-for-byte state undetected. Block
replay, block reorder and mid-file corruption ARE detected (sequence numbers are
inside the MAC and are checked against file position).
"""

import contextlib
import glob
import hashlib
import warnings
import os
import struct
import time
import zlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import crypto
from .errors import (ClosedVaultError, CorruptContainerError, ContainerReplacedError,
                     IntegrityError, NanomemError, NotEncryptedError,
                     PasswordRequiredError, ReadOnlyVaultError, WrongPasswordError)

# --- constants --------------------------------------------------------------
FILE_MAGIC = b"NANOMEM3"
FILE_VERSION = 3
READER_VERSION = 3
FILE_HEADER_SIZE = 256
FILE_HEADER_FMT = "<8sIIIIIIIId16sBBBBI16s32sI"
FILE_HEADER_PACKED = struct.calcsize(FILE_HEADER_FMT)      # 124

BLOCK_MAGIC = b"NMB3"
BLOCK_HEADER_FMT = "<4sIIIIIdQ16sIIII24x"
BLOCK_HEADER_SIZE = struct.calcsize(BLOCK_HEADER_FMT)      # 96
BLOCK_AUTH_LEN = BLOCK_HEADER_SIZE      # the WHOLE block header is authenticated
TRAILER_SIZE = 32
ALIGN = 64

KIND_DATA = 1
KIND_TOMBSTONE_RESERVED = 2                                # reserved for 3.1; never written

FLAG_ENCRYPTED = 1
FLAG_VEC_FP16 = 2
CIPHER_SHIFT = 2
CIPHER_MASK = 0b1100
# Row order in this file is the output of a global spherical k-means, so the
# opt-in cell router does not have to re-cluster it. It lives in the header (and
# so under the header CRC and the keyed header authenticator) because 3.0.2 kept
# it in a Python attribute initialised to False: every OPEN re-ran the one-time
# `compact(recluster=True)` that DECISIONS #4 mandates ONCE, rewriting the whole
# file, rotating `vault_uuid` and changing the inode, so concurrent opens raced
# and 9 of 12 failed with ContainerReplacedError from inside the constructor.
FLAG_RECLUSTERED = 16
CIPHER_NONE = 0
CIPHER_SHAKE_HMAC = 1

BLOCK_CAPACITY = 50
LANDMARKS_PER_BLOCK = 8
DEFAULT_EMBED_DIM = 768

V2_FILE_MAGIC = b"NANOMEM\x00"
TMP_SUFFIX = ".tmp-"
STALE_TMP_SECONDS = 3600
FILE_MODE = 0o600                       # vault files are owner-only
PROBE_CHUNK = 1 << 20                   # bytes read at a time when probing for corruption


def align_up(n: int, a: int = ALIGN) -> int:
    """Round ``n`` up to the next multiple of ``a``."""
    return ((int(n) + a - 1) // a) * a


# --- headers ----------------------------------------------------------------
@dataclass
class FileHeader:
    flags: int = 0
    embed_dim: int = DEFAULT_EMBED_DIM
    block_capacity: int = BLOCK_CAPACITY
    landmarks_per_block: int = LANDMARKS_PER_BLOCK
    created_unix: float = 0.0
    vault_uuid: bytes = b"\x00" * 16
    kdf_id: int = crypto.KDF_NONE
    scrypt_log2_n: int = crypto.DEFAULT_SCRYPT_LOG2_N
    scrypt_r: int = crypto.DEFAULT_SCRYPT_R
    scrypt_p: int = crypto.DEFAULT_SCRYPT_P
    kdf_salt: bytes = b"\x00" * 16
    header_auth: bytes = b"\x00" * 32
    format_version: int = FILE_VERSION

    @property
    def encrypted(self) -> bool:
        return bool(self.flags & FLAG_ENCRYPTED)

    @property
    def fp16(self) -> bool:
        return bool(self.flags & FLAG_VEC_FP16)

    @property
    def reclustered(self) -> bool:
        """True when row order is already the k-means layout the router wants."""
        return bool(self.flags & FLAG_RECLUSTERED)

    @property
    def cipher_id(self) -> int:
        return (self.flags & CIPHER_MASK) >> CIPHER_SHIFT

    @property
    def vector_dtype(self) -> str:
        return "float16" if self.fp16 else "float32"


@dataclass
class BlockMeta:
    index: int = 0
    offset: int = 0
    kind: int = KIND_DATA
    n: int = 0
    m: int = 0
    payload_len: int = 0
    written_unix: float = 0.0
    seq: int = 0
    nonce: bytes = b"\x00" * 16
    vec_section_len: int = 0
    rec_section_len: int = 0
    first_row: int = 0
    block_flags: int = 0

    @property
    def total_len(self) -> int:
        return BLOCK_HEADER_SIZE + self.payload_len + TRAILER_SIZE

    @property
    def block_id(self) -> str:
        return f"blk_{self.seq:08d}"


#: One row per block, exactly the fields :class:`BlockMeta` carries that are not
#: derivable from its position. This is what the arena cache stores so that a
#: cached open does not have to walk the file to rebuild the block list.
BLOCK_TABLE_DTYPE = np.dtype([
    ("offset", "<i8"), ("kind", "<i4"), ("n", "<i4"), ("m", "<i4"),
    ("payload_len", "<i4"), ("written_unix", "<f8"), ("seq", "<i8"),
    ("nonce", "u1", 16), ("vec_section_len", "<i4"), ("rec_section_len", "<i4"),
    ("first_row", "<i8"), ("block_flags", "<i4"), ("_pad", "<i4"),
])


def pack_block_table(blocks) -> np.ndarray:
    """``blocks`` (an iterable of :class:`BlockMeta`) as one structured array."""
    out = np.zeros(len(blocks), dtype=BLOCK_TABLE_DTYPE)
    for i, b in enumerate(blocks):
        out[i]["offset"] = int(b.offset)
        out[i]["kind"] = int(b.kind)
        out[i]["n"] = int(b.n)
        out[i]["m"] = int(b.m)
        out[i]["payload_len"] = int(b.payload_len)
        out[i]["written_unix"] = float(b.written_unix)
        out[i]["seq"] = int(b.seq)
        out[i]["nonce"] = np.frombuffer(bytes(b.nonce).ljust(16, b"\x00")[:16],
                                        dtype=np.uint8)
        out[i]["vec_section_len"] = int(b.vec_section_len)
        out[i]["rec_section_len"] = int(b.rec_section_len)
        out[i]["first_row"] = int(b.first_row)
        out[i]["block_flags"] = int(b.block_flags)
    return out


class BlockTable:
    """``Container.blocks`` when the arena came from a cache.

    A 71,433-row vault has 1,429 blocks, and building 1,429 dataclass instances
    at every open costs more than the whole O(1) open it would sit inside. The
    rows stay as a mapped structured array and a :class:`BlockMeta` is built
    only for the block somebody actually asks about -- which, for the search
    path, is none of them. Appends land in a Python tail, so a cached vault that
    is then written to keeps one flat, ordered block list.
    """

    __slots__ = ("_t", "_tail")

    def __init__(self, table):
        self._t = table
        self._tail = []

    def __len__(self) -> int:
        return int(self._t.shape[0]) + len(self._tail)

    def _meta(self, i: int) -> BlockMeta:
        r = self._t[i]
        return BlockMeta(index=int(i), offset=int(r["offset"]), kind=int(r["kind"]),
                         n=int(r["n"]), m=int(r["m"]),
                         payload_len=int(r["payload_len"]),
                         written_unix=float(r["written_unix"]), seq=int(r["seq"]),
                         nonce=bytes(r["nonce"]),
                         vec_section_len=int(r["vec_section_len"]),
                         rec_section_len=int(r["rec_section_len"]),
                         first_row=int(r["first_row"]),
                         block_flags=int(r["block_flags"]))

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        i = int(i)
        if i < 0:
            i += len(self)
        n = int(self._t.shape[0])
        if 0 <= i < n:
            return self._meta(i)
        j = i - n
        if j < 0 or j >= len(self._tail):
            raise IndexError(i)
        return self._tail[j]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def append(self, meta) -> None:
        self._tail.append(meta)


def _header_prefix88(h: FileHeader) -> bytes:
    return struct.pack(
        "<8sIIIIIIIId16sBBBBI16s",
        FILE_MAGIC, FILE_VERSION, READER_VERSION, FILE_HEADER_SIZE, int(h.flags),
        int(h.embed_dim), int(h.block_capacity), int(h.landmarks_per_block), 0,
        float(h.created_unix), bytes(h.vault_uuid), int(h.kdf_id),
        int(h.scrypt_log2_n), int(h.scrypt_r), int(h.scrypt_p), 0, bytes(h.kdf_salt),
    )


def pack_file_header(h: FileHeader) -> bytes:
    """Serialise a :class:`FileHeader` into exactly 256 bytes."""
    prefix = _header_prefix88(h)
    body = prefix + bytes(h.header_auth)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    packed = body + struct.pack("<I", crc)
    assert len(packed) == FILE_HEADER_PACKED, len(packed)
    return packed + b"\x00" * (FILE_HEADER_SIZE - FILE_HEADER_PACKED)


def unpack_file_header(buf: bytes, check_version: bool = True) -> FileHeader:
    """Parse and validate 256 header bytes. Raises :class:`CorruptContainerError`.

    ``check_version=False`` skips the forward-compatibility checks so the caller
    can verify the keyed header authenticator FIRST. 3.0.1 ran them before it,
    so flipping four bytes of a header and repairing the CRC produced "file
    requires reader version 9, this build is 3" instead of naming the tamper.
    """
    if len(buf) < FILE_HEADER_PACKED:
        raise CorruptContainerError(f"header too short ({len(buf)} bytes)")
    fields = struct.unpack(FILE_HEADER_FMT, buf[:FILE_HEADER_PACKED])
    (magic, version, min_reader, hdr_len, flags, embed_dim, block_cap, lmk,
     _r0, created, uuid, kdf_id, log2n, r, p, _r1, salt, auth, crc) = fields
    if magic != FILE_MAGIC:
        raise CorruptContainerError(f"unrecognised container magic {magic!r}")
    if zlib.crc32(buf[:120]) & 0xFFFFFFFF != crc:
        raise CorruptContainerError("file header CRC32 mismatch")
    if check_version:
        check_header_version(min_reader, hdr_len, flags)
    return FileHeader(flags=flags, embed_dim=embed_dim, block_capacity=block_cap,
                      landmarks_per_block=lmk, created_unix=created, vault_uuid=uuid,
                      kdf_id=kdf_id, scrypt_log2_n=log2n, scrypt_r=r, scrypt_p=p,
                      kdf_salt=salt, header_auth=auth, format_version=version)


def check_header_version(min_reader: int, hdr_len: int, flags: int) -> None:
    """Forward-compatibility checks, run AFTER the header authenticator."""
    if min_reader > READER_VERSION:
        raise CorruptContainerError(
            f"file requires reader version {min_reader}, this build is {READER_VERSION}")
    if hdr_len != FILE_HEADER_SIZE:
        raise CorruptContainerError(f"unexpected header_len {hdr_len}")
    if flags & ~(FLAG_ENCRYPTED | FLAG_VEC_FP16 | FLAG_RECLUSTERED | CIPHER_MASK):
        raise CorruptContainerError(f"unknown header flag bits set ({flags:#x})")


def pack_block_header(*, kind: int, n: int, m: int, payload_len: int, written_unix: float,
                      seq: int, nonce: bytes, vec_section_len: int, rec_section_len: int,
                      block_flags: int) -> bytes:
    """Serialise a 96-byte block header (CRC32 over its first 68 bytes)."""
    body = struct.pack("<4sIIIIIdQ16sIII", BLOCK_MAGIC, BLOCK_HEADER_SIZE, int(kind),
                       int(n), int(m), int(payload_len), float(written_unix), int(seq),
                       bytes(nonce), int(vec_section_len), int(rec_section_len),
                       int(block_flags))
    crc = zlib.crc32(body) & 0xFFFFFFFF
    out = body + struct.pack("<I", crc) + b"\x00" * 24
    assert len(out) == BLOCK_HEADER_SIZE
    return out


def unpack_block_header(buf: bytes):
    """Parse a block header. Returns ``None`` for a bad magic/CRC/kind (= end of file)."""
    if len(buf) < BLOCK_HEADER_SIZE:
        return None
    (magic, hdr_len, kind, n, m, payload_len, written, seq, nonce,
     vec_len, rec_len, bflags, crc) = struct.unpack(BLOCK_HEADER_FMT, buf[:BLOCK_HEADER_SIZE])
    if magic != BLOCK_MAGIC or hdr_len != BLOCK_HEADER_SIZE:
        return None
    if zlib.crc32(buf[:68]) & 0xFFFFFFFF != crc:
        return None
    if kind != KIND_DATA:
        return None
    return BlockMeta(kind=kind, n=n, m=m, payload_len=payload_len, written_unix=written,
                     seq=seq, nonce=nonce, vec_section_len=vec_len,
                     rec_section_len=rec_len, block_flags=bflags)


# --- string tables / records -------------------------------------------------
def pack_string_table(items) -> bytes:
    """``u32 count, u32 offsets[count+1], data``."""
    blobs = [bytes(b) for b in items]
    count = len(blobs)
    offsets = [0]
    acc = 0
    for b in blobs:
        acc += len(b)
        offsets.append(acc)
    out = bytearray()
    out += struct.pack("<I", count)
    out += np.asarray(offsets, dtype=np.uint32).tobytes()
    for b in blobs:
        out += b
    return bytes(out)


def _check_section(buf, off: int, need: int, what: str) -> None:
    """Every offset a records section claims must fit inside the payload.

    Without this, a forged count in a PLAINTEXT vault (whose trailer is an
    unkeyed SHA-256 anybody can recompute) surfaced as numpy's
    ``ValueError: buffer is smaller than requested size`` rather than a
    :class:`~nanomem.errors.NanomemError`. numpy bounds-checks, so this was never
    memory-unsafe -- it was the documented `except NanomemError` guard leaking.
    """
    if off < 0 or need < 0 or off + need > len(buf):
        raise CorruptContainerError(
            f"records section is malformed: {what} claims {need} bytes at offset "
            f"{off} of a {len(buf)}-byte payload")


def unpack_string_table(buf, off: int):
    """Returns ``(list_of_bytes, next_offset)``."""
    _check_section(buf, off, 4, "string table count")
    count = struct.unpack_from("<I", buf, off)[0]
    o = off + 4
    _check_section(buf, o, 4 * (count + 1), f"string table of {count} offsets")
    offsets = np.frombuffer(buf, dtype=np.uint32, count=count + 1, offset=o)
    o += 4 * (count + 1)
    data_start = o
    _check_section(buf, data_start, int(offsets[count]), "string table data")
    if count and not np.all(np.diff(offsets.astype(np.int64)) >= 0):
        raise CorruptContainerError("records section is malformed: string table "
                                    "offsets are not monotonic")
    items = [bytes(buf[data_start + int(offsets[i]):data_start + int(offsets[i + 1])])
             for i in range(count)]
    return items, data_start + int(offsets[count])


def unpack_string_table_spans(buf, off: int):
    """Like :func:`unpack_string_table` but returns ``(n, 2)`` spans instead of copies."""
    _check_section(buf, off, 4, "span table count")
    count = struct.unpack_from("<I", buf, off)[0]
    o = off + 4
    _check_section(buf, o, 4 * (count + 1), f"span table of {count} offsets")
    offsets = np.frombuffer(buf, dtype=np.uint32, count=count + 1, offset=o).astype(np.int64)
    o += 4 * (count + 1)
    _check_section(buf, o, int(offsets[count]), "span table data")
    if count and not np.all(np.diff(offsets) >= 0):
        raise CorruptContainerError("records section is malformed: span table "
                                    "offsets are not monotonic")
    spans = np.empty((count, 2), dtype=np.int64)
    spans[:, 0] = o + offsets[:count]
    spans[:, 1] = o + offsets[1:]
    return spans, o + int(offsets[count])


@dataclass
class RecordsView:
    n: int
    ts: np.ndarray
    rev: np.ndarray
    ids: List[bytes] = field(default_factory=list)
    groups: List[bytes] = field(default_factory=list)
    doc_spans: np.ndarray = None


def encode_records(ts, rev, ids, groups, docs) -> bytes:
    """Pack the records section: counts, columns and three string tables."""
    n = len(ids)
    out = bytearray()
    out += struct.pack("<I", n)
    out += np.asarray(ts, dtype=np.float64).tobytes()
    out += np.asarray(rev, dtype=np.int32).tobytes()
    out += pack_string_table([s.encode("utf-8") if isinstance(s, str) else s for s in ids])
    out += pack_string_table([s.encode("utf-8") if isinstance(s, str) else s for s in groups])
    out += pack_string_table(docs)
    return bytes(out)


def decode_records(buf) -> RecordsView:
    """Parse a records section; ``doc_spans`` are offsets into ``buf`` itself."""
    mv = memoryview(buf)
    _check_section(mv, 0, 4, "record count")
    n = struct.unpack_from("<I", mv, 0)[0]
    o = 4
    _check_section(mv, o, 12 * n, f"{n} timestamp/revision columns")
    ts = np.frombuffer(mv, dtype=np.float64, count=n, offset=o).copy()
    o += 8 * n
    rev = np.frombuffer(mv, dtype=np.int32, count=n, offset=o).copy()
    o += 4 * n
    ids, o = unpack_string_table(mv, o)
    groups, o = unpack_string_table(mv, o)
    spans, o = unpack_string_table_spans(mv, o)
    return RecordsView(n=n, ts=ts, rev=rev, ids=ids, groups=groups, doc_spans=spans)


# --- process / platform helpers ---------------------------------------------
@contextlib.contextmanager
def file_lock(f, exclusive: bool = True):
    """Advisory whole-file lock; a no-op where the platform provides neither API."""
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is not None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        return
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
    if msvcrt is None:
        # Neither API. Every cross-process guarantee in this module's docstring
        # is void here, so say so ONCE rather than pretend.
        _warn_unlocked("this platform provides neither fcntl nor msvcrt")
        yield
        return
    # msvcrt.locking() locks a byte RANGE of a file opened for writing. A shared
    # (reader) lock does not exist, and a file opened "rb" cannot be locked at
    # all, so a Windows reader cannot exclude a concurrent append. 3.0.1 wrapped
    # the whole thing in `except Exception: yield` and ran UNLOCKED without a
    # word -- including on LK_LOCK's ~10 s timeout, i.e. exactly when another
    # process held it. The lock is now taken over a byte PAST the immutable file
    # header (3.0.1 locked byte 0, inside the header it never writes) and a
    # failure is reported.
    if not exclusive:
        _warn_unlocked("msvcrt has no shared lock; this read is not serialised "
                       "against a concurrent append")
        yield
        return
    locked = False
    try:
        f.seek(FILE_HEADER_SIZE - 1)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        locked = True
    except Exception as exc:
        _warn_unlocked(f"msvcrt.locking failed ({exc})")
    try:
        yield
    finally:
        if locked:
            try:
                f.seek(FILE_HEADER_SIZE - 1)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass


_UNLOCKED_WARNED = set()


def _warn_unlocked(reason: str) -> None:
    """Warn once per reason that a vault operation ran without a file lock."""
    if reason in _UNLOCKED_WARNED:
        return
    _UNLOCKED_WARNED.add(reason)
    warnings.warn(
        f"nanomem: proceeding WITHOUT a file lock -- {reason}. Concurrent access "
        f"to one vault from several processes is not safe here.",
        RuntimeWarning, stacklevel=3)


def peak_rss_kb():
    """Process peak RSS in KiB (``ru_maxrss``; bytes on darwin, KiB on Linux)."""
    try:
        import resource
    except ImportError:
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    return int(ru / 1024) if sys.platform == "darwin" else int(ru)


def _darwin_rss_kb():
    """Current RSS in KiB on macOS via ``libproc.proc_pidinfo(PROC_PIDTASKINFO)``.

    ctypes is stdlib, so this adds no dependency. ``None`` if the call fails.
    """
    try:
        import ctypes
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        buf = ctypes.create_string_buffer(96)                  # sizeof(proc_taskinfo)
        n = libproc.proc_pidinfo(os.getpid(), 4, ctypes.c_uint64(0), buf, 96)
        if n < 16:
            return None
        _virt, resident = struct.unpack_from("<QQ", buf.raw, 0)
        return int(resident / 1024)
    except Exception:
        return None


def current_rss_kb():
    """Current resident size in KiB: ``/proc/self/statm`` on Linux, libproc on
    macOS, ``None`` on anything else. Used by ``stats()['process_rss_kb']``."""
    try:
        with open("/proc/self/statm", "r") as f:
            resident_pages = int(f.read().split()[1])
        return int(resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024)
    except Exception:
        pass
    import sys as _sys
    if _sys.platform == "darwin":
        return _darwin_rss_kb()
    return None


def detect_version(filepath: str):
    """``3``, ``2`` or ``None`` (missing / empty). Raises on an unknown magic."""
    try:
        if os.path.getsize(filepath) == 0:
            return None
    except OSError:
        return None
    with open(filepath, "rb") as f:
        magic = f.read(8)
    if magic == FILE_MAGIC:
        return 3
    if magic == V2_FILE_MAGIC:
        return 2
    raise CorruptContainerError(f"unrecognised container magic {magic!r} in {filepath}")


def cleanup_stale_temps(filepath: str, max_age_s: int = STALE_TMP_SECONDS) -> int:
    """Remove our own abandoned ``<path>.tmp-*`` files older than ``max_age_s``."""
    removed = 0
    now = time.time()
    for p in glob.glob(filepath + TMP_SUFFIX + "*"):
        try:
            if now - os.path.getmtime(p) > max_age_s:
                os.remove(p)
                removed += 1
        except OSError:
            pass
    return removed


def temp_path_for(filepath: str) -> str:
    return f"{filepath}{TMP_SUFFIX}{os.getpid()}-{os.urandom(4).hex()}"


def fsync_dir(path: str) -> None:
    """Best-effort directory fsync so a rename is durable (POSIX only)."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# --- container ---------------------------------------------------------------
class Container:
    """All file I/O for one v3 vault file. Holds no descriptor between calls."""

    def __init__(self, filepath: str, *, embed_dim: int = DEFAULT_EMBED_DIM,
                 password=None, vector_dtype: str = "float16",
                 block_capacity: int = BLOCK_CAPACITY,
                 landmarks_per_block: int = LANDMARKS_PER_BLOCK,
                 durable: str = "fsync", on_integrity_error: str = "raise",
                 readonly: bool = False, create: bool = True,
                 on_torn_tail: str = "warn"):
        self.filepath = os.path.abspath(filepath)
        password = crypto.check_password(password)
        self.durable = durable
        self.on_integrity_error = on_integrity_error
        self.on_torn_tail = str(on_torn_tail)
        self.read_only = bool(readonly)
        self.keys: Optional[crypto.KeyMaterial] = None
        self.blocks: List[BlockMeta] = []
        self.valid_end = FILE_HEADER_SIZE      # exact end of the last validated block
        self.scanned_end = FILE_HEADER_SIZE    # how far the file has been examined
        self.integrity_errors: List[int] = []
        self.truncated_tail_bytes = 0
        self.format_version_on_open = FILE_VERSION
        self._ino = None
        self._size = 0
        self._mtime_ns = None
        self._closed = False
        self._n_rows = 0
        self._n_positions = 0          # block POSITIONS consumed (incl. skipped ones)
        self._scan_progress = (FILE_HEADER_SIZE, 0, 0, None, 0)
        self._raw_header = None

        parent = os.path.dirname(self.filepath) or "."
        os.makedirs(parent, exist_ok=True)
        cleanup_stale_temps(self.filepath)

        exists = os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0
        if not exists:
            if not create:
                raise CorruptContainerError(f"no vault at {self.filepath}")
            if self.read_only:
                # 3.0.0 created the file anyway, so `readonly=True` on a missing
                # path left a 256-byte vault behind.
                raise ReadOnlyVaultError(
                    f"no vault at {self.filepath} and it is open read-only")
            self._raw_header = None
            self.header = self._new_header(embed_dim, vector_dtype, block_capacity,
                                           landmarks_per_block, password)
            # `_create_file` re-checks UNDER THE LOCK and returns False when
            # another process won the race, in which case this is an ordinary
            # open of an existing vault.
            exists = not self._create_file(password)
        if exists:
            with open(self.filepath, "rb") as f:
                raw = f.read(FILE_HEADER_SIZE)
            # Kept so that an arena cache can be bound to THIS header by digest
            # (see `snapshot_state`); it is 256 bytes and it is already in hand.
            self._raw_header = bytes(raw)
            self.header = unpack_file_header(raw, check_version=False)
            self._open_keys(password, raw)          # authenticator first ...
            (_m, _v, min_reader, hdr_len, flags) = (
                None, None,
                struct.unpack("<I", raw[12:16])[0], struct.unpack("<I", raw[16:20])[0],
                struct.unpack("<I", raw[20:24])[0])
            check_header_version(min_reader, hdr_len, flags)   # ... then the version
        self._stat()

    # -- creation ----------------------------------------------------------
    @staticmethod
    def _new_header(embed_dim, vector_dtype, block_capacity, landmarks_per_block,
                    password, reclustered: bool = False):
        flags = 0
        if str(vector_dtype).lower() in ("float16", "fp16", "f16", "half"):
            flags |= FLAG_VEC_FP16
        if reclustered:
            flags |= FLAG_RECLUSTERED
        if password:
            flags |= FLAG_ENCRYPTED | (CIPHER_SHAKE_HMAC << CIPHER_SHIFT)
        return FileHeader(
            flags=flags, embed_dim=int(embed_dim), block_capacity=int(block_capacity),
            landmarks_per_block=int(landmarks_per_block), created_unix=time.time(),
            vault_uuid=os.urandom(16),
            kdf_id=crypto.KDF_SCRYPT if password else crypto.KDF_NONE,
            kdf_salt=os.urandom(16) if password else b"\x00" * 16,
        )

    def _derive(self, password, header: FileHeader):
        """Derive at EXACTLY the header's cost. Never silently weakens the KDF.

        3.0.0 walked ``log2_n`` down to 14 on a ValueError/MemoryError. At create
        time that wrote a 4x cheaper KDF into the header without telling anyone;
        at open time it derived with the wrong ``n`` and surfaced as
        ``WrongPasswordError`` for a correct password. Both are now errors.
        """
        try:
            return crypto.derive_keys(password, header.kdf_salt, header.scrypt_log2_n,
                                      header.scrypt_r, header.scrypt_p), header.scrypt_log2_n
        except (ValueError, MemoryError) as exc:
            raise NanomemError(
                f"scrypt n=2^{header.scrypt_log2_n} r={header.scrypt_r} "
                f"p={header.scrypt_p} could not be computed here ({exc}); nanomem "
                f"will not fall back to a weaker KDF") from exc

    def _create_file(self, password) -> bool:
        """Create the vault file. Returns False if somebody else created it first.

        CREATION IS A RACE unless the existence check and the header write happen
        under one lock. 3.0.1 probed with ``os.path.exists`` and then opened
        ``O_CREAT | O_TRUNC``, so two processes starting on the same missing path
        both "created" it: the second truncated the first one's committed blocks
        and wrote a fresh ``vault_uuid``, and because the uuid is inside every
        block tag the file was then permanently unopenable
        (``IntegrityError: block 0 ... trailer mismatch``). Measured: 2 of 4
        trials unopenable, 1 of 10 silently half-empty. That is the FIRST thing a
        multi-process deployment does -- parallel ``nanomem add``, a multi-worker
        server -- so it is now O_CREAT WITHOUT O_TRUNC, flock, re-check the size,
        and only then write. Regression:
        tests/test_reopen_and_concurrency.py::test_concurrent_creation_of_one_vault.
        """
        if password:
            self.keys, used = self._derive(password, self.header)
            self.header.scrypt_log2_n = used
        self.header.header_auth = crypto.header_auth(self.keys, _header_prefix88(self.header))
        blob = pack_file_header(self.header)
        self._raw_header = bytes(blob)
        fd = os.open(self.filepath, os.O_RDWR | os.O_CREAT, FILE_MODE)
        with os.fdopen(fd, "r+b") as f:
            with file_lock(f, exclusive=True):
                if os.fstat(f.fileno()).st_size >= FILE_HEADER_SIZE:
                    if self.keys is not None:
                        self.keys.wipe()          # the winner's salt is different
                        self.keys = None
                    return False
                f.seek(0)
                f.write(blob)
                f.truncate(len(blob))
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
        fsync_dir(self.filepath)
        self.valid_end = FILE_HEADER_SIZE
        self.scanned_end = FILE_HEADER_SIZE
        return True

    def _open_keys(self, password, raw_header: bytes):
        """Verify the header authenticator, then act on the ENCRYPTED flag.

        The order matters: 3.0.0 branched on the flag first, so clearing the flag
        told a user with the right password that their vault "is not encrypted"
        instead of that it had been tampered with.
        """
        import hmac as _hmac
        h = self.header
        if not h.encrypted:
            expected = crypto.header_auth(None, raw_header[:88])
            if not _hmac.compare_digest(expected, bytes(h.header_auth)):
                raise CorruptContainerError("file header authenticator mismatch")
            if password:
                # A NanomemError subclass: `errors` promises that every error this
                # module raises can be caught with one `except NanomemError`, and
                # 3.0.1 broke that promise exactly here -- the CLI printed a raw
                # traceback for `--password` against a plaintext vault, and a
                # caller guarding an attacker-stripped header missed it entirely.
                raise NotEncryptedError(
                    "vault is not encrypted; use rekey(password) to encrypt it")
            return
        if h.cipher_id != CIPHER_SHAKE_HMAC:
            # `cipher_id` and `kdf_id` were parsed and never compared, so a file
            # written by a future build under the reserved HMAC-CTR suite would be
            # decrypted with the wrong construction and surface as "block MAC
            # mismatch" rather than "unsupported cipher". Both fields are inside
            # `header_auth`, so this is a forward-compatibility guard, not a
            # defence against an attacker.
            raise CorruptContainerError(
                f"unsupported cipher id {h.cipher_id} (this build implements "
                f"{CIPHER_SHAKE_HMAC}: SHAKE256-XOF + HMAC-SHA256)")
        if h.kdf_id != crypto.KDF_SCRYPT:
            raise CorruptContainerError(
                f"unsupported kdf id {h.kdf_id} (this build implements "
                f"{crypto.KDF_SCRYPT}: scrypt)")
        if not password:
            raise PasswordRequiredError(
                f"{self.filepath} is password-protected; pass password=...")
        self.keys, _ = self._derive(password, h)
        expected = crypto.header_auth(self.keys, raw_header[:88])
        if not _hmac.compare_digest(expected, bytes(h.header_auth)):
            raise WrongPasswordError(
                "wrong password for this vault (or its header was tampered with)")

    # -- stat / change detection -------------------------------------------
    def _stat(self):
        try:
            st = os.stat(self.filepath)
        except OSError:
            self._ino, self._size = None, 0
            return None
        self._ino, self._size = st.st_ino, st.st_size
        self._mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
        return st

    def modified(self) -> str:
        """One ``os.stat``: ``"same" | "appended" | "replaced" | "missing"``.

        Compared against ``scanned_end`` (bytes examined), NOT ``valid_end`` (end
        of the last block), because the two differ whenever the file ends in a
        torn tail -- and because ``valid_end`` used to be rounded up past EOF,
        which made every re-opened vault report "replaced" forever.
        """
        try:
            st = os.stat(self.filepath)
        except OSError:
            return "missing"
        if self._ino is not None and st.st_ino != self._ino:
            return "replaced"
        if st.st_size < self.scanned_end:
            return "replaced"
        if st.st_size > self.scanned_end:
            return "appended"
        mt = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
        if self._mtime_ns is not None and mt != self._mtime_ns:
            # Same size, same inode, different write time. Almost always this is
            # a crash remnant of exactly the length another process then replaced
            # ("same" forever, and a reader that never sees the new records);
            # a full reload is the only answer that is correct either way.
            return "replaced"
        return "same"

    @property
    def next_seq(self) -> int:
        """The ``seq`` the next appended block must carry (= its position)."""
        return self._n_positions

    # -- scanning -----------------------------------------------------------
    def _decode_payload(self, meta: BlockMeta, payload: bytes):
        D = self.header.embed_dim
        fp16 = bool(meta.block_flags & FLAG_VEC_FP16)
        lm_bytes = meta.m * D * 4
        vec_bytes = meta.n * D * (2 if fp16 else 4)
        if lm_bytes + vec_bytes + meta.rec_section_len != len(payload):
            raise IntegrityError(meta.index, meta.offset, "payload section lengths disagree")
        lm = (np.frombuffer(payload, dtype=np.float32, count=meta.m * D, offset=0)
              .reshape(meta.m, D).copy()) if meta.m else None
        vdt = np.float16 if fp16 else np.float32
        vec = (np.frombuffer(payload, dtype=vdt, count=meta.n * D, offset=lm_bytes)
               .reshape(meta.n, D).astype(np.float32))
        rec_bytes = bytes(payload[lm_bytes + vec_bytes:])
        return lm, vec, rec_bytes

    def _find_next_block(self, f, pos: int, size: int):
        """Offset of the next parseable block header after ``pos``, or ``None``.

        This is what separates "the file was cut off mid-write" (nothing
        parseable follows, so stop) from "a block header is corrupt" (a later
        block still reads, so the file is damaged in the middle and 3.0.0's
        ``break`` silently dropped every record after it).
        """
        scan_from = align_up(pos + 1)
        while scan_from + BLOCK_HEADER_SIZE <= size:
            f.seek(scan_from)
            chunk = f.read(min(PROBE_CHUNK, size - scan_from))
            if len(chunk) < BLOCK_HEADER_SIZE:
                return None
            k = 0
            while True:
                j = chunk.find(BLOCK_MAGIC, k)
                if j < 0:
                    break
                off = scan_from + j
                if off % ALIGN == 0 and off + BLOCK_HEADER_SIZE <= size:
                    if j + BLOCK_HEADER_SIZE <= len(chunk):
                        hdr = chunk[j:j + BLOCK_HEADER_SIZE]
                    else:
                        f.seek(off)
                        hdr = f.read(BLOCK_HEADER_SIZE)
                    if unpack_block_header(hdr) is not None:
                        return off
                k = j + 1
            scan_from += max(ALIGN, len(chunk) - BLOCK_HEADER_SIZE)
        return None

    def row_count_hint(self, f, pos: int, size: int) -> int:
        """Rows in ``[pos, size)``, read from block HEADERS only. A hint, never a fact.

        Walks the block chain with one 96-byte read per block and no payload,
        no HMAC and no decryption: ~1,429 reads for a 148 MB vault, measured at
        3.2 ms against a 178 ms reopen. It stops at the first header that does
        not parse, so a damaged or torn file simply yields a smaller number and
        the arena falls back to incremental growth. Nothing is authenticated
        here and nothing is loaded from it -- the only consequence of a wrong
        answer is an allocation of the wrong size.

        THAT IS STILL AN ALLOCATION, so the count is capped by what the bytes
        could possibly hold. ``n`` lives in a plaintext header whose CRC anyone
        can recompute, and the whole point of this method is that it runs BEFORE
        the block is authenticated; an unbounded ``n`` would let a one-byte edit
        of a 148 MB file ask the arena for an arbitrary number of rows. A block
        carries its vectors inside its own payload and fp16 is the narrowest
        vector this format stores, so a block cannot hold more than
        ``payload_len // (embed_dim * 2)`` rows -- and ``payload_len`` is already
        pinned to the real file size by the ``end > size`` check below. The cap
        is ~30x looser than any block a writer produces (50 rows against a bound
        near 1,500 for a full 768-d block), so it never binds on a real vault;
        it only refuses the absurd. Regression:
        tests/test_arena_residency.py::test_row_count_hint_is_capped_by_the_bytes_on_disk.
        """
        total = 0
        row_floor = max(1, int(self.header.embed_dim) * 2)
        try:
            here = int(pos)
            while here + BLOCK_HEADER_SIZE <= size:
                f.seek(here)
                meta = unpack_block_header(f.read(BLOCK_HEADER_SIZE))
                if meta is None:
                    break
                end = here + BLOCK_HEADER_SIZE + meta.payload_len + TRAILER_SIZE
                if end > size or meta.n < 0:
                    break
                total += min(int(meta.n), int(meta.payload_len) // row_floor)
                here = align_up(end)
        except (OSError, ValueError, struct.error):
            return 0
        finally:
            try:
                f.seek(pos)
            except OSError:
                pass
        return total

    def _scan_stream(self, f, sink) -> int:
        """The ONE scan implementation. ``f`` must already be locked (shared is
        enough for a read-only scan, exclusive for the append path).

        Sets ``valid_end`` to the exact end of the last validated block and
        ``scanned_end`` to the file size, so a reader and a writer always agree.
        """
        size = os.fstat(f.fileno()).st_size
        added = 0
        pos = align_up(self.valid_end)
        last_end = self.valid_end
        index = len(self.blocks)
        positions = self._n_positions
        first_row = self._n_rows
        resync = False
        import hmac as _hmac
        # ARENA RESIDENCY. Tell the sink how many rows are coming before the
        # first one arrives, so it allocates each row-indexed array once instead
        # of growing into it. This is a HINT from unauthenticated block headers:
        # it is never trusted for anything (the scan below still authenticates
        # every block and the sink still grows if the hint was short), it only
        # decides an allocation size. Measured at 71,433 rows: 643.6 MB of reopen
        # RSS becomes 286.0 MB (evidence/memory_results.json,
        # `summary.reopen_only.n71433`; the second run in `replicate` reads
        # 643.0 -> 285.3).
        reserve = getattr(sink, "reserve", None)
        if reserve is not None:
            hint = self.row_count_hint(f, pos, size)
            if hint > 0:
                try:
                    reserve(first_row + hint)
                except Exception:
                    pass
        try:
            return self._scan_loop(f, sink, size, pos, last_end, index, positions,
                                   first_row, resync, _hmac)
        except BaseException:
            # COMMIT WHAT WAS VALIDATED, THEN RE-RAISE. Blocks accepted before the
            # failure are already in ``self.blocks`` and already in the sink, so
            # leaving the offsets pointing at the pre-scan position made every
            # retry re-scan and re-append them: three ``count()`` calls after one
            # corrupted block turned 4 rows into 16, with 8 duplicate ids. The
            # committed state is exactly the prefix that succeeded, so a retry
            # re-raises on the same block instead of growing without bound.
            self._commit_scan(f, self._scan_progress)
            raise

    def _commit_scan(self, f, prog) -> None:
        last_end, positions, first_row, size, tail = prog
        self.valid_end = last_end
        # On the error path ``scanned_end`` is pulled back to the last good block
        # too, so ``modified()`` keeps answering "appended" and every subsequent
        # read re-raises instead of quietly serving a truncated vault.
        self.scanned_end = last_end if size is None else size
        self.truncated_tail_bytes = tail
        self._n_rows = first_row
        self._n_positions = positions
        try:
            st = os.fstat(f.fileno())
            self._ino, self._size = st.st_ino, st.st_size
            self._mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
        except OSError:
            pass

    def _scan_loop(self, f, sink, size, pos, last_end, index, positions, first_row,
                   resync, _hmac) -> int:
        added = 0
        # (last_end, positions, first_row, scanned_end-or-None, tail) -- what
        # `_commit_scan` writes back, kept current so the error path can commit
        # the prefix that really was validated.
        self._scan_progress = (last_end, positions, first_row, None, 0)
        while pos + BLOCK_HEADER_SIZE <= size:
            f.seek(pos)
            hdr_bytes = f.read(BLOCK_HEADER_SIZE)
            meta = unpack_block_header(hdr_bytes)
            if meta is None:
                nxt = self._find_next_block(f, pos, size)
                if nxt is None:
                    break                                   # torn tail: stop here
                if self.on_integrity_error == "raise":
                    raise IntegrityError(
                        index, pos, "unreadable block header with valid blocks after it")
                self.integrity_errors.append(index)
                index += 1
                positions += 1
                resync = True
                last_end = nxt
                pos = nxt
                self._scan_progress = (last_end, positions, first_row, None, 0)
                warnings.warn(
                    f"nanomem: skipping an unreadable block at offset {pos} in "
                    f"{self.filepath} and RESYNCING to the next block's own "
                    f"sequence number: every later block then loads normally, so "
                    f"an attacker with write access can excise a block cleanly. "
                    f"on_integrity_error='skip' is an INTEGRITY DOWNGRADE; the "
                    f"default 'raise' refuses the file instead.",
                    RuntimeWarning, stacklevel=2)
                continue
            end = pos + BLOCK_HEADER_SIZE + meta.payload_len + TRAILER_SIZE
            if end > size:
                break                                       # torn tail
            body = f.read(meta.payload_len)
            trailer = f.read(TRAILER_SIZE)
            meta.index = index
            meta.offset = pos
            expected = crypto.block_tag(self.keys, self.header.vault_uuid,
                                        hdr_bytes[:BLOCK_AUTH_LEN], body)
            if not _hmac.compare_digest(expected, trailer):
                if self.on_integrity_error == "raise":
                    raise IntegrityError(index, pos, "trailer mismatch")
                self.integrity_errors.append(index)
                index += 1
                positions += 1
                last_end = end
                pos = align_up(end)
                self._scan_progress = (last_end, positions, first_row, None, 0)
                continue
            if resync:
                positions = int(meta.seq)                    # resync after damage
                resync = False
            if int(meta.seq) != positions:
                # The header is authenticated, so this is not corruption: the
                # block is a replay of, or has been swapped with, another block
                # of the same generation. Sequence numbers are inside the MAC
                # precisely so this is cheap to check.
                if self.on_integrity_error == "raise":
                    raise IntegrityError(
                        index, pos,
                        f"block sequence {int(meta.seq)} at position {positions} "
                        f"(replayed or reordered block)")
                self.integrity_errors.append(index)
                index += 1
                positions += 1
                last_end = end
                pos = align_up(end)
                self._scan_progress = (last_end, positions, first_row, None, 0)
                continue
            payload = body if self.keys is None else crypto.open_block(
                self.keys, self.header.vault_uuid, hdr_bytes[:BLOCK_AUTH_LEN], meta.seq,
                body, trailer, index, pos, verified=True)   # compare_digest above
            lm, vec, rec_bytes = self._decode_payload(meta, payload)
            meta.first_row = first_row
            sink.add_block(meta, lm, vec, rec_bytes)
            self.blocks.append(meta)
            first_row += meta.n
            index += 1
            positions += 1
            added += 1
            last_end = end
            pos = align_up(end)
            self._scan_progress = (last_end, positions, first_row, None, 0)
        tail = max(0, size - last_end)
        self._scan_progress = (last_end, positions, first_row, size, tail)
        self._commit_scan(f, self._scan_progress)
        if tail and self.on_torn_tail != "ignore":
            # A tail is EITHER a crash mid-write OR damage to the last block's
            # header, and an append-only format with no commit counter cannot
            # tell them apart -- so nanomem says so instead of dropping the
            # records quietly. 3.0.1 reported this only through
            # `stats()['truncated_tail_bytes']`, which nobody reads: a one-bit
            # flip in the last block header took 10 of 50 records away with
            # `integrity_errors == []` and no warning at all.
            msg = (f"{self.filepath}: {tail} trailing bytes are not a readable "
                   f"block. They are either a crashed write or damage to the last "
                   f"block's header -- this format cannot tell those apart. Those "
                   f"records are NOT loaded, and the next append will remove them.")
            if self.on_torn_tail == "raise":
                raise IntegrityError(len(self.blocks), last_end, msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return added

    def scan(self, sink, from_offset=None) -> int:
        """Scan new blocks into ``sink`` under a shared lock. Returns how many.

        ``sink.add_block(meta, landmarks, vectors_f32, rec_bytes)`` is called for
        every complete, authenticated DATA block in file order.
        """
        if not os.path.exists(self.filepath):
            return 0
        if from_offset is not None:
            self.valid_end = int(from_offset)
        with open(self.filepath, "rb") as f:
            with file_lock(f, exclusive=False):
                return self._scan_stream(f, sink)

    # -- arena cache binding ------------------------------------------------
    def read_trailer(self, end_offset: int):
        """The 32 authenticator bytes that end the block finishing at ``end_offset``.

        One ``pread`` at a known offset, so it is O(1) whatever the file holds.
        It is the cheapest thing that says "this vault still ends where the
        cache thinks it ends, with the same last block": the trailer is an
        HMAC over that block's whole header and payload, so a cache written
        against a different final block, a truncated file or a file whose tail
        was rewritten cannot match it.
        """
        end_offset = int(end_offset)
        if end_offset <= FILE_HEADER_SIZE:
            return b""
        try:
            with open(self.filepath, "rb") as f:
                f.seek(end_offset - TRAILER_SIZE)
                got = f.read(TRAILER_SIZE)
        except OSError:
            return None
        return got if len(got) == TRAILER_SIZE else None

    def snapshot_state(self) -> dict:
        """Everything an arena cache needs to bind itself to this container.

        Called after a scan, with the container describing exactly the bytes the
        arena was built from.
        """
        blocks = list(self.blocks)
        trailer = self.read_trailer(self.valid_end)
        return {
            "vault_path": self.filepath,
            "vault_uuid": bytes(self.header.vault_uuid).hex(),
            "header_sha256": hashlib.sha256(self._raw_header or b"").hexdigest(),
            "valid_end": int(self.valid_end),
            "scanned_end": int(self.scanned_end),
            "size": int(self._size or 0),
            "n_positions": int(self._n_positions),
            "n_rows": int(self._n_rows),
            "truncated_tail_bytes": int(self.truncated_tail_bytes),
            "integrity_errors": [int(i) for i in self.integrity_errors],
            "block_capacity": int(self.header.block_capacity),
            "landmarks_per_block": int(self.header.landmarks_per_block),
            "last_trailer": (trailer or b"").hex(),
            "block_table": pack_block_table(blocks),
            # So that a cache which points AT this file rather than copying it
            # records the width the pointers are in. See nanomem.arena.
            "vault_fp16": bool(self.header.fp16),
            "vault_encrypted": bool(self.header.encrypted),
        }

    def accept_snapshot(self, snap) -> bool:
        """Bind ``snap`` to this vault and adopt its scan state, or refuse it.

        THE WHOLE VALIDATION IS HERE, and all of it is O(1):

          * the cache names this vault's uuid, and the SHA-256 of this vault's
            256-byte file header (which covers the uuid, the embed dim, the
            flags, the KDF parameters and the creation time -- so a cache
            written against a re-created or re-keyed vault is refused);
          * the geometry it was written for is the geometry this build reads
            (embed dim, block capacity, landmarks per block);
          * the vault is at least as long as the prefix the cache covers, and
            that prefix still ends in the same block, proved by reading that
            block's 32-byte trailer;
          * its own row and block counts are self-consistent.

        What it does NOT do is re-read the blocks and recompute their trailers:
        see :class:`nanomem.arena.ArenaSnapshot` for why that is a knob
        (``arena_cache="verify"``) and not a default, and for why "authenticate"
        is the wrong word for it in a plaintext vault, which is the only kind
        that ever gets a cache.

        ONE MORE CHECK LIVES OUTSIDE THIS METHOD, so "the whole validation is
        here" is no longer quite true: the engine also compares the vault's
        current (size, mtime) against the pair the cache recorded, and refuses a
        vault that was rewritten in place
        (:func:`nanomem.arena.vault_changed_since_cache`). It is outside because
        it needs the per-mode reason string the engine reports, not because it
        is optional.

        On acceptance the container adopts the cache's scan position, so the
        scan that follows reads only the blocks appended since -- which is what
        makes "another process appended while we were away" cost the appended
        rows and nothing else.
        """
        h = snap.header
        try:
            if h.get("vault_uuid") != bytes(self.header.vault_uuid).hex():
                return False
            if h.get("source_header_sha256") != hashlib.sha256(
                    self._raw_header or b"").hexdigest():
                return False
            if int(h.get("embed_dim", -1)) != int(self.header.embed_dim):
                return False
            if int(h.get("block_capacity", -1)) != int(self.header.block_capacity):
                return False
            if int(h.get("landmarks_per_block", -1)) != int(self.header.landmarks_per_block):
                return False
            valid_end = int(h["source_valid_end"])
            n_blocks = int(h["n_blocks"])
            n_rows = int(h["n_rows"])
            if valid_end < FILE_HEADER_SIZE or n_blocks < 0 or n_rows < 0:
                return False
            table = snap.array("blocks")
            if table.dtype != BLOCK_TABLE_DTYPE or table.shape != (n_blocks,):
                return False
            if n_blocks and int(table[-1]["first_row"]) + int(table[-1]["n"]) != n_rows:
                return False
            size = os.path.getsize(self.filepath)
            if size < valid_end:
                return False
            want = bytes.fromhex(h.get("source_last_trailer") or "")
            got = self.read_trailer(valid_end)
            if got is None or got != want:
                return False
        except (KeyError, ValueError, TypeError, OSError):
            return False

        self.blocks = BlockTable(table)
        self.valid_end = valid_end
        self.scanned_end = min(int(h.get("source_scanned_end", valid_end)), size)
        if self.scanned_end < valid_end:
            self.scanned_end = valid_end
        self._n_rows = n_rows
        self._n_positions = int(h.get("n_positions", n_blocks))
        self.truncated_tail_bytes = 0
        self.integrity_errors = [int(i) for i in (h.get("integrity_errors") or [])]
        return True

    def reauthenticate(self) -> bool:
        """Recompute every cached block's trailer against the file. ``arena_cache="verify"``.

        This is the check a cached open gives up, bought back at its real price:
        it reads every byte of every block the cache covers and recomputes its
        tag, exactly as a scan does, and skips only the decode, the record parse
        and the per-row interning that a scan also pays. It is O(bytes) and it
        is not the default; measured cost at each corpus size is in
        evidence/reopen_results.json.

        THE NAME OVERSTATES IT and is kept only because it is called from
        another module. With a passphrase the tag is an HMAC and this really is
        re-authentication. WITHOUT one it is an unkeyed SHA-256 that anyone who
        can write the file can recompute, so what this buys is corruption
        detection -- and a vault with a passphrase is refused an arena cache
        outright, so in every case this method is actually reached for, the
        unkeyed branch is the one running.
        """
        import hmac as _hmac
        try:
            with open(self.filepath, "rb") as f:
                with file_lock(f, exclusive=False):
                    for meta in self.blocks:
                        f.seek(int(meta.offset))
                        hdr_bytes = f.read(BLOCK_HEADER_SIZE)
                        if unpack_block_header(hdr_bytes) is None:
                            return False
                        body = f.read(int(meta.payload_len))
                        trailer = f.read(TRAILER_SIZE)
                        if len(body) != int(meta.payload_len) or len(trailer) != TRAILER_SIZE:
                            return False
                        expected = crypto.block_tag(self.keys, self.header.vault_uuid,
                                                    hdr_bytes[:BLOCK_AUTH_LEN], body)
                        if not _hmac.compare_digest(expected, trailer):
                            return False
        except OSError:
            return False
        return True

    def reset_scan_state(self):
        self.blocks = []
        self.valid_end = FILE_HEADER_SIZE
        self.scanned_end = FILE_HEADER_SIZE
        self.integrity_errors = []
        self.truncated_tail_bytes = 0
        self._n_rows = 0
        self._n_positions = 0

    # -- locking ------------------------------------------------------------
    def _assert_same_file(self, f) -> None:
        """Raise unless the open descriptor is still the file at ``filepath``.

        Checking only ``fstat(fd)`` against the remembered inode is not enough:
        after another process ``os.replace``d the vault, this descriptor still
        points at the (unlinked) old inode and every field still matches, so a
        write would land in a file nobody can ever open again. That is how a
        concurrent rebuild silently swallowed a successful ``add_fact``.
        """
        st = os.fstat(f.fileno())
        try:
            path_st = os.stat(self.filepath)
        except OSError:
            raise ContainerReplacedError(f"{self.filepath} disappeared")
        if path_st.st_ino != st.st_ino:
            raise ContainerReplacedError(f"{self.filepath} was replaced")
        if self._ino is not None and st.st_ino != self._ino:
            raise ContainerReplacedError(f"{self.filepath} was replaced")
        f.seek(0)
        if f.read(FILE_HEADER_SIZE)[48:64] != bytes(self.header.vault_uuid):
            raise ContainerReplacedError(f"{self.filepath} was replaced")

    @contextlib.contextmanager
    def exclusive(self, sink=None):
        """Hold the vault's exclusive lock across a multi-step operation.

        Used by :meth:`nanomem.engine.VaultEngine.replace_all` so that a rewrite
        and a concurrent append cannot interleave. Yields ``(f, state)`` with the
        same vocabulary as :meth:`modified`: ``"same"``, ``"appended"`` (and the
        new blocks have been read into ``sink`` if one was given) or
        ``"replaced"``.
        """
        with open(self.filepath, "r+b") as f:
            with file_lock(f, exclusive=True):
                state = "same"
                try:
                    self._assert_same_file(f)
                except ContainerReplacedError:
                    state = "replaced"
                if state == "same":
                    size = os.fstat(f.fileno()).st_size
                    if size < self.scanned_end:
                        state = "replaced"
                    elif size > self.scanned_end:
                        state = "appended"
                        if sink is not None:
                            self._scan_stream(f, sink)
                yield f, state

    # -- append -------------------------------------------------------------
    def append_block(self, blob, sink) -> BlockMeta:
        """Append one block. See the module docstring for the protocol.

        ``blob`` is either finished bytes or a callable ``make_blob(seq)``. Prefer
        the callable: a block's ``seq`` must equal its POSITION in the file, and
        the position is only known once the exclusive lock is held and any other
        writer's blocks have been read back -- so a blob sealed before the lock
        carries a seq that is already stale under concurrency. (``seq`` is inside
        the MAC, which is what makes replay and reordering detectable, so it
        cannot simply be patched up afterwards.)
        """
        self._assert_open("append to")
        if self.read_only:
            raise ReadOnlyVaultError(f"{self.filepath} is open read-only")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError("vault file was removed")
        with open(self.filepath, "r+b") as f:
            with file_lock(f, exclusive=True):
                self._assert_same_file(f)
                if os.fstat(f.fileno()).st_size > self.scanned_end:
                    self._scan_stream(f, sink)           # absorb other writers' blocks
                if callable(blob):
                    blob = blob(self._n_positions)
                size = os.fstat(f.fileno()).st_size
                if size > self.valid_end:
                    # No other appender can be mid-write -- we hold the exclusive
                    # lock and the scan above consumed everything that parses --
                    # so these bytes are unreadable. Discarding them is the only
                    # way forward, but it is PERMANENT, so it is never silent.
                    lost = size - self.valid_end
                    if self.on_torn_tail == "raise":
                        raise IntegrityError(
                            len(self.blocks), self.valid_end,
                            f"refusing to append over {lost} unreadable trailing "
                            f"bytes of {self.filepath} (on_torn_tail='raise')")
                    if self.on_torn_tail != "ignore":
                        warnings.warn(
                            f"{self.filepath}: discarding {lost} unreadable "
                            f"trailing bytes before appending; those records are "
                            f"gone for good.", RuntimeWarning, stacklevel=2)
                    os.ftruncate(f.fileno(), self.valid_end)
                    self.truncated_tail_bytes = 0
                start = align_up(self.valid_end)
                f.seek(self.valid_end)
                f.write(b"\x00" * (start - self.valid_end))
                f.write(blob)
                f.flush()
                self._sync(f)
                self.valid_end = start + len(blob)
                self.scanned_end = self.valid_end
                st = os.fstat(f.fileno())
                self._ino, self._size = st.st_ino, st.st_size
                self._mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
        meta = unpack_block_header(blob[:BLOCK_HEADER_SIZE])
        meta.index = len(self.blocks)
        meta.offset = start
        meta.first_row = self._n_rows
        self.blocks.append(meta)
        self._n_rows += meta.n
        self._n_positions += 1
        return meta

    def _sync(self, f):
        if self.durable == "none":
            return
        try:
            if self.durable == "full" and hasattr(os, "fcntl"):
                import fcntl
                fcntl.fcntl(f.fileno(), getattr(fcntl, "F_FULLFSYNC", 51))
            else:
                os.fsync(f.fileno())
        except (OSError, AttributeError, ImportError):
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    def sync(self):
        """Force one fsync of the whole file (used by ``durable='none'`` flushes)."""
        if self.durable == "none" or not os.path.exists(self.filepath):
            return
        with open(self.filepath, "r+b") as f:
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    # -- rewrite ------------------------------------------------------------
    def write_new_file(self, tmp_path: str, header: FileHeader, keys, blob_iter) -> int:
        """Write a complete new container to ``tmp_path``. Returns bytes written."""
        self._assert_open("rewrite")
        header = FileHeader(**{**header.__dict__})
        header.header_auth = crypto.header_auth(keys, _header_prefix88(header))
        written = 0
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
        with os.fdopen(fd, "wb") as f:
            f.write(pack_file_header(header))
            written = FILE_HEADER_SIZE
            for blob in blob_iter:
                pad = align_up(written) - written
                if pad:
                    f.write(b"\x00" * pad)
                    written += pad
                f.write(blob)
                written += len(blob)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        return written

    #: Windows refuses `os.replace` onto a path anything still has open, so the
    #: exclusive handle has to be let go first. A module-level flag rather than
    #: an inline `os.name` check so a test can exercise that path here.
    _REPLACE_NEEDS_CLOSED_HANDLE = os.name == "nt"

    def replace_with(self, tmp_path: str, retries: int = 5, holding=None) -> None:
        """``os.replace`` the temp file over this one.

        ``holding`` is the caller's open handle ON THE TARGET -- the one
        :meth:`exclusive` yields. POSIX does not care: `os.replace` swaps the
        directory entry and the open fd keeps the old inode, which is what makes
        the rewrite atomic for a concurrent reader. Windows refuses outright
        with `PermissionError [WinError 5]`, and it refused every time: 18 of
        the 50 failures in the first Windows CI run were `replace_all` and
        `compact` hitting this one line, with a retry loop that could never help
        because the handle was held for the whole operation, not momentarily.

        So on Windows the handle is closed first. The cost is real and worth
        naming: that handle carries the exclusive lock, so between closing it
        and the replace landing there is a window in which another writer could
        append to a file that is about to be replaced -- and `_assert_same_file`
        in the next `exclusive` is what catches that, raising
        `ContainerReplacedError` rather than losing the append silently. POSIX
        keeps the lock for the whole operation and has no such window.
        """
        if holding is not None and self._REPLACE_NEEDS_CLOSED_HANDLE:
            try:
                holding.close()
            except OSError:
                pass
        last = None
        for attempt in range(retries):
            try:
                os.replace(tmp_path, self.filepath)
                fsync_dir(self.filepath)
                return
            except PermissionError as exc:          # a mapping someone else holds
                last = exc
                time.sleep(0.02 * (2 ** attempt))
        raise last

    # -- lifecycle ----------------------------------------------------------
    def _assert_open(self, what: str) -> None:
        """Every write must go through this. ``close()`` wipes the keys, and 3.0.1
        then let a later write through with ``keys=None``: on a PASSWORD-PROTECTED
        vault ``build_block_blob`` happily wrote the record in CLEARTEXT, with a
        plaintext SHA-256 trailer, into a container whose header still said
        ENCRYPTED. The secret was on disk in the clear and the vault could never
        be opened again (``IntegrityError: trailer mismatch``), with no error at
        write time. Measured through the public ``Vault`` API.
        """
        if self._closed:
            raise ClosedVaultError(
                f"cannot {what} {self.filepath}: the vault is closed")

    def close(self) -> None:
        """Idempotent. Wipes key material; there is no handle to release."""
        if self.keys is not None:
            self.keys.wipe()
            self.keys = None
        self._closed = True


def build_block_blob(header: FileHeader, keys, seq: int, vectors: np.ndarray,
                     landmarks, ts, rev, ids, groups, docs) -> bytes:
    """Serialise one complete block (header + payload + trailer) ready to append.

    ``landmarks`` may be ``None``/empty, in which case ``m = 0`` is stored and a
    reader uses the block's own vectors as its landmarks -- that is what keeps a
    one-document chat flush down to a couple of kilobytes instead of padding it
    out to a full landmark table.
    """
    if bool(keys) != bool(header.encrypted):
        # AN INVARIANT, NOT A GUARD ON ONE PATH. This function branched purely on
        # `keys is not None` and never consulted the header, so a caller that lost
        # its keys wrote CLEARTEXT into a container whose header says ENCRYPTED
        # (3.0.1 did exactly that on a write after close(), bricking the file).
        # 3.0.2 fixed that one path with ClosedVaultError; this makes the whole
        # class impossible.
        raise NanomemError(
            "refusing to write a block whose encryption does not match the "
            f"container header (header encrypted={bool(header.encrypted)}, "
            f"keys={'present' if keys else 'absent'})")
    D = header.embed_dim
    n = int(vectors.shape[0])
    fp16 = header.fp16
    m = 0 if landmarks is None else int(np.asarray(landmarks).shape[0])
    if m and m >= n:
        m = 0                                   # vectors already are the landmarks
    lm_bytes = b""
    if m:
        lm_bytes = np.ascontiguousarray(landmarks, dtype=np.float32).tobytes()
    vdt = np.float16 if fp16 else np.float32
    vec_bytes = np.ascontiguousarray(vectors, dtype=vdt).tobytes()
    rec_bytes = encode_records(ts, rev, ids, groups, docs)
    payload = lm_bytes + vec_bytes + rec_bytes
    nonce = os.urandom(16) if keys is not None else b"\x00" * 16
    hdr = pack_block_header(
        kind=KIND_DATA, n=n, m=m, payload_len=len(payload), written_unix=time.time(),
        seq=seq, nonce=nonce, vec_section_len=m * D * 4 + n * D * (2 if fp16 else 4),
        rec_section_len=len(rec_bytes),
        block_flags=(FLAG_VEC_FP16 if fp16 else 0))
    if keys is not None:
        ct, tag = crypto.seal_block(keys, header.vault_uuid, hdr[:BLOCK_AUTH_LEN], seq, payload)
        return hdr + ct + tag
    return hdr + payload + crypto.plaintext_trailer(header.vault_uuid,
                                                    hdr[:BLOCK_AUTH_LEN], payload)


__all__ = [
    "FILE_MAGIC", "FILE_VERSION", "READER_VERSION", "FILE_HEADER_SIZE",
    "BLOCK_MAGIC", "BLOCK_HEADER_SIZE", "BLOCK_AUTH_LEN", "TRAILER_SIZE", "ALIGN",
    "KIND_DATA", "FILE_MODE",
    "KIND_TOMBSTONE_RESERVED", "FLAG_ENCRYPTED", "FLAG_VEC_FP16",
    "FLAG_RECLUSTERED", "CIPHER_NONE",
    "CIPHER_SHAKE_HMAC", "BLOCK_CAPACITY", "LANDMARKS_PER_BLOCK", "DEFAULT_EMBED_DIM",
    "FileHeader", "BlockMeta", "RecordsView", "Container", "align_up",
    "pack_file_header", "unpack_file_header", "pack_block_header", "unpack_block_header",
    "pack_string_table", "unpack_string_table", "encode_records", "decode_records",
    "detect_version", "file_lock", "peak_rss_kb", "current_rss_kb",
    "cleanup_stale_temps", "temp_path_for", "fsync_dir", "build_block_blob",
]
