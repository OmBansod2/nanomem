# nanomem core engine v3 — final implementation spec

Status: synthesis of the three judged designs. Base = Design 3 (highest overall: 8 / 7.5 / 7). Grafts and flaw fixes are listed in section 0 so every departure from a design is traceable. All numbers marked *measured* come from the orchestrator's findings or from files in `scratch/refound/` and the session scratchpad (`sweep71k.txt`, `level2.txt`, `lat.txt`, `sweep_routing_1190.txt`, `scale_results_current_engine.json`, `clean_chat_results_*.json`); nothing else is claimed.

## 0. What was taken from where, and which fatal flaws are resolved

Base (Design 3): immutable 256-byte header with `NANOMEM3` magic; 64-byte-aligned append-only blocks with a trailer that authenticates `vault_uuid || block header || payload`; scan-derived `valid_end` (no mutable header, no `committed_end` hint); copy-then-atomic-replace migration; read-only fallback; fp32 in-RAM arena, exhaustive cosine below `n_exhaustive`; C=50 blocks with L=8 spherical-k-means landmarks and max pooling above it; vectorised entity/temporal post-processing over interned int32 columns; only top-k hits materialised; measured `stats()`; written threat model; `(user_id, project, entity)` revision groups (the triple `Vault.merge` reconciles on at vault.py:967).

Grafted from Design 1: the append protocol (exclusive flock -> fstat -> incremental tail re-scan -> truncate only past the newly validated end -> single write -> fsync); no file handle or mmap held between calls; entity-history injection into the candidate pool so temporal superseding sees every revision even in routed mode; `min_score` applied directly to the legacy-calibrated score (no bisection); the routing sweep decision protocol with `router.DEFAULTS` bound to a results JSON and asserted by a test; fp16-on-disk default with an auto-fallback test on the 4x-distractor set; `durable` = fsync/full/none; stray `.tmp-*` cleanup; `scripts/sync_packages.py --verify`.

Grafted from Design 2: `score_mode="legacy"` default so the hard-coded thresholds 0.25 / 0.32 / 0.35 keep their meaning with zero call-site edits; `m = min(L, n)` landmarks (0 stored when `n <= L`, vectors double as landmarks) so per-turn chat flushes cost ~3.9 KB, plus `compact()` to coalesce under-filled blocks; `compact(recluster=True)` global spherical k-means re-layout (the only measured remedy for unordered ingestion: random-order routing 39.4 -> k-means layout 61.0 at a 15 % beam on 71k docs, equal to FAISS IVF 61.1); the generic attribute mechanism (structural classes + possessive head-noun extraction, no capitalised-word fallback, token-set containment matching, `SINGLE_VALUED`, lexicographic `(revision, timestamp)` ordering); `derive_entity_for_migrated`; a dedicated `K_chk` key; `search_batch`; the block-read counter assertion in the write-scaling test; verifying the legacy md5 id formula against golden ids.

Fatal flaws named by the judges and how each is closed:
1. D1 reused the v2 magic with a version bump; the shipping v2 reader ignores the version field and would read a v3 file as empty and append v2 blocks after it. -> New magic `b"NANOMEM3"` (verified: v2 raises on unrecognised magic).
2. D2/D3 append text ("ftruncate to committed_end / valid_end if the file extends past it") had a data-loss path for un-fsynced headers and for other processes' appends. -> Section 5.4 protocol: re-scan the tail under the exclusive lock first; truncate only bytes past the newly validated end; header never rewritten.
3. D2 shipped 256-row blocks at a 1.6 % candidate budget (measured 30.9 / 35.0 / 0.7 recall vs 62.1 exhaustive at 71k) and its routed test forced 50-row blocks. -> C=50, L=8, max pooling, 25 % beam (61.5 insertion / 61.6 k-means vs 62.1 exhaustive), engaged only above 50k docs and only when a layout gate passes; the routed test uses the shipped constants.
4. D2 fail-open integrity (search skipped bad blocks with a warning). -> Fail-closed by default; `on_integrity_error="skip"` opt-in and reported.
5. D1/D2 migration did `rename(path, .v2.bak)` then `replace(tmp, path)`, leaving `path` missing between the calls, after which the eager constructor creates an empty vault. -> `shutil.copyfile` to `.v2.bak` first, then `os.replace(tmp, path)`; the path always exists.
6. D3 changed the score scale (breaking 0.25/0.32 in cli/chat/Vault.forget and the 0.35 in `ingest_and_test_book.py:265` and `benchmark_multi_questions_in_one.py:133` across five copies). -> Legacy calibration stays the default; `score_mode="cosine"` + `legacy_score_to_cosine()` shipped; re-tune scheduled separately.
7. D3 `filter_cols` (one N-byte mask per distinct (key, value)) is O(N^2) on high-cardinality keys. -> D1's lazy walk in score order, bounded.
8. D1 block MAC did not bind anything that changes across rewrites, so an authenticated block from an earlier generation of the same file could be spliced back. -> `vault_uuid` rotates on every rewrite and is in both the keystream input and the MAC input, with `block_seq`.
9. D1/D3 kept v2's capitalised-phrase fallback (source of junk entity groups such as `Update`). -> Removed entirely.
10. D1 entity-only revision scope disagreed with `Vault.merge`'s `(user_id, project, entity)`. -> Triple used everywhere.
11. D1 padded every block to L landmarks and 4 KB pages (1-doc chat block ~28 KB on disk + always-resident landmarks). -> `min(L, n)` landmarks, 64-byte alignment, `compact()`.

Explicitly left out of 3.0 (measured or judged unjustified): tombstone blocks (kind reserved only; vault.py's rebuild dance and `replace_all` cover delete), a level-2 router (measured -1.0 to -1.7 pt at 25 % probe and slower in numpy: 2.09 ms vs 0.33 ms flat at 100k), partially-resident arena / block LRU / record LRU, two cipher suites, blake2b sorted id index, ctypes libproc RSS, `p=13` as a default (0.3-4 pt worse than max on every layout and budget measured; p=3 equals a single centroid), the pointwise x^13 "kernel" as anything but display calibration, any KB-scale RAM or sub-0.5 ms-at-any-N claim.

## 1. Module layout

```
nanomem_standalone/nanomem/
  engine.py      VaultEngine facade: compat surface, policy layer (entity tagging, anaphora tag,
                 auto-revision), search orchestration, stats. Re-exports matches_filter,
                 legacy_score, legacy_score_to_cosine, constants, error classes.        (~400 lines)
  container.py   v3 byte layout constants, FileHeader/BlockHeader pack/unpack, string tables,
                 scan (full/incremental), append protocol, rewrite (temp + os.replace),
                 file locking, RSS helpers.                                             (~450 lines)
  arena.py       Arena (resident fp32 vectors + int/float columns + record bytes + indexes)
                 and Memtable.                                                          (~250 lines)
  router.py      spherical_kmeans_landmarks, pool, LandmarkRouter, layout_check,
                 global_recluster_order, DEFAULTS, sweep helpers. numpy only, no nanomem imports.
  crypto.py      derive_keys, header_auth, keystream, seal_block, open_block, labels.
  entities.py    normalize_entity, detect_entity, query_intent, entities_match, matching_ids,
                 is_single_valued, has_temporal_cue, make_group_key, apply_temporal_boosts,
                 derive_entity_for_migrated.
  legacy_v2.py   read-only v2 reader (verbatim port) + migrate_v2_to_v3.
  errors.py      NanomemError, CorruptContainerError, IntegrityError, WrongPasswordError,
                 PasswordRequiredError, ReadOnlyVaultError, ContainerReplacedError.
  vault.py       5 small edits (section 3.4).   cli.py: 1 print line.
  __init__.py    additionally exports the error classes.
  classifier.py, embed.py, users.py, proxy.py, mcp.py (bundle copies only): unchanged.
nanomem_standalone/tests/       pytest suite (section 10); shipped in the sdist, excluded from the wheel.
nanomem_standalone/bench/       benchmark scripts + results/*.json (never imported by the package).
scripts/sync_packages.py        fan-out, wheel rebuild, md5 verification (section 11).
```
Runtime imports: `numpy`, stdlib (`os, io, struct, json, time, hashlib, hmac, zlib, threading, fcntl|msvcrt, shutil, resource, warnings, re, unicodedata, math`). `faiss` is imported only inside `bench/` scripts, guarded by try/except. A test asserts `{'faiss','torch','cryptography','sklearn','mlx'} & sys.modules == {}` after `import nanomem`.

## 2. v3 byte layout

All integers little-endian. One file at exactly `filepath`, created eagerly (header only, 256 bytes) by the constructor. No sidecars, except a transient `<path>.tmp-<pid>-<8hex>` during rewrite/migration and the one-time `<path>.v2.bak` of a migrated v2 file. Stray `.tmp-*` files matching our pattern and older than 1 h are removed at open.

### 2.1 File header (256 bytes, offset 0; written once at create/rewrite, never rewritten by appends)

```
off size field
0   8    magic = b"NANOMEM3"
8   4    u32 format_version = 3
12  4    u32 min_reader_version = 3      (reader refuses files whose value exceeds its READER_VERSION)
16  4    u32 header_len = 256
20  4    u32 flags: bit0 ENCRYPTED, bit1 VEC_FP16 (0 = fp32 vectors), bits 2-3 cipher_id
         (0 none, 1 SHAKE256-XOF + HMAC-SHA256), all other bits must be 0
24  4    u32 embed_dim (768)
28  4    u32 block_capacity (50)
32  4    u32 landmarks_per_block L (8)
36  4    u32 reserved0 = 0
40  8    f64 created_unix
48  16   vault_uuid = os.urandom(16)      (rotated on EVERY rewrite: replace_all, compact, rekey, migration)
64  1    u8 kdf_id: 0 none, 1 scrypt
65  1    u8 scrypt_log2_n (16)
66  1    u8 scrypt_r (8)
67  1    u8 scrypt_p (1)
68  4    u32 reserved1 = 0
72  16   kdf_salt = os.urandom(16) when encrypted (kept across rewrites so the same password works), zeros otherwise
88  32   header_auth: ENCRYPTED -> HMAC-SHA256(K_chk, b"NM3H" || header[0:88]); plaintext -> SHA-256(b"NM3H" || header[0:88])
120 4    u32 header_crc32 = zlib.crc32(header[0:120])
124 132  reserved, zeros
```
`FILE_HEADER_FMT = "<8sIIIIIIIId16sBBBBI16s32sI"` (124 bytes) followed by 132 zero bytes.
Version 2 files (`b"NANOMEM\x00"`, struct `"<8sIIII40s"`, version field 2) are recognised by magic and migrated (section 7). Any other magic -> `CorruptContainerError`.

### 2.2 Blocks

Blocks follow the header back-to-back; each starts at an offset that is a multiple of 64 (zero padding between blocks, <= 63 bytes). The scanner computes `next = align64(block_end)`.

Block header (96 bytes, always plaintext), `BLOCK_HEADER_FMT = "<4sIIIIIdQ16sIIII24x"`:
```
0   4   magic b"NMB3"
4   4   u32 header_len = 96
8   4   u32 kind: 1 DATA (2 TOMBSTONE is reserved: 3.0 neither writes it nor accepts it)
12  4   u32 doc_count n (1..block_capacity)
16  4   u32 n_landmarks m: 0 when n <= L (the loader uses the block's vectors as its landmarks), else L
20  4   u32 payload_len (ciphertext length == plaintext length)
24  8   f64 written_unix
32  8   u64 block_seq (0-based ordinal within this file generation; = index in toc; toc id = f"blk_{seq:08d}")
40  16  nonce (os.urandom(16) when ENCRYPTED, zeros otherwise)
56  4   u32 vec_section_len = m*D*4 + n*D*(2 if VEC_FP16 else 4)
60  4   u32 rec_section_len
64  4   u32 block_flags (bit0 VEC_FP16; must equal the file flag; others 0)
68  4   u32 header_crc32 = zlib.crc32(header[0:68])
72  24  reserved zeros
```

Payload (`payload_len` bytes; the unit of encryption):
```
[landmarks]  m x D float32, unit rows           (absent when m == 0)
[vectors]    n x D float16 | float32, unit rows (row i = global row first_row + i)
[records]    rec_section_len bytes:
   u32 n
   f64[n] timestamps
   i32[n] revisions
   string_table ids      (UTF-8)
   string_table groups   ("" or "<user_id>\x1f<project>\x1f<entity>", see entities.make_group_key)
   string_table docs     (UTF-8 JSON {"text": str, "source": str, "metadata": {...}},
                          json.dumps(ensure_ascii=False, separators=(",", ":"), default=str))
string_table := u32 count, u32 offsets[count+1] (relative to the table's data start), data bytes
```
Invariant checked by tests: `groups[i] == make_group_key(meta.get("user_id"), meta.get("project"), meta.get("entity"))` for every record; `ids[i] == metadata["id"]`.

Trailer (32 bytes, always present):
- plaintext: `SHA-256(b"NM3T" || vault_uuid || block_header[0:72] || payload)` (corruption detection only)
- encrypted: `HMAC-SHA256(K_mac, b"NM3T" || vault_uuid || block_header[0:72] || ciphertext)` (encrypt-then-MAC; header fields incl. nonce, seq, kind, lengths and the file generation are authenticated)

Size accounting (D=768): a full 50-doc block with ~500-char texts = 96 + 24,576 (landmarks) + 76,800 (fp16 vectors) + ~34,000 (records) + 32 ≈ 135 KB = 2.7 KB/doc = 0.76x raw text+fp32 (*measured* per-record JSON ~617 B); with fp32 vectors ≈ 212 KB = 4.25 KB/doc = 1.19x raw. A 1-doc chat flush = 96 + 0 + 1,536 + ~700 + 32 + pad ≈ 2.4 KB (fp16) / 3.9 KB (fp32) versus v2's 14.7 KB/doc.

### 2.3 Scan (open and incremental tail load)

```
pos = 256
while pos + 96 <= size:
    hdr = read(pos, 96)
    if hdr.magic != b"NMB3" or crc32(hdr[0:68]) != hdr.crc or hdr.kind != 1 or hdr.header_len != 96: break
    end = pos + 96 + hdr.payload_len + 32
    if end > size: break                                   # partial write
    body, trailer = read(pos+96, payload_len), read(end-32, 32)
    expected = trailer_for(header, body)                   # SHA-256 (plaintext) or HMAC over ciphertext (encrypted)
    if not hmac.compare_digest(expected, trailer):
        if on_integrity_error == "raise": raise IntegrityError(block_index, pos, "trailer mismatch")
        integrity_errors.append(block_index); pos = align64(end); continue
    payload = body if plaintext else open_block(...)       # decrypt only AFTER the tag verified
    sink.add_block(meta, payload)                           # arena ingest
    pos = align64(end)
valid_end = pos                                            # last byte of the last valid block (+ nothing else)
truncated_tail_bytes = size - valid_end
```
A partial trailing block is a crash remnant, never an error; it is reported and cut off by the next appender (section 5.4). A malformed block that is NOT the last one is an integrity failure (fail closed by default). Read-path tail scans take a shared `flock`, so a writer that holds the exclusive lock mid-write is never observed.

*Measured* open-cost components: crc32 3 ms/100 MB, SHA-256 36 ms/100 MB, SHAKE256 132 ms/100 MB, XOR 9 ms/100 MB, arena copy ~30 ms/100 MB, id index 12 ms/100k ids -> a plaintext 100 MB vault opens in ~0.1-0.15 s, an encrypted one in ~0.3 s + KDF.

## 3. Classes, signatures, docstrings, compatibility surface

### 3.1 `container.py`

```python
FILE_MAGIC = b"NANOMEM3"; FILE_VERSION = 3; READER_VERSION = 3; FILE_HEADER_SIZE = 256
BLOCK_MAGIC = b"NMB3"; BLOCK_HEADER_SIZE = 96; TRAILER_SIZE = 32; ALIGN = 64
KIND_DATA = 1; KIND_TOMBSTONE_RESERVED = 2
FLAG_ENCRYPTED = 1; FLAG_VEC_FP16 = 2; CIPHER_SHIFT = 2; CIPHER_NONE = 0; CIPHER_SHAKE_HMAC = 1
BLOCK_CAPACITY = 50; LANDMARKS_PER_BLOCK = 8; DEFAULT_EMBED_DIM = 768

@dataclass class FileHeader: flags, embed_dim, block_capacity, landmarks_per_block, created_unix, vault_uuid,
                             kdf_id, scrypt_log2_n, scrypt_r, scrypt_p, kdf_salt, header_auth
@dataclass class BlockMeta:  index, offset, kind, n, m, payload_len, written_unix, seq, nonce,
                             vec_section_len, rec_section_len, first_row

def pack_file_header(h: FileHeader) -> bytes
def unpack_file_header(b: bytes) -> FileHeader           # raises CorruptContainerError (magic/crc/version)
def pack_block_header(h) -> bytes; def unpack_block_header(b) -> BlockHeader | None   # None on bad magic/crc
def pack_string_table(items: list[bytes]) -> bytes
def unpack_string_table(buf: memoryview, off: int) -> tuple[list[bytes], int]
def encode_records(ts, rev, ids, groups, docs) -> bytes
def decode_records(buf: memoryview) -> RecordsView          # ts, rev, ids, groups, doc_spans (n,2) relative offsets
def align_up(n: int, a: int = ALIGN) -> int
def detect_version(filepath: str) -> int | None           # 2, 3 or None (missing/empty)
@contextmanager def file_lock(f, exclusive: bool)          # fcntl.flock / msvcrt.locking; no-op fallback
def current_rss_kb() -> int | None                          # /proc/self/statm if present, else None
def peak_rss_kb() -> int                                    # resource.getrusage(RUSAGE_SELF).ru_maxrss, bytes->KiB on darwin

class Container:
    """Owns all file I/O for one v3 file. Holds NO file descriptor or mmap between calls: every operation
    opens the file, does its work under the appropriate flock, and closes it. This is what makes close()
    trivially idempotent and os.remove()/os.replace() safe on every platform."""
    def __init__(self, filepath, *, embed_dim, password=None, vector_dtype="float16", block_capacity=50,
                 landmarks_per_block=8, durable="fsync", on_integrity_error="raise", migrate=True,
                 migrate_backup=True, readonly=False)
        # create eagerly if missing/empty; migrate if v2; verify header; derive keys; scan into a sink
    header: FileHeader; keys: KeyMaterial | None; blocks: list[BlockMeta]; valid_end: int
    integrity_errors: list[int]; truncated_tail_bytes: int; read_only: bool; format_version_on_open: int
    def scan(self, sink, from_offset=None) -> None          # section 2.3; shared lock for tail scans
    def modified(self) -> str                                # "same" | "appended" | "replaced" | "missing" (one os.stat)
    def append_block(self, blob: bytes, sink) -> BlockMeta   # section 5.4 protocol
    def write_new_file(self, tmp_path, header, keys, block_iter) -> None
    def replace_with(self, tmp_path) -> None                 # exclusive lock on current file, os.replace, dir fsync
    def close(self) -> None                                  # idempotent; wipes key material
```

### 3.2 `arena.py`

```python
class Arena:
    """Resident RAM structures built once at open and extended incrementally. Row r of every column is the
    same document; rows are in file order. All arrays grow by capacity doubling."""
    vec: np.ndarray (cap, D) float32, C-contiguous; n_rows: int
    ts: float64 (cap,); rev: int32 (cap,); entity_id: int32 (cap,) (-1 none); group_id: int32 (cap,) (-1 none)
    row_block: int32 (cap,); block_start: int64 (n_blocks+1,)   # block b owns rows [block_start[b], block_start[b+1])
    ids: list[str]; id_index: dict[str, int]                     # id -> FIRST row with that id (Vault.get semantics)
    entity_names: list[str]; entity_index: dict[str, int]        # interned normalised entity strings
    group_keys: list[str]; group_index: dict[str, int]
    group_max: dict[int, tuple[int, float]]                      # group_id -> (max_revision, max_timestamp)
    landmarks: float32 (cap_lm, D); land_start: int64 (n_blocks+1,)  # block b's landmarks = rows [land_start[b], land_start[b+1])
    rec_bytes: list[bytes]                                       # per block, the plaintext records section (owned bytes)
    doc_span: int64 (cap, 2)                                     # [start, end) of record r's JSON inside rec_bytes[row_block[r]]
    def __init__(self, embed_dim: int)
    def add_block(self, meta: BlockMeta, landmarks: np.ndarray, vectors_f32: np.ndarray, rec: RecordsView, rec_bytes: bytes) -> None
    def intern_entity(self, name: str | None) -> int; def intern_group(self, key: str | None) -> int
    def record(self, row: int) -> dict                            # json.loads of the span -> {"text","source","metadata"} (fresh objects)
    def rows_with_entity_ids(self, eids: np.ndarray, limit: int = 512) -> np.ndarray
    def resident_bytes(self) -> dict[str, int]                    # arena, landmarks, columns, records, index_estimated
    def reset(self) -> None

class Memtable:
    """Unflushed records; <= block_capacity items. Exposed unchanged as VaultEngine.memtable."""
    items: list[dict]     # {id, text, source, embedding (float32 unit), metadata, timestamp, revision, entity_id, group_id}
    vec: np.ndarray (block_capacity, D) float32; n: int
```

### 3.3 `router.py`, `crypto.py`, `entities.py`, `legacy_v2.py`, `errors.py`

```python
# router.py
DEFAULTS = dict(block_capacity=50, landmarks_per_block=8, pool="max", n_exhaustive=50_000,
                beam_frac=0.25, beam_min_blocks=4, layout_gate_min_coverage=0.90)   # must equal bench/results/route_sweep.json["chosen"]
def spherical_kmeans_landmarks(V: np.ndarray, L: int, seed: int, iters: int = 10) -> np.ndarray   # (min(L,n), D) unit rows
def pool(S: np.ndarray, land_start: np.ndarray, m: np.ndarray, p: float | str) -> np.ndarray    # (n_blocks,) block scores
class LandmarkRouter:
    def __init__(self, arena: Arena, *, p="max", beam_frac=0.25, beam_min_blocks=4)
    def route(self, q: np.ndarray) -> np.ndarray               # block indices (beam)
    def block_scores(self, q) -> np.ndarray                     # exposed for the sweep
def layout_check(arena: Arena, router: LandmarkRouter, sample: int = 256, seed: int = 0) -> float   # NN coverage in [0,1]
def global_recluster_order(V: np.ndarray, block_capacity: int, seed: int = 0, iters: int = 10) -> np.ndarray  # row permutation

# crypto.py
class KeyMaterial: k_enc: bytearray(32); k_mac: bytearray(32); k_chk: bytearray(32); def wipe(self)
def derive_keys(password: str, salt: bytes, log2_n: int, r: int, p: int) -> KeyMaterial
def header_auth(km: KeyMaterial | None, header_prefix88: bytes) -> bytes
def keystream(km: KeyMaterial, vault_uuid: bytes, nonce: bytes, seq: int, n: int) -> bytes
def seal_block(km, vault_uuid, header72_with_nonce: bytes, seq: int, payload: bytes) -> tuple[bytes, bytes]   # (ciphertext, tag)
def open_block(km, vault_uuid, header72: bytes, seq: int, ciphertext: bytes, tag: bytes) -> bytes             # raises IntegrityError
def plaintext_trailer(vault_uuid, header72, payload) -> bytes
CIPHER_LABEL_NONE = "none (plaintext)"
CIPHER_LABEL_SHAKE = "SHAKE256-XOF stream + HMAC-SHA256 tag (encrypt-then-MAC), scrypt-derived keys; stdlib construction, not AES, not a NIST AEAD"

# entities.py  (section 8)
def normalize_entity(s) -> str; def detect_entity(text) -> str | None; def is_pronoun_led(text) -> bool
def query_intent(query) -> str | None; def entity_class(entity) -> str | None; def entities_match(a, b) -> bool
def matching_ids(entity: str, entity_names: list[str]) -> np.ndarray; def is_single_valued(entity) -> bool
def has_temporal_cue(query) -> bool; def make_group_key(user_id, project, entity) -> str
def apply_temporal_boosts(cos, entity_id, rev, ts, entity_names, query_text, temporal_direction, mask) -> tuple[np.ndarray, str | None]
def derive_entity_for_migrated(meta: dict, text: str) -> str | None

# legacy_v2.py
V2_FILE_MAGIC = b"NANOMEM\x00"; V2_BLOCK_MAGIC = b"BLK\x00"; V2_CIPHER_SALT = b"NANOMEM_VAULT_AES256_PROJECTED_LATTICE_2026"  # read-only legacy constant
def iter_v2_records(filepath) -> Iterator[dict]      # {id, text, source, metadata, timestamp, revision, embedding}
def migrate_v2_to_v3(filepath, *, password, backup, container_kwargs) -> int

# errors.py
class NanomemError(Exception); class CorruptContainerError(NanomemError)
class IntegrityError(NanomemError): block_index: int; offset: int; reason: str
class WrongPasswordError(NanomemError); class PasswordRequiredError(NanomemError)
class ReadOnlyVaultError(NanomemError); class ContainerReplacedError(NanomemError)
```

### 3.4 `engine.py` — `VaultEngine` and the compatibility surface

```python
ENGINE_VERSION = "3.0.0"
def matches_filter(meta: dict, meta_filter: dict) -> bool
    """v2 semantics verbatim: every filter key must exist in meta; scalar match if val == v or str(val) == str(v);
    if either side is list/tuple/set, membership with str coercion."""
def legacy_score(c: np.ndarray | float) -> np.ndarray | float      # 0.6*c + 0.4*sign(c)*|c|**13 (strictly monotone)
def legacy_score_to_cosine(s: float) -> float                       # bisection inverse; 0.25 -> 0.417, 0.32 -> 0.533, 0.35 -> 0.58

class VaultEngine:
    def __init__(self, filepath: str, embed_dim: int = 768, *, password: str | None = None,
                 score_mode: str = "legacy", vector_dtype: str = "float16", n_exhaustive: int = 50_000,
                 block_capacity: int = 50, landmarks_per_block: int = 8, route_p: float | str = "max",
                 beam_frac: float = 0.25, beam_min_blocks: int = 4, layout_gate_min_coverage: float = 0.90,
                 durable: str = "fsync", on_integrity_error: str = "raise", migrate: bool = True,
                 migrate_backup: bool = True, readonly: bool = False):
        """Open or eagerly create the single-file vault at `filepath`. For an existing file the header's
        embed_dim/block_capacity/landmarks/vector_dtype win (a mismatch with the arguments issues a
        warnings.warn, as v2 did). Migrates v2 files on open (section 7). Never holds a file handle."""
    filepath: str; embed_dim: int
    memtable: list[dict]              # property -> Memtable.items (live list)
    container: _ContainerView         # property (section 3.5)
    def add_fact(self, text, embedding, source="user_input", metadata=None, timestamp=None, revision=None, id=None) -> str
        """Store one record; returns the doc id (str). Policy layer runs only when revision is None (section 5.1)."""
    def search(self, query_text, query_vec, top_k=3, metadata_filter=None, min_score=0.0, temporal_direction="current") -> list[dict]
        """Section 4. Returns fresh dicts {id, doc_id, text, source, metadata, score, cosine, timestamp, revision}
        sorted by score desc, len <= top_k, including unflushed records. min_score is applied to the
        pre-boost score on the configured scale."""
    def search_batch(self, query_vecs: np.ndarray, top_k: int = 10) -> list[list[tuple[str, float]]]
        """Cosine-only (no boosts, no filters) top-k ids and cosines for Q queries in one matmul; for
        decomposed / multi-hop callers. Not used by vault.py in 3.0."""
    def flush(self) -> None           # spill memtable to blocks until empty; idempotent
    def stats(self) -> dict           # section 9
    _matches_filter = staticmethod(matches_filter)
    def iter_records(self, include_embeddings: bool = True) -> Iterator[dict]
        """flush(); then yield {id, text, source, metadata, timestamp, revision[, embedding]} in file order,
        owned copies (embedding = arena row .copy())."""
    def get(self, id: str) -> dict | None       # O(1) via id_index, then memtable
    def count(self) -> int                       # flushed + pending
    def replace_all(self, records: Iterable[dict], *, password=KEEP, vector_dtype=KEEP) -> int
        """Atomic rewrite: temp file + fsync + exclusive lock + os.replace + directory fsync, then reload.
        Records are stored verbatim (id/text/source/metadata/timestamp/revision/embedding); the policy
        layer never runs here."""
    def compact(self, recluster: bool = False) -> dict
        """replace_all(iter_records()) merging under-filled blocks into full ones; with recluster=True the
        rows are first re-ordered by a global spherical k-means (section 4.6). Never automatic."""
    def rekey(self, new_password: str | None) -> None     # replace_all with new keys; None = decrypt to plaintext
    def layout_gate(self) -> dict                          # {"coverage", "passed", "checked_at_blocks"} (section 4.5)
    def close(self) -> None                                # idempotent; wipes keys; safe on a never-written engine
    def __enter__/__exit__
```

### 3.5 Compatibility surface required by vault.py, item by item

| Contract item | How v3 satisfies it |
|---|---|
| `VaultEngine(filepath=..., embed_dim=768)` keyword-called at vault.py:34 and after every rebuild (:853, :1067, :1137, :1192, :1436) | Same signature; extra kwargs are keyword-only with defaults. Eagerly writes the 256-byte header when the file is missing or empty (users.create_user, cli init, proxy /vault/init rely on `os.path.exists`). Opens without exclusive access and holds no handle, so a second `Vault(path).stats()` on an open file works and a new engine can be built on the same path immediately after `container.close()+os.remove()`. Single file, no sidecars. |
| `add_fact(...) -> float` (elapsed us) | Now returns the doc id `str` (hard constraint). Only `Vault.add` surfaces the value (cli prints it); proxy/server/mcp/chat ignore it. `metadata['id']` is always set with the v2 formula. |
| Explicit id/timestamp/revision/embedding stored verbatim; count preserved on rebuild | Policy layer gated on `revision is None`; no dedup; no vector rotation; fp16 -> renormalise -> fp16 is a fixed point (*measured*: 0 elements change per rebuild), so `add_batch(get_all_records())` reproduces the record list exactly. |
| Auto-revision for entity records when `revision is None` (demo.py, benchmark_200, Vault.chat) | `group_max[(user_id, project, entity)] + 1`, O(1) dict updated on add and rebuilt at open (`np.unique` + `np.maximum.at`). |
| `source == "chat_session"` entity tagging into `metadata['entity']` | Policy layer calls `entities.detect_entity(text)`; pronoun-led texts inherit the session's last entity per user_id as metadata only. |
| `search(...)` signature, result dict keys, ordering, unflushed visibility, `min_score` pre-boost, `metadata_filter` forces full scan, `temporal_direction`, thread-safety | Section 4. Fresh dict per hit with pure-Python types (`json.dumps` safe); memtable scored every query; filters never route; RLock held for the whole call. `score_mode="legacy"` keeps the 0.25/0.32/0.35 thresholds meaningful. |
| `flush()` semantics | After return `memtable_pending == 0`, every record enumerable through `container.toc + read_payload` and `iter_records`, fsync'd so another instance/process sees it via `modified()`. Idempotent. |
| `stats()` 8 legacy keys, JSON-serialisable, reflects other instances' appends, safe without the server lock | Section 9; calls `_reload_if_modified()`; RLock inside. |
| `_matches_filter(meta, filter)` cross-module private call (vault.py:713, :1050) | `staticmethod(matches_filter)` with v2 semantics; export's stricter `meta.get(k) == v` untouched. |
| `engine.container.toc` (vault.py:709, :742): iterable of dicts with `id, doc_count, offset, length` in stable on-disk order, complete after flush | `_ContainerView.toc` property -> `[{"id": f"blk_{seq:08d}", "doc_count": n, "offset": block offset, "length": 96+payload_len+32, "timestamp": written_unix, "first_row": ...}]` for every DATA block, file order. |
| `engine.container.read_payload(offset, length, doc_count, block_id)` returning `texts, sources, metadatas, timestamps, revisions, values` as OWNED copies | Looks the block up by offset (validated against block_id); `texts/sources/metadatas` from `json.loads` of each record (fresh objects; `'id'` injected into each metadata dict if absent); `timestamps` float64 copy; `revisions` int32 copy; `values` = `arena.vec[rows].copy()` (never shares memory with the arena). `keys/basis/mean` omitted (never read). |
| `engine.container.close()` before `os.remove` (vault.py:848, :1062, :1132, :1187, :1434, :1482) | Delegates to `engine.close()`; idempotent; nothing to release because no handle is held, so `os.remove` succeeds on Windows too. |
| `engine.memtable` (internal) | Live list property; item keys `id, text, source, embedding, metadata, timestamp, revision` (+ `entity_id, group_id`). |
| Dropped v2 internals (no external callers) | `entity_anchors, active_entity_*, route_blocks, _compact_memtable, _apply_temporal_priority, ContinuousContainer.*, _keystream, _crypto_transform, _safe_open, _file_lock` are gone; the v2 keystream lives only in `legacy_v2.py`. |
| Existing v2 .dat files open (memory_*.dat profiles, personal_memory.dat, hands_on_llm_vault.dat x3, my_demo_vault.dat) | Section 7: transparent one-time migration with the original preserved as `.v2.bak`; read-only fallback when the directory is not writable or `migrate=False`. |

Required edits outside `engine.py` (exact):
1. `vault.py` `Vault.add` (lines 129-198): return type `-> str`; single-chunk branch `return self.engine.add_fact(...)` (already the id); chunked branch `return parent_id` instead of the elapsed microseconds. Docstring updated. `self.last_id` unchanged.
2. `vault.py` `Vault.__init__` (line 24-38): add `password: Optional[str] = None`; `self._password = password or os.getenv("NANOMEM_PASSWORD")`; pass `password=self._password` to `VaultEngine` here and at the five rebuild sites (:853, :1067, :1137, :1192, :1436) and to `Vault(...)` opens in `merge`, `search_multi`, `export`/`split*`.
3. `vault.py` `_detect_entity` (line 859): body becomes `meta = metadata or {}; return str(meta["entity"]) if meta.get("entity") else detect_entity(text)` with `from .entities import detect_entity` (removes the fixture regexes at 870-890; merge at :933/:955 keeps calling it).
4. `vault.py` `Vault.close()` (line 1481): `self.flush(); self.engine.close()` (`container.close()` still works).
5. `cli.py` lines 138-139: `t0 = time.perf_counter(); doc_id = v.add(args.text, source=args.source); print(f"[nanomem] Stored fact {doc_id} in {(time.perf_counter()-t0)*1e6:.1f} us into '{v_path}'")`.
No other file in nanomem/, proxy.py, server.py, chat.py, users.py, mcp.py, demo.py changes. Optional phase 2 (behaviour-preserving, ~40 lines, not required for 3.0): collapse the five copy-pasted rebuild blocks into `Vault._rebuild(records) -> self.engine.replace_all(records)` (removes the crash window between `os.remove` and the rebuilt `add_batch`); `get_all_records`/`get` -> `engine.iter_records()`/`engine.get()`.

## 4. Search and routing

### 4.1 Score scale (compatibility)

Ranking is cosine over unit vectors; the shipped `0.60*cos + 0.40*sign(cos)*|cos|^13` is rank-identical (0/5000 argsort differences, *measured*) and survives only as a display calibration. `score_mode="legacy"` (default): `score = legacy_score(cos) + boosts`, `min_score` compared with `legacy_score(cos)` before boosts (exactly v2's rule). `score_mode="cosine"`: `score = cos + boosts`, `min_score` on `cos`. `cosine` is always returned raw. Documentation must call this a calibration, never a kernel; `legacy_score_to_cosine()` is shipped for the later re-tune of the three thresholds.

### 4.2 Mode selection

```
def _mode(self, metadata_filter):
    if metadata_filter is not None: return "exhaustive"            # filters never lose recall
    if self.force_mode: return self.force_mode                     # constructor/benchmark override
    if arena.n_rows <= n_exhaustive: return "exhaustive"            # default 50_000
    return "landmark" if self._gate["passed"] else "exhaustive"     # section 4.5
```
*Measured* exhaustive cost (fp32 matmul + argpartition, this machine, `lat.txt`): 1,190 docs 0.011 ms; 4,760 0.071; 10k 0.385 (0.21 idle); 20k 0.82; 32k 1.06; 50k 1.58; 71k 2.52; 100k 3.34; 200k 6.96. Below 50k every search is exact, so evidence-recall equals FAISS IndexFlatIP by construction (68.3 at 1,190; 45.8 at 4,760).

### 4.3 Search pseudo-code

```
def search(self, query_text, query_vec, top_k=3, metadata_filter=None, min_score=0.0, temporal_direction="current"):
    with self._lock:
        self._reload_if_modified()
        q = unit(np.asarray(query_vec, np.float32).ravel())          # any D-vector accepted; zero norm -> []
        if top_k <= 0 or q is None: return []
        N = arena.n_rows; k_cand = max(64, 8 * top_k)
        intent = entities.query_intent(query_text)                    # LRU-cached per query string
        intent_ids = entities.matching_ids(intent, arena.entity_names) if intent else None

        # 1. candidate rows + raw cosine
        mode = self._mode(metadata_filter)
        if mode == "exhaustive":
            rows = np.arange(N); cos = arena.vec[:N] @ q
        else:
            blocks = np.sort(self.router.route(q))                    # section 4.4
            rows = np.concatenate([np.arange(block_start[b], block_start[b+1]) for b in blocks])
            cos = np.empty(len(rows), np.float32)
            for b, slot in zip(blocks, slots): np.dot(arena.vec[block_start[b]:block_start[b+1]], q, out=cos[slot])
            # entity-history injection: every stored revision of the queried entity joins the pool
            if intent_ids is not None:
                extra = setdiff1d(arena.rows_with_entity_ids(intent_ids, limit=512), rows)
                rows = concat(rows, extra); cos = concat(cos, arena.vec[extra] @ q)
        # memtable rows (virtual ids >= MT_BASE)
        if memtable.n: rows = concat(rows, MT_BASE + arange(m)); cos = concat(cos, memtable.vec[:m] @ q)

        # 2. pre-boost threshold (v2 rule) and filter
        base = legacy_score(cos) if score_mode == "legacy" else cos
        mask = base >= min_score
        if metadata_filter is not None:                               # lazy walk in score order, bounded
            keep = zeros_like(mask); hits = 0
            for r in np.argsort(-cos):                                # argsort of N (0.5 ms at 10k, 3 ms at 50k)
                if not mask[r]: continue
                meta = record_meta(rows[r])                            # json.loads(span) or memtable dict, 1.6 us
                if matches_filter(meta, metadata_filter):
                    keep[r] = True; hits += 1
                    if hits >= k_cand and not entity_intent_pending: break
            mask &= keep                                               # entity rows of the intent group are also walked (bounded by 512)

        # 3. boosts (vectorised over the candidate arrays; section 8.5)
        ent = column(entity_id, rows); rv = column(rev, rows); tsv = column(ts, rows)
        boost, top_entity = entities.apply_temporal_boosts(cos, ent, rv, tsv, arena.entity_names, query_text,
                                                            temporal_direction, mask, intent_ids)
        # apply_temporal_boosts: +0.25 where ent in intent_ids; determine top_entity = intent or entity of the
        # best masked (base+0.25) candidate; if top_entity is not the intent, in landmark mode inject its rows
        # (one more rows_with_entity_ids call, then recompute for those rows); +1.0 * rank-normalised
        # (revision, timestamp) within the comparable group when is_single_valued(top_entity) or
        # temporal_direction == "historical" or has_temporal_cue(query_text).
        final = np.where(mask, base + boost, -np.inf)

        # 4. top-k and materialisation
        k = min(top_k, int(mask.sum()));  if k == 0: return []
        idx = np.argpartition(-final, k-1)[:k]; idx = idx[np.argsort(-final[idx])]
        hits = []
        for i in idx:
            r = rows[i]; rec = memtable.items[r - MT_BASE] if r >= MT_BASE else arena.record(r)
            hits.append({"id": rec_id, "doc_id": rec_id, "text": rec["text"], "source": rec["source"],
                         "metadata": dict(rec["metadata"]), "score": float(final[i]), "cosine": float(cos[i]),
                         "timestamp": float(tsv[i]), "revision": int(rv[i])})
        return hits
```
Costs beyond the matmul (*measured* by the judges on this machine): column boosts ~0.03 ms, hit materialisation ~0.02 ms for top_k=4, so p50 at 10k docs is ~0.3-0.45 ms depending on load (0.21-0.39 ms matmul).

### 4.4 Landmark router

Landmarks per block, computed at spill time (*measured* 0.47-0.68 ms for 50x768, L=8):
```
def spherical_kmeans_landmarks(V, L, seed, iters=10):     # V: (n, D) unit rows
    if n <= L: return V.copy()                            # (stored as m=0; vectors double as landmarks)
    rng = np.random.default_rng(seed)                     # seed = block_seq -> deterministic rebuilds
    idx = [rng.integers(n)]
    for _ in range(L-1):                                  # k-means++ init on cosine distance
        d = np.clip(1 - (V @ V[idx].T).max(axis=1), 1e-9, None); idx.append(rng.choice(n, p=d/d.sum()))
    M = V[idx].copy()
    for _ in range(iters):
        a = np.argmax(V @ M.T, axis=1)
        for l in range(L):
            if (a == l).any(): M[l] = V[a == l].mean(0)
            else: M[l] = V[np.argmin((V @ M.T).max(axis=1))]          # re-seed empty cluster with the farthest row
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M
```
Routing:
```
def route(self, q):
    S = arena.landmarks[:M_total] @ q                                    # 16k rows -> 0.33 ms at 100k docs (measured)
    if p == "max": B = np.maximum.reduceat(S, land_start[:-1])
    else: R = np.clip(S, 0, None) ** p; B = (np.add.reduceat(R, land_start[:-1]) / m_b) ** (1/p)   # power mean, p tunable
    beam = clip(ceil(beam_frac * n_blocks), beam_min_blocks, n_blocks)
    return np.argpartition(-B, beam-1)[:beam]
```
The pooling exponent is the one place where a nonlinearity over similarities changes a ranking (it aggregates L similarities per block). *Measured* on 71k HotpotQA paragraphs (2,000 queries, exhaustive 62.1): insertion order C=50 L=8 -> max 57.1 / 58.4 / 60.1 / 60.6 / 61.5 at 1.6 / 5 / 10 / 15 / 25 % beam; p=13 -> 57.0 / 58.3 / 59.6 / 60.1 / 61.1; p=3 -> 37.8 / 46.2 / 51.5 / 54.5 / 57.3 (= single centroid 37.1 / 45.7 / 51.3 / 54.0 / 57.1); FAISS IVFFlat nlist=1429 at matched nprobe: 51.1 / 58.1 / 60.6 / 61.1 / 61.8. So `pool="max"`, `L=8`, `C=50`, `beam_frac=0.25` ship (loss -0.6 pt insertion / -0.5 k-means vs exhaustive; +4.4 over single centroid at the same budget). `p` stays a constructor/benchmark knob (`route_p`) and is swept, never defaulted to 13. Routing is a constant-factor win (100k docs: 0.33 ms landmark scan + 357 blocks x ~2.4 us + ~0.9 ms per-block dots ≈ 1.3-1.5 ms vs 1.9-3.3 ms exhaustive), not asymptotic; that is what the docs must say.

### 4.5 Layout gate (routing engages only when the block layout supports it)

The 1,190-doc numbers (70.8-71.7 at beam 2-4) are an insertion-order artefact (both gold paragraphs share a block 98.3 % of the time, *measured*); on a random permutation the same router scores 12.5 at beam 4 and 39.4 at a 15 % beam on 71k docs. Therefore:
```
def layout_check(arena, router, sample=256, seed=0):
    rows = rng.choice(N, sample, replace=False); Q = arena.vec[rows]
    S = Q @ arena.vec[:N].T; S[arange(sample), rows] = -inf            # ~0.2-0.3 s at 100k docs
    nn = argmax(S, axis=1)                                             # exact nearest other row
    return mean(row_block[nn[i]] in router.route(Q[i]) for i)          # fraction of nearest neighbours the beam would reach
```
Run once at open when `N > n_exhaustive`, after every `compact()`, and whenever `n_blocks` has grown by >= 10 % since the last check. `passed = coverage >= layout_gate_min_coverage` (0.90 provisional; `bench/route_sweep.py` reports this metric for insertion / k-means / random layouts so the constant is set from data and a test asserts the gate passes on the insertion-order 71k corpus and fails on its random permutation). When the gate fails the engine stays exhaustive (linear cost, exact recall) and `stats()["routing_gate"]` says so; the remedy is `compact(recluster=True)`.

### 4.6 `compact(recluster=True)`

`global_recluster_order`: spherical k-means with K = ceil(N / block_capacity) over all rows (10 iterations, seed 0; *measured* 3 s at 71k), rows ordered by (cluster, original row); then `replace_all(records in that order)`. Restores routable blocks after unordered ingestion (random 39.4 -> 61.0 at 15 %, 47.5 -> 61.6 at 25 %). Changes file order, hence `get_all_records`/`/dump` order (documented). Never automatic; exposed as `Vault.compact(recluster=...)` and an optional `nanomem compact [--recluster]` cli subcommand.

### 4.7 Sweep protocol that binds the defaults

`bench/route_sweep.py` (port of `scratch/refound/sweep_routing.py` and the scratchpad `sweep71k.py`): corpora = cached 1,190 (insertion + random permutation), the perturbed 4,760 set, and `scratch/refound/hotpot_train_8k.json` subsets at 10k and 71,433 paragraphs (insertion, random, k-means layouts); grid C in {50, 256}, L in {1, 4, 8, 16, 32}, pool in {1, 2, 3, 5, 7, 9, 13, 21, max}, beam_frac in {0.016, 0.05, 0.10, 0.15, 0.25, 0.50}; metrics evidence-recall@4 and @10, gold-block coverage, NN-coverage (gate metric), candidates scanned, p50 ms; baselines at equal candidates scanned: single centroid (L=1), exhaustive, FAISS IndexFlatIP, FAISS IndexIVFFlat with nlist = n_blocks x L and nprobe matched; paired bootstrap 95 % CI (10k resamples) and McNemar discordant counts vs exhaustive (helpers copied from `benchmark_nanomem_multihop_scaled_1k.py`). Decision rules: `pool = "max"` unless a finite p beats it by > 0.5 pt on both large corpora; L = smallest value within 1.0 pt of the best L at the chosen beam; `beam_frac` = smallest grid value whose CI lower bound of (recall - exhaustive) >= -1.0 pt on the 71k insertion-order corpus. Output `bench/results/route_sweep.json` with `"chosen"`; `router.DEFAULTS` must equal it (test).

## 5. Write path, indexes, memtable, compaction, durability

### 5.1 `add_fact` (O(1) amortised; never reads a block)

```
def add_fact(self, text, embedding, source="user_input", metadata=None, timestamp=None, revision=None, id=None) -> str:
    with self._lock:
        self._reload_if_modified()
        meta = dict(metadata or {})
        ts = float(timestamp if timestamp is not None else meta.get("timestamp", time.time()))
        doc_id = str(id or meta.get("id") or "doc_" + hashlib.md5(f"{text}_{ts}".encode()).hexdigest()[:10])   # v2 formula
        meta["id"] = doc_id
        v = np.asarray(embedding, np.float32).reshape(-1)
        if v.size != self.embed_dim: raise ValueError(f"embedding has {v.size} dims, vault has {self.embed_dim}")
        v = v / (np.linalg.norm(v) + 1e-8)                             # fresh Ollama vector or a read_payload row
        is_chat = source == "chat_session" or meta.get("source") == "chat_session"
        uid = meta.get("user_id")
        if revision is None:                                           # POLICY LAYER (Vault.add from chat/proxy/cli only)
            if is_chat and not meta.get("entity"):
                ent = entities.detect_entity(text)
                if ent is None and entities.is_pronoun_led(text) and self._last_entity.get(uid):
                    ent = self._last_entity[uid]; meta["anaphora_resolved"] = True     # metadata only; vector untouched
                if ent: meta["entity"] = ent
            if meta.get("entity"): self._last_entity[uid] = meta["entity"]
            if meta.get("entity"):
                g = entities.make_group_key(uid, meta.get("project"), meta["entity"])
                gid = arena.intern_group(g)
                rev = arena.group_max.get(gid, (0, 0.0))[0] + 1        # O(1); includes pending rows
            else:
                rev = int(meta.get("revision", 1))
        else:
            rev = int(revision)                                        # RAW STORE: rebuild / merge / demo values verbatim
        eid = arena.intern_entity(meta.get("entity")); gid = arena.intern_group(make_group_key(uid, meta.get("project"), meta.get("entity")))
        memtable.append({... , "entity_id": eid, "group_id": gid}); memtable.vec[m] = v
        if gid >= 0: arena.group_max[gid] = (max(prev_rev, rev), max(prev_ts, ts))
        if memtable.n >= block_capacity: self._spill()
        return doc_id
```
No dedup (merge(deduplicate=False) promises zero loss), no anaphora rotation (v2's 30-degree re-rotation on every replay is gone). ~10-20 us per add, flat in N (v2 *measured* 0.41 ms/add at 500 docs rising to 3.85 ms at 4,000).

### 5.2 Spill (one block)

```
def _spill(self):
    batch = memtable.items[:C]; V = memtable.vec[:len(batch)]
    M = spherical_kmeans_landmarks(V, L, seed=next_seq) if len(batch) > L else None
    rec = encode_records(ts, rev, ids, groups, [json.dumps({"text","source","metadata"}, ...).encode() for ...])
    payload = (M.tobytes() if M is not None else b"") + V.astype(disk_dtype).tobytes() + rec
    header = pack_block_header(kind=1, n, m=len(M) if M is not None else 0, payload_len, written_unix, seq, nonce, ...)
    if encrypted: ct, tag = seal_block(keys, uuid, header[0:72], seq, payload); blob = header + ct + tag
    else:         blob = header + payload + plaintext_trailer(uuid, header[0:72], payload)
    meta = container.append_block(blob, sink=self._sink)              # section 5.4
    arena.add_block(meta, landmarks=M or V, vectors_f32=V, rec=decode_records(rec), rec_bytes=rec)
    drop batch from memtable
```
Per full block: k-means 0.5-0.7 ms + JSON 0.12 ms + hash/MAC 0.1-0.35 ms + write+fsync ~0.1-0.8 ms (*measured*) ≈ 1.5-2 ms => ~40 us/doc amortised.

### 5.3 `flush()`

`while memtable.n: _spill()`. Idempotent on an empty memtable. After return `memtable_pending == 0`, every record is in the arena, the toc, and on disk (fsync'd per `durable`), so any instance on the same path sees it after its next `_reload_if_modified()`.

### 5.4 Append protocol (the D1 protocol; header never rewritten)

```
def append_block(self, blob, sink):
    with open(path, "r+b") as f, file_lock(f, exclusive=True):
        st = os.fstat(f.fileno())
        if st.st_ino != self._ino or read_uuid(f) != self.header.vault_uuid:
            raise ContainerReplacedError                               # engine: full reload (memtable kept), retry once
        if st.st_size > self.valid_end:
            self.scan(sink, from_offset=self.valid_end)                # ingest other processes' valid appends
            if st.st_size > self.valid_end:                            # only a torn tail can remain
                os.ftruncate(f.fileno(), self.valid_end); self.truncated_tail_bytes = 0
        start = align_up(self.valid_end, 64)
        f.seek(self.valid_end); f.write(b"\x00" * (start - self.valid_end) + blob); f.flush()
        if durable == "fsync": os.fsync(f.fileno())
        elif durable == "full": fcntl.fcntl(f.fileno(), fcntl.F_FULLFSYNC) if darwin else os.fsync(...)
        self.valid_end = start + len(blob); self._stat = os.fstat(f.fileno())
        return BlockMeta(...)
```
`durable="none"` skips the per-block fsync (bulk ingest) and `flush()` performs one fsync at the end. *Measured*: fsync 0.08-0.8 ms per block, F_FULLFSYNC 4.0 ms.

### 5.5 `_reload_if_modified()` (top of add_fact/search/flush/stats/toc/read_payload; one `os.stat`)

`"same"` -> nothing. `"appended"` (same inode, size > valid_end) -> shared-locked tail scan from `valid_end` (a torn tail is left alone and counted). `"replaced"` (inode changed, or uuid changed, or size < valid_end) -> full reload from the file (memtable kept). `"missing"` -> reads continue from RAM; `add_fact`/`flush` raise `FileNotFoundError("vault file was removed")`.

### 5.6 `replace_all(records)` and `compact()`

Write a complete new file to `<path>.tmp-<pid>-<8hex>` (same header params, NEW `vault_uuid`, same `kdf_salt`, fresh nonces, seq from 0, `block_capacity` rows per block) streaming the records; fsync it; take the exclusive lock on the current file; `os.replace(tmp, path)`; fsync the directory (POSIX); release; reload. A crash before `os.replace` leaves the old file untouched; the temp is removed at the next open. Windows: `os.replace` fails while another process holds the file open; retried 5x with backoff, then raises (v2's `os.remove` had the same limitation; documented).

### 5.7 Concurrency

In-process: one `threading.RLock` per engine around every public method (search holds it for its full sub-millisecond duration; the numpy matmul releases the GIL). Cross-process: exclusive `flock` for appends and rewrites; shared `flock` for tail scans; readers detect appends/replacements by `os.stat`. Documented non-guarantee: two processes auto-assigning revisions for the same group at the same instant can produce equal revisions; the resolver then orders by timestamp.

## 6. Encryption

Default: plaintext, exactly like SQLite. `stats()["encrypted_at_rest"] = False`, `["cipher"] = "none (plaintext)"`. The v2 XOR keystream (compiled-in salt, no secret, no nonce, no tag) is retired; its constant survives only in `legacy_v2.py` for reading old files.

Optional password (`VaultEngine(..., password=...)`, `Vault(..., password=...)`, env `NANOMEM_PASSWORD` read by Vault only). All stdlib.

KDF (once per open):
```
pw = unicodedata.normalize("NFKC", password).encode("utf-8")
master = hashlib.scrypt(pw, salt=kdf_salt, n=2**log2_n, r=r, p=p, maxmem=256*1024*1024, dklen=32)   # defaults log2_n=16, r=8, p=1
K_enc = hmac.new(master, b"nanomem3:enc", hashlib.sha256).digest()
K_mac = hmac.new(master, b"nanomem3:mac", hashlib.sha256).digest()
K_chk = hmac.new(master, b"nanomem3:chk", hashlib.sha256).digest()
```
n=2^16 costs ~100 ms per open on this machine (*measured* 2^15 = 46-55 ms, 2^17 = 182-213 ms) and 64 MiB; it is chosen over 2^15 because keys are derived once per open and 2^15 is below current OWASP guidance; parameters are stored in the header so they can be raised on `rekey`. At create time, if `hashlib.scrypt` raises `ValueError`/`MemoryError` for the default, the engine retries with `log2_n - 1` down to 14, stores the value used, and warns.

Header: `header_auth = HMAC-SHA256(K_chk, b"NM3H" || header[0:88])` binds uuid, dims, flags, cipher_id, KDF params and salt; verified with `hmac.compare_digest` in the constructor BEFORE any block is read: mismatch -> `WrongPasswordError`; `ENCRYPTED` flag without a password -> `PasswordRequiredError`; password given for a plaintext file -> `ValueError("vault is not encrypted; use rekey(password)")`.

Per block:
```
nonce = os.urandom(16)
ks = hashlib.shake_256(b"NM3K" || K_enc || vault_uuid || nonce || seq.to_bytes(8, "little")).digest(len(payload))
ct = (np.frombuffer(payload, np.uint8) ^ np.frombuffer(ks, np.uint8)).tobytes()
tag = hmac.new(K_mac, b"NM3T" || vault_uuid || header[0:72] || ct, hashlib.sha256).digest()
```
Open order: header CRC -> tag (`compare_digest`) -> only then XOR and parse. Never decrypt unauthenticated bytes. (`K_enc, nonce, vault_uuid, seq`) is unique per block and per file generation, so keystream never repeats across rewrites and an authenticated block from another generation or another vault cannot be spliced in.

Why SHAKE256 instead of the suggested HMAC-SHA256 counter mode: same stdlib footprint and the same security argument (a keyed PRF stream under a unique nonce), but pure-Python HMAC-CTR yields 32 bytes per call (*measured* 5.9 ms per 160 KB, 27-33 MB/s) while one `shake_256().digest(n)` call yields the whole block keystream (*measured* 0.167 ms per 160 KB, ~1 GB/s), a 35x difference on every open, flush and rewrite. A keyed sponge with a secret prefix is the construction NIST standardised as KMACXOF256 (SP 800-185, with framing bytes Python cannot add because it has no cSHAKE); SHAKE256 is a FIPS 202 XOF with 512-bit capacity and no length-extension issue. Integrity is HMAC-SHA256 (FIPS 198-1). `cipher_id` bits in the header flags leave room for an HMAC-CTR suite if a reviewer insists, without a format bump.

Hot-path cost: search and add_fact do zero crypto (the arena and record bytes are plaintext in RAM); spill/flush pays one SHAKE + one HMAC + one XOR per block (~0.35 ms per 212 KB); open pays the KDF + ~2.5 ms/MB.

Error surfacing: `WrongPasswordError` / `PasswordRequiredError` / `CorruptContainerError` at open; `IntegrityError(block_index, offset, reason)` on tag or trailer mismatch of any complete non-tail block, at open or on a tail reload; default `on_integrity_error="raise"` fails closed; `"skip"` lists skipped indices in `stats()["integrity_errors"]` and search never silently drops blocks otherwise. `rekey(new_password)` = `replace_all` with new keys (None = decrypt to plaintext); `keys.wipe()` overwrites the bytearrays on close.

Threat model (to be quoted verbatim in the README and USER_MANUAL_DEVELOPER security section): Protects the confidentiality and integrity of every stored vector, text, id, metadata, timestamp and revision against an adversary who obtains the .dat file without the passphrase, and detects any bit modification, block reordering, or block substitution between vaults or file generations. Does NOT protect against: an adversary who can read process memory or swap while the vault is open (the arena is plaintext in RAM); rollback or truncation (trailing blocks can be deleted; there is no authenticated block count); traffic analysis of plaintext block headers (block count, per-block doc counts, payload sizes and write times are visible); weak passphrases (scrypt 2^16 slows an offline attacker to roughly 10 guesses per second per core on this hardware; a four-word diceware passphrase is the documented minimum); side channels in Python's hashlib. Not a NIST-approved AEAD mode, not FIPS-validated, not AES; the `cryptography` package is deliberately not a dependency. The stats string is `CIPHER_LABEL_SHAKE`; no "256-bit" marketing wording anywhere.

## 7. v2 migration

Detection at open: first 8 bytes `b"NANOMEM\x00"` with version 2 in `"<8sIIII40s"`.

`legacy_v2.iter_v2_records(path)` is a verbatim port of the v2 reader: header `"<8sIIII40s"` (magic, version, embed_dim, coord_dim, block cap); block header `"<4s32sIIfI"` (magic `b"BLK\x00"`, block id 32s, n_docs, ts u32, radius f32, payload_sz) followed by the centroid (`embed_dim*4` bytes); keystream = SHA-256 chain `h0 = SHA256(block_id32 || V2_CIPHER_SALT)`, `h_{i+1} = SHA256(h_i || block_id32)` in 32-byte chunks; payload = `"<II"` (meta_len, r_actual) + meta JSON `{texts, sources, metadatas}` + keys (n x 64 f32) + values (n x D f32) + basis (r_actual x D) + mean (D) + timestamps (n f64) + revisions (n i32), 4 KB-padded; stops at the first bad magic / short centroid / incomplete payload exactly as v2's `reload()` does. For each record: `id = metadatas[i].get("id") or "doc_" + md5(f"{text}_{timestamps[i]}").hexdigest()[:10]` with the numpy float64 formatted by the f-string exactly as `Vault.get_all_records` does (*verified* to reproduce 80/80 ids in `book_v2_golden.json`); `embedding = values[i]` re-normalised; text/source/metadata/timestamp/revision verbatim; keys/basis/mean discarded (never read by anything).

Entity repair (Design 2, adjusted so column and metadata agree): `derive_entity_for_migrated(meta, text)` returns `normalize_entity(meta["entity"])` if that value matches `^[a-z0-9_]+$` (an explicit or class-like tag), otherwise `detect_entity(text)`. If the result differs from the stored `metadata["entity"]`, the migrated record's metadata gets `entity = repaired` and `entity_v2 = original` (so the group column, `Vault.merge`'s `_detect_entity(meta)` and the temporal resolver all see the same value; the original is also preserved in `.v2.bak`). Revisions are NOT renumbered; v2's junk-group revisions (e.g. `Update` -> rev 1) therefore tie on revision and are ordered by timestamp, which is what the lexicographic resolver does.

Procedure (`migrate=True`, default): (1) `shutil.copyfile(path, path + ".v2.bak")` (or `.v2.bak.<unix>` if it exists; skipped with `migrate_backup=False` / env `NANOMEM_KEEP_V2_BACKUP=0`); (2) stream `iter_v2_records` through `write_new_file` into `<path>.tmp-...` (with the constructor's password/vector_dtype/L); (3) fsync; (4) `os.replace(tmp, path)` under the exclusive lock; (5) log one line to stderr: `nanomem: migrated <path> to v3 (<n> docs); original kept at <path>.v2.bak`. The vault path exists at every instant. Read-only directory or `migrate=False`: the v2 records are loaded into the arena, `stats()["read_only"] = True` and `["format_version"] = 2`, searches work, `add_fact`/`flush` raise `ReadOnlyVaultError`. Header-only 64-byte v2 files (personal_memory.dat) become header-only v3 files. v3 -> v2 downgrade is unsupported; `.v2.bak` is the escape hatch. Live files: user profile `memory_*.dat` (list_users triggers migration per profile), `personal_memory.dat`, `hands_on_llm_vault.dat` x3, `my_demo_vault.dat`. *Measured* cost: chat_v2.dat (14 docs) 3.5 ms parse; book_v2.dat (845 docs, 6.35 MB) 74 ms parse + ~30 ms write.

## 8. Generic entity and temporal logic (`entities.py`)

Pure functions; module-level tuple/dict constants; no proper nouns, brands, breeds, titles or product names. All matching on NFKC-normalised, lower-cased text.

8.1 `normalize_entity(s)`: NFKC, lower, non-alphanumerics -> `_`, collapse, strip, `favourite` -> `favorite`, cap 48 chars.

8.2 Structural classes (ordered; first match wins):
```
emergency_contact  r"\bemergency contact\b"
credential         r"\b(api[\s_-]?key|secret[\s_-]?key|password|passcode|passphrase|access[\s_-]?token|auth[\s_-]?token|bearer token|jwt|pin code|private key)\b"
email_address      EMAIL_RE = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
url                r"https?://\S+"
ip_address         r"\b\d{1,3}(\.\d{1,3}){3}\b"
phone_number       (r"\b(phone|mobile|cell|cellphone|telephone|whatsapp|tel)\b" and PHONE_RE) or (PHONE_RE and r"\bmy\b")
                   PHONE_RE = r"(?<![\w])\+?\d[\d\s().-]{6,}\d(?![\w])" filtered to >= 8 digits (accepts "020-4455-7788", which v2 rejected)
birthday           r"\b(birthday|born on|date of birth|dob)\b"
allergy            r"\b(allergic|allerg(y|ies))\b"
location           r"\b(live in|living in|moved to|moving to|based in|relocated to|my (home |new |current )?address|neighbou?rhood|apartment in|house in|hometown)\b"
career             r"\b(work(s|ing)? (as|at|for)|job title|my (new |current )?(title|role|position)|role is|promoted|promotion|hired (at|as)|joined \w+ as|employer|my company|profession)\b"
name               r"\b(my name is|call me|i go by|everyone calls me)\b"
```
8.3 Slot entities (generic possessive/ownership extraction; replaces v2's pet/gaming/keyboard/routine regexes):
```
P1  r"\bmy (?P<slot>[a-z][a-z'\- ]{1,40}?)\s+(?:is|are|was|were|will be|has been|=|:)\s"
P2  r"\bmy (?P<slot>[a-z][a-z'\- ]{1,40}?)'s name is"
P3  r"\bi (?:use|drive|own|have|play|prefer|take|wear|ride)\s+(?:a|an|the|my)?\s*(?P<slot>[a-z][a-z'\- ]{1,40}?)(?:\s+(?:named|called|is|that|which)|[,.]|$)"
P4  r"\b(?:every|each) (morning|evening|night|day|week|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b" -> "routine"
```
Normalisation of a slot: drop leading modifiers in `MODIFIER_STOP = {new, old, current, own, little, big, first, second, main, primary, daily, usual, other, latest, go-to}`; keep `favorite/favourite, work, personal, home`; singularise a trailing `s` (len > 3, not `ss`); collapse to `_`; cap at 3 tokens; reject if the head token is in `GENERIC_HEADS = {day, time, life, week, morning, evening, thing, question, plan, idea, point, turn, problem, guess}` ("my day was great" -> None). Then map through `SYNONYM_CLASSES`:
```
phone_number  <- {phone, mobile, cell, cellphone, telephone, number, whatsapp, phone_number, mobile_number, cell_number, contact_number}
email_address <- {email, e_mail, mail, email_address}
career        <- {job, work, title, job_title, role, position, employer, company, career, profession, workplace}
location      <- {home, address, city, location, apartment, house, flat, neighborhood, neighbourhood, hometown, residence}
pet           <- {dog, cat, puppy, kitten, pet, animal, hamster, rabbit, parrot, bird, fish}
allergy       <- {allergy, allergic, allergies};  birthday <- {birthday, birthdate, dob, date_of_birth}
credential    <- {api_key, token, password, secret, credential, passphrase, pin};  name <- {name, preferred_name, nickname, full_name}
```
`detect_entity(text)`: structural class, else normalised slot, else None. No capitalised-name fallback of any kind. `is_pronoun_led(text) = ^(he|she|it|they|his|her|its|their|that)\b`.

8.4 `query_intent(query)`: cue-word table (phone/mobile/cell/telephone/number -> phone_number; email/mail -> email_address; live/living/address/city/where do -> location; work/job/career/company/title/employed/employer -> career; pet/pets/dog/cat/animal -> pet; allergic/allergy/allergies -> allergy; birthday/born/birthdate -> birthday; api key/token/password/secret/credential -> credential; "emergency contact"; routine/schedule; "my name"/"who am i" -> name), else a slot from `\b(what|which|who|where|when|how)\b.*\b(?:my|his|her|their)\s+(?P<slot>[a-z][a-z'\- ]{1,40}?)(?:\?|$|\s+(?:is|was|are|do|does|did|now|currently|originally|these days|again))`, `(about|of) my (?P<slot>...)`, or `what (?P<slot>...) do i (use|have|drive|ride|play|own)`, normalised as in 8.3. LRU(256) per query string.

8.5 Matching and temporal resolution: `entities_match(a, b)` after normalisation = equal, or same synonym class, or token-set containment either way with modifier tokens removed ("phone" vs "phone_number"; "game" vs "favorite_video_game"; demo.py's explicit `entity="phone"` matches the `phone_number` intent), or credential fuzzy (any of key/secret/token/auth in both). `matching_ids(entity, entity_names)` returns the int ids of stored names that match; cached per `(entity, len(entity_names))`.
`SINGLE_VALUED = {phone_number, email_address, location, career, credential, emergency_contact, birthday, name, url, ip_address, routine}` plus any entity starting with `favorite_`; `is_single_valued(e)` uses `entity_class`. `TEMPORAL_CUES = {current, currently, now, latest, newest, updated, new, today, original, originally, first, previous, previously, old, before, "used to", earlier}`.
`apply_temporal_boosts` (vectorised): (1) `+0.25` where `entity_id in intent_ids`; (2) `top_entity = intent or entity_names[entity_id[argmax(masked base + 0.25 boost)]]` (None if that row has no entity); (3) if `top_entity` and (`is_single_valued(top_entity)` or `temporal_direction == "historical"` or `has_temporal_cue(query)`): `comp = mask & isin(entity_id, matching_ids(top_entity))`; if `comp.sum() > 1`: rank rows lexicographically by `(rev, ts)` with ties sharing a rank, `u = rank / (n_distinct - 1)`, `boost[comp] += u` for "current" or `1 - u` for "historical". Magnitudes are v2's, so demo.py still prints a score above 1 (documented as "score = calibrated cosine + boosts; see hit['cosine']"). `min_score` was applied before any boost.
Revision groups: `make_group_key(user_id, project, entity) = f"{str(user_id or '').strip()}\x1f{str(project or '').strip()}\x1f{normalize_entity(entity)}"`; auto revision = `group_max + 1`; explicit revision wins; `Vault.merge(reconcile_revisions=True)` keeps computing its own reconciled revisions and passes them explicitly. Non-English text yields no entity and plain calibrated-cosine ranking (documented; explicit `metadata['entity']` works in any language).

Chat golden walk-through with these rules (`scratch/refound/golden/chat_v2_golden.json`, migrated): "current phone number" -> intent phone_number; group = "My phone number is 020-4455-7788." (rev 1, t1) and the migrated "Update: my new phone number is 020-9911-2233" (v2 entity `Update` repaired to phone_number, rev 1, t2) -> equal revisions -> timestamp order -> current 020-9911-2233, historical 020-4455-7788 (v2 failed the historical case). "Who is my emergency contact and their number?" -> single-valued -> rev 2 -> 98220-99887 (v2 failed: no temporal cue). Expected 12/12; the gate is >= 11/12 because this is a walk-through until measured.

## 9. `stats()` fields and how each is measured

Legacy keys (exact set preserved, same types): `file_path` (str, abspath), `file_size_mb` (`os.path.getsize/1e6`, 3 dp), `total_documents` (`arena.n_rows + memtable.n`), `compacted_blocks` (DATA block count), `memtable_pending` (`memtable.n`), `active_heap_ram_kb` (sum of `arena.resident_bytes()` = `vec.nbytes[:n]` + `landmarks.nbytes[:M]` + all column `.nbytes[:n]` + `sum(len(b) for b in rec_bytes)` + `memtable.vec.nbytes[:m]` + `index_bytes_estimated` (= `sys.getsizeof(id_index) + 64 * n_rows`, labelled estimated), divided by 1024, rounded to 0.1; never a constant, never clamped), `encrypted_at_rest` (header flag), `cipher` (label string from crypto.py).
New keys: `format_version` (3, or 2 in read-only v2 fallback), `engine_version` ("3.0.0"), `embed_dim`, `vector_dtype_on_disk`, `block_capacity`, `landmarks_per_block`, `score_scale` ("legacy" | "cosine"), `routing_mode` ("exhaustive" | "landmark"), `routing_gate` ({"coverage": float|None, "passed": bool|None, "checked_at_blocks": int}), `n_exhaustive`, `beam_fraction`, `beam_blocks` (current beam size or 0), `arena_bytes`, `landmark_bytes`, `column_bytes`, `record_bytes`, `memtable_bytes`, `index_bytes_estimated`, `process_peak_rss_kb` (`resource.getrusage(RUSAGE_SELF).ru_maxrss`, converted from bytes on darwin; None where `resource` is unavailable), `process_rss_kb` (`/proc/self/statm` resident pages x page size on Linux; None on macOS), `integrity_errors` (list[int]), `truncated_tail_bytes` (int), `read_only` (bool), `durable` (str), `kdf` ("scrypt n=65536 r=8 p=1" | None), `vault_uuid` (hex). All values JSON-serialisable; `stats()` takes the RLock and calls `_reload_if_modified()` so appends by other instances are reflected; it performs no I/O beyond `os.stat`/`getsize` (and a shared-locked tail scan when the file grew).

## 10. Test plan, acceptance thresholds, benchmark scripts

Location `nanomem_standalone/tests/` (pytest; numpy only; Ollama never called: vectors come from the cached npz files or seeded random unit vectors; faiss used only when importable). Every threshold is an `assert`. Every benchmark JSON records `n, corpus, k, seed, machine, numpy version, embedder actually used, engine constants`. Gates marked [G] must be green on macOS and Linux before any package copy is synced.

Format and durability
1. [G] `test_create_eager`: missing path -> exactly one 256-byte file; second construction on the open file works; `close()`, `os.remove`, construct again works; directory contains only the .dat after any sequence of adds/flushes.
2. [G] `test_block_roundtrip`: 1..200 records with unicode text, nested metadata, explicit ids/timestamps/revisions; flush; reopen; `iter_records` equals input (ids, ts to 1e-9, revisions, metadata equality; embeddings cosine > 0.99999 under fp16, allclose 1e-6 under fp32); `container.toc + read_payload` returns owned copies (`np.shares_memory` False; mutating returned dicts/arrays does not change a re-read); `stats()['memtable_pending'] == 0`; flush on empty is a no-op; the groups table equals `make_group_key(meta...)` for every record.
3. [G] `test_rebuild_idempotent`: `replace_all(iter_records())` three times -> identical records each time, identical file size between rounds 2 and 3 (only uuid/nonces/timestamps differ); the unchanged vault.py update dance run 3 times on a chat-style vault (pronoun-led facts, duplicate texts) keeps count and revisions stable.
4. [G] `test_no_dedup_no_rotation`: same text+entity 10 times with explicit revision -> 10 records; a pronoun-led chat fact's stored embedding equals its input (cos > 0.999999) and carries `anaphora_resolved`.
5. [G] `test_truncation_recovery`: 20 blocks; for 50 random cut points (block header, mid-payload, mid-trailer, padding) open succeeds with exactly the complete blocks, `truncated_tail_bytes > 0`, the next append lands at `valid_end` and the file re-scans cleanly.
6. [G] `test_bit_flip_detection`: one byte flipped in header/payload/trailer of a random non-tail block -> `IntegrityError` naming the block index (plaintext SHA-256 and encrypted HMAC); `on_integrity_error="skip"` opens with the index in `stats()['integrity_errors']`; an immutable header byte flip -> `CorruptContainerError`/`WrongPasswordError`.
7. [G] `test_atomic_replace`: `os.replace` monkeypatched to raise mid-`replace_all` -> original unchanged, temp removed at next open; a manually planted stale `.tmp-*` is removed at open.
8. [G] `test_concurrent_processes`: 4 subprocesses each add 1,000 records with flush every 50 into one vault for 5 s; parent opens: `total_documents == 4000`, no `IntegrityError`, all ids present; a reader process polling `stats()` sees monotonically non-decreasing counts; a second in-process instance opened before A's flush sees A's records after `A.flush()`.
9. [G] `test_thread_safety`: 4 add+flush threads, 4 search threads, 1 stats thread for 3 s: no exceptions, final count correct, every result list internally consistent.
10. [G] `test_append_never_truncates_valid_blocks`: process B appends a valid block while A holds a stale `valid_end`; A's next append re-scans and its block lands after B's; nothing lost. Header bytes are byte-identical before and after 100 appends.
11. [G] `test_embed_dim`: a 384-d vector into a 768-d file raises `ValueError` naming both dims; opening a 768-d file with `embed_dim=384` warns and adopts 768.

Encryption
12. [G] `test_kdf_and_password`: `derive_keys` deterministic for fixed salt/params; wrong password -> `WrongPasswordError` before any block read (monkeypatched read counter == 0); no password on an encrypted file -> `PasswordRequiredError`; password on a plaintext file -> `ValueError`; NFKC-equivalent passphrases open the same vault.
13. [G] `test_encrypt_roundtrip`: 1,000 records; reopen with the password -> identical records; the file bytes contain none of 50 known text substrings, ids, metadata values, or the fp16/fp32 byte patterns of 20 stored vectors; 300 blocks -> 300 distinct nonces; `replace_all` produces a new `vault_uuid` and new nonces; a block copied from the pre-rewrite file into the post-rewrite file fails its tag.
14. [G] `test_encrypted_hot_path_no_crypto`: monkeypatched `hashlib`/`hmac` counters show zero calls during 200 searches on a resident encrypted vault; p50 within 10 % of the plaintext vault on the same data. `rekey` round-trips encrypted -> new password -> plaintext -> encrypted.
15. [G] `test_stats_cipher_fields`: plaintext -> `encrypted_at_rest False`, `cipher "none (plaintext)"`, `kdf None`; encrypted -> `True`, the exact `CIPHER_LABEL_SHAKE`, `"scrypt n=65536 r=8 p=1"`.

Compatibility
16. [G] `test_v2_migration_golden`: copy `scratch/refound/golden/book_v2.dat` (845 docs, 17 blocks) to tmp; open -> `format_version 3`, 845 records, texts/ids/sources/metadata/timestamps/revisions identical to a direct `iter_v2_records` read, embeddings cosine > 0.99999; `.v2.bak` byte-identical to the source; all 80 golden ids present; for the 20 golden queries the top-4 ids equal exhaustive numpy cosine over the migrated vectors (authoritative check); `chat_v2.dat` -> 14 records, `expected_top1_substring` correct on >= 11/12 queries (target 12/12; v2 got 10/12); second open is a plain v3 open; `NANOMEM_KEEP_V2_BACKUP=0` leaves no backup; a 64-byte v2 file migrates to an empty v3 vault; `migrate=False` and a `chmod 0o555` directory -> searches work, `add_fact` raises `ReadOnlyVaultError`.
17. [G] `test_vault_contract_unchanged`: the shipped vault.py (an unmodified copy) against the new engine: add/add_batch/get/get_all_records/update/delete (id, ids, text_exact, text_contains, where, source)/export(purge)/split*/unmerge/prune/merge(deduplicate=False and True, reconcile_revisions=True)/search_multihop/search_multi/stats/close all work with the expected record counts; `get_all_records` order == insertion order; values rows survive `close()+os.remove()` and `with Vault(other)` exit; `merge(deduplicate=False)` of two vaults with duplicate texts loses nothing.
18. [G] `test_vault_edits`: with the five listed edits: `Vault.add` returns the id (both branches) and sets `last_id`; cli prints it; `_detect_entity` contains no regex literal; `password=` reaches the engine.
19. [G] `test_search_result_types`: `json.dumps(hits)` succeeds; keys exactly `id, doc_id, text, source, metadata, score, cosine, timestamp, revision` with Python types; fresh dict objects per call; sorted desc; `len <= top_k`; unflushed records returned (demo.py sequence: current -> rev 2 "9899999999", historical -> rev 1 "9811111111"); `min_score` applied before boosts on the legacy scale (a record with legacy_score 0.24 and a +1.0 boost is excluded at `min_score=0.25`); `score_mode="cosine"` changes the scale and `legacy_score_to_cosine(0.25) == 0.417 +- 1e-3`, `(0.32) == 0.533`, `(0.35) == 0.58`.
20. [G] `test_matches_filter_parity`: 200 random (meta, filter) cases against the v2 function copied into the test; `metadata_filter={'user_id': ...}` search returns only matching rows and reaches `top_k` when enough exist, including for a rare value present in exactly one record.
21. [G] `test_container_view`: toc entries have `id/doc_count/offset/length`; `read_payload(offset, length, doc_count, id)` returns the six keys with `len == doc_count`, `'id'` present in every metadata dict.

Entity / temporal
22. [G] `test_no_fixture_vocabulary`: grep `nanomem/engine.py, entities.py, vault.py, container.py, arena.py, router.py, crypto.py, legacy_v2.py` for the case-insensitive tokens {barnaby, ergodox, disco elysium, catan, golden retriever} and the benchmark_200 chaff phrase list -> zero hits (strict); `classifier.py` is `xfail` until its companion cleanup lands; also asserts "Update: my new phone number is 1234567890" -> phone_number and "I bought a Royal Enfield Hunter 350" -> None.
23. [G] `test_detect_entity_generic`: 40 hand-written generic sentences -> expected classes/slots; 20 negatives -> None; `entities_match` table (phone/phone_number, game/favorite_video_game, key/api_key, address/location); `normalize_entity` aliases.
24. [G] `test_auto_revision_o1`: 5,000 chat facts across 50 `(user_id, project, entity)` groups: revisions 1..k per group; time per add in adds 3,900-4,000 <= 1.5x that in adds 400-500 and both < 100 us; a block-read counter stays at zero during adds.
25. [G] `test_temporal_direction`: 3 revisions of a generic slot; "current" -> highest revision; "historical" with a temporal cue -> revision 1; a non-single-valued class without a cue gets no superseding boost; two pets coexist; equal revisions fall back to timestamps; in forced landmark mode an old low-cosine revision of the queried entity is still in the result (entity-history injection).
26. [G] `test_clean_chat_no_regression`: run `scratch/refound/benchmark_clean_chat.py <pkg> v3` (3 personas, n=36) and its held-out variant (`clean_chat_benchmark_heldout.json`, n=24) through the migrated package: temporal verification with gold stores must be >= the current-engine baseline (top1 50.0 / top3 86.1 and 33.3 / 95.8 respectively, *measured*); target +10 pt top1; classifier numbers are recorded, not gated.

Retrieval, routing, latency, size, RAM
27. [G] `test_hotpot_1190_recall` (skips without `scratchpad/embeds.npz` + `scratch/hotpotqa_scaled_1k.json`): ingest the 1,190 vectors via `add_fact`; 120 queries, `top_k=4`; evidence-recall@4 >= 68.3 and the ranked id sets equal brute-force numpy cosine for all 120 queries; the same with `vector_dtype="float32"`.
28. [G] `test_hotpot_4x_recall`: the orchestrator's perturbed-copy construction (3 perturbed copies per doc, seed logged) at 4,760 docs: recall@4 >= 45.8 under the default fp16 and ranked ids == brute force; if it fails only under fp16 the test prints the tie and the ship rule flips `vector_dtype` default to `"float32"` (decision recorded in `bench/results/`).
29. [G] `test_routed_mode_shipped_constants`: forced landmark mode (`n_exhaustive=0`) with the SHIPPED constants (C=50, L=8, max, beam_frac 0.25, beam_min 4) on the 1,190 set: recall@4 >= 68.3; on a 10k train subset (`hotpot_train_8k` insertion order, 500 questions): recall within 1.5 pt of exhaustive; single-centroid emulation (`landmarks_per_block=1`) at the same beam is >= 3 pt worse (guards the landmark table); `layout_gate()` passes on the insertion-order corpus and fails on its random permutation; `compact(recluster=True)` on the permuted vault makes it pass and brings recall within 1.5 pt of exhaustive.
30. [G] `test_latency_targets`: seeded unit vectors, 200 warm queries, `time.perf_counter` around `VaultEngine.search` (temporal logic on, `top_k=4`): 10k docs p50 < 0.5 ms and p95 < 1.0 ms; 1,190 docs p50 < 0.2 ms; 50k p50 < 2.0 ms; 100k routed p50 < 100k exhaustive p50 (ratio reported); encrypted within 10 % of plaintext. The test records `os.getloadavg()`; *measured* floor is the matmul (0.21-0.39 ms at 10k depending on load).
31. [G] `test_write_complexity`: ms/add in windows of 500 adds at 500, 4,000 and 20,000 docs (chat style, entity set, `revision=None`): max/min ratio <= 1.5 (v2: 9.4x).
32. [G] `test_index_size`: 1,190 HotpotQA paragraphs: fp16 file <= 0.85x and fp32 file <= 1.3x of `sum(len(text.encode())) + N*768*4`; migrated book vault <= 2.6 MB (fp16); 1-doc chat blocks <= 4 KB each and `compact()` shrinks 200 of them to ceil(200/50) blocks.
33. [G] `test_stats_measured`: `active_heap_ram_kb` equals the sum of array nbytes + record bytes + the labelled index estimate within 5 % at 0 / 1k / 10k docs and grows with N; two vaults of different sizes report different values; `process_peak_rss_kb >= active_heap_ram_kb`; `routing_mode` flips when `n_exhaustive` is lowered below N; no field is a constant; all `json.dumps`-able; `stats()` from a thread under concurrent add/flush never raises.
34. [G] `test_defaults_bound_to_sweep`: `router.DEFAULTS == json.load(bench/results/route_sweep.json)["chosen"]`.
35. [G] `test_no_forbidden_imports` and `test_wheel_contents` (new modules present, tests excluded, `assets/manifold_prototypes.npz` included, md5 of `nanomem/*.py` identical across the five package copies via `scripts/sync_packages.py --verify`).

Benchmark scripts (print + JSON to `bench/results/`; run manually; their tables replace every latency/RAM/cipher claim in the six doc copies)
- `bench/route_sweep.py`: section 4.7 grid on 1,190 / 4,760 / 10k / 71,433 with insertion, random and k-means layouts, FAISS IVF baseline, gate metric, paired bootstrap; writes `route_sweep.json` with `"chosen"`.
- `bench/head_to_head.py`: per corpus {1,190; 4,760 perturbed; 10k; 71,433} rows for v3-exhaustive, v3-landmark (shipped constants), v2 engine (imported from the `.v2.bak`-era wheel), FAISS IndexFlatIP, FAISS IndexIVFFlat (nlist = n_blocks x L, nprobe matched to the same scanned fraction); evidence-recall@4/@10 with paired bootstrap 95 % CI and McNemar vs FAISS flat, p50/p95 ms (vector in hand, Ollama round-trip ~9 ms reported separately), index bytes, `active_heap_ram_kb`, peak-RSS delta. Acceptance: v3-exhaustive recall == FAISS flat on every corpus; v3-landmark >= FAISS IVF - 1.0 pt at equal candidates on 71k.
- `bench/scale_v3.py` (port of `scratch/refound/scale_baseline_current.py`): 10,000 and 71,433-paragraph subsets, 500 questions; ingest_s, index_MB, engine p50/p95, engine recall, exhaustive numpy p50/recall, FAISS flat p50/recall, peak_rss_MB, rss_delta_MB. Current-engine baseline (*measured*, `scale_results_current_engine.json`): 10k -> 44.0 ms p50, 24.0 % recall vs 70.4 exhaustive/FAISS, 71.8 MB index, rss delta 218 MB; 71k -> 461 ms, 5.8 % vs 60.6, 511 MB index, rss delta 330 MB. Acceptance at 10k: engine recall == 70.4 (exhaustive), p50 < 0.5 ms, index <= 1.3x raw (fp32) with the fp16 figure reported; at 71k: landmark recall >= 60.6 - 1.0 with the gate passed, p50 and RSS reported.
- `bench/write_scaling.py` (ms/add and ms/flush vs N to 50k, plaintext and encrypted, durable fsync/full/none), `bench/latency_vs_n.py`, `bench/rss_probe.py` (subprocess per N in {1,190; 5k; 10k; 50k}: ru_maxrss delta vs numpy-only baseline next to `active_heap_ram_kb`).
- Existing scripts to re-run and record: `nanomem_standalone/demo.py` (must print rev 2 then rev 1; converted to assert), `benchmark_nanomem_multihop_scaled_1k.py --no-llm` (single-pass recall must equal the exhaustive 68.3 at top_k=4; the old 70.8 was routing noise), `benchmark_nanomem_multihop_ab.py` (recorded), `scratch/refound/benchmark_clean_chat.py` (3-persona and held-out), `benchmark_200_friendship_chat.py` (informational only, contaminated; printed as out-of-sample with the embedder recorded), `ingest_and_test_book.py` against the local PDF only (not shipped) or against the migrated golden book vault (15 keyword checks), `nanomem_standalone/server.py` smoke via curl `/health /ingest /search`.

## 11. Sync and rebuild steps for all package copies

1. Land the new modules and the five edits in `nanomem_standalone/nanomem/` (canonical) and `nanomem_standalone/tests/`, `nanomem_standalone/bench/`; run the [G] suite on macOS and Linux.
2. `python3 scripts/sync_packages.py --sync`: copies `nanomem/*.py` + `assets/manifold_prototypes.npz` to `Launch 1/shared/nanomem` (wheel source; no mcp.py by design), `nanomem_mac_bundle/nanomem` and `Launch 1/Mac/nanomem_mac_bundle/nanomem` (keep their mcp.py), `Launch 1/nanomem_standalone/nanomem`; deletes `Launch 1/shared/build/` and `nanomem.egg-info/` so the staging copy is regenerated, not hand-edited.
3. Rebuild the wheel from `Launch 1/shared` (`python3 -m build --wheel` or `setup.py bdist_wheel`; setuptools `find_packages` picks up the new modules; `tests/` and `bench/` are not inside the package); fan the wheel out to the six canonical locations (repo root, `nanomem_standalone/`, `nanomem_mac_bundle/`, `Launch 1/nanomem_standalone/`, `Launch 1/Mac/nanomem_mac_bundle/`, `Launch 1/shared/dist/`); delete the six stale wheels in `Launch 1/{Mac,Linux,Windows}/` and their `dist/`; fix the three platform build scripts so they no longer `rm -rf dist` inside `shared/`.
4. Migrate the shipped fixtures once (`hands_on_llm_vault.dat` x3 -> v3 with `.v2.bak` deleted from the bundle) or leave them v2 to migrate on first open; delete the stray `Launch 1/Mac/nanomem_mac_bundle/my_demo_vault.dat`; drop the never-read 146 MB `assets/model.bin` from both mac bundles; re-zip `Launch 1/nanomem_standalone.zip` and `Launch 1/nanomem_mac_bundle.zip` without `__pycache__`.
5. Sibling scripts duplicated per bundle (chat.py x5, server.py x5, demo.py x4, benchmark_200 x5, ingest_and_test_book.py x3, query_book.py x3, benchmark_multi_questions_in_one.py x3): no API change is required for 3.0 because thresholds keep their scale; copy any converted-to-assert versions.
6. Docs (README, USER_MANUAL, USER_MANUAL_PERSONAL, USER_MANUAL_DEVELOPER, SERVICES_AND_API_SPECIFICATION, MULTIHOP guide, x5-6 copies) get the measured tables from `bench/results/` and the threat-model text; every "160 KB", "0.44 ms", "256-bit", "encrypted at rest by default", "sub-linear" and "4D engine" wording is replaced; `server.py:82`'s service string loses "4D".
7. `python3 scripts/sync_packages.py --verify` exits 0 only when every `nanomem/*.py` and the six canonical wheels have identical md5; CI runs it after the test suite. Note `.gitignore` blocks `*.npz`/`*.bin`, so `assets/manifold_prototypes.npz` must be whitelisted (`!nanomem/assets/*.npz`) before the package can be committed.

## 12. Expected numbers (targets vs. what this design will report)

| Target | v2 (measured) | v3 expected | Basis |
|---|---|---|---|
| recall@4, 1,190 docs >= 68.3 | 71.7 (routing artefact, p=0.125) | 68.3 exhaustive == FAISS flat | exact scan |
| recall@4, 4,760 docs >= 45.8 | 45.0 | 45.8 == FAISS flat (fp16 gated by test 28) | exact scan; fp16 changed 2/2000 top-4 sets at 71k |
| p50 at 10k docs < 0.5 ms | 44.0 ms (24 % recall) | 0.3-0.45 ms | matmul 0.21-0.39 + ~0.06 post |
| growth beyond 10k | linear in decrypted blocks | exact and linear to 50k (1.6-2 ms at 50k); landmark mode ~1.3-1.5 ms at 100k (vs 1.9-3.3 exhaustive), linear thereafter | lat.txt; NOT sub-linear (see open decisions) |
| writes O(1) | 0.41 -> 3.85 ms/add | ~15-60 us/add flat | dict revision index; per-block spill amortised |
| index size <= 1.3x raw | 1.89x (book) | 0.69-0.76x fp16, 1.07-1.19x fp32 | section 2.2 |
| RAM | 109 MB RSS at 5k; stats says 160 KB | ~4.6 MB engine at 1,190; ~38 MB at 10k; ~380 MB at 100k; reported, plus process RSS | arena 3 KB/doc + landmarks + records |
| encryption | obfuscation labelled encryption | plaintext default; opt-in authenticated stream, zero crypto per query | section 6 |
| open | 13 ms at 845 docs | ~5 ms at 1,190; ~0.15 s at 100k plaintext, +100 ms KDF encrypted | section 2.3 |
