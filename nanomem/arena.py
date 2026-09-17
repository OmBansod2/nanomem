"""
nanomem.arena
~~~~~~~~~~~~~
The resident RAM side of a vault: one vector arena plus parallel columns, built
once at open and extended incrementally as blocks arrive.

Row ``r`` of every array is the same document, and rows are in file order, so a
search is one ``(n, D) @ (D,)`` matmul followed by vectorised column work. Only
the ``top_k`` winners are ever materialised back into Python dicts.

RESIDENCY. Vectors are fp16 on disk (DECISIONS.md core-5) and the arena upcasts
every one of them into a resident fp32 array. As committed (git 6aa6923, engine
3.0.4) that array grew by DOUBLING, which cost 643.6 MB of RSS to SERVE a
71,433-row vault -- and the measurement says the fp32 arithmetic is not what
paid for it:

  * the doubling overshoot -- the fp32 arena really allocated 131,072 rows,
    384.0 MB to hold 209.3 MB of vectors, and
  * the copies doubling leaves behind: 383.8 MB of discarded arrays that this
    allocator does not hand back.

:meth:`Arena.reserve` removes BOTH at once for a vault that is OPENED: the
container counts the rows from block headers before it reads a single one, the
arena allocates each array exactly once, and nothing is ever copied. Measured on
a reopened 71,433-document vault: 643.6 MB of resident RSS becomes 286.0 MB,
with all 500 queries' top-10 unchanged (scratch/refound/memory_results.json,
``summary.reopen_only.n71433``).

That fixed the SERVER and left the LOADER, which is where nanomem lost: a
process that INGESTS 71,433 documents one ``add_fact()`` at a time is told
nothing about how many are coming, so it grew by doubling and peaked at 826.4 MB
of ru_maxrss to build-then-serve, against 230.6 MB for FAISS IndexFlatIP and
290.5 MB for Chroma's default HNSW on the same corpus
(scratch/refound/competitors_standard_results.json). THIS CHANGE fixes it in
the allocator rather than in the caller: :class:`_VectorStore` holds the
vectors in a lazily committed RESERVATION and grows by re-viewing it, so growth
never copies and the array is still cut to the exact row count. Measured by
scratch/refound/bench_ingest_ram.py, which imports the committed package from a
git checkout as its own baseline arm, two independent runs each (``summary``,
``replicate``):

  71,433 docs         loader peak RSS       server RSS   p50       top-10
                      median of 4 [range]                            changed
  committed, no hint  820.5 [819.7-823.8]     289.5 MB  1.8702 ms      --
  committed + hint    288.5 [288.0-294.6]     287.6 MB    --         0 / 500
  reserved, no hint   298.4 [287.9-308.3]     287.6 MB  1.8622 ms    0 / 500
  reserved + hint     289.5 [287.6-308.6]     287.7 MB  1.8718 ms    0 / 500
  Vault.add_batch     313.7 [313.4-314.6]     285.4 MB    --         0 / 500

  10,000 docs         loader peak RSS       server RSS   p50       top-10
  committed, no hint  103.7 [103.5-104.0]      42.6 MB  0.3848 ms      --
  committed + hint     43.1 [ 42.5- 43.3]      42.6 MB    --         0 / 500
  reserved, no hint    42.7 [ 42.2- 42.9]      43.5 MB  0.3846 ms    0 / 500
  reserved + hint      42.8 [ 42.5- 43.0]      42.7 MB  0.3982 ms    0 / 500
  Vault.add_batch      45.2 [ 45.0- 45.4]      42.6 MB    --         0 / 500

Read the loader column at its median and its range, which is why both are
printed: the baseline's four runs land inside 0.5% of each other, but the
reserved arms at 71,433 rows scatter between 288 and 310 MB -- 7.1% of spread,
on an ingest transient that the committed-plus-hint arm shows too, so it is not
the reservation. Two further runs of the same build, taken after this docstring
was written (``post_report_verification``), read 293.6 and 310.0 against a
baseline of 823.3. ``Vault.add_batch``
reads 15 MB above the loop for a reason that is the benchmark's, not the
arena's: that arm materialises all 71,433 record dicts in a list before it
calls, and the list is charged to it (``records_built_peak_mb``).

The p50 column is the median of seven cycles of a paired duel, re-timed on the
vaults the builds left on disk, with a bare numpy matvec measured in every cycle
because the box was not quiet (that floor read 1.483 ms against the 0.97 ms it
offers quiet). The paired new/baseline ratio is 0.997 at 71,433 rows and 1.000
at 10,000 -- no measurable cost either way. Exactness does not depend on the
machine and is checked directly: the SHA-256 of the concatenated fp32 score
vectors of ALL 500 queries against ALL rows is identical for every arm at both
sizes (``exactness``), and no arm changed one of the 500 top-10 lists at either
size.

Against the field, on the numbers those competitors' own file reports
(``headline``, which reads them out of competitors_standard_results.json): at
10,000 documents 42.7 MB is the lowest of every arm measured, FAISS
IndexFlatIP's 50.3 MB included (0.85x). At 71,433 it is 298.4 MB against FAISS's
230.6 -- 1.29x, and still LAST against FAISS -- and 1.03x Chroma's default HNSW
at 290.5, which is inside this arm's own 7% spread. It is below the other four
arms that file measures (0.89x sqlite-vec, 0.94x tuned HNSW, 0.34x and 0.45x the
two Chroma brute-force arms).

The remaining ~290 MB is no longer the allocator: the loader's own floor is now
the SERVER, 287.6 MB, which is what a process that only reads the file pays.
Of it, 209.3 MiB is the vectors themselves -- the same 209.3 MiB of fp32 FAISS
holds -- 43.8 MiB is the document text nanomem returns and FAISS does not store
at all, and 8.9 MiB is the columns and the id index, for 262.0 MiB of vault
against the 289 MB the process shows
(``arms.n71433.*.serve.engine_stats``; the 1.10x gap between the two is
``active_heap_ram_method``'s, and is measured, not assumed). Getting under FAISS
from here means attacking one of those three, not the allocator.

A caller that KNOWS its row count should still say so, and now the library says
it for them: ``Vault.add_batch`` hints its own length, ``ingest_directory``
hints an estimate from the bytes it is about to read, and
``VaultEngine.reserve_rows`` / ``reserve_additional_rows`` are the explicit
form. With the reserved arena underneath, the hint is worth 3% at 71,433
rows (298.4 -> 289.5 MB, medians of four runs each, inside the spread) instead
of 2.85x.

On top of that the arena can hold its vectors in a narrower form; ``residency``
picks which, and every mode below fp32 scans in CHUNKS with fp32 accumulation,
which is why ``float16`` is not an approximation. Measured at 71,433 rows
(server shape = resident RSS of a reopened vault; p50 = median over 5 cycles of
the whole mode set, ``latency_duel.n71433``, on a QUIET machine -- see the
warning below):

  mode             server RSS   phys_footprint   p50      top-10 changed
  ``float32``        286.0 MB        286 MB     1.85 ms    0 / 500
  ``float16``        193.1 MB        193 MB     4.21 ms    0 / 500
  ``float16_mmap``   193.1 MB         88 MB     4.23 ms    0 / 500
  ``int8``           249.6 MB        145 MB    15.79 ms    7 / 500

  ``float32``       the 3.0.3 layout. Exact, fastest, largest. THE DEFAULT: it
                    is the only mode that costs nothing on the axis nanomem
                    actually wins, and ``reserve`` already took most of the RAM
                    off it.
  ``float16``       half the arena, 2.28x the p50 (measured, paired per cycle;
                    1.69x at 10,000 rows). The stored values are the SAME fp16
                    values that are on disk, so nothing is quantised that was not
                    quantised already; the per-chunk upcast is lossless and the
                    score vectors were measured bitwise-identical to ``float32``
                    at the shipped ``scan_chunk`` (``scan_chunk_sweep``). Flip to
                    this when RAM matters more than 2.4 ms per query.
  ``float16_mmap``  the same fp16 values in a private, unlinked sidecar file that
                    is memory-mapped instead of held in anonymous RAM. Identical
                    arithmetic; the pages are clean and the OS can evict them,
                    which is why its phys_footprint is 88 MB against 286 MB --
                    the smallest footprint of any mode, at 2.29x the p50.
  ``int8``          per-row symmetric int8 (+ one fp32 scale per row) for the
                    scan, then an EXACT fp32 re-rank of the top ``rerank_pool``
                    rows read back through the fp16 sidecar. Approximate below
                    the pool -- see :meth:`scores`. MEASURED A LOSS ON EVERY
                    AXIS: it needs the fp16 sidecar to re-rank, so it holds MORE
                    than ``float16`` (249.6 MB against 193.1 MB), it is 8.5x
                    ``float32``'s p50 because an int8 scan still converts every
                    row to fp32 for BLAS, and it is the only mode that changes an
                    answer. It is kept because it was measured, not because it
                    is recommended.

A RATIO MEASURED UNDER LOAD IS NOT LOAD-INDEPENDENT, and this table is the
evidence. Every multiple above was first measured on a box at a 1-minute load
average of 16.9-19.4 and read 1.55x / 1.61x / 5.48x; re-measured by the same
script, same corpus, same paired-per-cycle protocol, at load 1.6-2.7 it reads
2.28x / 2.29x / 8.5x, and a second quiet run agrees (2.26x / 2.25x / 8.36x,
``latency_duel.replicate``). Contention flatters the SLOWER arm -- when every
arm is waiting on the machine, the one that needs more CPU loses relatively less
-- so the loaded reading understated the cost of every narrow mode by about 1.5x
and it is the quiet numbers that are quoted here. The absolute p50s moved too:
``float32`` reads 3.92 ms loaded and 1.85 ms quiet.

``resident_bytes()`` reports what is actually held, split into anonymous and
mapped, so the number can be checked rather than believed.
"""

import hashlib
import json
import mmap
import os
import struct
import sys
import tempfile
import time
import zlib

import numpy as np

from . import container as _container
from .errors import VaultShrankError

#: Must equal :data:`nanomem.container.TMP_SUFFIX`; the arena cache writes its
#: temp file as a ``<vault>.tmp-arena-*`` sibling so that the container's own
#: stale-temp sweeper (and ``users.delete_user``) already know how to clean it
#: up. tests/test_arena_cache.py asserts the two constants agree.
_TMP_SUFFIX = ".tmp-"

#: Residency modes ``Arena`` accepts, cheapest-arithmetic first.
RESIDENCY_MODES = ("float32", "float16", "float16_mmap", "int8")

#: Rows converted per chunk by the narrow scan kernels. 4,096 x 768 x 4 B is a
#: 12.6 MB reusable staging buffer. Swept over 256..65,536 at both corpus sizes
#: (memory_results.json, ``scan_chunk_sweep``): at 71,433 rows 4,096 was the
#: fastest of the nine (4.54 ms p50 against 4.63 at 1,024 and 8.27 at 65,536),
#: and the sweep is the reason the size is not raised further -- the staging
#: buffer is itself resident, so a 65,536-row chunk would cost 192 MB of RAM to
#: save 105 MB of arena. Chunk size cannot change an answer, and every row of the
#: sweep asserts that; note that the smallest chunk (256) was the one row whose
#: scores were NOT bitwise identical to the fp32 arena (max |diff| 5.96e-08, one
#: fp32 ULP), which is BLAS blocking, not loss.
DEFAULT_SCAN_CHUNK = 4096

#: How many rows ``int8`` re-ranks exactly. Swept over 32..8,192 in
#: scratch/refound/memory_results.json (``int8_pool_sweep``). The sweep's answer
#: is that the pool is NOT the binding constraint: at 71,433 rows every pool in
#: that range still disagreed with the fp32 engine on 1 to 4 of 500 queries, and
#: EVERY one of those disagreements had a cosine gap of exactly 0.0 -- a tie
#: between two bit-identical duplicate paragraphs, which either order answers
#: correctly. Discounting ties, a pool of 32 already changes nothing at either
#: size. 512 is kept as the conservative default of a mode that is not
#: recommended anyway (see the module docstring).
DEFAULT_RERANK_POOL = 512

_DTYPE = {"float32": np.float32, "float16": np.float16,
          "float16_mmap": np.float16, "int8": np.int8}


def _next_capacity(cap: int, need: int, row_bytes: int) -> int:
    """Capacity-doubling growth, for the PER-ROW COLUMNS.

    The vectors left this policy for :class:`_VectorStore`; the columns -- 40
    bytes a row against the vectors' 3,072 -- kept it, because at 71,433 rows
    everything this function governs is 2.7 MiB of live bytes
    (ingest_ram_results.json, ``column_bytes``).

    Doubling looks wasteful -- it really does allocate 131,072 rows to hold
    71,433 -- and the obvious fix (grow in fixed ~32 MB steps) was implemented,
    measured, and REJECTED, because on this platform the cost of growing an array
    is not the overshoot, it is the copy left behind. Growing a 768-d fp32 arena
    to 71,433 rows, one 50-row block at a time, one subprocess per policy,
    measured by ``ru_maxrss``. Every row below is ingest_ram_results.json's
    ``growth_policy`` except the two policies it does not run -- 32 MB steps and
    x1.5 -- which are memory_results.json's:

        policy      final capacity   growths   bytes freed   peak RSS
        double         131,072 rows       12      383.8 MB    591.8 MB
        32 MB steps     76,454 rows        7      672.0 MB    880.9 MB
        x1.5            94,479 rows       18      553.0 MB    761.0 MB
        exact fit       71,433 rows    1,429  149,458.9 MB  13,505.9 MB
        reserved view   71,433 rows        2        0.0 MB    208.9 MB
        reserve()       71,433 rows        1        0.0 MB    208.9 MB

    Peak tracks (live + freed) almost exactly: the allocator hands almost none of
    it back. Doubling frees the fewest bytes of any policy that has to
    reallocate, which is why it is still here for the columns -- and the real
    answer is not to reallocate at all, which is what the reserved view does for
    the vectors.
    """
    if need <= cap:
        return cap
    new_cap = max(64, int(cap))
    while new_cap < need:
        new_cap *= 2
    return new_cap


def _grow(arr: np.ndarray, need: int) -> np.ndarray:
    """Capacity growth for a column array (see :func:`_next_capacity`)."""
    cap = arr.shape[0]
    if need <= cap:
        return arr
    row_bytes = arr.dtype.itemsize * int(np.prod(arr.shape[1:], dtype=np.int64) or 1)
    new_cap = _next_capacity(cap, need, row_bytes)
    shape = (new_cap,) + arr.shape[1:]
    out = np.zeros(shape, dtype=arr.dtype)
    out[:cap] = arr
    return out


#: Smallest reservation worth making, in bytes. Below this the address-space
#: bookkeeping costs more than the arena it holds.
RESERVE_MIN_BYTES = 2 << 20

#: How much headroom an UNHINTED reservation takes, as a multiple of the rows
#: that forced it. Address space, not RAM: an untouched page of an anonymous
#: mapping is not resident, so the only thing this buys is the right to grow
#: without copying, and the only thing it costs is virtual address space. 16x
#: leaves three reservations between the first row and 200,000 of them, so a
#: 71,433-row ingest copies 2.3 MB and 37 MB once each and nothing after that.
RESERVE_FACTOR = 16

#: Hard ceiling on that headroom, in bytes. Without it a 500,000-row corpus
#: would reserve ~24 GB of address space to hold 1.5 GB of vectors. With it the
#: reservation never runs more than half a gigabyte ahead of the rows, and a
#: corpus large enough to outgrow that pays one extra copy per 512 MB instead.
RESERVE_HEADROOM_CAP_BYTES = 512 << 20

_MAP_ANON = getattr(mmap, "MAP_ANONYMOUS", getattr(mmap, "MAP_ANON", 0))


class _VectorStore:
    """A contiguous ``(capacity, D)`` vector array that GROWS WITHOUT COPYING.

    The array is a view over an anonymous mapping that was reserved larger than
    the rows in it. Reserving address space is not the same as spending memory:
    an untouched page of an anonymous mapping is not resident, so a reservation
    costs nothing until a row is written into it, and growing the array is then
    a new view over the same pages -- no allocation, no copy, nothing left
    behind. Measured by ``ru_maxrss``, growing a 768-d fp32 arena to 71,433 rows
    50 rows at a time, one subprocess per policy (memory_results.json,
    ``growth_policy``, and scratch/refound/ingest_ram_results.json):

        policy                    final capacity   copies   peak RSS delta
        capacity doubling (as was)  131,072 rows       12       591.8 MB
        exact fit, reallocating      71,433 rows    1,429    13,505.9 MB
        reserved view (this class)   71,433 rows        2       208.9 MB

    The reservation is the ONLY thing that is oversized. The array itself is cut
    to the exact row count, so ``vec.nbytes`` is what the rows really are and
    ``stats()`` cannot overstate the arena by 1.83x the way capacity doubling
    could.

    When the reservation is outgrown a bigger one is made, the live rows are
    copied once, and the old mapping is UNMAPPED -- which is the other half of
    the problem: a freed numpy buffer is measurably not handed back to the OS on
    this platform (that is why doubling's peak tracks live + everything it ever
    discarded), while an ``munmap`` is. Reaching 71,433 rows 50 at a time costs
    two such copies, of 2.3 MB and 37 MB, and the peak above is the live rows
    exactly.

    THE OTHER WAY TO MAKE GROWTH FREE is to give up on one array: hold the arena
    as a list of fixed-size chunks and append a new one when it fills. That
    needs no reservation at all and copies nothing either -- and it is not what
    this class does, because the scan pays for it. One ``(n, D) @ (D,)`` BLAS
    call becomes ceil(n/chunk) of them, and measured on the 71,433-row corpus
    with the same vectors, paired in cycles against the contiguous scan
    (ingest_ram_results.json, ``chunked_alternative``): 0.9634 ms contiguous
    against 1.0207 / 1.0847 / 1.1915 / 1.5541 ms at 32,768 / 16,384 / 8,192 /
    4,096-row chunks -- 1.06x to 1.61x, on the axis nanomem is SECOND on. The
    scores are bitwise identical at every chunk size, so this is a pure
    latency-for-simplicity trade and it was declined: the reserved view buys the
    same RAM for 0.997x the p50 (``latency_duel``).

    If the platform will not give us an anonymous mapping the class falls back to
    a plain zero-filled array with the same view discipline: the growth still
    does not copy, only the release is at the allocator's discretion. If it will
    not give us the HEADROOM either -- a strict overcommit policy, an rlimit --
    :meth:`_adopt` retries with exactly the rows that are needed, so the arena
    still works, it just reallocates more often.
    """

    def __init__(self, embed_dim: int, dtype) -> None:
        self.embed_dim = int(embed_dim)
        self.dtype = np.dtype(dtype)
        self.row_bytes = self.embed_dim * self.dtype.itemsize
        self._map = None
        self._base = None                 # 1-D view over the whole reservation
        self.reservation_rows = 0
        self.mapped_reservation = False
        self.copies = 0                   # growth copies this store has paid for
        #: True while ``array`` is a read-only view into an arena cache mapping
        #: rather than memory this store owns. The first row appended after that
        #: moves the live rows into a fresh reservation (``ensure`` -> ``_adopt``),
        #: which is the only time a cached arena is ever copied.
        self.mapped_array = False
        self.array = np.zeros((0, self.embed_dim), dtype=self.dtype)

    # -- reservation --------------------------------------------------------
    def _reserve_map(self, rows: int):
        """Reserve address space for ``rows`` rows; ``(mapping, 1-D view)``."""
        rows = max(1, int(rows))
        nbytes = rows * self.row_bytes
        try:
            m = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | _MAP_ANON,
                          prot=mmap.PROT_READ | mmap.PROT_WRITE)
            base = np.frombuffer(m, dtype=self.dtype,
                                 count=rows * self.embed_dim)
            if not base.flags.writeable:
                raise BufferError("anonymous mapping came back read-only")
            return m, base, True
        except MemoryError:
            raise
        except Exception:
            # No anonymous mapping here (or it was refused). A zero-filled array
            # is lazily committed too, so the growth story is unchanged; only
            # the release is weaker.
            return None, np.zeros(rows * self.embed_dim, dtype=self.dtype), False

    def _view(self, rows: int) -> None:
        rows = int(rows)
        self.array = self._base[:rows * self.embed_dim].reshape(rows, self.embed_dim)

    def _adopt(self, rows: int, view_rows: int, n_live: int) -> None:
        """Move to a ``rows``-row reservation, carrying ``n_live`` rows over."""
        try:
            m, base, mapped = self._reserve_map(rows)
        except MemoryError:
            # The headroom is a convenience, never a requirement: a platform that
            # will not hand out the reservation (a strict overcommit policy, a
            # small address space, an rlimit) still gets a working arena sized to
            # the rows that actually exist.
            if view_rows >= rows:
                raise
            m, base, mapped = self._reserve_map(view_rows)
            rows = view_rows
        old_map, old_base, old_arr = self._map, self._base, self.array
        self._map, self._base = m, base
        self.mapped_array = False
        self.reservation_rows, self.mapped_reservation = int(rows), mapped
        self._view(view_rows)
        if n_live:
            self.array[:n_live] = old_arr[:n_live]
            self.copies += 1
        del old_arr, old_base
        if old_map is not None:
            try:
                old_map.close()           # munmap: hands the pages back NOW
            except BufferError:
                pass                      # somebody still holds a view; GC gets it

    def reserve(self, rows: int, n_live: int) -> None:
        """Size the array to exactly ``rows`` rows because a caller SAID so.

        An authoritative count (the container's block-header row count on a
        reopen, or ``reserve_rows`` from a loader that knows its corpus) gets an
        exact reservation: no headroom, nothing to overstate.
        """
        rows = int(rows)
        if rows <= self.array.shape[0]:
            return
        if rows <= self.reservation_rows:
            self._view(rows)
            return
        self._adopt(rows, rows, int(n_live))

    def ensure(self, need: int, n_live: int) -> None:
        """Make room for ``need`` rows when nobody said how many were coming."""
        need = int(need)
        if need <= self.array.shape[0]:
            return
        if need <= self.reservation_rows:
            self._view(need)              # free: the pages are already reserved
            return
        need_bytes = need * self.row_bytes
        headroom = min(need_bytes * (RESERVE_FACTOR - 1), RESERVE_HEADROOM_CAP_BYTES)
        target_bytes = max(need_bytes + headroom, RESERVE_MIN_BYTES)
        target = max(need, -(-int(target_bytes) // self.row_bytes),
                     self.reservation_rows * 2)
        self._adopt(target, need, int(n_live))

    def adopt_mapped(self, arr: np.ndarray) -> None:
        """Serve ``arr`` -- a read-only view into an arena cache -- as the arena.

        Nothing is allocated and nothing is copied: the scan that would have
        produced these rows is replaced by a mapping of the bytes it produced
        last time. The reservation is dropped (there is none behind a mapping),
        so the next append takes the ordinary ``ensure`` -> ``_adopt`` path and
        pays exactly one copy to move into anonymous memory.
        """
        if arr.shape[1] != self.embed_dim or arr.dtype != self.dtype:
            raise ValueError(f"cached vectors are {arr.shape[1]}-d {arr.dtype}, "
                             f"arena wants {self.embed_dim}-d {self.dtype}")
        self.close()
        self.array = arr
        self.mapped_array = True

    # -- reporting ----------------------------------------------------------
    def reservation_bytes(self) -> int:
        return int(self.reservation_rows) * self.row_bytes

    def close(self) -> None:
        self.array = np.zeros((0, self.embed_dim), dtype=self.dtype)
        self.mapped_array = False
        self._base = None
        if self._map is not None:
            try:
                self._map.close()
            except BufferError:
                pass
            self._map = None
        self.reservation_rows = 0


class _Sidecar:
    """A private, unlinked fp16 vector file the arena reads through ``mmap``.

    Unlinked at creation, so it cannot be left behind by a crash and cannot be
    read by another process; the fd keeps it alive. Written with ``pwrite`` (so
    the pages the arena writes are page cache, not process RSS) and mapped
    read-only for the scan, so the resident cost is clean, evictable pages.
    """

    def __init__(self, embed_dim: int, directory=None):
        self.embed_dim = int(embed_dim)
        self.row_bytes = self.embed_dim * 2
        d = directory or tempfile.gettempdir()
        try:
            fd, path = tempfile.mkstemp(prefix="nanomem-arena-", suffix=".f16", dir=d)
        except OSError:
            fd, path = tempfile.mkstemp(prefix="nanomem-arena-", suffix=".f16")
        os.unlink(path)
        self._fd = fd
        self.path_hint = path
        self.n_rows = 0
        self._mm = None
        self._mm_rows = 0

    def write(self, start: int, rows_f16: np.ndarray) -> None:
        buf = np.ascontiguousarray(rows_f16, dtype=np.float16).tobytes()
        off = int(start) * self.row_bytes
        wrote = 0
        while wrote < len(buf):
            wrote += os.pwrite(self._fd, buf[wrote:], off + wrote)
        self.n_rows = max(self.n_rows, int(start) + int(rows_f16.shape[0]))
        self._mm = None                 # stale: remapped on the next read

    def view(self) -> np.ndarray:
        """Read-only ``(n_rows, D)`` fp16 view of the sidecar."""
        if self._mm is None or self._mm_rows != self.n_rows:
            self._mm = None
            if self.n_rows:
                self._mm = np.memmap(os.fdopen(os.dup(self._fd), "rb"),
                                     dtype=np.float16, mode="r",
                                     shape=(self.n_rows, self.embed_dim))
            else:
                self._mm = np.zeros((0, self.embed_dim), dtype=np.float16)
            self._mm_rows = self.n_rows
        return self._mm

    def mapped_bytes(self) -> int:
        return int(self.n_rows) * self.row_bytes

    def close(self) -> None:
        self._mm = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --- the arena cache: a mappable snapshot of everything above ----------------
#: Suffix of the arena cache that sits next to a vault file.
ARENA_CACHE_SUFFIX = ".arena"
ARENA_CACHE_MAGIC = b"NMARENA1"
ARENA_CACHE_VERSION = 4

#: What `engine.add_fact` writes into a record's metadata when the CALLER
#: named the entity. Matched as bytes on the ingest path, so it must stay in
#: step with that json.dumps -- `separators=(",", ":")`, no space.
DECLARED_MARK = b'"entity_declared":true'

#: Every section starts on a page boundary, so a mapped array is page-aligned
#: (and therefore aligned for any dtype this format stores) and a section can be
#: faulted in without dragging its neighbour's pages along.
ARENA_CACHE_ALIGN = 4096
ARENA_CACHE_HEADER = 4096

#: What ``arena_cache=`` accepts. See :class:`ArenaSnapshot` for what each means.
ARENA_CACHE_MODES = ("off", "map", "copy", "verify")

#: Vaults smaller than this never get a cache: their whole scan is under a
#: millisecond and the file would cost more than it saves.
ARENA_CACHE_MIN_ROWS = 256

#: Rows the cache may fall behind the vault before an open rewrites it. The
#: measured rescan rate is ~2.2 us/row at 71,433 rows, so 4,096 rows is ~9 ms of
#: scan -- worth paying repeatedly rather than rewriting the whole cache after
#: every small append.
ARENA_CACHE_REFRESH_ROWS = 4096

#: Residencies whose vectors live in ``_VectorStore`` and can therefore be
#: mapped straight out of the cache. ``float16_mmap`` and ``int8`` keep their own
#: private sidecar and are not cached.
ARENA_CACHE_RESIDENCIES = ("float32", "float16")

#: WHERE A CACHED ARENA'S VECTORS COME FROM. The first two put a second copy of
#: every vector in the sidecar; the last two do not, and cost nothing on disk
#: beyond the block table the sidecar already carried.
#:
#:   ``"cache"``        the sidecar holds the arena's own vectors, at the
#:                      arena's own width. 3.0.6 behaviour. (An fp16 sidecar is
#:                      not a fourth value: it is ``residency="float16"``, which
#:                      already writes its fp16 arena straight into this
#:                      section.)
#:   ``"offsets"``      the sidecar holds NO vectors. The vault is mapped and
#:                      every scan gathers out of its own vector regions.
#:   ``"offsets_ram"``  as ``offsets``, but the mapped vault is upcast into one
#:                      anonymous fp32 array the FIRST time a vector is read --
#:                      lazily, so the open stays O(1) and only a process that
#:                      actually searches pays for it.
ARENA_VECTOR_SOURCES = ("cache", "offsets", "offsets_ram")

#: Where a cached arena's per-block RECORD sections come from. ``"cache"``
#: copies them into the sidecar (3.0.6); ``"vault"`` reads them out of the
#: mapped vault, which is where they already are.
ARENA_RECORD_SOURCES = ("cache", "vault")


class _MappedStrings:
    """A string column read out of the cache mapping instead of a Python list.

    ``ids`` is 71,433 strings at 71,433 rows and NONE of them is needed to
    answer a query -- only the handful that win. Holding them as ``offsets`` +
    one blob of utf-8 inside the mapping means an open decodes zero of them;
    ``ids[r]`` decodes one on demand. What that laziness saves shows up in the
    open (0.000193 s cached against 0.1634 s scanned) and what it defers shows
    up in the first write after one, which has to build the tables it skipped:
    0.0474 s against 0.0124 s at 71,433 rows
    (scratch/refound/reopen_results.json, ``first_write_after_open.n71433``).

    Rows appended after the snapshot land in a plain Python tail, so a vault
    that is opened from the cache and then written to behaves exactly as before
    for its new rows.
    """

    __slots__ = ("_off", "_data", "_n_mapped", "_tail")

    def __init__(self, off=None, data=None):
        self._off = off
        self._data = data
        self._n_mapped = 0 if off is None else int(off.shape[0]) - 1
        self._tail = []

    def __len__(self) -> int:
        return self._n_mapped + len(self._tail)

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += self._n_mapped + len(self._tail)
        if 0 <= i < self._n_mapped:
            return bytes(self._data[int(self._off[i]):int(self._off[i + 1])]).decode("utf-8")
        j = i - self._n_mapped
        if j < 0:
            raise IndexError(i)
        return self._tail[j]

    def __iter__(self):
        for i in range(self._n_mapped):
            yield bytes(self._data[int(self._off[i]):int(self._off[i + 1])]).decode("utf-8")
        for s in self._tail:
            yield s

    def append(self, s) -> None:
        self._tail.append(s)

    def extend(self, items) -> None:
        self._tail.extend(items)

    @property
    def mapped_bytes(self) -> int:
        return 0 if self._off is None else int(self._off[-1])

    @property
    def anon_bytes(self) -> int:
        return int(sum(len(s) for s in self._tail))


class _MappedBlobs:
    """The per-block record sections, sliced out of the mapping on demand.

    :meth:`Arena.record` reads one document's JSON out of one block's section,
    so the section never has to be a ``bytes`` object that this process owns:
    43.9 MiB of document text at 71,433 rows stays in clean, file-backed,
    evictable pages and only the pages a query actually returns are ever
    faulted in.
    """

    __slots__ = ("_off", "_data", "_n_mapped", "_tail")

    def __init__(self, off=None, data=None):
        self._off = off
        self._data = data
        self._n_mapped = 0 if off is None else int(off.shape[0]) - 1
        self._tail = []

    def __len__(self) -> int:
        return self._n_mapped + len(self._tail)

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += self._n_mapped + len(self._tail)
        if 0 <= i < self._n_mapped:
            return self._data[int(self._off[i]):int(self._off[i + 1])]
        j = i - self._n_mapped
        if j < 0:
            raise IndexError(i)
        return self._tail[j]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def append(self, b) -> None:
        self._tail.append(b)

    @property
    def mapped_bytes(self) -> int:
        return 0 if self._off is None else int(self._off[-1])

    @property
    def anon_bytes(self) -> int:
        return int(sum(len(b) for b in self._tail))


class _VaultBlobs:
    """The per-block record sections, read out of the MAPPED VAULT.

    The same interface as :class:`_MappedBlobs` over different bytes: the vault
    already holds every record section, so the sidecar copying them (43.8 MiB at
    71,433 rows) bought nothing but a second set of pages to fault in.
    """

    __slots__ = ("_backing", "_off", "_len", "_n_mapped", "_tail")

    def __init__(self, backing=None, off=None, length=None):
        self._backing = backing
        self._off = off
        self._len = length
        self._n_mapped = 0 if off is None else int(off.shape[0])
        self._tail = []

    def __len__(self) -> int:
        return self._n_mapped + len(self._tail)

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += self._n_mapped + len(self._tail)
        if 0 <= i < self._n_mapped:
            o, ln = int(self._off[i]), int(self._len[i])
            return self._backing.bytes_at(o, ln)
        j = i - self._n_mapped
        if j < 0:
            raise IndexError(i)
        return self._tail[j]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def append(self, b) -> None:
        self._tail.append(b)

    @property
    def mapped_bytes(self) -> int:
        return 0 if self._len is None else int(self._len.sum())

    @property
    def anon_bytes(self) -> int:
        return int(sum(len(b) for b in self._tail))


class _VaultBacking:
    """A read-only mapping of the VAULT, addressed by the block table.

    THE POINT OF THIS CLASS. A vault stores every vector once, fp16, inside its
    blocks. The arena cache stored them a SECOND time, fp32, in a sidecar -- and
    that sidecar was 257.5 MiB beside a 148.7 MiB vault, which cost nanomem the
    one axis it had led on. Nothing in the sidecar's vector section is
    information: it is an upcast of bytes that are already on the same disk,
    forty-eight bytes further along. This class replaces the copy with the only
    thing that was ever missing -- WHERE each block's vectors start -- and the
    block table the sidecar already carried (114 KiB at 71,433 rows) is where
    those offsets come from, so the replacement costs zero new bytes.

    WHAT THE LAYOUT ALLOWS AND WHAT IT DOES NOT. Inside one block the ``n`` rows
    are one contiguous run of ``n * D`` fp16 (``container.build_block_blob``:
    payload = landmarks + vectors + records), and blocks appear in row order, so
    a block IS a zero-copy ``(n, D)`` numpy view. ACROSS blocks they are not
    contiguous and not evenly spaced: each run is followed by that block's record
    section, its 32-byte trailer and 64-byte alignment padding, so on the 71,433-
    row corpus the 1,429 vector runs are 0-of-1,428 adjacent and have 179
    distinct strides. There is therefore NO strided view of the whole vault, and
    every read here is a per-block loop. Measured, not assumed.

    ALIGNMENT. Block offsets are 64-aligned and the block header is 96 bytes, so
    a vector run starts at 32 mod 64 -- 32-byte aligned, and an exact multiple of
    both 2 and 4, which is all numpy needs to view it as fp16 or fp32. Landmark
    tables, when a block has one, are ``m * D * 4`` bytes and do not disturb that.

    ENCRYPTED VAULTS CANNOT BE MAPPED AT ALL, which is not a new restriction:
    the block payload of an encrypted vault is ciphertext, there are no fp16
    vectors in it to point at, and the arena cache is already refused outright
    for an encrypted vault (:meth:`VaultEngine._arena_cache_eligible`).

    MIXED VECTOR WIDTHS ARE REFUSED. A vault whose blocks do not all carry the
    same ``FLAG_VEC_FP16`` cannot be presented as one array of one dtype, so
    :meth:`open` returns ``None`` and the caller falls back to a full scan.
    """

    __slots__ = ("path", "_mm", "_base", "_f", "_fd", "_need", "dtype",
                 "itemsize", "embed_dim", "n_rows", "n_blocks", "vec_start",
                 "block_start", "rec_off", "rec_len", "map_bytes", "_closed")

    def __init__(self, path, mm, base, dtype, embed_dim, block_start, vec_start,
                 rec_off, rec_len, fd=-1, need=0):
        self.path = path
        self._mm = mm
        self._f = None
        self._fd = int(fd)
        self._need = int(need)
        self._base = base
        self.dtype = np.dtype(dtype)
        self.itemsize = int(self.dtype.itemsize)
        self.embed_dim = int(embed_dim)
        self.block_start = block_start
        self.vec_start = vec_start            # in ELEMENTS of ``dtype``
        self.rec_off = rec_off                # in BYTES
        self.rec_len = rec_len
        self.n_blocks = int(vec_start.shape[0])
        self.n_rows = int(block_start[-1]) if block_start.shape[0] else 0
        self.map_bytes = int(len(mm))
        self._closed = False

    # -- opening ------------------------------------------------------------
    @classmethod
    def open(cls, path, table, embed_dim, valid_end, *, fp16: bool):
        """Map ``path`` and derive every block's vector and record span.

        ``table`` is the cache's own ``blocks`` section, which
        :meth:`Container.accept_snapshot` has already bound to this vault (uuid,
        file-header digest, geometry, and the 32-byte trailer of the last block
        it covers). Returns ``None`` for anything not usable -- a vault that
        cannot be mapped, a table that does not agree with the file, a mixed
        vector width -- because a backing that cannot be built is a cache that
        cannot be used, never an error the caller has to handle.
        """
        try:
            n_blocks = int(table.shape[0])
            off = table["offset"].astype(np.int64)
            n = table["n"].astype(np.int64)
            m = table["m"].astype(np.int64)
            rec_len = table["rec_section_len"].astype(np.int64)
            flags = table["block_flags"].astype(np.int64)
            D = int(embed_dim)
            is16 = (flags & _container.FLAG_VEC_FP16) != 0
            if n_blocks and not (bool(is16.all()) or not bool(is16.any())):
                return None               # mixed widths: no single dtype fits
            blk_fp16 = bool(is16.all()) if n_blocks else bool(fp16)
            if blk_fp16 != bool(fp16):
                return None               # the header and the blocks disagree
            dt = np.float16 if blk_fp16 else np.float32
            isz = 2 if blk_fp16 else 4
            vec_off = off + _container.BLOCK_HEADER_SIZE + m * D * 4
            vec_len = n * D * isz
            rec_off = vec_off + vec_len
            end = rec_off + rec_len
            if n_blocks:
                if int(off.min()) < _container.FILE_HEADER_SIZE:
                    return None
                if int(end.max()) > int(valid_end):
                    return None
                if int((vec_off % isz).max()) != 0:
                    return None
            block_start = np.zeros(n_blocks + 1, dtype=np.int64)
            np.cumsum(n, out=block_start[1:])
            size = os.path.getsize(path)
            if size < int(valid_end):
                return None
            # The last byte any read here can address. Kept so that
            # :meth:`_intact` can tell a vault that lost the bytes we point at
            # from one whose torn tail was trimmed behind us -- the second is
            # nanomem's own recovery and must not raise.
            need = int(end.max()) if n_blocks else int(valid_end)
            fd = os.open(path, os.O_RDONLY)
            try:
                mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            except Exception:
                os.close(fd)
                return None
            try:
                base = np.frombuffer(mm, dtype=dt, count=size // isz)
            except Exception:
                mm.close()
                os.close(fd)
                return None
            # The fd stays OPEN on purpose: ``os.fstat`` on it names the same
            # inode the mapping is of, which ``os.stat(path)`` would not after a
            # rename or a replace.
            return cls(path, mm, base, dt, D, block_start, vec_off // isz,
                       rec_off, rec_len, fd, need)
        except Exception:
            return None

    # -- reads --------------------------------------------------------------
    def _intact(self) -> None:
        """Raise if the vault no longer contains the bytes this mapping addresses.

        WHY THIS EXISTS. ``mmap`` binds to a length, not to a file: truncate the
        file and the pages past the new end are still mapped and still
        addressable, and touching one raises SIGBUS. A signal is not an
        exception -- it kills the process, with no traceback, no ``except``
        anywhere that can see it, and nothing written to say why. The layouts
        that copy the vectors and the record sections into the sidecar never
        read the vault after the open, so the same damage is survivable there;
        these layouts read it on every query and every record, so the exposure
        is the price of not copying and it is checked rather than crashed.

        WHAT IS CHECKED. One ``fstat`` against ``_need``, the last byte the
        block table can address -- not against the mapped length. nanomem's own
        torn-tail recovery (:mod:`nanomem.container`) truncates only what lies
        PAST the last valid block, so it shrinks the file without ever taking a
        byte this backing points at, and it does not raise here. Measured cost
        is one ``fstat`` per scan, per gather and per record read.

        WHAT IS NOT PROMISED. This is a guard, not a guarantee. A truncation
        that lands between this check and the page touch still faults; nothing
        short of reading through ``pread`` instead of a mapping can close that
        window, and that is the ``arena_cache_vectors="cache"`` layout.
        """
        if self._fd < 0:
            return
        try:
            size = os.fstat(self._fd).st_size
        except OSError:
            return                        # cannot tell: do not invent a failure
        if size < self._need:
            raise VaultShrankError(
                "%s shrank to %d bytes under a live mapping that addresses %d; "
                "the arena is reading vectors/records out of this file "
                "(arena_cache_vectors='offsets'/'offsets_ram', "
                "arena_cache_records='vault'). Reopen the vault, or use "
                "arena_cache_vectors='cache' to keep a copy in the sidecar."
                % (self.path, int(size), int(self._need)))

    def bytes_at(self, off: int, length: int):
        self._intact()
        return memoryview(self._mm)[int(off):int(off) + int(length)]

    def block_view(self, b: int) -> np.ndarray:
        """Block ``b``'s vectors as a zero-copy ``(n_b, D)`` view."""
        lo, hi = int(self.block_start[b]), int(self.block_start[b + 1])
        o = int(self.vec_start[b])
        return self._base[o:o + (hi - lo) * self.embed_dim].reshape(hi - lo,
                                                                   self.embed_dim)

    def read_rows(self, lo: int, hi: int, dst: np.ndarray, staging=None) -> None:
        """:meth:`fill`, but the bytes arrive by ``read`` instead of the mapping.

        FOR THE ONE-SHOT UPCAST ONLY, and the reason is ru_maxrss. Pulling
        104.6 MiB of vectors THROUGH the mapping faults every one of those pages
        into this process, and a high-water mark never comes back down, so the
        arm that upcasts into anonymous fp32 was charged for the fp16 source and
        the fp32 result at the same time. Reading them instead leaves the same
        bytes in the page cache, where a second process can still have them and
        this process is not billed for them.
        """
        D, isz = self.embed_dim, self.itemsize
        k_total = int(hi) - int(lo)
        if k_total <= 0:
            return
        self._intact()
        if self._f is None:
            self._f = open(self.path, "rb", buffering=0)
        if staging is None or staging.shape[0] < k_total * D:
            staging = np.empty(k_total * D, dtype=self.dtype)
        buf = staging[:k_total * D]
        mv = memoryview(buf).cast("B")
        bs = self.block_start
        b = int(np.searchsorted(bs, lo, side="right")) - 1
        r, fill = int(lo), 0
        while r < hi:
            if not 0 <= b < self.n_blocks:
                raise IndexError("row %d is not in any block of %s" % (r, self.path))
            bend = min(int(bs[b + 1]), int(hi))
            k = bend - r
            if k > 0:
                off = (int(self.vec_start[b]) + (r - int(bs[b])) * D) * isz
                want = k * D * isz
                at = fill * D * isz
                self._f.seek(off)
                got = 0
                while got < want:
                    n = self._f.readinto(mv[at + got:at + want])
                    if not n:
                        raise OSError("short read of %d bytes at %d in %s"
                                      % (want - got, off + got, self.path))
                    got += n
                fill += k
                r = bend
            b += 1
        np.copyto(dst.reshape(-1), buf)

    def fill(self, lo: int, hi: int, dst: np.ndarray) -> None:
        """Copy rows ``[lo, hi)`` into ``dst`` (``(hi-lo, D)``), converting dtype.

        One flat ``copyto`` per block touched: the runs are contiguous inside a
        block and nowhere else, so this is the loop the layout forces.
        """
        self._intact()
        D = self.embed_dim
        flat = dst.reshape(-1)
        b = int(np.searchsorted(self.block_start, lo, side="right")) - 1
        r, fill = int(lo), 0
        bs = self.block_start
        while r < hi:
            if not 0 <= b < self.n_blocks:
                raise IndexError("row %d is not in any block of %s"
                                 % (r, self.path))
            bend = min(int(bs[b + 1]), int(hi))
            k = bend - r
            if k > 0:
                o = int(self.vec_start[b]) + (r - int(bs[b])) * D
                np.copyto(flat[fill * D:(fill + k) * D], self._base[o:o + k * D])
                fill += k
                r = bend
            b += 1

    def take(self, rows: np.ndarray, dst: np.ndarray) -> None:
        """Gather arbitrary ``rows`` into ``dst`` (``(len(rows), D)``).

        Grouped by block, because a fancy index into a contiguous block view is
        one numpy call while a byte-level gather across the whole mapping is
        two orders of magnitude slower (measured: 12.8 ms against 1.6 ms for
        10,714 rows at 71,433).
        """
        if rows.size == 0:
            return
        self._intact()
        blk = np.searchsorted(self.block_start, rows, side="right") - 1
        order = np.argsort(blk, kind="stable")
        sb = blk[order]
        bounds = np.flatnonzero(np.diff(sb)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [sb.shape[0]]))
        bs = self.block_start
        for s, e in zip(starts, ends):
            b = int(sb[s])
            sel = order[s:e]
            local = rows[sel] - int(bs[b])
            dst[sel] = self.block_view(b)[local]

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        self._closed = True
        self._base = None
        fd, self._fd = self._fd, -1
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        f, self._f = self._f, None
        if f is not None:
            try:
                f.close()
            except OSError:
                pass
        mm, self._mm = self._mm, None
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                pass            # a view is still live; the mapping goes with it


def arena_cache_path(vault_path: str) -> str:
    """Where the cache for ``vault_path`` lives. It is derived data: deleting it
    costs one slow open and nothing else."""
    return str(vault_path) + ARENA_CACHE_SUFFIX


def _string_table(items):
    """``(offsets int64[k+1], list of blobs)`` for a sequence of strings."""
    blobs = [s.encode("utf-8") if isinstance(s, str) else bytes(s) for s in items]
    off = np.zeros(len(blobs) + 1, dtype=np.int64)
    if blobs:
        np.cumsum(np.fromiter((len(b) for b in blobs), dtype=np.int64,
                              count=len(blobs)), out=off[1:])
    return off, blobs


class ArenaSnapshot:
    """A memory-mappable copy of a built arena, written next to the vault.

    WHY. Opening a vault used to cost a full pass over the container: every
    block's payload read, its trailer recomputed, its vectors upcast, its records
    parsed and one Python dict entry made per row. Measured on the 71,433-row
    corpus (scratch/refound/reopen_results.json, ``headline.n71433``), that is
    0.1634 s against 0.000193 s for a cached open -- 846x. It is O(rows) and it
    is paid on EVERY open, which is why sqlite-vec (0.0014 s) and Chroma
    (0.0017 s) beat nanomem by two orders of magnitude on reopen: they map or
    lazily page their storage and nanomem rebuilt its own. A cached open is
    0.14x sqlite-vec's, so this is the axis nanomem lost and now wins.

    WHAT IT COSTS, because none of it is free. The open that WRITES the cache
    pays for it: 0.3051 s against a 0.1659 s plain open at 71,433 rows, of
    which 0.1315 s is the write (``rebuild.cache_map.n71433``) -- 1.84x, paid
    once. The cache is
    257.5 MiB beside a 148.7 MiB vault in 3.1.0, so a vault that kept one
    occupied 406.2 MiB -- nanomem's on-disk figure had been the smallest of every
    arm measured (148.7 MiB against sqlite-vec's 258.0 and Chroma's 488-564,
    competitors_standard_results.json) and with a cache it was not. THAT WAS THE
    TRADE THIS FEATURE MADE, and it is no longer the only one on offer.

    209.3 MiB of that file was an fp32 upcast of vectors the vault already holds
    in fp16 and 43.8 MiB was a verbatim copy of the vault's record sections, and
    neither copy is information: the block table this file already carried says
    where both live. ``vectors="offsets"`` and ``records="vault"`` keep the
    offsets instead and let the engine map the vault (:class:`_VaultBacking`),
    which takes the sidecar to 4.38 MiB and the pair to 153.05 MiB at 71,433
    rows -- under sqlite-vec's 258.0 and 4.4 MiB above a vault with no sidecar at
    all. That is the shipped default (``arena_cache_vectors="offsets_ram"``);
    what it costs is in scratch/refound/sidecar_size_results.json and in the
    ``arena_cache_vectors`` block of :class:`~nanomem.engine.VaultEngine`.

    ONE SENTENCE THAT WAS HERE WAS WRONG, and is kept because it is the reason
    nobody tried this for a release: "an fp16 sidecar would halve it and put the
    O(rows) upcast back in the OPEN, which is the whole thing being deleted."
    The upcast does not go back in the open. Measured at 71,433 rows,
    ``residency="float16"`` -- which writes an fp16 arena straight into this
    file's ``vec`` section -- reopens in 0.000229 s against 0.000187 s for the
    fp32 sidecar; both are O(1) and both beat sqlite-vec's 0.0014 s. The
    conversion lands on the QUERY, where it is x2.26 of the fp32 arena's p50 in
    the shipped ``screen="off"`` configuration (paired duel). That is a real
    cost and it is why the default upcasts ONCE instead. It is not the cost the
    sentence claimed. ``arena_cache="off"`` writes nothing. And the
    first WRITE after a cached open pays for the lazy tables it skipped:
    0.0474 s against 0.0124 s (``first_write_after_open``). Deleting the cache
    costs one slow open and nothing else.

    WHAT IT IS. One file, ``<vault>.arena``, laid out so that every array the
    arena holds is a page-aligned section that ``mmap`` can hand to numpy with
    no copy and no parse: the vectors in the arena's resident dtype, the
    per-row columns, the block table, the record sections, and the id / entity
    / group string tables. Opening it is a stat, a 4 KiB header read, a 32-byte
    read from the vault, and one ``mmap`` -- constant work, whatever the row
    count.

    WHAT IT IS NOT. It is NOT the vault. The container stays the source of
    truth: a cache that is missing, stale, truncated, from another vault, or
    unparseable is dropped and rebuilt without a word, and the rebuilt arena is
    the same arena the scan always produced. Nothing is ever read from the
    cache that is not also derivable from the container.

    WHAT IT DOES NOT RE-READ, and a withdrawal. A scan reads every block and
    recomputes its 32-byte trailer before a single row of it is believed. A
    cached open does not: it checks that the cache is bound to THIS vault at
    THIS length -- vault uuid, a SHA-256 of the vault's 256-byte file header,
    the row and block counts, the vector dtype, the 32-byte trailer of the last
    block the cache covers, and the vault's (size, mtime) as of the scan the
    cache was built from -- and then trusts the bytes. Those checks are O(1).
    So in ``map`` and ``copy`` two things are taken on trust: an edit to the
    VAULT that the (size, mtime) binding does not catch (an edit to the prefix
    of a vault that was afterwards appended to), and any edit at all to the
    CACHE. Measured, not assumed -- rewriting one row of a cache's ``vec``
    section plants a row that comes back at cosine 1.0000 and takes the top-1
    slot from the true winner (``reopen_results.json`` -> ``cache_integrity``).

    WITHDRAWN, in the exact words this docstring used: "the vault's per-block
    HMACs make tampering with the vault detectable, so a ``<vault>.arena``
    beside it is a strictly weaker, unauthenticated path to the same answers."
    That is false in the only configuration a sidecar can exist in. A cache is
    refused outright for an ENCRYPTED vault, so every vault that has one is
    PLAINTEXT, and a plaintext vault's block trailer is an unkeyed SHA-256 that
    "detects accidental corruption only and can be recomputed by anyone"
    (nanomem/crypto.py, THREAT_MODEL). Rewriting one row's fp16 vector in the
    block body and recomputing that trailer plants a row that a full scan with
    ``arena_cache="off"`` accepts and returns at cosine 0.999997, taking top-1
    from the true winner (``cache_integrity.vault_forgery``, all three corpora;
    tests/test_arena_cache.py::test_a_plaintext_vault_is_forgeable_with_no_cache_in_sight).
    The sidecar does not lower the vault's threat model. It inherits it, and in
    plaintext that model is corruption, not adversaries. Everything below is
    therefore about CORRUPTION, and the word authentication does not belong to
    any of it.

    Nor is it fixable cheaply, which is the real reason there is a knob here at
    all: checking bytes is O(bytes) by construction. At 71,433 rows re-reading
    the 148.7 MiB vault costs 0.0605 s and digesting the 257.5 MiB cache another
    0.0866 s (SHA-256 at 2,975 MiB/s; ``cache_integrity.corpora.n71433``).
    Either one alone is dozens of times sqlite-vec's ENTIRE 0.0014 s reopen, so
    an O(1) open and "the bytes were checked" cannot both be had, whatever the
    checking is made of. The knob:

      ``arena_cache="map"``      the default. Map the cache, bind it to the
                                vault, re-read neither. O(1).
      ``arena_cache="copy"``     the same, then copy the vectors into anonymous
                                RAM once, so no query ever takes a page fault.
                                Same trust as ``map``.
      ``arena_cache="verify"``   map the cache, check it against the content
                                digest in its own header, AND re-read every
                                block of the vault and recompute its trailer,
                                before serving. Both, because they cover
                                different files: re-reading the vault cannot see
                                a cache edit at all. BOTH ARE CHECKSUMS. The
                                cache's digest is unkeyed, it sits in the header
                                of the file it describes, and that header is
                                protected only by a CRC32, so a ~10-line forgery
                                updates both and this mode serves the planted
                                row at cosine 1.0 (``cache_integrity``;
                                tests/test_arena_cache.py::test_verify_mode_is_hijacked_by_a_forged_content_digest).
                                It detects corruption in either file and it
                                detects no tampering at all.
                                SAY THE PRICE OUT LOUD -- checking both costs
                                0.1478 s at 71,433 rows against 0.1656 s for
                                simply rescanning the vault, measured in the
                                SAME phase so the two are comparable
                                (``cache_integrity.corpora.n71433``): 0.89x, i.e.
                                slightly CHEAPER than a scan, because it skips
                                the decode, the record parse and the per-row
                                interning. It is still ~850x a cached open, so
                                what it buys is not speed: it is the mapped
                                memory profile (phys_footprint 9 MB against
                                287 MB) at a scan's price, with the same
                                corruption checking a scan does.
      ``arena_cache="off"``      no cache is read or written; 3.0.5 behaviour.

    THE ONE CASE THAT WAS CLOSED, rather than only described. A vault edited in
    place at the same length with its trailer left stale is refused by ``off``
    and ``verify`` (``IntegrityError``) and used to be served by ``map`` with no
    error, no warning and no ``integrity_errors`` entry -- a silent downgrade of
    the DEFAULT. The (size, mtime) binding above now refuses that cache and the
    scan that follows raises, so all three modes agree. It is one stat and it is
    a corruption check: ``os.utime`` defeats it, which is a passing test. What
    this open re-read is reported rather than implied --
    ``arena_cache_info()["vault_blocks_checked"]`` is ``"all"``,
    ``"appended tail only"`` or ``"none"``. The check is one ``os.stat``:
    measured at +4.08 us at 71,433 rows and +6.42 us at 10,000 against the same
    package with the check deleted, on a 0.000175 s cached open
    (``binding_cost``, 1.024x and 1.038x; two earlier paired runs read +1.69 us
    and +0.81 us, so the honest reading is "a few microseconds, at or below the
    run-to-run spread of the open itself"), and it changes
    no answer -- the fp32 score digest over 200 queries is identical across all
    four modes at n10000 and n71433 (``post_fix_check``).

    ENCRYPTED VAULTS NEVER GET ONE, in any mode. The cache holds vectors and
    document text in the clear, so writing one next to an encrypted vault would
    put its plaintext on the same disk the encryption exists to protect. An
    encrypted vault also pays ~100 ms of scrypt at every open (DECISIONS core-6),
    which is the cost a cached open could not remove anyway.
    """

    __slots__ = ("path", "header", "_mm", "sections", "n_rows", "n_blocks",
                 "embed_dim", "dtype", "_arrays", "_closed")

    def __init__(self, path: str, header: dict, mm):
        self.path = path
        self.header = header
        self._mm = mm
        self.sections = header["sections"]
        self.n_rows = int(header["n_rows"])
        self.n_blocks = int(header["n_blocks"])
        self.embed_dim = int(header["embed_dim"])
        self.dtype = np.dtype(header["dtype"])
        self._arrays = {}
        self._closed = False

    # -- reading ------------------------------------------------------------
    def array(self, name: str) -> np.ndarray:
        """The named section as a read-only numpy view into the mapping."""
        got = self._arrays.get(name)
        if got is not None:
            return got
        spec = self.sections[name]
        shape = tuple(int(x) for x in spec["shape"])
        count = 1
        for s in shape:
            count *= s
        dt = _json_to_dtype(spec["dtype"])
        if count == 0:
            out = np.zeros(shape, dtype=dt)
        else:
            out = np.frombuffer(self._mm, dtype=dt, count=count,
                                offset=int(spec["off"])).reshape(shape)
        self._arrays[name] = out
        return out

    def blob(self, name: str):
        """A memoryview over a byte section (no copy, no decode)."""
        spec = self.sections[name]
        off, ln = int(spec["off"]), int(spec["len"])
        return memoryview(self._mm)[off:off + ln]

    def strings(self, name: str) -> list:
        """A byte section + its offsets, decoded into a real list of ``str``."""
        off = self.array(name + "_off")
        data = self.blob(name + "_data")
        return [bytes(data[int(off[i]):int(off[i + 1])]).decode("utf-8")
                for i in range(off.shape[0] - 1)]

    @property
    def total_bytes(self) -> int:
        return int(self.header["total_bytes"])

    def content_sha256(self) -> str:
        """Hash everything after the header, to compare against the digest the
        header carries. O(bytes), so ``verify`` is the only mode that calls it --
        and it MUST call it, because it is the only check that reads the cache's
        own bytes at all.

        IT IS A CHECKSUM. The digest it is compared against lives in the header
        of this same file and is unkeyed, and that header is protected only by a
        CRC32, so an editor of this file can recompute both and this check then
        passes on the edited bytes (measured; pinned by
        tests/test_arena_cache.py::test_verify_mode_is_hijacked_by_a_forged_content_digest).
        It catches a cache that was DAMAGED -- a half-written file, a bad sector,
        a truncation -- which is the same class of thing a plaintext vault's own
        unkeyed trailers catch."""
        h = hashlib.sha256()
        h.update(memoryview(self._mm)[ARENA_CACHE_HEADER:self.total_bytes])
        return h.hexdigest()

    def close(self) -> None:
        self._closed = True
        self._arrays = {}
        mm, self._mm = self._mm, None
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                pass          # a view is still live; the mapping goes with it

    # -- opening ------------------------------------------------------------
    @classmethod
    def open(cls, path: str):
        """Parse and map ``path``. Returns ``None`` for anything not usable.

        EVERY failure returns ``None``. A cache is derived data, so there is no
        error here that the caller cannot answer by rebuilding, and raising
        would turn a corrupt cache into an unopenable vault.
        """
        try:
            size = os.path.getsize(path)
            if size < ARENA_CACHE_HEADER:
                return None
            with open(path, "rb") as f:
                raw = f.read(ARENA_CACHE_HEADER)
            hdr = _parse_cache_header(raw)
            if hdr is None or int(hdr["total_bytes"]) != size:
                return None
            fd = os.open(path, os.O_RDONLY)
            try:
                mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            finally:
                os.close(fd)      # the mapping keeps the file alive by itself
            return cls(path, hdr, mm)
        except Exception:
            return None

    # -- writing ------------------------------------------------------------
    @staticmethod
    def write(path: str, arena: "Arena", state: dict, *, durable: bool = True,
              digest: bool = True, vectors: str = "cache",
              records: str = "cache") -> dict:
        """Write the cache for ``arena`` atomically. Returns what it measured.

        ``vectors`` and ``records`` decide what this file DUPLICATES. Both
        default to the 3.0.6 answer, "everything". ``vectors="offsets"`` (and
        its ``"offsets_ram"`` twin, which differs only in what the READER does)
        writes no vector section at all and leaves the block table -- already
        here, 114 KiB at 71,433 rows -- to say where each block's vectors are in
        the vault. ``records="vault"`` does the same for the record sections.
        See :class:`_VaultBacking` for what the vault's layout does and does not
        allow. The reader is told which was written by the ``vec_source`` and
        ``rec_source`` header fields, so an old cache and a new one are told
        apart by their contents, not by a flag the caller has to remember.

        ``state`` is :meth:`nanomem.container.Container.snapshot_state` -- the
        binding to the vault this arena was scanned from. The write goes to a
        ``<vault>.tmp-arena-*`` sibling (so the container's own stale-temp
        sweeper owns the wreckage of a crashed write) and is put in place with
        ``os.replace``, which is atomic: a reader either sees the whole old
        cache or the whole new one, and a reader that has already mapped the old
        one keeps reading it from the unlinked inode. Two processes writing at
        once is therefore safe and the last one wins.
        """
        n, D = int(arena.n_rows), int(arena.embed_dim)
        nb = int(arena.n_blocks)
        _vault_stat = _stat_pair(state.get("vault_path"),
                                 state.get("vault_size_at_scan"),
                                 state.get("vault_mtime_ns_at_scan"))
        secs = []                       # (name, kind, payload)

        def arr(name, a):
            secs.append((name, "array", np.ascontiguousarray(a)))

        def blobs(name, items):
            off, bl = _string_table(items)
            arr(name + "_off", off)
            secs.append((name + "_data", "blobs", bl))

        vectors = str(vectors or "cache").lower()
        records = str(records or "cache").lower()
        if vectors not in ARENA_VECTOR_SOURCES:
            raise ValueError(f"vectors must be one of {ARENA_VECTOR_SOURCES!r}, "
                             f"got {vectors!r}")
        if records not in ARENA_RECORD_SOURCES:
            raise ValueError(f"records must be one of {ARENA_RECORD_SOURCES!r}, "
                             f"got {records!r}")
        vec_source = "vault" if vectors.startswith("offsets") else "cache"
        if vec_source == "cache":
            # An arena that is itself reading a mapped vault has no vector array
            # to copy; it has to own its rows before it can hand them over. Only
            # reachable when a caller writes a copying cache for an arena that
            # was opened from an offset one.
            arena._materialise_vault_vectors()
            vec_dtype = np.dtype(arena.vec.dtype)
            arr("vec", arena.vec[:n])
        else:
            vec_dtype = np.dtype(np.float16 if state.get("vault_fp16", True)
                                 else np.float32)
        for col in ("ts", "rev", "entity_id", "group_id", "row_block"):
            arr(col, getattr(arena, col)[:n])
        arr("doc_span", arena.doc_span[:n])
        arr("block_start", arena.block_start[:nb + 1])
        arr("land_start", arena.land_start[:nb + 1])
        arr("landmarks", arena.landmarks[:int(arena.land_start[nb])]
            if arena.landmarks.shape[0] else np.zeros((0, D), dtype=np.float32))
        arr("blocks", state["block_table"])
        if records == "cache":
            blobs("rec", [arena.rec_bytes[i] for i in range(nb)])
        blobs("ids", [arena.ids[i] for i in range(n)])
        blobs("entities", list(arena.entity_names))
        blobs("groups", list(arena.group_keys))
        gmr, gmt = arena.group_max_columns()
        arr("group_max_rev", gmr)
        arr("group_max_ts", gmt)
        arr("group_declared", arena.group_declared_column())

        layout, off = {}, ARENA_CACHE_HEADER
        for name, kind, payload in secs:
            if kind == "array":
                nbytes = int(payload.nbytes)
                shape = [int(x) for x in payload.shape]
                dt = _dtype_to_json(payload.dtype)
            else:
                nbytes = int(sum(len(b) for b in payload))
                shape = [nbytes]
                dt = "|u1"
            layout[name] = {"off": off, "len": nbytes, "dtype": dt, "shape": shape}
            off += nbytes
            off = ((off + ARENA_CACHE_ALIGN - 1) // ARENA_CACHE_ALIGN) * ARENA_CACHE_ALIGN
        total = off

        hdr = {
            "format_version": ARENA_CACHE_VERSION,
            "engine": state.get("engine_version"),
            "created_unix": time.time(),
            "vault_uuid": state["vault_uuid"],
            "source_header_sha256": state["header_sha256"],
            "source_valid_end": int(state["valid_end"]),
            "source_scanned_end": int(state["scanned_end"]),
            "source_last_trailer": state["last_trailer"],
            "source_size": int(state["size"]),
            "truncated_tail_bytes": int(state["truncated_tail_bytes"]),
            # The vault as it stood when the scan that built this arena ended.
            # The engine captures it the instant the scan returns; falling back
            # to a stat here keeps a direct caller (a test, a repair tool)
            # honest rather than letting it write a cache that cannot bind.
            "source_stat_size": int(_vault_stat[0]),
            "source_mtime_ns": int(_vault_stat[1]),
            "integrity_errors": list(state["integrity_errors"]),
            "embed_dim": D,
            "dtype": np.dtype(vec_dtype).str,
            "vec_source": vec_source,
            "vec_mode": vectors,
            "rec_source": "vault" if records == "vault" else "cache",
            "residency": arena.residency,
            "block_capacity": int(state["block_capacity"]),
            "landmarks_per_block": int(state["landmarks_per_block"]),
            "n_rows": n,
            "n_blocks": nb,
            "n_positions": int(state["n_positions"]),
            "n_entities": len(arena.entity_names),
            "n_groups": len(arena.group_keys),
            "total_bytes": total,
            "content_sha256": None,
            "sections": layout,
        }

        tmp = "%s%sarena-%d-%s" % (state["vault_path"], _TMP_SUFFIX, os.getpid(),
                                  os.urandom(4).hex())
        t0 = time.perf_counter()
        h = hashlib.sha256() if digest else None
        pad = b"\x00" * ARENA_CACHE_ALIGN
        written = 0
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(b"\x00" * ARENA_CACHE_HEADER)     # header written last
                at = ARENA_CACHE_HEADER
                for name, kind, payload in secs:
                    want = int(layout[name]["off"])
                    if at < want:
                        f.write(pad[:want - at])
                        if h is not None:
                            h.update(pad[:want - at])
                        at = want
                    if kind == "array":
                        if payload.nbytes:
                            # `.cast("B")` refuses a zero in the shape, and an
                            # empty section is a real case (no landmarks, no
                            # interned entities), so it is skipped outright.
                            mv = memoryview(payload).cast("B")
                            f.write(mv)
                            if h is not None:
                                h.update(mv)
                            at += len(mv)
                    else:
                        for b in payload:
                            if not len(b):
                                continue
                            mv = memoryview(b).cast("B")
                            f.write(mv)
                            if h is not None:
                                h.update(mv)
                            at += len(mv)
                if at < total:
                    f.write(b"\x00" * (total - at))
                    if h is not None:
                        h.update(b"\x00" * (total - at))
                    at = total
                written = at
                hdr["content_sha256"] = h.hexdigest() if h is not None else None
                f.flush()
                f.seek(0)
                f.write(_pack_cache_header(hdr))
                f.flush()
                if durable:
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
            os.replace(tmp, path)
            if durable:
                _fsync_parent(path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return {"path": path, "bytes": written, "write_s": time.perf_counter() - t0,
                "sections": {k: v["len"] for k, v in layout.items()},
                "content_sha256": hdr["content_sha256"], "durable": bool(durable)}


def _stat_pair(path, size=None, mtime_ns=None):
    """``(size, mtime_ns)`` for ``path``, preferring values captured earlier.

    ``(-1, -1)`` when nothing can be established, which the reader treats as a
    cache that does not bind (see :func:`vault_changed_since_cache`).
    """
    if size is not None and mtime_ns is not None:
        try:
            return int(size), int(mtime_ns)
        except (TypeError, ValueError):
            pass
    try:
        st = os.stat(path)
        return int(st.st_size), int(st.st_mtime_ns)
    except (OSError, TypeError):
        return -1, -1


def vault_changed_since_cache(vault_path: str, header: dict) -> str:
    """Has the vault been written since the cache was taken? O(1), one stat.

    Returns one of:

      ``"unchanged"``  same length, same mtime: nothing has written to the vault
                       since the scan this cache was built from, so the prefix
                       the cache covers is the prefix that was read.
      ``"grew"``       longer: an append. The appended tail is scanned (and its
                       trailers checked) by the open that follows; the PREFIX is
                       taken on trust, which is the residual hole -- an edit to
                       the prefix that is followed by an append is not caught.
      ``"rewritten"``  same length, different mtime, or shorter: something
                       rewrote the file in place. The cache is refused and the
                       open falls back to a full scan, which is what notices a
                       stale trailer and raises.
      ``"unknown"``    the cache records no (size, mtime), or the vault cannot
                       be stat'ed. Treated as a refusal: an absent binding is
                       not a passed one, and the cost of being wrong is one slow
                       open.

    WHAT THIS IS WORTH, precisely. It closes the case where a vault is edited in
    place at the same length and its trailer left stale -- before it, ``map``
    served that vault with no error at all while ``off`` and ``verify`` raised
    ``IntegrityError``. It is a CORRUPTION check and nothing more: mtime is
    attacker-writable with one ``os.utime`` call, which is pinned as a passing
    test (``test_the_stat_binding_is_defeated_by_restoring_mtime``). It is also
    best-effort against a filesystem whose mtime granularity is coarser than the
    gap between two writes; APFS stores nanoseconds.
    """
    try:
        rec_size = int(header.get("source_stat_size", -1))
        rec_mtime = int(header.get("source_mtime_ns", -1))
    except (TypeError, ValueError):
        return "unknown"
    if rec_size < 0 or rec_mtime < 0:
        return "unknown"
    try:
        st = os.stat(vault_path)
    except OSError:
        return "unknown"
    if int(st.st_size) == rec_size:
        return "unchanged" if int(st.st_mtime_ns) == rec_mtime else "rewritten"
    return "grew" if int(st.st_size) > rec_size else "rewritten"


def _pack_cache_header(hdr: dict) -> bytes:
    body = json.dumps(hdr, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(body) + 24 > ARENA_CACHE_HEADER:
        raise ValueError("arena cache header does not fit in one page")
    fixed = ARENA_CACHE_MAGIC + struct.pack("<II", ARENA_CACHE_VERSION, len(body))
    crc = zlib.crc32(fixed + body) & 0xFFFFFFFF
    out = fixed + struct.pack("<I", crc) + body
    return out + b"\x00" * (ARENA_CACHE_HEADER - len(out))


def _parse_cache_header(raw: bytes):
    """Header dict, or ``None`` for anything that is not our intact header."""
    if len(raw) < 24 or raw[:8] != ARENA_CACHE_MAGIC:
        return None
    version, jlen = struct.unpack("<II", raw[8:16])
    if version != ARENA_CACHE_VERSION or jlen <= 0 or 24 + jlen > len(raw):
        return None
    crc = struct.unpack("<I", raw[16:20])[0]
    body = raw[20:20 + jlen]
    if zlib.crc32(raw[:16] + body) & 0xFFFFFFFF != crc:
        return None
    try:
        hdr = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    need = ("vault_uuid", "n_rows", "n_blocks", "embed_dim", "dtype", "sections",
            "total_bytes", "source_valid_end", "source_last_trailer",
            "source_header_sha256", "source_stat_size", "source_mtime_ns")
    if not isinstance(hdr, dict) or any(k not in hdr for k in need):
        return None
    return hdr


def _dtype_to_json(dt):
    """A dtype in a form JSON can hold and numpy can read back EXACTLY.

    ``dtype.str`` is lossy for a structured dtype -- the block table comes back
    as ``|V80``, which then fails to match and silently costs a cached open --
    so a structured dtype travels as its ``descr``, which round-trips fields and
    subarrays. ``numpy.lib.format`` has the canonical pair of functions for this
    and is deliberately NOT used: touching it imports a numpy submodule that
    nothing else here needs, which measured 6 ms on the first open of a process
    -- 30x the whole cached open it would have sat inside.
    """
    dt = np.dtype(dt)
    return dt.str if dt.fields is None else [list(f) for f in dt.descr]


def _json_to_dtype(spec):
    """Inverse of :func:`_dtype_to_json` (JSON turns every tuple into a list)."""
    def tup(x):
        return tuple(tup(i) for i in x) if isinstance(x, list) else x
    if isinstance(spec, str):
        return np.dtype(spec)
    return np.dtype([tup(f) for f in spec])


def _fsync_parent(path: str) -> None:
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



class Arena:
    """Resident vectors, columns, record bytes and the id / entity / group indexes."""

    def __init__(self, embed_dim: int, *, residency: str = "float32",
                 scan_chunk: int = DEFAULT_SCAN_CHUNK,
                 rerank_pool: int = DEFAULT_RERANK_POOL,
                 sidecar_dir=None):
        residency = str(residency or "float32").lower()
        if residency not in RESIDENCY_MODES:
            raise ValueError(f"residency must be one of {RESIDENCY_MODES!r}, "
                             f"got {residency!r}")
        self.embed_dim = int(embed_dim)
        self.residency = residency
        self.store_dtype = _DTYPE[residency]
        self.scan_chunk = max(64, int(scan_chunk))
        self.rerank_pool = max(0, int(rerank_pool))
        self.sidecar_dir = sidecar_dir
        self.n_rows = 0
        #: Vectors live in a reservation that is grown by re-viewing, never by
        #: copying (see :class:`_VectorStore`). ``self.vec`` is the live view.
        self._store = _VectorStore(self.embed_dim, self.store_dtype)
        # int8 only: one fp32 scale per row, so a stored row decodes as
        # ``vec[r] * vscale[r]``.
        self.vscale = np.zeros(0, dtype=np.float32)
        self._sidecar = None
        self._buf = None
        #: The mapped :class:`ArenaSnapshot` this arena was opened from, if any.
        self._snapshot = None
        #: The mapped VAULT this arena reads its vectors and/or record sections
        #: out of (:class:`_VaultBacking`), when the sidecar declined to copy
        #: them. ``None`` for a scanned arena and for a 3.0.6-shaped cache.
        self._vault = None
        #: True while the vectors themselves come from ``_vault`` rather than
        #: from ``_store``. Cleared the moment they are materialised -- by the
        #: lazy upcast of ``offsets_ram``, or by the first append.
        self._vault_vectors = False
        #: True when the vault-backed vectors should be upcast into one
        #: anonymous fp32 array at first touch instead of being gathered per
        #: query (``arena_cache_vectors="offsets_ram"``).
        self._vault_upcast = False
        #: True while the per-row columns are still views into that mapping. The
        #: first append copies them into anonymous arrays (``_grow``) and clears
        #: it, so the byte report never calls a mapped page "heap" or the other
        #: way round.
        self._mapped_columns = False
        self.ts = np.zeros(0, dtype=np.float64)
        self.rev = np.zeros(0, dtype=np.int32)
        self.entity_id = np.zeros(0, dtype=np.int32)
        self.group_id = np.zeros(0, dtype=np.int32)
        self.row_block = np.zeros(0, dtype=np.int32)
        self.doc_span = np.zeros((0, 2), dtype=np.int64)
        self.block_start = np.zeros(1, dtype=np.int64)
        #: Row -> document id. A list while the arena is built by scanning, a
        #: mapped string table (plus a Python tail) when it came from a cache.
        self.ids = _MappedStrings()
        self.landmarks = np.zeros((0, self.embed_dim), dtype=np.float32)
        self.land_start = np.zeros(1, dtype=np.int64)
        #: Per-block record sections: owned ``bytes`` when scanned, memoryviews
        #: into the cache mapping when not.
        self.rec_bytes = _MappedBlobs()
        self.n_blocks = 0
        self.reserved_rows = 0
        #: THE FOUR LAZY TABLES. Every one of them is a Python dict or list that
        #: the scan used to fill one row at a time -- 0.044 s and ~19 MB at
        #: 71,433 rows -- and NONE of them is read to answer a query: the search
        #: path reads the ``entity_id`` / ``group_id`` COLUMNS, which are mapped.
        #: They are built on first touch instead, so a cached open pays for them
        #: only if something actually writes to the vault or resolves an entity
        #: name. ``None`` means "not built yet"; the cache, if there is one,
        #: holds the bytes they are built from.
        #: ``None`` means "not built yet, build it from the cache on first
        #: touch"; :meth:`attach_snapshot` is what sets them to ``None``.
        self._id_index = {}
        self._entity_names = []
        self._entity_index = {}
        self._group_keys = []
        self._group_index = {}
        self._group_max = {}
        #: Parallel to ``group_keys``: did the CALLER name this group's entity,
        #: or did the lexical tagger infer it? The relevance floor needs to know
        #: (``engine._apply_group_floor``) and nothing else does.
        self._group_declared = []

    # -- residency plumbing -------------------------------------------------
    @property
    def vec(self) -> np.ndarray:
        """The resident vector array. Row ``r`` is document ``r``, and the array
        is cut to the rows that exist -- the headroom is in the reservation
        behind it, not in this array's shape."""
        return self._store.array

    @property
    def mapped(self) -> bool:
        """True when the vectors live in a mapped file rather than anonymous RAM."""
        return self.residency == "float16_mmap"

    @property
    def needs_sidecar(self) -> bool:
        """True when this mode keeps an exact fp16 copy outside the arena."""
        return self.residency in ("float16_mmap", "int8")

    def _sidecar_store(self) -> "_Sidecar":
        if self._sidecar is None:
            self._sidecar = _Sidecar(self.embed_dim, self.sidecar_dir)
        return self._sidecar

    def _stage(self, n: int) -> np.ndarray:
        """Reusable ``(<=scan_chunk, D)`` fp32 staging buffer for the narrow modes."""
        if self._buf is None:
            self._buf = np.empty((self.scan_chunk, self.embed_dim), dtype=np.float32)
        return self._buf[:n]

    # -- vault-backed vectors ----------------------------------------------
    @property
    def vault_backed(self) -> bool:
        """True while this arena's vectors are read out of the mapped vault."""
        return self._vault_vectors and self._vault is not None

    def _materialise_vault_vectors(self) -> None:
        """Move the vault-backed rows into ``_store`` and stop reading the vault.

        Paid exactly once, by whoever needs the vectors in one array: the lazy
        upcast of ``offsets_ram`` on its first query, or the first APPEND to a
        vault-backed arena, which cannot write row ``n`` into a read-only
        mapping of somebody else's file. The vault mapping is deliberately NOT
        closed -- the record sections may still be reading out of it.
        """
        if not self.vault_backed:
            return
        n = int(self.n_rows)
        vault = self._vault
        self._vault_vectors = False       # before the fill: it reads _store now
        self._store.reserve(n, 0)
        dst = self._store.array
        c = max(1, int(self.scan_chunk))
        staging = np.empty(min(n, c) * self.embed_dim, dtype=vault.dtype)
        for s in range(0, n, c):
            e = min(s + c, n)
            vault.read_rows(s, e, dst[s:e], staging)
        self.reserved_rows = max(self.reserved_rows, n)

    def _vault_chunked_dot(self, q: np.ndarray, out: np.ndarray) -> None:
        """``out[i] = row_i . q`` with the rows gathered from the mapped vault.

        The staging buffer, its size and the ``buf @ q`` call are the ones
        :meth:`_chunked_dot` uses, so the arithmetic -- and therefore the score
        bits -- are the full fp32 arena's; only where the bytes come from
        changes.
        """
        n = out.shape[0]
        c = self.scan_chunk
        vault = self._vault
        for s in range(0, n, c):
            e = min(s + c, n)
            buf = self._stage(e - s)
            vault.fill(s, e, buf)
            out[s:e] = buf @ q

    # -- lazy tables --------------------------------------------------------
    #
    # Each of these is materialised on first touch and is a perfectly ordinary
    # dict/list afterwards, so every caller that appends to one keeps working.
    # The point is WHEN: a cached open touches none of them, and a query touches
    # at most ``entity_names``.
    @property
    def id_index(self) -> dict:
        """``doc id -> row``. Built lazily: only ``get(id)`` and the write path
        need it, and it is the single most expensive thing a scan used to build
        (71,433 dict entries)."""
        if self._id_index is None:
            idx = {}
            ids = self.ids
            for r in range(len(ids)):
                idx[ids[r]] = r           # LAST wins, as the scan did
            self._id_index = idx
        return self._id_index

    @property
    def entity_names(self) -> list:
        """Interned entity names, indexed by ``entity_id``."""
        if self._entity_names is None:
            self._entity_names = (self._snapshot.strings("entities")
                                  if self._snapshot is not None else [])
        return self._entity_names

    @property
    def entity_index(self) -> dict:
        if self._entity_index is None:
            self._entity_index = {k: i for i, k in enumerate(self.entity_names)}
        return self._entity_index

    @property
    def group_keys(self) -> list:
        """Interned ``(user_id, project, entity)`` keys, indexed by ``group_id``."""
        if self._group_keys is None:
            self._group_keys = (self._snapshot.strings("groups")
                                if self._snapshot is not None else [])
        return self._group_keys

    @property
    def group_index(self) -> dict:
        if self._group_index is None:
            self._group_index = {k: i for i, k in enumerate(self.group_keys)}
        return self._group_index

    @property
    def group_declared(self) -> list:
        """``group_id -> bool``: the entity was DECLARED, not inferred.

        A cache written before format 4 has no such section, and a vault whose
        records predate the marker has no evidence either way. Both read as
        ``False``, which is the conservative answer -- it is what the engine did
        before this existed, so an old vault keeps its old ranking.
        """
        if self._group_declared is None:
            arr = (self._snapshot.array("group_declared")
                   if self._snapshot is not None else None)
            self._group_declared = ([bool(x) for x in arr] if arr is not None
                                    else [])
        n = len(self.group_keys)
        if len(self._group_declared) < n:
            self._group_declared.extend([False] * (n - len(self._group_declared)))
        return self._group_declared

    def group_declared_column(self) -> np.ndarray:
        """``group_declared`` as a dense uint8 array, for the cache writer."""
        d = self.group_declared
        return np.asarray(d, dtype=np.uint8) if d else np.zeros(0, dtype=np.uint8)

    @property
    def group_max(self) -> dict:
        """``group_id -> (max revision, max timestamp)``, for auto-revisioning."""
        if self._group_max is None:
            snap = self._snapshot
            rev = snap.array("group_max_rev")
            ts = snap.array("group_max_ts")
            self._group_max = {i: (int(rev[i]), float(ts[i]))
                               for i in range(rev.shape[0])}
        return self._group_max

    def group_max_columns(self):
        """``group_max`` as two dense arrays, for the cache writer."""
        n = len(self.group_keys)
        rev = np.zeros(n, dtype=np.int32)
        ts = np.zeros(n, dtype=np.float64)
        for gid, (r, t) in self.group_max.items():
            if 0 <= int(gid) < n:
                rev[int(gid)] = int(r)
                ts[int(gid)] = float(t)
        return rev, ts

    # -- arena cache --------------------------------------------------------
    def attach_snapshot(self, snap: "ArenaSnapshot", *, copy_vectors: bool = False,
                        vault: "_VaultBacking" = None) -> None:
        """Serve this arena out of a mapped :class:`ArenaSnapshot`.

        Called instead of replaying the container's blocks, and it is where the
        O(rows) open becomes O(1): every array below is a view into the mapping,
        the four Python tables are deferred, and the string columns are decoded
        one element at a time by whoever asks. The caller has already checked
        that the snapshot belongs to this vault (see
        ``Container.accept_snapshot``); the only thing checked here is that its
        SHAPE is the one this arena can serve, because that is what would corrupt
        an answer rather than merely being stale.

        ``copy_vectors=True`` is ``arena_cache="copy"``: take the same mapping
        and copy the vectors once into anonymous memory, trading the open's
        constant time and the mapping's clean pages for queries that never take
        a page fault.

        ``vault`` is a :class:`_VaultBacking` and is REQUIRED when the snapshot
        says its ``vec_source`` or ``rec_source`` is ``"vault"``: that cache
        holds offsets into the vault rather than copies of what they point at,
        and this method checks that the two agree on the row count, the block
        count and every block boundary before a single offset is believed. With
        vault-sourced vectors ``copy_vectors`` still means "into anonymous
        memory", but it happens LAZILY, at the first vector read rather than
        here, so the open it would otherwise lengthen stays O(1) -- 0.000232 s
        at 71,433 rows, with the upcast showing up in the first query
        (0.0227 s against 0.0117 s) and nowhere else.

        THE DEFAULT IS TO READ THE MAPPING, and the measurement is why. At
        71,433 rows (``headline.n71433``), map against copy:

          reopen              0.000193 s   vs 0.0217 s   (113x)
          first query after   0.0107 s     vs 0.0042 s   (the page faults)
          time to first answer 0.0109 s    vs 0.0259 s   (2.4x for map)
          steady-state p50    1.7806 ms    vs 1.8538 ms  (4% apart, map ahead)
          peak ru_maxrss      258.2 MB     vs 467.5 MB
          phys_footprint      9 MB         vs 219 MB     (24x)

        Mapping wins the open outright, wins time-to-first-answer even after
        paying every page fault in the first query, and costs nothing per query
        once warm -- the paired duel, arms alternated cycle by cycle on the same
        vault, puts the mapped scan at 0.9983x the scanned one at this size
        (``latency_duel.n71433``; an earlier run on a busier machine read
        1.0059x, so read this as "no measurable cost either way"). ``copy`` exists for
        the caller who wants the first query to be as fast as the second and
        will spend 219 MB of dirty anonymous RAM on it.
        """
        if self.residency not in ARENA_CACHE_RESIDENCIES:
            raise ValueError(f"residency {self.residency!r} does not keep its "
                             f"vectors in the arena cache")
        if self.n_rows or self.n_blocks:
            raise ValueError("attach_snapshot on a non-empty arena")
        if snap.embed_dim != self.embed_dim:
            raise ValueError(f"cache holds {snap.embed_dim}-d vectors, "
                             f"arena is {self.embed_dim}-d")
        n, nb = snap.n_rows, snap.n_blocks
        vec_source = str(snap.header.get("vec_source", "cache"))
        rec_source = str(snap.header.get("rec_source", "cache"))
        if vec_source == "vault" or rec_source == "vault":
            if vault is None:
                raise ValueError("cache stores offsets into the vault but no "
                                 "vault mapping was supplied")
            if vault.n_rows != n or vault.embed_dim != self.embed_dim:
                raise ValueError(f"vault mapping holds {vault.n_rows} x "
                                 f"{vault.embed_dim}, cache says {n} x "
                                 f"{self.embed_dim}")
            if vault.n_blocks != nb:
                raise ValueError("vault mapping and cache disagree on block count")
            if not np.array_equal(vault.block_start,
                                  snap.array("block_start")[:nb + 1]):
                # The offsets are only rows if the two agree on which rows are in
                # which block. They can diverge exactly once -- a vault with a
                # SKIPPED block, where the container keeps counting positions the
                # arena never received -- and then every row after it would be
                # read out of the wrong place.
                raise ValueError("vault mapping and cache disagree on block boundaries")
        if vec_source == "vault":
            vec = None
        else:
            vec = snap.array("vec")
            if vec.shape != (n, self.embed_dim) or vec.dtype != np.dtype(self.store_dtype):
                raise ValueError(f"cache vectors are {vec.shape} {vec.dtype}, "
                                 f"arena wants {(n, self.embed_dim)} {self.store_dtype}")
        cols = {c: snap.array(c) for c in ("ts", "rev", "entity_id", "group_id",
                                           "row_block", "doc_span")}
        for c, a in cols.items():
            want = (n, 2) if c == "doc_span" else (n,)
            if a.shape != want:
                raise ValueError(f"cache column {c} is {a.shape}, wanted {want}")
        block_start = snap.array("block_start")
        land_start = snap.array("land_start")
        if block_start.shape != (nb + 1,) or land_start.shape != (nb + 1,):
            raise ValueError("cache block table does not match its block count")
        if nb and int(block_start[nb]) != n:
            raise ValueError("cache block_start does not end at its row count")
        ids_off = snap.array("ids_off")
        if ids_off.shape != (n + 1,):
            raise ValueError("cache string tables do not match the row count")
        if rec_source == "cache":
            rec_off = snap.array("rec_off")
            if rec_off.shape != (nb + 1,):
                raise ValueError("cache string tables do not match the row count")

        if vec_source == "vault":
            # Nothing is read here. The mapping is already open and the block
            # table already says where every row is; the first READ decides
            # whether they are gathered per query or upcast once.
            self._store.close()
            self._vault = vault
            self._vault_vectors = True
            self._vault_upcast = bool(copy_vectors)
        elif copy_vectors:
            self._store.reserve(n, 0)
            self._store.array[:n] = vec
        else:
            self._store.adopt_mapped(vec)
        if rec_source == "vault":
            self._vault = vault
        self.ts, self.rev = cols["ts"], cols["rev"]
        self.entity_id, self.group_id = cols["entity_id"], cols["group_id"]
        self.row_block, self.doc_span = cols["row_block"], cols["doc_span"]
        self.block_start, self.land_start = block_start, land_start
        self.landmarks = snap.array("landmarks")
        self.rec_bytes = (_VaultBlobs(vault, vault.rec_off, vault.rec_len)
                          if rec_source == "vault"
                          else _MappedBlobs(rec_off, snap.blob("rec_data")))
        self.ids = _MappedStrings(ids_off, snap.blob("ids_data"))
        self._id_index = None
        self._entity_names = self._entity_index = None
        self._group_keys = self._group_index = self._group_max = None
        self._group_declared = None
        self.n_rows, self.n_blocks = n, nb
        self.reserved_rows = n
        self._mapped_columns = True
        self._snapshot = snap

    @property
    def from_cache(self) -> bool:
        """True when this arena was mapped from a cache instead of scanned."""
        return self._snapshot is not None

    def reserve(self, n_rows: int) -> None:
        """Pre-size every row-indexed array to hold ``n_rows`` rows, EXACTLY.

        For a count somebody knows: the container calls this with the row count
        it read out of the block headers before it scans, and
        ``VaultEngine.reserve_rows`` passes a loader's corpus size. The count is
        taken at its word, so nothing is over-allocated and ``stats()`` reports
        the arena at its true size. Measured at 71,433 rows on an M4 Pro: 643.6
        MB of resident RSS becomes 286.0 MB, with all 500 queries' top-10
        bit-for-bit unchanged (scratch/refound/memory_results.json,
        ``summary.reopen_only.n71433``). It is only a hint: a short or missing
        hint still grows correctly, and the container caps it by the bytes the
        file actually holds so a forged header cannot forge an allocation.

        Use :meth:`hint_rows` when the number is an ESTIMATE rather than a count.
        """
        n = int(n_rows)
        if n <= self.vec.shape[0] and n <= self.reserved_rows:
            return
        self.reserved_rows = max(self.reserved_rows, n)
        if self.residency != "float16_mmap":
            self._store.reserve(n, self.n_rows)
        self._presize_columns(n)

    def hint_rows(self, n_rows: int) -> None:
        """Pre-size for ``n_rows`` rows that are ESTIMATED, not counted.

        Same effect as :meth:`reserve` on the arrays, but the reservation behind
        the vectors keeps its growth headroom, so a caller that hints repeatedly
        -- one hint per file of a directory ingest, one per batch of a stream --
        does not pay a copy per hint the way an exact reservation would. The
        hint is free to be wrong in either direction: too small and growth takes
        over, too large and the extra rows are pages nobody ever touches. An
        estimate large enough that the allocation itself fails is DROPPED, not
        raised: a hint that breaks the ingest it was meant to speed up would be
        worse than no hint, and the arena grows perfectly well without one.
        """
        n = int(n_rows)
        if n <= self.vec.shape[0] and n <= self.reserved_rows:
            return
        try:
            if self.residency != "float16_mmap":
                self._store.ensure(n, self.n_rows)
            self._presize_columns(n)
        except (MemoryError, ValueError, OverflowError):
            return
        self.reserved_rows = max(self.reserved_rows, n)

    def _presize_columns(self, n: int) -> None:
        """Grow the per-row columns (and the int8 scales) to ``n`` rows."""
        if self.residency == "int8" and self.vscale.shape[0] < n:
            sc = np.zeros(n, dtype=np.float32)
            sc[:self.n_rows] = self.vscale[:self.n_rows]
            self.vscale = sc
        for name in ("ts", "rev", "entity_id", "group_id", "row_block"):
            col = getattr(self, name)
            if col.shape[0] < n:
                self._mapped_columns = False
                out = np.zeros(n, dtype=col.dtype)
                out[:self.n_rows] = col[:self.n_rows]
                setattr(self, name, out)
        if self.doc_span.shape[0] < n:
            self._mapped_columns = False
            out = np.zeros((n, 2), dtype=self.doc_span.dtype)
            out[:self.n_rows] = self.doc_span[:self.n_rows]
            self.doc_span = out

    def _store_rows(self, start: int, vectors_f32: np.ndarray) -> None:
        """Write ``vectors_f32`` at row ``start`` in whatever form this mode holds."""
        n = int(vectors_f32.shape[0])
        end = start + n
        if self.needs_sidecar:
            self._sidecar_store().write(start, vectors_f32.astype(np.float16))
        if self.residency == "float16_mmap":
            return
        if self.residency == "int8":
            scale = np.abs(vectors_f32).max(axis=1).astype(np.float32)
            scale[scale == 0.0] = 1.0
            q = np.rint(vectors_f32 / scale[:, None] * 127.0)
            self.vec[start:end] = np.clip(q, -127, 127).astype(np.int8)
            self.vscale[start:end] = scale / 127.0
        else:
            self.vec[start:end] = vectors_f32.astype(self.store_dtype, copy=False)

    # -- interning ----------------------------------------------------------
    def intern_entity(self, name) -> int:
        """Return the int id of a normalised entity name (-1 when there is none)."""
        if not name:
            return -1
        key = str(name)
        got = self.entity_index.get(key)
        if got is None:
            got = len(self.entity_names)
            self.entity_names.append(key)
            self.entity_index[key] = got
        return got

    def intern_group(self, key, declared: bool = False) -> int:
        """Return the int id of a ``(user_id, project, entity)`` group key.

        ``declared`` records that the CALLER named the entity on this write.
        DECLARED WINS over a later inferred write to the same key: one caller
        naming the attribute is evidence the tagger's guess never is, and a
        group that is half-declared is still a group somebody declared.
        """
        if not key:
            return -1
        got = self.group_index.get(key)
        if got is None:
            got = len(self.group_keys)
            self.group_keys.append(key)
            self.group_index[key] = got
        d = self.group_declared
        while len(d) <= got:
            d.append(False)
        if declared:
            d[got] = True
        return got

    def note_group(self, gid: int, rev: int, ts: float) -> None:
        """Keep ``group_max[gid] = (max revision, max timestamp)`` current."""
        if gid < 0:
            return
        prev = self.group_max.get(gid)
        if prev is None:
            self.group_max[gid] = (int(rev), float(ts))
        else:
            self.group_max[gid] = (max(prev[0], int(rev)), max(prev[1], float(ts)))

    # -- ingest -------------------------------------------------------------
    def add_block(self, meta, landmarks, vectors_f32, rec_bytes) -> None:
        """Append one decoded block. ``meta`` is a :class:`~nanomem.container.BlockMeta`."""
        from .container import decode_records

        rec = decode_records(rec_bytes)
        n = rec.n
        if n != vectors_f32.shape[0]:
            raise ValueError(f"block {meta.index}: {n} records vs {vectors_f32.shape[0]} vectors")
        start = self.n_rows
        need = start + n

        # A vault-backed arena reads its vectors out of somebody else's
        # read-only mapping, and row ``start`` cannot be written into one. The
        # rows move into anonymous memory once, here, which is the same single
        # copy a cache-backed arena pays on its first append.
        if self.vault_backed:
            self._materialise_vault_vectors()
        if self.residency != "float16_mmap":
            # Growth, not allocation: the rows land in address space that was
            # already reserved, so nothing is copied and nothing is discarded.
            self._store.ensure(need, start)
        if self.residency == "int8":
            self.vscale = _grow(self.vscale, need)
        self._store_rows(start, np.asarray(vectors_f32, dtype=np.float32))

        # Every one of these returns a NEW anonymous array when the current one
        # is a mapped, read-only cache view, which is what makes appending to a
        # cached arena correct without a special case.
        self._mapped_columns = False
        self.ts = _grow(self.ts, need)
        self.rev = _grow(self.rev, need)
        self.entity_id = _grow(self.entity_id, need)
        self.group_id = _grow(self.group_id, need)
        self.row_block = _grow(self.row_block, need)
        self.doc_span = _grow(self.doc_span, need)

        # ``row_block`` indexes THIS list, not the container's block numbering.
        # The two diverge as soon as a block is skipped: with
        # ``on_integrity_error="skip"`` the container keeps counting positions
        # (including the damaged one) while the arena only stores the blocks it
        # received, so ``meta.index`` pointed one buffer too far and every row
        # after a skipped block returned another block's text -- then ran off the
        # end of the list with an IndexError. Regression:
        # tests/test_container.py::test_skipped_block_does_not_shift_later_records.
        block_idx = len(self.rec_bytes)
        self.rec_bytes.append(rec_bytes)

        self.ts[start:need] = rec.ts
        self.rev[start:need] = rec.rev
        self.row_block[start:need] = block_idx
        self.doc_span[start:need] = rec.doc_spans
        # ONE PROPERTY LOOKUP PER BLOCK, not per row. `ids` and `id_index` are
        # now indirections (a mapped string table; a dict that may not exist
        # yet), and resolving them 50 times a block -- 71,433 times over a
        # reopen -- was measurably slower than the plain attributes they
        # replaced. Hoisting them, and decoding the ids in one list
        # comprehension, puts the scan back under what it cost before the cache
        # existed (scratch/refound/reopen_results.json, `scan_path_duel`).
        names = [b.decode("utf-8") for b in rec.ids]
        self.ids.extend(names)
        idx = self.id_index
        gids = self.group_id
        eids = self.entity_id
        revs, tss = rec.rev, rec.ts
        for i in range(n):
            # LAST wins. Ids are supposed to be unique, but `Vault.add` derives
            # one from the text alone, so re-adding the same text produces a
            # second row with the same id. First-wins made `get(id)` return the
            # superseded copy and `search_multihop` bridge from it.
            idx[names[i]] = start + i
            gkey = rec.groups[i].decode("utf-8")
            # Provenance without parsing. The marker is written by `add_fact`
            # into the record's own metadata JSON, which is already in this
            # block's payload; `json.dumps(separators=(",", ":"))` emits it
            # exactly like this. A substring test over ~200 bytes per record
            # costs nothing next to the JSON parse that reading it properly
            # would, and this loop runs 71,433 times over a reopen.
            lo, hi = int(rec.doc_spans[i][0]), int(rec.doc_spans[i][1])
            gid = self.intern_group(gkey, DECLARED_MARK in rec_bytes[lo:hi])
            gids[start + i] = gid
            ent = gkey.rsplit("\x1f", 1)[-1] if gkey else ""
            eids[start + i] = self.intern_entity(ent)
            self.note_group(gid, int(revs[i]), float(tss[i]))

        self.n_rows = need
        self.block_start = np.concatenate([self.block_start, [need]])

        # Landmarks are optional: they exist only for the PageLandmarkRouter
        # ablation. When a block stores none (m == 0, the default), nothing is
        # duplicated in RAM -- the ablation reads the block's own rows instead.
        m = 0 if landmarks is None else int(np.asarray(landmarks).shape[0])
        if m:
            lm = np.asarray(landmarks, dtype=np.float32)
            if self.landmarks.shape[0] == 0:
                self.landmarks = lm.copy()
            else:
                self.landmarks = np.vstack([self.landmarks, lm])
        self.land_start = np.concatenate([self.land_start, [self.land_start[-1] + m]])
        self.n_blocks = block_idx + 1

    # -- vector reads -------------------------------------------------------
    def exact16(self) -> np.ndarray:
        """The fp16 vectors, from wherever this mode keeps them."""
        if self.vault_backed:
            out = np.empty((self.n_rows, self.embed_dim), dtype=np.float16)
            self._vault.fill(0, self.n_rows, out)
            return out
        if self.needs_sidecar:
            return self._sidecar_store().view()
        if self.residency == "float16":
            return self.vec[:self.n_rows]
        return self.vec[:self.n_rows].astype(np.float16)

    def vectors(self, rows) -> np.ndarray:
        """fp32 copies of ``rows``, exactly as they are stored on disk."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        if self.vault_backed:
            if self._vault_upcast:
                self._materialise_vault_vectors()
            else:
                out = np.empty((rows.shape[0], self.embed_dim), dtype=np.float32)
                self._vault.take(rows, out)
                return out
        if self.residency == "float32":
            return np.ascontiguousarray(self.vec[rows], dtype=np.float32)
        if self.residency == "int8":
            return np.ascontiguousarray(self._sidecar_store().view()[rows],
                                        dtype=np.float32)
        if self.residency == "float16_mmap":
            return np.ascontiguousarray(self._sidecar_store().view()[rows],
                                        dtype=np.float32)
        return np.ascontiguousarray(self.vec[rows], dtype=np.float32)

    def vector(self, row: int) -> np.ndarray:
        """One fp32 vector, freshly owned (never a view into the arena)."""
        return self.vectors(np.asarray([int(row)], dtype=np.int64))[0]

    def matrix(self, lo: int = 0, hi=None) -> np.ndarray:
        """fp32 ``[lo:hi)`` of the arena.

        A VIEW under ``float32`` residency (what 3.0.3 handed out) and a fresh
        fp32 array under every other mode, so a caller that mutates it must copy
        first either way.
        """
        hi = self.n_rows if hi is None else int(hi)
        lo = int(lo)
        if self.vault_backed and self._vault_upcast:
            self._materialise_vault_vectors()
        if self.residency == "float32" and not self.vault_backed:
            return self.vec[lo:hi]
        return np.ascontiguousarray(self._decode_range(lo, hi), dtype=np.float32)

    def _decode_range(self, lo: int, hi: int) -> np.ndarray:
        if self.vault_backed:
            out = np.empty((max(0, hi - lo), self.embed_dim), dtype=np.float32)
            if out.shape[0]:
                self._vault.fill(lo, hi, out)
            return out
        if self.residency == "int8":
            return self._sidecar_store().view()[lo:hi].astype(np.float32)
        if self.residency == "float16_mmap":
            return self._sidecar_store().view()[lo:hi].astype(np.float32)
        return self.vec[lo:hi].astype(np.float32)

    def page_matrices(self):
        """Per-block fp32 vector matrices, for the PageLandmarkRouter ablation."""
        return [self.matrix(int(self.block_start[b]), int(self.block_start[b + 1]))
                for b in range(self.n_blocks)]

    # -- scan kernels -------------------------------------------------------
    def _chunked_dot(self, src, q: np.ndarray, out: np.ndarray) -> None:
        """``out[i] = src[i] . q`` in fp32, converting ``scan_chunk`` rows at a time.

        Every row's dot product is computed by the same fp32 BLAS call it would
        get from a full fp32 arena; chunking changes which rows share a call, not
        the arithmetic inside one. Measured over all 500 queries of the 71,433-row
        corpus, the resulting score vectors are BITWISE identical to the fp32
        arena's (scratch/refound/memory_results.json).
        """
        n = out.shape[0]
        c = self.scan_chunk
        for s in range(0, n, c):
            e = min(s + c, n)
            buf = self._stage(e - s)
            np.copyto(buf, src[s:e])
            out[s:e] = buf @ q

    def scores(self, q: np.ndarray) -> np.ndarray:
        """Cosine of every resident row against ``q`` (already L2-normalised).

        EXACT for ``float32``, ``float16`` and ``float16_mmap``: those three hold
        the same fp16-on-disk values and accumulate in fp32.

        APPROXIMATE for ``int8``, with one guarantee: the ``rerank_pool`` rows
        with the highest approximate score are re-scored EXACTLY from the fp16
        sidecar, and every row outside that pool is shifted down so it cannot
        outrank the pool. So the top-k is exact whenever the true top-k is inside
        the pool -- which is what the pool sweep in memory_results.json measures --
        but a score BELOW the pool is not a cosine and must not be shown to a
        user or compared against a threshold.
        """
        n = self.n_rows
        q = np.ascontiguousarray(q, dtype=np.float32)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        if self.vault_backed:
            if self._vault_upcast:
                self._materialise_vault_vectors()
            else:
                out = np.empty(n, dtype=np.float32)
                self._vault_chunked_dot(q, out)
                return out
        if self.residency == "float32":
            return np.asarray(self.vec[:n] @ q, dtype=np.float32)
        out = np.empty(n, dtype=np.float32)
        if self.residency == "int8":
            self._chunked_dot(self.vec[:n], q, out)
            out *= self.vscale[:n]
            return self._rerank(out, q)
        src = self._sidecar_store().view() if self.mapped else self.vec[:n]
        self._chunked_dot(src, q, out)
        return out

    def _rerank(self, approx: np.ndarray, q: np.ndarray) -> np.ndarray:
        n = approx.shape[0]
        pool = int(min(max(1, self.rerank_pool), n))
        if pool >= n:
            exact = np.empty(n, dtype=np.float32)
            self._chunked_dot(self._sidecar_store().view()[:n], q, exact)
            return exact
        idx = np.argpartition(-approx, pool - 1)[:pool]
        # Same kernel as the full scan, so a pool larger than ``scan_chunk``
        # cannot disagree with the full scan by a last-ULP BLAS-blocking
        # difference. (It could before, and it showed up in the pool sweep as a
        # LARGER pool flipping one tie.)
        exact = np.empty(pool, dtype=np.float32)
        self._chunked_dot(self._sidecar_store().view()[idx], q, exact)
        out = approx.copy()
        # Push everything outside the pool below the pool's worst EXACT score,
        # preserving its relative order, so selection can never mix the two
        # scales. ``float32`` spacing near 1.0 is ~6e-8; 1e-6 clears it.
        lo = float(exact.min())
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        if mask.any():
            shift = float(out[mask].max()) - lo + 1e-6
            if shift > 0.0:
                out[mask] -= shift
        out[idx] = exact
        return out

    def scores_rows(self, rows, q: np.ndarray) -> np.ndarray:
        """Cosine of the listed rows against ``q`` (the routed-candidate path)."""
        rows = np.asarray(rows, dtype=np.int64)
        q = np.ascontiguousarray(q, dtype=np.float32)
        if rows.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self.residency == "float32" and not self.vault_backed:
            return np.asarray(self.vec[rows] @ q, dtype=np.float32)
        return np.asarray(self.vectors(rows) @ q, dtype=np.float32)

    def score_matrix(self, Q: np.ndarray) -> np.ndarray:
        """``(n_rows, n_queries)`` cosines, for :meth:`VaultEngine.search_batch`."""
        n = self.n_rows
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        if self.vault_backed and self._vault_upcast:
            self._materialise_vault_vectors()
        if self.residency == "float32" and not self.vault_backed:
            return np.asarray(self.vec[:n] @ Q.T, dtype=np.float32)
        out = np.empty((n, Q.shape[0]), dtype=np.float32)
        vault = self._vault if self.vault_backed else None
        src = (self._sidecar_store().view() if self.needs_sidecar
               else self.vec[:n])
        c = self.scan_chunk
        for s in range(0, n, c):
            e = min(s + c, n)
            buf = self._stage(e - s)
            if vault is not None:
                vault.fill(s, e, buf)
            else:
                np.copyto(buf, src[s:e])
            out[s:e] = buf @ Q.T
        return out

    # -- reads --------------------------------------------------------------
    def record(self, row: int) -> dict:
        """Deserialise row ``row`` into a fresh ``{"text","source","metadata"}`` dict."""
        b = self.rec_bytes[int(self.row_block[row])]
        lo, hi = self.doc_span[row]
        return json.loads(bytes(memoryview(b)[int(lo):int(hi)]).decode("utf-8"))

    def metadata(self, row: int) -> dict:
        return self.record(row).get("metadata") or {}

    def rows_with_entity_ids(self, eids, limit: int = 512) -> np.ndarray:
        """Rows whose entity is one of ``eids`` (entity-history injection pool)."""
        if self.n_rows == 0 or eids is None or len(eids) == 0:
            return np.zeros(0, dtype=np.int64)
        hit = np.flatnonzero(np.isin(self.entity_id[:self.n_rows], np.asarray(eids, dtype=np.int32)))
        if hit.size > limit:
            hit = hit[-limit:]
        return hit.astype(np.int64)

    def resident_bytes(self) -> dict:
        """Byte budget of everything this arena actually holds. All measured.

        Every array is counted at its ALLOCATED size, not the used slice.
        ``arena_used_bytes`` is the slice, for the size-per-document arithmetic.
        The two are now equal for the vectors of an unhinted ingest, because the
        vector array is cut to the rows that exist and the headroom sits in a
        reservation behind it (:class:`_VectorStore`); under 3.0.3's capacity
        doubling they differed by up to 1.83x and reporting the slice would have
        under-counted the largest allocation in the process by that much. The
        int8 scale column and the small per-row columns still grow in steps, so
        for those two the distinction is still live.

        ``arena_bytes`` is the ANONYMOUS memory the vector array spans. Every
        page of it that holds a row has been written and is resident; a tail that
        a hint reserved ahead of the rows has not been and is not.
        ``arena_reservation_bytes`` is the address space reserved behind it:
        untouched pages of an anonymous mapping are not resident, so this number
        is not RAM and must never be added to the others.
        ``arena_mapped_bytes`` is the file-backed sidecar (``float16_mmap`` /
        ``int8``), which shows up in RSS once touched but is clean and evictable,
        so it must not be added to ``arena_bytes`` and called "RAM" either.
        """
        n, m = self.n_rows, int(self.land_start[-1]) if self.land_start.size else 0
        cached = self._store.mapped_array
        cols_mapped = self._mapped_columns
        arena_alloc = (0 if cached else int(self.vec.nbytes)) + int(self.vscale.nbytes)
        arena_used = int(n * self.embed_dim * self.vec.dtype.itemsize)
        if self.residency == "int8":
            arena_used += int(n * 4)
        mapped = int(self._sidecar.mapped_bytes()) if self._sidecar is not None else 0
        if cached:
            mapped += int(self.vec.nbytes)
        vault_vec = 0
        if self.vault_backed:
            # The vault's own vector regions, mapped. Clean, file-backed and
            # evictable, like every other mapped number here -- and NOT a second
            # copy of anything, which is the whole point.
            vault_vec = int(n * self.embed_dim * self._vault.itemsize)
            mapped += vault_vec
        landmarks = int(self.landmarks.nbytes) or m * self.embed_dim * 4
        columns = int(self.ts.nbytes + self.rev.nbytes + self.entity_id.nbytes
                      + self.group_id.nbytes + self.row_block.nbytes
                      + self.doc_span.nbytes)
        # Record sections and the id table are file-backed under a cached open,
        # so they are reported as MAPPED, not as heap. That is not bookkeeping:
        # those pages are clean and the OS can drop them, and at 71,433 rows the
        # records alone are 43.9 MiB that a scanned arena holds in anonymous RAM.
        records = int(getattr(self.rec_bytes, "anon_bytes", 0)
                      if isinstance(self.rec_bytes, (_MappedBlobs, _VaultBlobs))
                      else sum(len(b) for b in self.rec_bytes))
        rec_mapped = int(getattr(self.rec_bytes, "mapped_bytes", 0))
        ids_mapped = int(getattr(self.ids, "mapped_bytes", 0))
        if cols_mapped:
            columns = 0                   # every column above is a mapped view
        index_est = int(sys.getsizeof(self._id_index) + 64 * n) if self._id_index is not None else 0
        stage = int(self._buf.nbytes) if self._buf is not None else 0
        return {"arena_bytes": arena_alloc, "arena_used_bytes": arena_used,
                "arena_from_cache": bool(self._snapshot is not None),
                "arena_vault_backed": bool(self.vault_backed),
                "arena_vault_vector_bytes": vault_vec,
                "arena_vectors_mapped": bool(cached or self.vault_backed),
                "arena_cache_mapped_bytes": (int(self.vec.nbytes) if cached
                                             else vault_vec),
                "record_mapped_bytes": rec_mapped, "id_table_mapped_bytes": ids_mapped,
                "column_mapped_bytes": int(
                    self.ts.nbytes + self.rev.nbytes + self.entity_id.nbytes
                    + self.group_id.nbytes + self.row_block.nbytes
                    + self.doc_span.nbytes) if cols_mapped else 0,
                "arena_reservation_bytes": self._store.reservation_bytes(),
                "arena_reservation_is_mapped": bool(self._store.mapped_reservation),
                "arena_growth_copies": int(self._store.copies),
                "arena_mapped_bytes": mapped, "arena_residency": self.residency,
                "arena_dtype": np.dtype(self.store_dtype).name,
                "scan_buffer_bytes": stage,
                "landmark_bytes": int(landmarks), "column_bytes": columns,
                "record_bytes": records, "index_bytes_estimated": index_est}

    def close(self) -> None:
        """Release the sidecar fd (a no-op for the anonymous modes).

        The vector reservation is deliberately NOT released here: ``close()`` is
        called by ``VaultEngine.close()``, after which ``stats()`` and the
        already-returned hit dicts must still read. It is unmapped when the arena
        itself is dropped, which is what makes a build-then-reopen process hand
        the ingest arena back before the second one is built.
        """
        if self._sidecar is not None:
            self._sidecar.close()
            self._sidecar = None
        # The arena cache mapping is NOT dropped here for the same reason the
        # vector reservation is not: ``stats()`` and already-returned hits must
        # still read after ``close()``. ``mmap.close()`` refuses while a numpy
        # view is live anyway, so the pages go when the arena itself does.

    def reset(self) -> None:
        self.close()
        snap, self._snapshot = self._snapshot, None
        if snap is not None:
            snap.close()
        vault, self._vault = self._vault, None
        self._vault_vectors = self._vault_upcast = False
        if vault is not None:
            vault.close()
        self._store.close()
        self.__init__(self.embed_dim, residency=self.residency,
                      scan_chunk=self.scan_chunk, rerank_pool=self.rerank_pool,
                      sidecar_dir=self.sidecar_dir)


class Memtable:
    """Unflushed records: at most ``block_capacity`` of them, scored on every query."""

    def __init__(self, embed_dim: int, block_capacity: int):
        self.embed_dim = int(embed_dim)
        self.block_capacity = int(block_capacity)
        self.items = []
        self.vec = np.zeros((self.block_capacity, self.embed_dim), dtype=np.float32)

    @property
    def n(self) -> int:
        return len(self.items)

    def append(self, item: dict, vector: np.ndarray) -> None:
        i = len(self.items)
        if i >= self.vec.shape[0]:
            out = np.zeros((max(self.block_capacity, i * 2), self.embed_dim), dtype=np.float32)
            out[:i] = self.vec[:i]
            self.vec = out
        self.vec[i] = vector
        self.items.append(item)

    def drop_front(self, k: int) -> None:
        k = int(k)
        rest = len(self.items) - k
        if rest > 0:
            self.vec[:rest] = self.vec[k:k + rest]
        self.items = self.items[k:]

    def clear(self) -> None:
        self.items = []

    def nbytes(self) -> int:
        return int(self.n * self.embed_dim * 4)


__all__ = ["Arena", "Memtable", "RESIDENCY_MODES", "DEFAULT_SCAN_CHUNK",
           "DEFAULT_RERANK_POOL"]
