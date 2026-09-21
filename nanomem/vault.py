"""
nanomem.vault
~~~~~~~~~~~~~
Primary public API for the nanomem continuous memory engine.
Provides a clean, intuitive 4-method interface: add(), search(), chat(), and ask().
"""

import os
import warnings
import time
import json
import urllib.parse
import urllib.request
import numpy as np
from typing import List, Dict, Any, Optional
from .engine import VaultEngine
from .embed import EmbeddingProvider
from .entities import detect_entity
from .errors import NanomemError, NotEncryptedError, PasswordRequiredError

_KEEP_PW = object()      # "inherit this vault's password" sentinel
_UNSET_PW = object()     # "the caller said nothing about a password" sentinel


def resolve_vault_path(name, root: str, suffix: str = ".dat") -> str:
    """Resolve a client-supplied vault NAME inside ``root``. Never escapes it.

    A network endpoint that takes a vault path from a request body is an
    arbitrary-path file-creation primitive, because :class:`Container` creates
    the file and ``os.makedirs`` its parents. 3.0.2's ``/v1/vault/init`` and
    ``/load`` both did exactly that: ``{"vault": "../../elsewhere/x.dat"}``
    created a vault and its parent directory outside the working directory.

    The name is taken as a path RELATIVE to ``root`` -- absolute paths, ``..``
    segments and anything that resolves outside ``root`` are refused, as is a
    name that does not end in ``suffix``.
    """
    root = os.path.realpath(os.path.abspath(root))
    raw = str(name or "").strip()
    if not raw:
        raise ValueError("vault name must not be empty")
    if os.path.isabs(raw) or raw.startswith("~"):
        raise ValueError(f"vault name must be relative to {root!r}, got {raw!r}")
    if not raw.endswith(suffix):
        raise ValueError(f"vault name must end in {suffix!r}, got {raw!r}")
    target = os.path.realpath(os.path.join(root, raw))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f"vault name {raw!r} resolves outside {root!r}")
    return target


class TextHopBridge:
    """
    Default `search_multihop` bridge: re-embed the question together with the
    hop-1 document text.

    `bridge(q_vec, d1_vec, question, d1_text)` returns a unit vector for the
    second pass. Any object with that method can be passed as
    `Vault.search_multihop(..., bridge=...)`; `q_vec` and `d1_vec` are supplied
    so a latent-space bridge can be used instead of re-embedding text.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def bridge(self, q_vec, d1_vec, question, d1_text):
        if not d1_text:
            return None
        v = self.embedder.embed(f"{question}\n{d1_text}")
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None


# Local servers that speak the OpenAI chat-completions API. Matched on the
# PARSED PORT, never on a substring of the URL: through 0.6.0 the test was
# `"8000" in target_base`, so a host called `web8000.internal` on port 11434 was
# routed to /v1/chat/completions and failed, while llama.cpp on 8080 -- which
# does speak that API -- was routed to Ollama's /api/generate and also failed.
OPENAI_COMPATIBLE_PORTS = frozenset({1234, 8000, 8080, 5000, 4891, 11435})


# The most sub-queries a composite question is allowed to become. Each one is a
# separate full scan, so this is a ceiling on what a single `search` can cost.
# Measured over 145,051 distinct queries from every non-quarantined corpus on
# the development tree: 99.95% decompose to 8 or fewer, largest real one is 25,
# and a pasted 1,800-word FAQ is 400. Past this the text is treated as one
# query, which is both cheaper and more faithful to what it is.
MAX_SUB_QUERIES = 8


def _chunk_index(chunk_id: str) -> int:
    """Where a chunk sits in its document, for ordering.

    Two id shapes reach here. `add()` writes ``{parent}_chunk_N``; `ingest_file`
    writes ``{basename}:{start}-{end}``, and from 0.7.18 those carry a
    ``parent_id`` too, so they have to sort by their START LINE or a reassembled
    file comes back with its lines shuffled. An unrecognised id sorts last
    instead of raising.
    """
    text = str(chunk_id)
    tail = text.rsplit("_chunk_", 1)
    if len(tail) == 2 and tail[1].isdigit():
        return int(tail[1])
    span = text.rsplit(":", 1)
    if len(span) == 2:
        start = span[1].split("-", 1)[0]
        if start.isdigit():
            return int(start)
    return 1 << 30


_CJK_RANGES = (
    (0x3040, 0x30FF),   # Hiragana, Katakana
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x20000, 0x2FA1F), # CJK Extension B..F and Compatibility Supplement
)


def _is_dense_script(text: str, threshold: float = 0.2) -> bool:
    """Is this text a script that does not put spaces between words?

    Through 0.7.17 this was ``any(ord(c) > 0x2E80 for c in text)``. Every emoji
    sits above that threshold, so ONE emoji anywhere in an English document
    routed the whole thing down the character splitter, which cuts every
    ``MAX_FACT_CHARS_CJK`` characters with no regard for word boundaries.
    Measured on a 3,000-token ASCII document with a single emoji inserted:
    chunks 6 -> 19 and 13 tokens PERMANENTLY LOST, with others stored truncated
    mid-word ("tok004" for "tok00499"). Emoji are ordinary content in chat logs,
    notes and commit messages, so this was silent data loss on everyday input.

    Two things are fixed here. The ranges are the actual CJK and Hangul blocks
    rather than "anything above U+2E80", so symbols and emoji no longer qualify.
    And it is a PROPORTION rather than an existence test, so one Chinese
    character quoted in an English paragraph does not change how the paragraph is
    split -- which the old test also got wrong, just less visibly.
    """
    if not text:
        return False
    dense = 0
    counted = 0
    for ch in text:
        if ch.isspace():
            continue
        counted += 1
        o = ord(ch)
        for lo, hi in _CJK_RANGES:
            if lo <= o <= hi:
                dense += 1
                break
    if counted == 0:
        return False
    return (dense / counted) >= threshold


#: Metadata keys this library writes itself. A caller who sets one is not adding
#: a field, they are overwriting a structural link -- and `parent_id` in
#: particular is an ordinary thing to want (tickets, threads, comments). Two rows
#: tagged `{"parent_id": "TICKET-42"}` were read back as ONE chunked document:
#: `get("TICKET-42")` returned both texts joined, `exists` said True for a
#: document nobody added, and `delete(id="TICKET-42")` removed both. Filtering on
#: these keys is supported and documented; SETTING them is refused.
RESERVED_METADATA_KEYS = frozenset({
    "id", "parent_id", "chunk_index", "chunk_total", "is_chunked",
    "parent_sha256", "embed_backend",
})

#: Why each one is refused, so the error can say something useful.
_RESERVED_WHY = {
    "id": ("it is read as the DOCUMENT id, so three different documents tagged "
           "{'id': 'ticket-4711'} all came back under that one handle -- `get` "
           "returned only the newest, `exists` said True, and `delete(id=...)` "
           "removed all three. Pass `id=` to choose a document's id; that is "
           "what it is for"),
}
_RESERVED_DEFAULT_WHY = ("they link a split document's chunks, and overriding them "
                         "makes unrelated records read back as one document")


def _jsonable_metadata(meta):
    """Make metadata storable without silently turning it into a string.

    Metadata is serialised as JSON, and a `set` is not JSON-serialisable -- so a
    set-valued tag was stored as its Python REPR, `"{'x', 'y'}"`, and a filter
    for `"x"` then matched nothing. A set is an obvious way to express "these
    tags", so it is converted to a sorted list, which filters by membership the
    way the caller meant. Tuples go the same way. Anything else is untouched.
    """
    if not isinstance(meta, dict):
        return meta
    out = {}
    for k, v in meta.items():
        if isinstance(v, (set, frozenset)):
            out[k] = sorted(v, key=lambda x: (str(type(x)), str(x)))
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _reject_reserved_metadata(meta) -> None:
    if not isinstance(meta, dict):
        return
    clash = sorted(RESERVED_METADATA_KEYS & set(meta))
    if clash:
        why = _RESERVED_WHY.get(clash[0], _RESERVED_DEFAULT_WHY)
        raise ValueError(
            "metadata key(s) %s are read by nanomem itself and cannot be set by a "
            "caller: %s. Rename the field (for example %r -> %r). Filtering on "
            "these keys is still supported."
            % (", ".join(repr(k) for k in clash), why, clash[0], "my_" + clash[0]))


def _doc_fingerprint(text: str) -> str:
    """A stable fingerprint of EXACTLY the text a caller passed to `add()`.

    `split_large_text` REBUILDS text rather than slicing it -- paragraphs are
    stripped, units are re-joined with " " and "\n\n" -- so the chunks cannot be
    concatenated back into the original, and `_join_chunks` is a reconstruction,
    not an inverse. Three ways it differs, all measured: every newline becomes a
    space, a passage that repeats is dropped by the overlap detector (one
    sentence repeated 400 times lost 29% of the document), and indentation is
    gone.

    0.7.16 resolved `delete(text_exact=...)` on a chunked document by comparing
    against that reconstruction, and the test written for it used a document of
    unique space-joined tokens -- the one shape where the reconstruction happens
    to round-trip. Every real document over 500 words has newlines, so the fix
    covered almost nothing and still returned 0. Found by the fourth black-box
    review (F1).

    So the match is made against what was WRITTEN, not against what can be
    rebuilt. `hashlib` is stdlib; this adds no dependency.
    """
    import hashlib
    return hashlib.sha256(str(text).strip().encode("utf-8")).hexdigest()


def _join_chunks(texts: List[str], max_overlap_words: int = 200) -> str:
    """Rejoin a split document, removing the overlap bridge between neighbours.

    `split_large_text` repeats roughly ``overlap_words`` between consecutive
    chunks so no sentence is orphaned, and it breaks on paragraph and sentence
    boundaries -- so the repeat is rarely exactly that many words. The real
    overlap is found instead of assumed: the longest suffix of one chunk that is
    also a prefix of the next. Concatenating without this repeats text, which on
    a document that mentions something once inside the bridge reads as if it
    were said twice.
    """
    out: List[str] = []
    for t in texts:
        w = str(t).split()
        if not out:
            out = w
            continue
        limit = min(len(out), len(w), max_overlap_words)
        k = 0
        for n in range(limit, 0, -1):
            if out[-n:] == w[:n]:
                k = n
                break
        out.extend(w[k:])
    return " ".join(out)


def _llm_endpoint(base_url: str):
    """Resolve a base URL to ``(flavour, url)``.

    ``flavour`` is ``"anthropic"``, ``"openai"`` or ``"ollama"``, and decides
    the request SHAPE, not just the path -- the three differ in payload and in
    how the key is sent.

    Resolution order, most explicit first:

    1. an Anthropic host, or a path already ending ``/messages``;
    2. a path already ending ``/chat/completions`` -- used verbatim, which is
       how an Azure OpenAI deployment URL (with its ``api-version`` query) or
       any other non-standard route is supported: pass the whole thing;
    3. a path ending ``/v1`` -- the convention every OpenAI-compatible vendor
       follows (OpenAI, Groq, Together, Mistral, DeepSeek, OpenRouter,
       Fireworks, LM Studio, vLLM);
    4. a known OpenAI-compatible local PORT;
    5. otherwise Ollama's native ``/api/generate``.
    """
    base = (base_url or "").rstrip("/")
    parsed = urllib.parse.urlparse(base if "://" in base else "http://" + base)
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    try:
        port = parsed.port
    except ValueError:                      # malformed port; treat as unset
        port = None

    if host.endswith("anthropic.com") or path.endswith("/messages"):
        if path.endswith("/messages"):
            return "anthropic", base
        return "anthropic", (f"{base}/messages" if path.endswith("/v1")
                             else f"{base}/v1/messages")
    if path.endswith("/chat/completions"):
        return "openai", base
    if path.endswith("/v1"):
        return "openai", f"{base}/chat/completions"
    if port in OPENAI_COMPATIBLE_PORTS or host.endswith("openai.azure.com"):
        return "openai", f"{base}/v1/chat/completions"
    if path.endswith("/api/generate"):
        return "ollama", base
    return "ollama", f"{base}/api/generate"


class Vault:
    """
    A persistent memory vault for AI applications and agents, backed by a single
    local binary file.

    Every stored vector is held in RAM as fp32 and searched with one dense
    matmul, which is what makes search exact. On disk the default is fp16
    (``vector_dtype="float32"`` to change it), so the resident fp32 array of a
    vault that was written by an earlier process is fp16-derived: measured worst
    per-row cosine against the original 0.9999998, 0 discordant top-4 on the
    validation corpora. The cost is roughly 3-4 KB of RAM per document, reported
    by ``stats()["active_heap_ram_kb"]`` (allocated bytes, measured, never a
    constant) rather than assumed.

    MEASURED on 10,000 HotpotQA paragraphs in the shape a caller actually runs
    (a vault opened from an existing file, 500 questions,
    ``evidence/scale_results_v3r3.json``): 22.0 MB on disk, 0.344 ms p50
    search, evidence recall@4 70.4% -- identical to an exhaustive scan (0.272 ms)
    and to FAISS IndexFlatIP (0.391 ms) on the same data. At 71,433 paragraphs:
    155.9 MB, 1.809 ms p50, 60.6% -- again identical to both. The index is 0.609x
    of (raw UTF-8 text + fp32 vectors), but roughly half of that saving is
    precision, not format: against raw text + fp16 vectors, the shape nanomem
    actually stores, the same index is 1.056x
    (``evidence/headtohead_v3.json`` -> ``index_size``). Search is exact and
    LINEAR; there is no fixed millisecond guarantee.
    """

    def __init__(
        self,
        path: str = "memory.dat",
        embed_model: str = "nomic-embed-text",
        base_url: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        password=_UNSET_PW,
        on_torn_tail: str = "warn",
        embedder: Optional[Any] = None,
        vector_dtype: str = "float16",
        group_floor_sim: float = 0.0,
        intent_margin: float = 0.15
    ):
        # THREE DOCUMENTED ARGUMENTS THAT DID NOT EXIST.
        # Each was described in the README or this class's own docstring and
        # each raised TypeError, because all three were implemented further down
        # and never plumbed up to here: `vector_dtype` and `group_floor_sim` are
        # `VaultEngine` arguments, and `embedder` only needed the provider to be
        # something a caller could supply rather than something built in place.
        # Loud failures, so nothing was silently wrong -- but the README's
        # "+19.0 points of top-1" tuning step could not be applied at all.
        self.path = os.path.abspath(path)
        if embedder is not None:
            missing = [a for a in ("embed", "embed_batch", "dim")
                       if not hasattr(embedder, a)]
            if missing:
                raise TypeError(
                    "embedder= needs .embed, .embed_batch and .dim; "
                    f"{type(embedder).__name__} is missing {', '.join(missing)}")
            self.embedder = embedder
        else:
            self.embedder = EmbeddingProvider(model=embed_model, base_url=base_url)
        # EXPLICIT None IS AN OPT-OUT, not "unset". `export(target_password=None)`
        # deliberately writes a plaintext target, and 3.0.2 still picked up
        # NANOMEM_PASSWORD for it -- so the opt-out was unreliable and raised
        # NotEncryptedError against an existing plaintext target. Only an omitted
        # argument consults the environment.
        if password is _UNSET_PW:
            self._password = os.getenv("NANOMEM_PASSWORD") or None
        else:
            self._password = password
        self.engine = VaultEngine(filepath=self.path, embed_dim=self.embedder.dim,
                                  password=self._password,
                                  on_torn_tail=on_torn_tail,
                                  vector_dtype=vector_dtype,
                                  group_floor_sim=group_floor_sim,
                                  intent_margin=intent_margin)
        self.llm_base_url = llm_base_url or os.getenv("NANOMEM_LLM_URL", os.getenv("LLM_BASE_URL"))
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.last_id: Optional[str] = None

    #: Bytes of source text per ingested chunk, used ONLY to pre-size the arena
    #: for a directory ingest before anything has been read. See
    #: :meth:`ingest_directory` for the measurement behind the number.
    BYTES_PER_CHUNK_ESTIMATE = 2048

    MAX_FACT_WORDS = 500       # High threshold for space-separated text (~2,500 chars)
    MAX_FACT_CHARS_CJK = 1500  # High threshold for dense non-spaced CJK scripts
    OVERLAP_WORDS = 40         # Overlap bridge between split chunks

    @classmethod
    def split_large_text(cls, text: str, max_words: int = 500, overlap_words: int = 40) -> List[str]:
        """
        Hierarchical Semantic Auto-Splitter:
        Breaks oversized text along natural semantic boundaries (paragraphs -> sentences)
        with an overlap bridge between consecutive chunks to ensure zero context fragmentation.
        Supports both space-separated (Latin, Hindi, Arabic) and dense non-spaced (CJK) scripts.
        """
        clean = str(text).strip()
        if not clean:
            return []

        words = clean.split()
        char_count = len(clean)
        is_cjk = _is_dense_script(clean)

        if not is_cjk and len(words) <= max_words:
            return [clean]
        if is_cjk and char_count <= cls.MAX_FACT_CHARS_CJK:
            return [clean]

        import re
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean) if p.strip()]
        if not paragraphs:
            paragraphs = [clean]

        raw_units = []
        for p in paragraphs:
            p_words = p.split()
            p_cjk = _is_dense_script(p)
            if (not p_cjk and len(p_words) > max_words) or (p_cjk and len(p) > cls.MAX_FACT_CHARS_CJK):
                sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？\n])\s+", p) if s.strip()]
                raw_units.extend(sentences if sentences else [p])
            else:
                raw_units.append(p)

        units = []
        for u in raw_units:
            u_words = u.split()
            u_cjk = _is_dense_script(u)
            if not u_cjk and len(u_words) > max_words:
                for i in range(0, len(u_words), max_words):
                    units.append(" ".join(u_words[i : i + max_words]))
            elif u_cjk and len(u) > cls.MAX_FACT_CHARS_CJK:
                for i in range(0, len(u), cls.MAX_FACT_CHARS_CJK):
                    units.append(u[i : i + cls.MAX_FACT_CHARS_CJK])
            else:
                units.append(u)

        chunks = []
        cur_chunk = []
        cur_words = 0
        cur_chars = 0

        for unit in units:
            unit_w = len(unit.split())
            unit_c = len(unit)
            would_exceed = (not is_cjk and cur_words + unit_w > max_words) or (is_cjk and cur_chars + unit_c > cls.MAX_FACT_CHARS_CJK)
            if would_exceed and cur_chunk:
                chunk_str = "\n\n".join(cur_chunk)
                chunks.append(chunk_str)

                tail_words = " ".join(cur_chunk).split()
                if not is_cjk and len(tail_words) > overlap_words:
                    bridge = " ".join(tail_words[-overlap_words:])
                    cur_chunk = [bridge, unit]
                    cur_words = len(bridge.split()) + unit_w
                    cur_chars = len(bridge) + unit_c
                elif is_cjk and len(chunk_str) > 120:
                    bridge = chunk_str[-120:]
                    cur_chunk = [bridge, unit]
                    cur_words = len(bridge.split()) + unit_w
                    cur_chars = len(bridge) + unit_c
                else:
                    cur_chunk = [unit]
                    cur_words = unit_w
                    cur_chars = unit_c
            else:
                cur_chunk.append(unit)
                cur_words += unit_w
                cur_chars += unit_c

        if cur_chunk:
            chunks.append("\n\n".join(cur_chunk))

        return chunks if chunks else [clean]

    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "user_input",
        timestamp: Optional[float] = None,
        revision: Optional[int] = None,
        id: Optional[str] = None
    ) -> str:
        """
        Store a document, note, or fact into memory and return its document id.
        Transparently auto-splits oversized text (> 500 words / 1500 CJK chars)
        into linked atomic chunks with semantic overlap; the parent id is returned
        for a chunked document. Returns "" for empty input.

        The generated id is ``doc_<md5(text)[:10]>`` -- a function of the TEXT
        alone -- so adding the identical text twice stores two rows that share one
        id. ``get(id)`` and ``search_multihop``'s anchor lookup return the most
        recent of them; ``delete(id=...)`` removes all of them; ``update(id=...)``
        rewrites the first. Pass an explicit ``id=`` if you need one row per call.
        """
        if not text or not str(text).strip():
            return ""
        clean_text = str(text).strip()

        _reject_reserved_metadata(metadata)
        meta = _jsonable_metadata(dict(metadata or {}))
        import hashlib
        # NOT `meta.get("id")`. That was an undocumented side channel which
        # overrode both halves of this method's documented contract -- that the
        # id is "a function of the TEXT alone", and that `id=` is how you supply
        # one. `id` is now a reserved key, refused above, so a caller who means
        # "my ticket number" gets told to rename the field instead of silently
        # losing two of their three documents. `add_batch` still reads it: that
        # is the round-trip path for merge and split, where the metadata being
        # replayed is nanomem's own.
        doc_id = str(id or f"doc_{hashlib.md5(clean_text.encode('utf-8')).hexdigest()[:10]}")
        meta["id"] = doc_id
        # DO NOT WRITE BACK INTO THE CALLER'S DICT. Until 0.7.18 this set
        # `metadata["id"]` on the object the caller passed in, so reusing one dict
        # across several `add()` calls -- `meta = {"user_id": "alice"}` and then
        # three facts -- fed the FIRST document's id back in as the second's, and
        # every later document collapsed onto it. Three distinct facts, one id:
        # `get` returned only the newest, `delete(id=)` removed all three, and
        # `update(id=)` rewrote the first. It also contradicted this method's own
        # documented rule that the id is a function of the TEXT alone. The id is
        # the return value; mutating the argument was never documented.
        self.last_id = doc_id

        chunks = self.split_large_text(clean_text, max_words=self.MAX_FACT_WORDS, overlap_words=self.OVERLAP_WORDS)
        if len(chunks) <= 1:
            vec = self.embedder.embed(clean_text)
            # Which encoder produced this vector. Rows written while the endpoint
            # was unreachable are encoded lexically and are not comparable with
            # rows the model embedded; without this there is no way to find them
            # afterwards. Only recorded when it is NOT the model, so ordinary rows
            # carry no extra metadata.
            if getattr(self.embedder, "last_backend", "model") != "model":
                meta["embed_backend"] = self.embedder.last_backend
            return self.engine.add_fact(
                text=clean_text,
                embedding=vec,
                source=source,
                metadata=meta,
                timestamp=timestamp,
                revision=revision,
                id=doc_id
            )

        parent_id = doc_id
        total_chunks = len(chunks)
        # Hoisted out of the loop below. It was computed per chunk, so a
        # 60,000-word document hashed 72 MB instead of 0.6 MB -- 119x the work,
        # for a value that is identical on every chunk by definition.
        parent_fp = _doc_fingerprint(clean_text)
        vecs = self.embedder.embed_batch(chunks)
        backend = getattr(self.embedder, "last_backend", "model")

        for idx, ch in enumerate(chunks):
            chunk_meta = dict(meta)
            chunk_id = f"{parent_id}_chunk_{idx+1}"
            chunk_meta["id"] = chunk_id
            chunk_meta["chunk_index"] = idx + 1
            chunk_meta["chunk_total"] = total_chunks
            chunk_meta["parent_id"] = parent_id
            chunk_meta["is_chunked"] = True
            chunk_meta["parent_sha256"] = parent_fp
            if backend != "model":
                chunk_meta["embed_backend"] = backend

            self.engine.add_fact(
                text=ch,
                embedding=vecs[idx],
                source=source,
                metadata=chunk_meta,
                timestamp=timestamp,
                revision=revision,
                id=chunk_id
            )
        return parent_id

    def ingest_file(
        self,
        file_path: str,
        lines_per_chunk: int = 50,
        overlap_lines: int = 10,
        metadata: Optional[Any] = None,
        source: Optional[str] = None
    ) -> int:
        """
        Ingest an entire code, markdown, or text file into memory.
        Preserves exact code formatting, indentation, and line numbers for coding agents.
        Supports custom document-level or section-level metadata and source references.
        """
        if isinstance(metadata, dict):
            _reject_reserved_metadata(metadata)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        binary_exts = {
            ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tiff",
            ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar",
            ".bin", ".exe", ".dll", ".so", ".dylib", ".mp3", ".mp4", ".wav",
            ".docx", ".pptx", ".xlsx", ".doc", ".xls", ".ppt",
            ".parquet", ".feather", ".arrow", ".h5", ".hdf5",
            ".pkl", ".pickle", ".safetensors", ".onnx", ".pt", ".pth"
        }
        ext = os.path.splitext(file_path)[1].lower()
        if ext in binary_exts:
            raise ValueError(
                f"Binary / office container format '{ext}' cannot be ingested directly into text memory. "
                "For PDFs, Word/Excel documents, audio, and images, extract plain text or markdown first, then ingest the resulting text file."
            )

        basename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_lines = f.readlines()
        except Exception:
            return 0

        if not raw_lines:
            return 0

        chunks = []
        # Once for the file, not once per chunk: this both joins and hashes.
        file_fp = _doc_fingerprint("".join(raw_lines))
        import hashlib as _hashlib
        file_doc_id = "doc_" + _hashlib.md5(
            os.path.abspath(file_path).encode("utf-8")).hexdigest()[:10]
        stride = max(1, lines_per_chunk - overlap_lines)
        code_exts = {
            ".py", ".ts", ".js", ".tsx", ".jsx", ".rs", ".go", ".cpp", ".c", ".h", ".hpp",
            ".java", ".cs", ".rb", ".php", ".swift", ".kt", ".sh", ".bash", ".sql", ".html", ".css"
        }
        is_code = ext in code_exts

        def _by_line_budget(lines):
            """Break a slice on LINE boundaries so no piece exceeds the word cap.

            Anything still over the cap reaches `add_batch`, which re-splits it
            with the word-based splitter and loses the formatting this method
            promises. Splitting here keeps whole lines, so indentation survives.
            """
            out, cur, cur_words = [], [], 0
            for ln in lines:
                w = len(ln.split())
                if cur and cur_words + w > self.MAX_FACT_WORDS:
                    out.append(cur); cur, cur_words = [], 0
                cur.append(ln); cur_words += w
            if cur:
                out.append(cur)
            return out or [lines]

        for start_idx in range(0, len(raw_lines), stride):
            window = raw_lines[start_idx : start_idx + lines_per_chunk]
            if not window:
                break
            for slice_lines in _by_line_budget(window):
                # KEEP THE FORMATTING THIS METHOD PROMISES. Two things used to
                # break it. `.strip()` removed the leading indentation of a chunk's
                # first line, which in Python is the difference between a method and
                # a module-level function. And a 50-line slice of long lines can
                # exceed `MAX_FACT_WORDS`, at which point `add_batch` re-split it with
                # `split_large_text` -- the word-based splitter that rebuilds text --
                # and the indentation went with it. Measured on 200 lines of ~30
                # tokens each: indentation preserved with short lines, GONE with long
                # ones, while the docstring promises "exact code formatting,
                # indentation, and line numbers" either way.
                chunk_text = "".join(slice_lines).rstrip("\n")
                if chunk_text.strip():
                    start_line = start_idx + 1
                    end_line = min(start_idx + len(slice_lines), len(raw_lines))
                    chunk_id = f"{basename}:{start_line}-{end_line}"

                    if callable(metadata):
                        try:
                            chunk_meta = dict(metadata(chunk_text, start_line, end_line) or {})
                        except Exception:
                            chunk_meta = {}
                    elif isinstance(metadata, dict):
                        chunk_meta = dict(metadata)
                    else:
                        chunk_meta = {}

                    chunk_meta["id"] = chunk_id
                    chunk_meta["file_path"] = file_path
                    chunk_meta["filename"] = basename
                    chunk_meta["start_line"] = start_line
                    chunk_meta["end_line"] = end_line
                    chunk_meta["is_code"] = is_code
                    # The same fingerprint `add()` stamps on a split document, so
                    # `delete(text_exact=<the file's text>)` reaches an ingested file
                    # too. `delete` advertises exact-text targeting, and an ingested
                    # file was the last multi-row document it could not match -- it
                    # returned 0 and removed nothing, silently.
                    chunk_meta["parent_sha256"] = file_fp
                    # ONE INGESTED FILE IS ONE DOCUMENT. Without this every
                    # chunk looked like a separate VERSION of the entity, and
                    # `forget_superseded(keep=1)` on a four-chunk document
                    # deleted three of them -- measured: the query that answered
                    # "are shipping charges refundable?" correctly before the
                    # call answered with a DIFFERENT passage after, which this
                    # method's own docstring calls worse than an empty result.
                    # `add()` has always stamped this; `ingest_file` never did.
                    chunk_meta["parent_id"] = file_doc_id

                    chunk_source = source or f"{basename}:{start_line}-{end_line}"
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": chunk_meta,
                        "source": chunk_source,
                        "preserve_indent": True,
                    })

        if chunks:
            self.add_batch(chunks, batch_size=32, _replaying_stored_records=True)
        return len(chunks)

    def ingest_directory(
        self,
        dir_path: str,
        extensions: Optional[List[str]] = None,
        ignore_dirs: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Recursively indexes an entire codebase or repository for coding agents.
        Skips build artifacts, git history, and caches automatically.
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        default_exts = {
            ".py", ".ts", ".js", ".tsx", ".jsx", ".rs", ".go", ".cpp", ".c", ".h", ".hpp",
            ".java", ".cs", ".rb", ".php", ".swift", ".kt", ".sh", ".bash", ".sql",
            ".md", ".json", ".yaml", ".yml", ".toml", ".html", ".css",
            ".txt", ".csv", ".tsv", ".log", ".rst", ".transcript"
        }
        allowed_exts = set(extensions) if extensions else default_exts
        # EXTENDS the automatic list, it does not replace it. This read
        # `set(ignore_dirs or [defaults])`, so naming ONE directory to skip
        # silently re-enabled every default -- `ignore_dirs=["src"]` indexed
        # `node_modules`. The docstring says build artifacts, git history and
        # caches are skipped "automatically", which is not something a caller
        # adding one more directory is asking to turn off.
        _AUTOMATIC_SKIPS = (
            ".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".idea",
            ".vscode", "dist", "build", "target", ".next", ".nuxt", "coverage", ".pytest_cache")
        ignored = set(_AUTOMATIC_SKIPS) | {str(x) for x in (ignore_dirs or [])}

        targets = []
        est_bytes = 0
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in ignored and not d.startswith(".")]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in allowed_exts and not f.startswith("."):
                    full_path = os.path.join(root, f)
                    targets.append(full_path)
                    try:
                        est_bytes += os.path.getsize(full_path)
                    except OSError:
                        pass

        # ARENA SIZING, FROM AN ESTIMATE. The real chunk count is only known
        # after every file has been read and split, which is the work itself, so
        # the hint is taken from the one number a directory walk already has:
        # total bytes, divided by BYTES_PER_CHUNK_ESTIMATE. MEASURED ratio of
        # bytes to chunks at the default 50-line/10-overlap split
        # (evidence/ingest_ram_results.json, ``directory_estimate``):
        # 1,640 B/chunk over 23 source files, 1,755 over the 50 files of the
        # nanomem package, 2,339 over 15,361 files of a whole repository and
        # 9,703 over a documentation tree of very long lines. 2,048 sits inside
        # that spread and leans low ON PURPOSE: under-estimating now costs
        # nothing (growth re-views a reservation instead of copying), while
        # over-estimating would inflate what ``stats()`` reports as the arena.
        # On those four trees the estimate lands at 0.80x, 0.86x, 1.14x and
        # 4.73x of the real chunk count -- the last being what a tree of
        # 9.7 kB lines does to any byte-based guess, and still only a hint.
        if est_bytes > 0:
            self.engine.reserve_additional_rows(
                max(len(targets), est_bytes // self.BYTES_PER_CHUNK_ESTIMATE))

        total_files = 0
        total_chunks = 0
        for full_path in targets:
            try:
                n = self.ingest_file(full_path)
                if n > 0:
                    total_files += 1
                    total_chunks += n
            except Exception:
                continue

        return {"files_indexed": total_files, "chunks_indexed": total_chunks}

    def add_batch(
        self,
        records: List[Dict[str, Any]],
        batch_size: int = 64,
        _replaying_stored_records: bool = False
    ) -> int:
        """
        High-throughput batch ingestion for books, datasets, and corpora.
        If embeddings are already present (e.g. during vault merge or split),
        they are transferred directly, without re-calling the embedding model.
        """
        if not records:
            return 0
        # THE SAME GUARD `add()` HAS. A reserved key reaching storage through
        # this door does exactly what it does through that one: three records
        # carrying {"id": "ticket-4711"} collapsed onto one handle. The guard
        # landed on `add()` alone in 0.7.19 -- which is how a fix ends up
        # covering one of four write surfaces.
        #
        # `_replaying_stored_records` is set ONLY by merge, split, export and the
        # chunk writers, where the metadata is nanomem's own and legitimately
        # carries these keys.
        if not _replaying_stored_records:
            for _r in records:
                _reject_reserved_metadata(_r.get("metadata"))

        total = len(records)
        # What was HANDED IN, versus what actually reached the vault. `total` is
        # the arena-sizing hint below; it was also the return value, so a batch
        # containing a record with empty text reported storing it. A caller
        # reconciling counts had no way to see the gap.
        stored = 0
        # ARENA SIZING. A bulk caller knows how many rows are coming; the engine
        # does not, and nothing in the library used to tell it. Measured over
        # a 71,433-document ingest of 768-d vectors, peak ru_maxrss of the whole
        # build-then-serve process, two runs each
        # (evidence/ingest_ram_results.json, summary + replicate),
        # median of four runs: 820.5 MB through the capacity doubling this
        # engine shipped with, 313.7 MB through this method today -- 15 MB of
        # which is the caller's own list of 71,433 record dicts, not the
        # vault's.
        # It is only a hint -- a wrong one costs nothing but pages nobody
        # touches.
        self.engine.reserve_additional_rows(total)
        dropped_empty = 0
        for i in range(0, total, batch_size):
            chunk = records[i : i + batch_size]
            # `ingest_file` promises "exact code formatting, indentation, and
            # line numbers". A blanket `.strip()` removed the leading indentation
            # of every chunk's FIRST line -- in Python the difference between a
            # method and a module-level function -- so a record may ask to keep
            # it. Trailing whitespace is still trimmed either way.
            texts = [(str(r.get("text", "")).rstrip()
                      if r.get("preserve_indent") else str(r.get("text", "")).strip())
                     for r in chunk]
            has_all_vecs = all("embedding" in r and r["embedding"] is not None for r in chunk)
            if has_all_vecs:
                vecs = [r["embedding"] for r in chunk]
            else:
                vecs = self.embedder.embed_batch(texts)

            for j, r in enumerate(chunk):
                if not texts[j]:
                    dropped_empty += 1
                    continue
                meta = dict(r.get("metadata", {}))
                import hashlib
                doc_id = str(r.get("id") or meta.get("id") or f"doc_{hashlib.md5(f'{texts[j]}_{time.time()}_{j}'.encode()).hexdigest()[:10]}")
                meta["id"] = doc_id
                # LONG TEXT IS SPLIT HERE TOO, exactly as `add()` splits it.
                # This method is documented for "books, datasets and corpora"
                # and stored whatever it was given verbatim, so one embedding
                # had to stand for a whole document. The vector then represents
                # the document's BULK and not its details, and a fact stated once
                # near the end becomes unreachable: measured on a 48,466-char
                # manual holding one shutdown code, against 40 competing
                # documents, `add()` returned it at rank 1 and `add_batch` could
                # not place it in the top 5 -- a safety bulletin that merely
                # shared vocabulary won instead. The record was in the vault the
                # whole time; search answered with something else.
                #
                # NOT when the caller supplied embeddings. That is a TRANSFER --
                # `merge`, `split`, a rebuild -- where the rows are already
                # whole and re-splitting them would invent records the source
                # never had.
                if not has_all_vecs:
                    pieces = self.split_large_text(
                        texts[j], max_words=self.MAX_FACT_WORDS,
                        overlap_words=self.OVERLAP_WORDS)
                else:
                    pieces = [texts[j]]
                if len(pieces) > 1:
                    piece_fp = _doc_fingerprint(texts[j])   # once per record, not per chunk
                    piece_vecs = self.embedder.embed_batch(pieces)
                    for k, piece in enumerate(pieces):
                        cmeta = dict(meta)
                        cid = f"{doc_id}_chunk_{k + 1}"
                        cmeta.update({"id": cid, "chunk_index": k + 1,
                                      "chunk_total": len(pieces),
                                      "parent_id": doc_id, "is_chunked": True,
                                      "parent_sha256": piece_fp})
                        stored += 1
                        self.engine.add_fact(
                            text=piece, embedding=piece_vecs[k],
                            source=r.get("source", "batch_ingestion"),
                            metadata=cmeta, timestamp=r.get("timestamp"),
                            revision=r.get("revision"), id=cid)
                    continue
                stored += 1
                self.engine.add_fact(
                    text=texts[j],
                    embedding=vecs[j],
                    source=r.get("source", "batch_ingestion"),
                    metadata=meta,
                    timestamp=r.get("timestamp"),
                    revision=r.get("revision"),
                    id=doc_id
                )
        self.flush()
        if dropped_empty:
            warnings.warn(
                "nanomem: add_batch was given %d record(s) with no text and "
                "stored %d of %d. The return value is what reached the vault."
                % (dropped_empty, stored, total), RuntimeWarning, stacklevel=2)
        return stored

    @staticmethod
    def decompose_query(query: str) -> List[str]:
        """
        Decomposes composite multi-part questions into focused sub-queries.
        Handles both clause-based conjunctions and comma-separated topic lists:
        1. 'What is BPE, and how does KV caching accelerate attention?'
           -> ['What is BPE', 'how does KV caching accelerate attention']
        2. 'What is BPE, KV cache, and LoRA?'
           -> ['What is BPE', 'What is KV cache', 'What is LoRA']
        """
        import re
        q = str(query).strip()
        raw_segments = re.split(r"[\?;]+", q)
        segments = []

        split_pat = re.compile(
            r"(?:,\s*|\s+)and\s+(?=(?:how|what|why|where|when|which|who|explain|describe|does|can|could|is|are)\b)|"
            r"(?:,\s*|\s+)also\s+(?=(?:how|what|why|where|when|which|who|explain|describe|does|can|could|is|are)\b)|"
            r"\b(?:also|plus|additionally)\b\s*[:,]?\s*",
            re.IGNORECASE
        )

        for seg in raw_segments:
            sub_segs = split_pat.split(seg)
            for s in sub_segs:
                s_clean = s.strip(" \t\n\r,.;:-?")
                if len(s_clean.split()) >= 2:
                    segments.append(s_clean)

        if len(segments) > 1:
            return segments

        # Check for list-type questions: e.g. "What is X, Y, and Z?" or "Explain A, B, and C"
        prefix_match = re.match(
            r"^(what\s+is|what\s+are|explain|describe|tell\s+me\s+about|how\s+does|how\s+do)\s+(.+)$",
            q,
            re.IGNORECASE
        )
        if prefix_match:
            prefix = prefix_match.group(1).strip()
            rest = prefix_match.group(2).strip(" ?.")
            items = re.split(r",\s*(?:and\s+)?|\s+and\s+", rest, flags=re.IGNORECASE)
            items = [it.strip() for it in items if it.strip()]
            if len(items) >= 2:
                return [f"{prefix} {it}" for it in items]

        return [query]

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
        temporal_direction: str = "current",
        decompose: bool = True,
        multihop: bool = False,
        alpha: float = 0.35,
        num_hops: int = 2,
        beam_width: int = 2,
        as_of: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant facts, documents and citations.

        ``as_of`` (a unix timestamp) answers the question against the vault AS
        IT STOOD at that moment: records written after it are not candidates, so
        a superseded value is returned wherever it was still the current one.
        It forces an exhaustive scan and turns the PCA screen off, because
        masking a routed or screened shortlist would drop exactly the older
        record the query is asking for. Proven against physically truncated
        vaults in ``evidence/temporal_as_of_results.json``.

        Search is an exact, LINEAR scan: latency grows with the corpus and the
        measured p50 is published per size in
        ``evidence/scale_results_v3r3.json`` -- there is no fixed
        millisecond guarantee here. Supports multi-query decomposition for
        composite multi-part questions, and `multihop=True` for a second,
        text-bridged pass (`search_multihop`); alpha-steering was measured worse
        than query-only and removed (DECISIONS 4.7).
        """
        if multihop:
            return self.search_multihop(
                query=query,
                top_k=top_k,
                filter=filter,
                min_score=min_score,
                temporal_direction=temporal_direction,
                alpha=alpha,
                num_hops=num_hops,
                beam_width=beam_width,
                as_of=as_of
            )


        # ASKING FOR NOTHING GETS NOTHING, AND ASKING FOR MORE THAN 50 GETS IT.
        # This was `max(1, min(50, int(top_k)))`, which did two silent things.
        # `top_k=0` returned ONE result, so a pagination loop whose remaining
        # count reached zero got a phantom row. And every request above 50 was
        # capped with nothing said, so a caller could not tell "only 50 matched"
        # from "we truncated you" -- on a 120-record corpus, top_k=120 returned
        # 50. The cap bought almost nothing: measured on 2,000 records, the
        # engine returns top_k=2000 in 17.0 ms against 7.7 ms for top_k=50.
        if top_k is None:
            safe_top_k = 3
        else:
            safe_top_k = int(top_k)
            if safe_top_k <= 0:
                return []
        # A QUESTION ASKED AFTER WORD 100 USED TO BE THROWN AWAY.
        # This was `" ".join(words[:100])`, which discarded everything past the
        # hundredth word of a query and said nothing. It is the shape people
        # actually send that this ruins: context pasted first and the real
        # question LAST. Measured against nomic-embed-text over five such
        # queries, rank 1 moved on 2 of 5 with a shared preamble and 1 of 5 with
        # distinct ones, and in every case the surviving text contained none of
        # the question -- so a correct answer there was the leftover filler
        # landing near the right record, not retrieval. The bare questions
        # scored 5/5. See benchmarks/query_truncation_results.json, which ships.
        #
        # There is no right number to put here. nanomem bundles no weights and
        # embeds against whatever endpoint answers, so the real limit belongs to
        # a model this library cannot interrogate -- nomic-embed-text carries
        # 8192 tokens, far more than 100 words, and enforces that itself. A
        # 50,000-word query returns in 0.18 s at the correct width. So the cap
        # goes, and the model's own limit is the only one left.
        clean_query = str(query).strip()
        if not clean_query:
            return []

        sub_queries = self.decompose_query(clean_query) if decompose else [clean_query]

        # AND A PASTED DOCUMENT IS NOT A COMPOSITE QUESTION.
        # Every sub-query costs its own full linear scan, so decomposition is a
        # multiplier on search cost. `decompose_query` splits on `?` and `;`,
        # which a pasted email thread or FAQ is full of: a 1,800-word FAQ came
        # back as 400 sub-queries, i.e. 400 scans. The 100-word cap was
        # incidentally holding this down and removing it would have turned a
        # correctness fix into a latency cliff -- but the cliff was already
        # reachable at 100 words, which yielded 22.
        #
        # 8 is measured, not chosen: over 145,051 distinct queries from every
        # non-quarantined corpus on the development tree, 99.95% decompose to 8
        # or fewer and the largest real one is 25. Past 8 this is not a question
        # with parts, it is a document, and the whole text is the better query.
        # It falls back to one query rather than keeping the first 8, because
        # dropping sub-queries silently is the same defect as the one above.
        if len(sub_queries) > MAX_SUB_QUERIES:
            sub_queries = [clean_query]

        if len(sub_queries) == 1:
            q_vec = self.embedder.embed(clean_query)
            return self.engine.search(
                query_text=clean_query,
                query_vec=q_vec,
                top_k=safe_top_k,
                metadata_filter=filter,
                min_score=min_score,
                temporal_direction=temporal_direction,
                as_of=as_of
            )

        # Multi-query beam execution with round-robin interleaving
        sub_vecs = self.embedder.embed_batch(sub_queries)
        sub_hits = []
        per_k = max(2, safe_top_k)
        for idx, sq in enumerate(sub_queries):
            hits = self.engine.search(
                query_text=sq,
                query_vec=sub_vecs[idx],
                top_k=per_k,
                metadata_filter=filter,
                min_score=min_score,
                temporal_direction=temporal_direction,
                as_of=as_of
            )
            sub_hits.append((sq, hits))

        merged_results: List[Dict[str, Any]] = []
        seen_texts = set()
        max_depth = max((len(h[1]) for h in sub_hits), default=0)

        # WHICH SUB-QUERY CAME FIRST MUST NOT DECIDE RANK 1.
        # The interleave exists so a genuine multi-part question gets hits for
        # every part instead of top_k hits for its strongest clause. But it
        # walked `sub_hits` in INPUT ORDER, so rank 1 was always the best hit of
        # whatever text appeared first -- and for a pasted email, a chat turn, or
        # any question asked after context, that is the preamble. Measured over
        # five unrelated questions behind one 46-word preamble: all five returned
        # the SAME record, and it was exactly the record the preamble alone
        # returns. The question contributed nothing.
        #
        # 0.7.10 removed the 100-word query cap for this same shape and missed
        # this, because its test and its release check both passed
        # `decompose=False` to isolate the truncation -- the one path on which
        # this cannot happen.
        #
        # The levels are kept, so each sub-query still contributes its best hit
        # before any contributes a second; only the order WITHIN a level changes,
        # from input position to score. A composite question still answers every
        # part, and rank 1 is now the best hit anywhere.
        def _rank(h):
            v = h.get("score")
            if v is None:
                v = h.get("cosine", 0.0)
            return float(v)

        for d in range(max_depth):
            level = [(sq, hits[d]) for sq, hits in sub_hits if d < len(hits)]
            level.sort(key=lambda pair: _rank(pair[1]), reverse=True)
            for sq, h in level:
                if h["text"] in seen_texts:
                    continue
                seen_texts.add(h["text"])
                h["sub_query"] = sq
                merged_results.append(h)

        return merged_results[:safe_top_k]

    def history(self, query: str, max_len: Optional[int] = None,
                min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Every value a fact has held, oldest first; the last one is current.

        ``search`` answers "what is it now"; this answers "what has it been".
        Each entry carries ``text, timestamp, revision, cosine, superseded``.
        A fact that never changed has a one-element history, which is an answer,
        not an empty result.
        """
        q = str(query).strip()
        if not q:
            return []
        return self.engine.history(q, self.embedder.embed(q),
                                   max_len=max_len, min_score=min_score)

    def changes(self, since: float, until: Optional[float] = None,
                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """What the vault learned in ``(since, until]``, oldest first.

        No query and no embedding call: this reads the resident timestamp
        column. ``revision > 1`` marks a record that superseded an earlier value
        of the same attribute when it was written.
        """
        return self.engine.changes(since, until=until, limit=limit)

    def volatility(self, min_revisions: int = 2,
                   now: Optional[float] = None) -> List[Dict[str, Any]]:
        """How often each remembered fact actually changes.

        Read straight off the revision log: per fact, how many times it has been
        restated, the intervals between those restatements, the median interval,
        and how long the current value has stood unconfirmed (``age``). No model,
        no query, no embedding call.

        Use it to decide which facts are probably stale and worth re-confirming
        with the user. Facts the tagger never grouped are excluded, because they
        are not restatements of one thing.
        """
        return self.engine.volatility(min_revisions=min_revisions, now=now)

    def staleness(self, now: Optional[float] = None, shrink: float = 1.0,
                  min_revisions: int = 2,
                  assume_memoryless: bool = False) -> List[Dict[str, Any]]:
        """:meth:`volatility` plus a modelled probability the value is out of date.

        ``p_superseded`` is ``None`` unless ``assume_memoryless=True``. That is
        not caution: the per-fact rate only beat a single corpus-wide rate on
        data generated to match its own memoryless assumption, and tied
        elsewhere, so it does not ship on by default. See
        :meth:`VaultEngine.staleness` for the measured table.
        """
        return self.engine.staleness(now=now, shrink=shrink,
                                     min_revisions=min_revisions,
                                     assume_memoryless=assume_memoryless)

    def search_multihop(
        self,
        query: str,
        top_k: int = 3,
        filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
        temporal_direction: str = "current",
        alpha: float = 0.35,
        num_hops: int = 2,
        beam_width: int = 2,
        bridge: Optional[Any] = None,
        as_of: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Two-pass retrieval for bridge ("A -> B -> answer") questions. Opt-in.

        Pass 1 is an ordinary search. Pass 2 re-embeds the question together with
        the text of the hop-1 winner -- `f"{question}\n{hop1_text}"` -- and
        searches with that vector, then merges: the top ceil(k/2) single-pass
        hits plus the top floor(k/2) bridge hits, ranked by cosine and
        de-duplicated by id, so the budget is the same `k` documents.

        MEASURED, equal budget, evidence recall@4, on a re-opened vault
        (evidence/multihop_texthop_v3r2_1190.json, 1,190 paragraphs, 120
        questions): single pass 68.3 -> text-hop 79.2. The orchestrator's prior
        run of the same mechanism: 1,190 docs 68.3 -> 73.3 (+5.0), hop-2 recall@4
        93.8 vs 87.9; 71,433 docs 60.4 -> 63.5 (+3.1), McNemar discordant 167 vs
        117.

        It HURTS when the hop-1 anchor is wrong (questions with a wrong anchor:
        25% -> 14%), which is why it stays opt-in. Simply raising `top_k` is the
        larger lever, and it is measured here too: single pass at top-8 scores
        90.0 on the same 1,190 documents, against 79.2 for the bridge at top-4.

        The previous alpha-steering bridge (`q + alpha * v_anchor`) is gone: it
        measured significantly WORSE than the query alone, CI [-0.050, -0.033].
        `alpha`, `num_hops` and `beam_width` are accepted for compatibility and
        no longer change the result.

        `bridge` may be any object with
        `bridge(q_vec, d1_vec, question, d1_text) -> unit vector`, so a learned
        latent field can be attached without editing this file. The default is
        `TextHopBridge(self.embedder)`.
        """
        # Same rule as `search`: nothing asked for, nothing returned, and no
        # silent cap at 50.
        if top_k is None:
            safe_top_k = 3
        else:
            safe_top_k = int(top_k)
            if safe_top_k <= 0:
                return []
        # Same rule as `search`: the query is not truncated here either. The
        # reasoning and the measurement are in `search`; this path never
        # decomposed, so it needs no fan-out bound.
        clean_query = str(query).strip()
        if not clean_query:
            return []

        q_vec = self.embedder.embed(clean_query)
        k_single = -(-safe_top_k // 2)              # ceil(k/2)
        k_bridge = safe_top_k - k_single            # floor(k/2)

        single = self.engine.search(
            query_text=clean_query,
            query_vec=q_vec,
            top_k=safe_top_k,
            metadata_filter=filter,
            min_score=min_score,
            temporal_direction=temporal_direction,
            # BOTH HOPS take the cutoff. `search(as_of=..., multihop=True)`
            # delegated here and simply did not pass it -- this method did not
            # even accept it -- so a question asked "as it stood then" was
            # answered with a record written after the cutoff. Measured: a fact
            # timestamped 1.8e9 came back for `as_of=1.7e9`. The bridge hop
            # needs it as much as the first: bridging THROUGH a future record
            # reaches conclusions the vault could not have supported then.
            as_of=as_of
        )
        if not single or k_bridge <= 0:
            return single[:safe_top_k]

        anchor = single[0]
        bridger = bridge if bridge is not None else TextHopBridge(self.embedder)
        anchor_rec = self.engine.get(anchor.get("id")) or {}
        try:
            b_vec = bridger.bridge(q_vec, anchor_rec.get("embedding"),
                                   clean_query, anchor.get("text", ""))
        except Exception:
            return single[:safe_top_k]
        if b_vec is None:
            return single[:safe_top_k]

        hop2 = self.engine.search(
            query_text=clean_query,
            query_vec=b_vec,
            top_k=safe_top_k + k_bridge + 2,
            metadata_filter=filter,
            min_score=min_score,
            temporal_direction=temporal_direction,
            as_of=as_of
        )

        chosen: List[Dict[str, Any]] = []
        seen = set()
        for h in single[:k_single]:
            hid = h.get("id") or h.get("text", "")[:80]
            if hid in seen:
                continue
            seen.add(hid)
            h = dict(h)
            h["hop"] = 1
            chosen.append(h)
        for h in hop2:
            if len(chosen) >= safe_top_k:
                break
            hid = h.get("id") or h.get("text", "")[:80]
            if hid in seen:
                continue
            seen.add(hid)
            h = dict(h)
            h["hop"] = 2
            h["anchor_id"] = anchor.get("id")
            chosen.append(h)
        for h in single:
            if len(chosen) >= safe_top_k:
                break
            hid = h.get("id") or h.get("text", "")[:80]
            if hid in seen:
                continue
            seen.add(hid)
            h = dict(h)
            h["hop"] = 1
            chosen.append(h)

        chosen.sort(key=lambda x: x.get("cosine", x.get("score", 0.0)), reverse=True)
        return chosen[:safe_top_k]

    @staticmethod
    def should_store(text: str) -> bool:
        """
        Cognitive Latent Manifold Memory Filter.
        Intelligently classifies utterances into persistent memories vs conversational flux
        across diverse occupations, mental states, and personalities using 768D manifold projection.
        Zero brittle keywords, zero system prompts.
        """
        try:
            from .classifier import get_classifier
            return get_classifier().should_store(text)
        except Exception:
            return False

    @staticmethod
    def inspect_memory(text: str) -> Dict[str, Any]:
        """
        Inspect the latent manifold energy, margin, and classification decision for a text.
        """
        try:
            from .classifier import get_classifier
            return get_classifier().inspect(text)
        except Exception as e:
            return {"should_store": False, "error": str(e)}

    def chat(
        self,
        user_message: str,
        user_id: Optional[str] = None,
        role: str = "user",
        llm: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stream chat messages with automated coreference resolution,
        temporal MVCC tracking, intelligent noise filtering, and optional LLM synthesis.
        """
        meta = {"role": role}
        if user_id:
            meta["user_id"] = user_id

        # 1. Retrieve relevant past context
        context_candidates = self.search(
            query=user_message,
            top_k=3,
            filter={"user_id": user_id} if user_id else None
        )

        # 2. Filter & Store only if message contains valuable semantic information
        stored = False
        if self.should_store(user_message):
            self.add(text=user_message, metadata=meta, source="chat_session")
            self.flush()
            stored = True

        # 3. If an LLM model name is provided, synthesize the grounded response
        response_text = None
        if llm:
            response_text = self._call_llm(user_message, context_candidates, model=llm)

        return {
            "message": user_message,
            "stored": stored,
            "context": context_candidates,
            "reply": response_text
        }

    def ask(
        self,
        question: str,
        llm: str = "llama3.2",
        top_k: int = 3,
        filter: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cite: bool = True,
        multihop: bool = False
    ) -> Dict[str, Any]:
        """
        1-line Grounded RAG synthesis: retrieves pin-point context and generates cited answer.
        Supports any LLM model and port/API (Ollama, LM Studio, vLLM, OpenAI).
        Set `cite=False` to generate clean conversational responses without citation tags or sources.
        Set `multihop=True` to enable latent tangent steering for nested relational questions.
        """
        candidates = self.search(query=question, top_k=top_k, filter=filter, multihop=multihop)
        answer = self._call_llm(question, candidates, model=llm, base_url=base_url, api_key=api_key, cite=cite)
        return {
            "question": question,
            "answer": answer,
            "citations": candidates if cite else []
        }


    def get_all_records(
        self,
        include_embeddings: bool = True,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract every stored record (text, metadata, source and the pre-computed
        vector) in stored order. Flushes pending facts first.
        Optional `where` filters by metadata using engine filter semantics.
        """
        records = []
        for r in self.engine.iter_records(include_embeddings=include_embeddings):
            meta = r.get("metadata") or {}
            if where is not None and not self.engine._matches_filter(meta, where):
                continue
            meta["id"] = r["id"]
            rec = {
                "id": r["id"],
                "text": r["text"],
                "metadata": meta,
                "source": r["source"],
                "timestamp": float(r["timestamp"]),
                "revision": int(r["revision"]),
            }
            if include_embeddings and "embedding" in r:
                rec["embedding"] = r["embedding"]
            records.append(rec)
        return records

    def _chunks_of(self, doc_id: str) -> List[Dict[str, Any]]:
        """The chunks of a chunked document, in order, or [] if ``doc_id`` is not
        a parent id.

        `add()` returns the PARENT id for text it split, and that id is not a
        stored row -- the rows are `{parent}_chunk_N`. So every by-id call
        silently missed: `get` returned None, `exists` False, `update` False,
        `delete` 0, and the document stayed on disk. Nothing raised, which is
        why a caller deleting a document could believe it had.
        """
        pid = str(doc_id).strip()
        if not pid:
            return []
        out = []
        for r in self.get_all_records():
            if (r.get("metadata") or {}).get("parent_id") == pid:
                out.append(r)
        out.sort(key=lambda r: _chunk_index(r.get("id", "")))
        return out

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory record, fact, or chunk by its unique ID.
        Returns the record dictionary if found, or None if not present.

        For a stored row the text is exactly what was stored. For the PARENT id of
        a document `add()` split, there is no such row, and the text returned is a
        RECONSTRUCTION rebuilt from the chunks -- not the document that was added.
        It differs in three measured ways: every newline becomes a space, a passage
        that repeats can be dropped by the overlap detector (one sentence repeated
        400 times lost 29% of the document), and leading indentation is gone. This
        docstring said "exact" through 0.7.16 and that was wrong.

        ``metadata["reconstructed"]`` is True on such a result, and
        ``metadata["reconstruction_exact"]`` says whether it round-tripped, checked
        against a fingerprint of the original. ``metadata["chunk_ids"]`` names the
        rows that hold the exact stored content. If you need the file's formatting
        preserved, ingest it with :meth:`ingest_file`, which chunks on line
        boundaries and does not rebuild text.
        """
        clean_id = str(id).strip()
        if not clean_id:
            return None
        self.flush()
        rec = self.engine.get(clean_id)
        if rec is None:
            chunks = self._chunks_of(clean_id)
            if not chunks:
                return None
            # The whole document, not one chunk. Chunks carry a 40-word overlap
            # bridge, so they are joined by removing the real overlap between
            # each neighbouring pair rather than by assuming a fixed width --
            # `split_large_text` breaks on paragraph and sentence boundaries, so
            # the bridge is rarely exactly 40 words.
            first = chunks[0]
            meta = dict(first.get("metadata") or {})
            meta["id"] = clean_id
            meta["chunk_ids"] = [c.get("id") for c in chunks]
            meta.pop("parent_id", None)
            # SAY THAT THIS IS A RECONSTRUCTION, AND WHETHER IT SURVIVED.
            # A caller cannot otherwise tell this from a stored row, and for a
            # coding agent the difference is source code with its indentation
            # silently removed. The fingerprint is already stored for `delete`,
            # so the honest answer costs a comparison.
            rebuilt = _join_chunks([c.get("text", "") for c in chunks])
            stamp = (chunks[0].get("metadata") or {}).get("parent_sha256")
            meta["reconstructed"] = True
            if stamp:
                meta["reconstruction_exact"] = (_doc_fingerprint(rebuilt) == stamp)
            return {
                "id": clean_id,
                "text": rebuilt,
                "metadata": meta,
                "source": first.get("source"),
                "timestamp": float(first.get("timestamp", 0.0)),
                "revision": int(first.get("revision", 1)),
            }
        meta = dict(rec.get("metadata") or {})
        meta["id"] = clean_id
        return {
            "id": clean_id,
            "text": rec["text"],
            "metadata": meta,
            "source": rec["source"],
            "timestamp": float(rec["timestamp"]),
            "revision": int(rec["revision"]),
        }

    def exists(self, id: str) -> bool:
        """Check whether a fact or document chunk exists in the vault by its unique ID."""
        return self.get(id) is not None

    def __contains__(self, id: str) -> bool:
        """Enable Pythonic 'if doc_id in vault:' existence checks."""
        return self.exists(id)

    def __len__(self) -> int:
        """Enable Pythonic 'len(vault)' to get total number of stored records."""
        return int(self.stats().get("total_documents", 0))

    def update(
        self,
        id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> bool:
        """
        In-place atomic update of a fact or document chunk by its unique ID.
        If `text` is changed, recomputes the vector embedding cleanly.
        Updates metadata and increments the record revision.
        Returns True if the record was found and updated, False otherwise.
        """
        _reject_reserved_metadata(metadata)
        clean_id = str(id).strip()
        if not clean_id:
            return False

        self.flush()
        if not os.path.exists(self.path):
            return False

        all_recs = self.get_all_records(include_embeddings=True)
        if not all_recs:
            return False

        target_idx = None
        for idx, r in enumerate(all_recs):
            if r.get("id") == clean_id or (r.get("metadata") or {}).get("id") == clean_id:
                target_idx = idx
                break

        if target_idx is None:
            # A parent id names a document, not a row. Replacing it means
            # dropping every chunk and re-adding, because new text splits into a
            # different number of chunks than the old text did.
            chunks = self._chunks_of(clean_id)
            if not chunks:
                return False
            if text is None:
                for c in chunks:
                    self.update(c.get("id"), metadata=metadata, source=source)
                return True
            first = chunks[0]
            new_meta = dict(first.get("metadata") or {})
            # Drop EVERY key the library writes itself, not just three of them.
            # This metadata came off a stored chunk, so it carries the structural
            # links; re-adding with them would both re-assert a stale document
            # shape and now trip `_reject_reserved_metadata`.
            for k in set(RESERVED_METADATA_KEYS) | {"id"}:
                new_meta.pop(k, None)
            if metadata is not None:
                new_meta.update(metadata)
            self.delete(id=clean_id)
            self.add(str(text).strip(),
                     metadata=new_meta or None,
                     source=source if source is not None else first.get("source"),
                     timestamp=first.get("timestamp"),
                     id=clean_id)
            self.flush()
            return True

        target_rec = all_recs[target_idx]
        cur_meta = dict(target_rec.get("metadata") or {})
        if metadata is not None:
            cur_meta.update(metadata)
        cur_meta["id"] = clean_id

        new_text = str(text).strip() if text is not None else target_rec.get("text", "")
        new_source = source if source is not None else target_rec.get("source", "user_input")

        # If text changed, recompute embedding; otherwise keep precomputed vector
        if text is not None and text.strip() != target_rec.get("text", ""):
            new_emb = self.embedder.embed(new_text)
        else:
            new_emb = target_rec.get("embedding")
            if new_emb is None:
                new_emb = self.embedder.embed(new_text)

        updated_rec = {
            "id": clean_id,
            "text": new_text,
            "source": new_source,
            "metadata": cur_meta,
            "embedding": new_emb,
            # KEEP WHEN IT WAS WRITTEN. This was `time.time()`, so ANY update --
            # including relabelling a `source` and touching nothing else --
            # re-dated the record to now. Measured: relabelling a 300-day-old
            # row moved it 300 days forward, which made it the answer to "where
            # do I work" ahead of two newer values, after which
            # `forget_superseded(keep=1)` permanently deleted both of them and
            # reported success. That is the README's opening scenario -- an
            # assistant confidently repeating an address you left two years ago --
            # produced by the library itself.
            #
            # `update` is documented as an IN-PLACE update by id. A record's
            # timestamp is when the fact was written, not when its row was last
            # touched; `add()` is how you record a new value at a new time.
            "timestamp": float(target_rec.get("timestamp") or time.time()),
            "revision": int(target_rec.get("revision", 1)) + 1
        }

        all_recs[target_idx] = updated_rec

        self._rebuild(all_recs)
        return True

    @staticmethod
    def _detect_entity(text: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Entity for a record: an explicit metadata tag, else the generic tagger."""
        meta = metadata or {}
        if meta.get("entity"):
            return str(meta["entity"])
        return detect_entity(text)

    def merge(
        self,
        other_vault_path: str,
        incoming_user_id: Optional[str] = None,
        incoming_project: Optional[str] = None,
        deduplicate: bool = False,
        reconcile_revisions: bool = True
    ) -> Dict[str, int]:
        """
        Combines another .dat vault file into this vault.
        - ZERO DATA LOSS BY DEFAULT: `deduplicate` is False by default.
          Every chunk, file, and fact from both vaults is preserved with 100% fidelity.
          If two different projects both contain `main.py` (or shared boilerplate code),
          neither is dropped. Both files and chunks remain intact.
        - incoming_project: Optionally tags all incoming records with a project namespace.
        - incoming_user_id: Optionally tags all incoming records with a user namespace.
        - Revision Reconciliation: Dynamic facts (e.g. phone numbers, addresses) are preserved
          in full and ordered chronologically with monotonic revisions (Rev 1, Rev 2), ensuring
          current queries retrieve the latest state without deleting history.
        - Opt-In Deduplication: If explicitly set to `deduplicate=True`, exact matches within
          the SAME user and project scope will be deduplicated.
        Returns a summary dictionary: {"incoming": int, "added": int, "duplicates_skipped": int}.
        """
        if not os.path.exists(other_vault_path):
            raise FileNotFoundError(f"Vault to merge not found: {other_vault_path}")

        existing = self.get_all_records()
        with self._open_source(other_vault_path) as other:
            incoming = other.get_all_records()

        if not incoming:
            return {"incoming": 0, "added": 0, "duplicates_skipped": 0}

        # Build existing signature set and entity maps scoped by (user_id, project, entity)
        existing_signatures = set()
        existing_entity_map = {}
        for r in existing:
            meta = r.get("metadata") or {}
            uid = meta.get("user_id")
            proj = meta.get("project")
            txt_norm = r["text"].strip().lower()
            ent = meta.get("entity") or self._detect_entity(r["text"], meta)
            existing_signatures.add((uid, proj, ent, txt_norm))
            if ent:
                existing_entity_map.setdefault((uid, proj, ent), []).append(r)

        to_insert = []
        dups_skipped = 0

        for inc in incoming:
            if not inc.get("metadata"):
                inc["metadata"] = {}

            # Tag incoming provenance
            inc["metadata"]["source_vault"] = os.path.basename(other_vault_path)
            if incoming_user_id:
                inc["metadata"]["user_id"] = incoming_user_id
            if incoming_project:
                inc["metadata"]["project"] = incoming_project

            uid = inc["metadata"].get("user_id")
            proj = inc["metadata"].get("project")
            txt_norm = inc["text"].strip().lower()
            ent = inc["metadata"].get("entity") or self._detect_entity(inc["text"], inc["metadata"])

            if ent:
                inc["metadata"]["entity"] = ent

            # Deduplication is opt-in and strictly scoped
            sig = (uid, proj, ent, txt_norm)
            if deduplicate and sig in existing_signatures:
                dups_skipped += 1
                continue

            # Multi-user & project-safe revision reconciliation
            entity_key = (uid, proj, ent)
            if reconcile_revisions and ent and entity_key in existing_entity_map:
                existing_for_ent = existing_entity_map[entity_key]
                max_existing_ts = max(r.get("timestamp", 0.0) for r in existing_for_ent)
                max_existing_rev = max(r.get("revision", 1) for r in existing_for_ent)
                inc_ts = inc.get("timestamp", 0.0)

                if inc_ts >= max_existing_ts:
                    inc["revision"] = max_existing_rev + 1
                else:
                    min_existing_rev = min(r.get("revision", 1) for r in existing_for_ent)
                    inc["revision"] = max(1, min_existing_rev - 1)

                existing_for_ent.append(inc)

            to_insert.append(inc)
            existing_signatures.add(sig)

        if to_insert:
            self.add_batch(to_insert, batch_size=64, _replaying_stored_records=True)

        return {
            "incoming": len(incoming),
            "added": len(to_insert),
            "duplicates_skipped": dups_skipped
        }

    def delete(
        self,
        id: Optional[Any] = None,
        ids: Optional[List[str]] = None,
        text_exact: Optional[str] = None,
        text_contains: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> int:
        """
        Permanently remove specific memory records from the vault.
        Supports targeting by unique ID or list of IDs, exact text, substring, metadata filter ('where'), or source document.

        `text_exact` on a document that `add()` SPLIT matches two things: the text
        exactly as it was passed to `add()` (compared by fingerprint, so newlines,
        indentation and repeated passages all count), and the whitespace-normalised
        reconstruction that `get(parent_id)["text"]` returns. It is not a substring
        or fuzzy match: a document is removed only when one of those two matches in
        full, and then every one of its chunks goes.
        Returns the number of deleted records.

        COST. There are no tombstones in 3.0 (DECISIONS #11), so a delete is a
        full atomic rewrite of the container with pre-computed vectors: measured
        6.6 ms at 1,000 records and 65.2 ms at 10,000 (768-d), and about 1 s at
        71k -- linear in vault size, not "sub-millisecond execution" as this
        docstring claimed through 3.0.2. The rewrite holds the container's
        exclusive lock for its whole duration, so concurrent appenders block.
        Batch your deletions: one call with a list of ids costs one rewrite.
        See evidence/rewrite_cost_v3r4.json.
        """
        self.flush()
        if not os.path.exists(self.path):
            return 0
        all_recs = self.get_all_records(include_embeddings=True)
        if not all_recs:
            return 0

        target_ids = set()
        if id is not None:
            if isinstance(id, (list, tuple, set)):
                target_ids.update(str(x).strip() for x in id if str(x).strip())
            else:
                target_ids.add(str(id).strip())
        if ids is not None:
            target_ids.update(str(x).strip() for x in ids if str(x).strip())

        # A document `add()` split is stored as `{parent}_chunk_N` rows, and no
        # single row holds the text the caller handed us -- so `text_exact` with
        # that text matched nothing and returned 0, on a mode the docstring
        # advertises. Found by the third black-box review (M2). The by-id forms
        # were repaired for chunked documents in 0.7.11; this is the same defect
        # on the by-text form. Resolve to parent ids with the SAME de-overlap
        # join `get()` uses, so what reconstructs is what deletes.
        text_ids = set()
        if text_exact is not None:
            want = text_exact.strip()
            if not any(r.get("text", "").strip() == want for r in all_recs):
                # MATCH WHAT WAS WRITTEN, NOT WHAT CAN BE REBUILT. `split_large_text`
                # rebuilds text rather than slicing it, so `_join_chunks` is a
                # reconstruction and comparing against it matched only documents of
                # unique space-joined words -- see `_doc_fingerprint`.
                fp = _doc_fingerprint(want)
                groups: Dict[str, List[Dict[str, Any]]] = {}
                for r in all_recs:
                    m = r.get("metadata") or {}
                    if m.get("parent_sha256") == fp:
                        # A split document is named by its parent; an ingested file
                        # has no parent row, so the chunk is named directly.
                        text_ids.add(str(m.get("parent_id") or r.get("id") or ""))
                    pid = m.get("parent_id")
                    if pid:
                        groups.setdefault(str(pid), []).append(r)
                text_ids.discard("")
                for pid, rows in groups.items():
                    if pid in text_ids:
                        continue
                    rows.sort(key=lambda r: _chunk_index(r.get("id", "")))
                    rebuilt = _join_chunks([r.get("text", "") for r in rows]).strip()
                    if rebuilt == want:
                        text_ids.add(pid)
                        continue
                    # LEGACY ROWS ONLY. A vault written before 0.7.18 has no
                    # fingerprint, and its reconstruction differs from the document
                    # by whitespace alone for every shape except a repeated passage
                    # (measured: newline-joined, CSV and indented code all match once
                    # whitespace is collapsed; a repeated passage does not, because
                    # that content is genuinely not in the vault). Without this the
                    # 0.7.17 fix reaches only documents written after upgrading --
                    # which is nobody's existing data.
                    if any((r.get("metadata") or {}).get("parent_sha256") for r in rows):
                        continue
                    if " ".join(rebuilt.split()) == " ".join(want.split()):
                        text_ids.add(pid)

        # AN EMPTY FILTER IS A PROGRAMMING ERROR, NOT "EVERYTHING".
        # `_matches_filter(meta, {})` is vacuously true, so `delete(where={})`
        # erased the whole vault -- and an empty dict is what you get when the
        # filter was BUILT and every condition dropped out. `delete()` with no
        # arguments already deletes nothing; this is the same intent expressed
        # through a variable, and it must not be the one spelling that wipes the
        # file. Refused loudly rather than ignored, because a caller who really
        # means "remove everything" should say so in a way that reads like it.
        if where is not None and not where:
            raise ValueError(
                "delete(where={}) would match every record and erase the whole "
                "vault. An empty filter is almost always a filter that was built "
                "and came out empty. Pass a filter with at least one condition, "
                "or if you really mean to remove everything, delete the vault "
                "file itself.")

        kept = []
        deleted_count = 0
        for r in all_recs:
            drop = False
            rec_id = str(r.get("id") or (r.get("metadata") or {}).get("id") or "")
            meta = r.get("metadata") or {}

            # EVERY CRITERION THE CALLER GAVE MUST MATCH. This was an `elif`
            # chain, so the first one that applied decided and the rest were
            # ignored: `delete(source="report.txt", where={"team": "b"})`
            # deleted BOTH rows of that source, including the one the filter
            # excluded. Narrowing is the safe direction for a delete -- a caller
            # who names two things means the intersection, and the failure mode
            # of the old reading was deleting data they asked to keep.
            checks = []
            if target_ids:
                checks.append(rec_id in target_ids
                              or meta.get("parent_id") in target_ids)
            if text_exact is not None:
                # One criterion, two ways to satisfy it: the row IS the text, or
                # the row belongs to a document whose text this is (resolved by
                # fingerprint above). `text_ids` is deliberately separate from
                # `target_ids` -- folding it in made "delete by text" and
                # "delete by id" two criteria that a chunk could never satisfy
                # at once, and `delete(text_exact=<a split document>)` stopped
                # deleting anything.
                checks.append(r.get("text", "").strip() == text_exact.strip()
                              or rec_id in text_ids
                              or meta.get("parent_id") in text_ids)
            if text_contains is not None:
                checks.append(text_contains.lower() in r.get("text", "").lower())
            if source is not None:
                src_str = str(r.get("source", ""))
                fn_str = str(meta.get("filename", ""))
                fp_str = str(meta.get("file_path", ""))
                clean_src = str(source).strip()
                # NAMES, NOT SUBSTRINGS. `clean_src in src_str` and
                # `fp_str.endswith(clean_src)` meant `delete(source="notes.txt")`
                # also deleted everything from `meeting_notes.txt` -- measured,
                # two rows deleted where one was named. For a delete, a loose
                # match is data loss, and the caller has no way to see what else
                # it caught. `ingest_file` writes `"{basename}:{start}-{end}"`,
                # so that one prefix form stays; matching a path compares the
                # basename rather than the tail of the string.
                checks.append(clean_src == src_str or
                              clean_src == fn_str or
                              clean_src == fp_str or
                              src_str.startswith(f"{clean_src}:") or
                              (fp_str and os.path.basename(fp_str) == clean_src))
            if where is not None:
                checks.append(bool(self.engine._matches_filter(meta, where)))

            drop = bool(checks) and all(checks)

            if drop:
                deleted_count += 1
            else:
                kept.append(r)

        if deleted_count == 0:
            return 0

        self._rebuild(kept)
        return deleted_count

    def forget(self, query: str, top_k: int = 1, min_score: float = 0.42) -> List[Dict[str, Any]]:
        """
        Semantic forget: searches for the closest matching memory to query and erases it from disk.
        Returns the deleted memory records.
        """
        candidates = self.search(query, top_k=top_k, min_score=min_score)
        if not candidates:
            return []
        deleted = []
        for c in candidates:
            n = self.delete(text_exact=c["text"])
            if n > 0:
                deleted.append(c)
        return deleted

    def _open_source(self, path: str) -> "Vault":
        """Open an EXISTING vault to read from (``merge``).

        Tries this vault's password first, because the common case is merging two
        vaults of the same owner, and falls back to opening it in the clear when
        the file turns out to be plaintext. A file that needs a different
        passphrase raises, it is never guessed at.
        """
        try:
            return self._sibling(path)
        except NotEncryptedError:
            return self._sibling(path, None)

    def _sibling(self, path: str, password=_KEEP_PW) -> "Vault":
        """A second vault that inherits THIS vault's protection.

        ``export`` / ``split*`` / ``merge`` all write a second file, and 3.0.1
        opened it as a bare ``Vault(path)``: splitting a password-protected vault
        wrote every record into a NEW PLAINTEXT vault, in the clear, with no
        warning -- a silent confidentiality failure that contradicts
        ``crypto.THREAT_MODEL``. Pass ``password=None`` explicitly to opt out
        (``export(..., target_password=None)``).
        """
        pw = self._password if password is _KEEP_PW else password
        return Vault(path, embed_model=self.embedder.model,
                     base_url=self.embedder.base_url, password=pw)

    def export(
        self,
        target_vault_path: str,
        where: Optional[Dict[str, Any]] = None,
        source_doc: Optional[str] = None,
        purge: bool = False,
        target_password=_KEEP_PW
    ) -> int:
        """
        Exports matching records to target_vault_path.
        If `purge=True`, performs a real split: removes exported records from this source vault.

        The target INHERITS this vault's password by default, so exporting out of
        an encrypted vault cannot silently produce a plaintext one. Pass
        ``target_password=None`` to write a plaintext target on purpose, or a
        string to use a different passphrase.
        """
        all_recs = self.get_all_records(include_embeddings=True)
        if not all_recs:
            return 0

        matching = []
        remaining = []

        for r in all_recs:
            is_match = True
            meta = r.get("metadata") or {}
            if where is not None:
                if not all(meta.get(k) == v for k, v in where.items()):
                    is_match = False
            if source_doc is not None:
                src = str(r.get("source", ""))
                fn = str(meta.get("filename", ""))
                fp = str(meta.get("file_path", ""))
                if source_doc not in src and source_doc not in fn and not fp.endswith(source_doc):
                    is_match = False

            if is_match:
                matching.append(r)
            else:
                remaining.append(r)

        if not matching:
            return 0

        with self._sibling(target_vault_path, target_password) as target:
            target.add_batch(matching, batch_size=64, _replaying_stored_records=True)
            target.flush()

        if purge:
            self._rebuild(remaining)

        return len(matching)

    def split(self, filter_dict: Dict[str, Any], target_vault_path: str, purge: bool = False) -> int:
        """
        Divides this vault by exporting records matching filter_dict into a new .dat vault file.
        If `purge=True`, performs a real split by deleting matching records from this vault.
        """
        return self.export(target_vault_path=target_vault_path, where=filter_dict, purge=purge)

    def split_by_doc(self, filename: str, target_vault_path: str, purge: bool = False) -> int:
        """
        Extracts all chunks belonging to a specific document or file into a standalone vault.
        If `purge=True`, removes the extracted document chunks from this vault (Option 2).
        """
        return self.export(target_vault_path=target_vault_path, source_doc=filename, purge=purge)

    def unmerge(self, vault_name_or_project: str, target_vault_path: Optional[str] = None) -> int:
        """
        Cleanly detaches a previously merged vault or project with zero data misplacement.
        Extracts all matching records to target_vault_path (if provided) and purges them from this vault.
        """
        all_recs = self.get_all_records(include_embeddings=True)
        if not all_recs:
            return 0
        clean_key = vault_name_or_project.strip()
        matching = []
        remaining = []
        for r in all_recs:
            meta = r.get("metadata") or {}
            sv = str(meta.get("source_vault", ""))
            proj = str(meta.get("project", ""))
            if clean_key in [sv, proj, os.path.basename(sv)]:
                matching.append(r)
            else:
                remaining.append(r)

        if not matching:
            return 0

        if target_vault_path:
            with self._sibling(target_vault_path) as target:
                target.add_batch(matching, batch_size=64, _replaying_stored_records=True)
                target.flush()

        self._rebuild(remaining)
        return len(matching)

    def split_by_source(self, source_name: str, target_vault_path: str) -> int:
        """
        Divides this vault by extracting all chunks belonging to a specific source document or code file.
        """
        all_recs = self.get_all_records()
        matching = [
            r for r in all_recs
            if (r.get("metadata") or {}).get("filename") == source_name
            or (r.get("metadata") or {}).get("file") == source_name
            or str((r.get("metadata") or {}).get("file", "")).endswith(source_name)
            or source_name in str(r.get("source", ""))
            or (r.get("metadata") or {}).get("file_path", "").endswith(source_name)
        ]
        if not matching:
            return 0

        with self._sibling(target_vault_path) as target:
            return target.add_batch(matching, batch_size=64, _replaying_stored_records=True)

    def split_by_date(
        self,
        target_vault_path: str,
        before_timestamp: Optional[float] = None,
        after_timestamp: Optional[float] = None
    ) -> int:
        """
        Divides this vault chronologically (e.g. for archival of old memories or seasonal projects).
        """
        all_recs = self.get_all_records()
        matching = []
        for r in all_recs:
            ts = float(r.get("timestamp", 0.0))
            if before_timestamp is not None and ts > before_timestamp:
                continue
            if after_timestamp is not None and ts < after_timestamp:
                continue
            matching.append(r)

        if not matching:
            return 0

        with self._sibling(target_vault_path) as target:
            return target.add_batch(matching, batch_size=64, _replaying_stored_records=True)

    def split_by_key(self, metadata_key: str, output_dir: str) -> Dict[str, int]:
        """
        Automatically partitions this vault into multiple separate .dat files based on the values of a metadata key.
        E.g. split_by_key("category", "./categories") creates "./categories/<val>.dat".
        """
        import re
        os.makedirs(output_dir, exist_ok=True)
        all_recs = self.get_all_records()
        groups: Dict[str, List[Dict[str, Any]]] = {}

        # GROUP BY THE VALUE, NAME THE FILE SEPARATELY. Grouping by the
        # SANITISED name merged distinct values that happen to sanitise alike --
        # `"a/b"` and `"a_b"` both became `a_b.dat`, so two teams' records were
        # written into one vault with nothing to say so. Grouping on the real
        # value keeps them apart; only the FILENAME is sanitised, and a
        # collision there gets a short digest suffix so the partition stays
        # one-file-per-value.
        import hashlib as _hashlib
        used = {}
        for r in all_recs:
            val = (r.get("metadata") or {}).get(metadata_key)
            if val is None:
                continue
            key = str(val)
            if key not in used:
                safe_name = re.sub(r"[^\w\-_]", "_", key).strip("_")
                if not safe_name:
                    safe_name = "value"
                if safe_name in used.values():
                    safe_name = "%s-%s" % (
                        safe_name,
                        _hashlib.md5(key.encode("utf-8")).hexdigest()[:6])
                used[key] = safe_name
            groups.setdefault(used[key], []).append(r)

        results = {}
        for grp_name, recs in groups.items():
            target_path = os.path.join(output_dir, f"{grp_name}.dat")
            with self._sibling(target_path) as target:
                count = target.add_batch(recs, batch_size=64, _replaying_stored_records=True)
                results[grp_name] = count
        return results

    @classmethod
    def search_multi(cls, vault_paths: List[str], query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Federated search: queries multiple separate vault files simultaneously without merging them on disk.
        Recommended for modular read-only doc sets (up to 3-5 vaults).
        """
        if len(vault_paths) > 5:
            import warnings
            warnings.warn(
                f"Querying {len(vault_paths)} vaults simultaneously may cause context dilution, "
                "lower accuracy, and increased latency. For best quality (95%+ accuracy), consider "
                "merging related files into a single partitioned vault using Vault.merge().",
                UserWarning,
                stacklevel=2
            )

        all_hits = []
        for vp in vault_paths:
            if os.path.exists(vp):
                with cls(vp) as v:
                    hits = v.search(query, top_k=top_k)
                    for h in hits:
                        h["vault_file"] = os.path.basename(vp)
                        all_hits.append(h)
        all_hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return all_hits[:top_k]

    def _call_llm(
        self,
        prompt: str,
        candidates: List[Dict[str, Any]],
        model: str = "llama3.2",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cite: bool = True
    ) -> str:
        """
        Query local Ollama, LM Studio, vLLM, or OpenAI compatible model with context budget protection
        and graceful error recovery.
        """
        # 1. Budget context snippets to avoid blowing up LLM context window (max ~800 words)
        context_snippets = []
        total_ctx_words = 0
        MAX_CONTEXT_WORDS = 800

        for i, c in enumerate(candidates):
            c_words = len(c["text"].split())
            if total_ctx_words + c_words > MAX_CONTEXT_WORDS and context_snippets:
                # Add truncated excerpt if budget allows
                remaining = max(30, MAX_CONTEXT_WORDS - total_ctx_words)
                short_text = " ".join(c["text"].split()[:remaining]) + "..."
                if cite:
                    context_snippets.append(f"[{i+1}]: {short_text}")
                else:
                    context_snippets.append(f"• {short_text}")
                break
            if cite:
                src = c.get("source", "knowledge_base")
                context_snippets.append(f"[{i+1}] (Source: {src}): {c['text']}")
            else:
                context_snippets.append(f"• {c['text']}")
            total_ctx_words += c_words

        context_str = "\n".join(context_snippets)

        # Budget user prompt (max 1000 words)
        p_words = prompt.split()
        safe_prompt = prompt if len(p_words) <= 1000 else " ".join(p_words[:1000]) + "..."

        if cite:
            system_instructions = (
                "You are a helpful, precise assistant. Answer the user prompt using ONLY the provided facts. "
                "Cite facts using bracketed numbers like [1] or [2] where appropriate."
            )
        else:
            system_instructions = (
                "You are a helpful, precise assistant. Answer the user prompt naturally and conversationally using ONLY the provided facts. "
                "Do not include bracketed citation numbers or source references."
            )

        target_base = (base_url or self.llm_base_url or "http://localhost:11434").rstrip("/")
        cur_key = api_key or self.api_key

        # Determine endpoint and protocol.
        flavour, url = _llm_endpoint(target_base)
        user_content = f"Facts:\n{context_str}\n\nQuestion: {safe_prompt}"
        headers = {"Content-Type": "application/json"}
        if flavour == "anthropic":
            # Claude is NOT OpenAI-compatible: /v1/messages, a top-level
            # `system`, a required `max_tokens`, and x-api-key rather than a
            # bearer token. Through 0.6.0 an api.anthropic.com base_url ended in
            # "/v1", matched the OpenAI branch, and was POSTed to
            # /v1/chat/completions -- which 404s, so the caller saw "[Model not
            # found]" for a model that exists.
            payload_data = {"model": model, "max_tokens": 1024,
                            "system": system_instructions,
                            "messages": [{"role": "user", "content": user_content}]}
            if cur_key:
                headers["x-api-key"] = cur_key
            headers["anthropic-version"] = "2023-06-01"
        elif flavour == "openai":
            payload_data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_content}
                ],
                "stream": False
            }
            if cur_key:
                headers["Authorization"] = f"Bearer {cur_key}"
        else:
            full_prompt = f"{system_instructions}\n\nFacts:\n{context_str}\n\nUser Prompt: {safe_prompt}"
            payload_data = {"model": model, "prompt": full_prompt, "stream": False}
            if cur_key:
                headers["Authorization"] = f"Bearer {cur_key}"

        payload = json.dumps(payload_data).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0].get("message", {}).get("content", "").strip()
                if isinstance(data.get("content"), list):        # Anthropic messages
                    return "".join(b.get("text", "") for b in data["content"]
                                   if isinstance(b, dict)).strip()
                return data.get("response", "").strip()
        except urllib.error.HTTPError as e:
            try:
                err_data = json.loads(e.read().decode("utf-8", errors="replace"))
                err_msg = err_data.get("error", {}).get("message", err_data.get("error", str(e)))
            except Exception:
                err_msg = str(e)
            if "context" in str(err_msg).lower() or "length" in str(err_msg).lower():
                if len(candidates) > 1:
                    return self._call_llm(safe_prompt, candidates[:1], model=model, base_url=base_url, api_key=api_key)
                return f"[Context length exceeded for '{model}'. Try a model with larger context window.]"
            if e.code == 404:
                return f"[Model '{model}' not found at {url}. Make sure model is loaded/available.]"
            return f"[LLM error ({e.code}): {err_msg}]"
        except urllib.error.URLError:
            return f"[Facts retrieved ({len(candidates)} citations). LLM server not reachable at {url}.]"
        except Exception as e:
            return f"[Context retrieved ({len(candidates)} facts). LLM call skipped: {e}]"

    def _current_value_ids(self, records) -> frozenset:
        """Ids that make up the NEWEST revision of a tagged fact.

        These are the records that answer a question. Everything else is either
        a superseded value or an untagged document, and neither is what a
        caller loses an answer by deleting.

        ALL of them, not one: a value long enough to be chunked is spread over
        several records that share an entity, a revision and a timestamp, and
        protecting one of them protects an arbitrary quarter of a document.
        """
        newest: Dict[tuple, tuple] = {}
        by_version: Dict[tuple, list] = {}
        for r in records:
            meta = r.get("metadata") or {}
            ent = meta.get("entity")
            if not ent:
                continue
            key = (meta.get("user_id"), meta.get("project"), ent)
            # A CHUNKED VALUE IS ONE VALUE. `add()` splits a long record into
            # `{parent}_chunk_N` and every chunk inherits `entity`, so a policy
            # stored as four chunks used to look like four revisions of itself.
            # Keeping one ID per group then kept ONE CHUNK -- and since the
            # chunks tie on revision and timestamp, the survivor was whichever
            # came first, routinely a paragraph of boilerplate rather than the
            # one holding the answer. Measured before this: a three-version
            # refund policy went 12 records -> 1, and the surviving record did
            # not contain the eligibility window the vault was being asked for.
            version = meta.get("parent_id") or r.get("id")
            # TIME FIRST, ARRIVAL ONLY TO BREAK A TIE -- the same order 0.7.4
            # gave the ranker. This was (revision, timestamp), i.e. arrival
            # order, so on a vault written out of order the two halves of the
            # system disagreed about which record was current: `search` returned
            # the newest BY TIME and retention protected the newest BY ARRIVAL,
            # then deleted the one search had just called the answer.
            rank = (float(r.get("timestamp", 0.0)), int(r.get("revision", 1)))
            by_version.setdefault((key, version), []).append(r.get("id"))
            if key not in newest or rank > newest[key][0]:
                newest[key] = (rank, version)
        out = set()
        for key, (_rank, version) in newest.items():
            out.update(i for i in by_version.get((key, version), ()) if i is not None)
        return frozenset(out)

    def forget_superseded(self, keep: int = 1, older_than_days: Optional[float] = None,
                          min_revisions: int = 2, dry_run: bool = False) -> Dict[str, Any]:
        """Drop OLD REVISIONS of facts that have been restated, keeping the newest.

        The retention rule this store actually wants. A vault used as a dump
        grows a long tail of values that were true once: five addresses, four
        phone numbers, three employers. Only the newest of each answers a
        question, and the rest are there so `history` and `as_of` can answer
        "what was it before". Past a point you stop wanting all of them.

        ``keep=1`` leaves only the current value of each fact. ``keep=3`` leaves
        the current one and the two before it. A fact restated fewer than
        ``min_revisions`` times is not touched at all, and ``older_than_days``
        restricts it further to revisions older than that.

        WHAT IT WILL NEVER DELETE, which is the entire reason it exists rather
        than you calling :meth:`prune`:

        * the CURRENT value of any fact, at any age. `prune(older_than_days=N)`
          selects on age alone and is blind to revisions, so it will happily
          delete a fact that has been true and unchanged for two years -- and
          the search that used to answer it then returns a DIFFERENT fact
          rather than nothing, which is worse than an empty result.
        * any record with no entity, since a record that is not part of a
          revision chain has no newer version to be superseded by.

        Returns ``{"groups", "deleted", "kept", "ids", "dry_run"}``:
        ``deleted`` and ``ids`` are the revisions this call removes, ``kept``
        the revisions left across every tagged fact, ``groups`` the chains
        touched. With ``dry_run=True`` nothing is written and every one of those
        numbers describes what the real call WOULD do -- worth running first on
        a vault you care about, because deletion here is a full rewrite and
        there is no undo. Check ``dry_run`` in the result, not ``deleted``, to
        tell a preview from a run.
        """
        keep = max(1, int(keep))
        self.flush()
        cutoff = (time.time() - float(older_than_days) * 86400.0
                  if older_than_days is not None else None)

        groups: Dict[tuple, list] = {}
        for r in self.engine.iter_records():
            meta = r.get("metadata") or {}
            ent = meta.get("entity")
            if not ent:
                continue                       # not a revision chain
            key = (meta.get("user_id"), meta.get("project"), ent)
            groups.setdefault(key, []).append(r)

        doomed = []
        touched = 0
        for key, recs in groups.items():
            # COLLAPSE CHUNKS INTO VERSIONS FIRST. `add()` splits a long record
            # into `{parent}_chunk_N`, each inheriting `entity`, so counting
            # RECORDS counted one policy's four paragraphs as four revisions of
            # it. keep=1 then kept a single chunk of the current version and
            # deleted the rest of it -- the precise thing this method's own
            # docstring promises never to do. A version is what a caller wrote
            # once, so that is the unit `keep` and `min_revisions` now count.
            versions: Dict[Any, list] = {}
            for r in recs:
                meta = r.get("metadata") or {}
                versions.setdefault(meta.get("parent_id") or r.get("id"), []).append(r)
            # Newest LAST, by time, with the arrival counter breaking ties --
            # see `_current_value_ids`. Sorting by revision first deleted the
            # record `search` calls current whenever the two disagreed.
            ordered = sorted(
                versions.values(),
                key=lambda rs: (float(rs[0].get("timestamp", 0.0)),
                                int(rs[0].get("revision", 1))))
            if len(ordered) < max(2, int(min_revisions)):
                continue
            older = ordered[:-keep]            # every version but the newest `keep`
            if cutoff is not None:
                # a version goes only if ALL of it is past the cutoff
                older = [rs for rs in older
                         if all(float(r.get("timestamp", 0.0)) < cutoff for r in rs)]
            if older:
                touched += 1
                doomed.extend(r["id"] for rs in older for r in rs)

        # ``deleted`` is what this call describes, on a dry run as much as a real
        # one. 0.7.1 reported 0 for a dry run, which read as "nothing to clean"
        # to the obvious caller -- `if plan["deleted"]: v.forget_superseded(...)`
        # -- and silently skipped the cleanup. A preview whose headline number
        # disagrees with the run it previews is worse than no preview. ``dry_run``
        # in the result is how you tell the two apart.
        kept = sum(len(r) for r in groups.values()) - len(doomed)
        out = {"groups": touched, "deleted": len(doomed), "kept": kept,
               "ids": doomed, "dry_run": bool(dry_run)}
        if doomed and not dry_run:
            self.delete(ids=doomed)
        return out

    def prune(self, target_freed_bytes: Optional[int] = None,
              older_than_days: Optional[int] = None,
              keep_current: bool = True) -> int:
        """Reclaim disk space by removing the oldest records.

        RETURNS BYTES FREED, NOT A RECORD COUNT. This is the one thing the
        docstring never said, and `delete()` -- documented immediately above,
        with the same `-> int` -- returns "the number of deleted records". So
        the natural reading next to it is wrong: pruning 9 records from a small
        vault returns 15339, and
        ``print(f"pruned {v.prune(older_than_days=365)} records")`` reports
        fifteen thousand of them. The name ``target_freed_bytes`` is the hint;
        it should not have had to be one.

        ``keep_current=True``, the default, exempts the CURRENT value of every
        tagged fact whatever its age. Until 0.7.1 there was no such exemption
        and the selection was age alone, which on a memory vault destroyed
        answers silently:

            before  prune(older_than_days=365):
              "what is my blood type" -> "My blood type is O negative."
            after:
              "what is my blood type" -> "My locker code is 5555."

        That fact had been true and unchanged for 700 days. Losing it is bad;
        the query then returning A DIFFERENT FACT is worse, because nothing
        tells the caller an answer went missing. 0.7.0 shipped a safe
        alternative and a warning in this docstring, which was the wrong call --
        a documented trap is still a trap.

        THE DEFAULT COSTS THE DOCUMENT CASE NOTHING. Only a record that is the
        newest revision of a tagged chain is exempt, and an ingested document
        carries no entity, so a corpus being aged out prunes exactly as it did.
        The behaviour changes only where it used to delete the answer.

        ``keep_current=False`` restores age-alone selection for a caller who
        means it. To drop old values while keeping current ones, prefer
        :meth:`forget_superseded`, which is built for it and can keep more than
        one.
        """
        self.flush()
        if not os.path.exists(self.path):
            return 0
        current_sz = os.path.getsize(self.path)
        records = self.get_all_records()
        if not records:
            return 0

        now = time.time()
        protected = self._current_value_ids(records) if keep_current else frozenset()
        kept_records = []
        for r in records:
            if older_than_days is not None and r.get("id") not in protected:
                age_days = (now - r.get("timestamp", now)) / 86400.0
                if age_days > older_than_days:
                    continue
            kept_records.append(r)

        if target_freed_bytes and target_freed_bytes > 0:
            bytes_to_free = min(current_sz, target_freed_bytes)
            avg_rec_sz = max(100, current_sz // max(1, len(records)))
            recs_to_drop = min(len(kept_records), max(1, bytes_to_free // avg_rec_sz))
            # Drop from the front as before, but step over anything protected:
            # freeing bytes is a budget, and no budget is worth an answer.
            dropped, head = 0, []
            for r in kept_records:
                if dropped < recs_to_drop and r.get("id") not in protected:
                    dropped += 1
                    continue
                head.append(r)
            kept_records = head

        if len(kept_records) == len(records):
            return 0

        self._rebuild(kept_records)
        new_sz = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        return max(0, current_sz - new_sz)

    def _rebuild(self, records: List[Dict[str, Any]]) -> int:
        """
        Replace the vault's contents with `records`, atomically.

        This is the single mutation path behind update/delete/export(purge)/
        unmerge/prune. It writes a complete new container to a temp file and
        `os.replace`s it over the old one, so there is no window in which the
        vault is missing or half-written -- the previous implementation removed
        the file and re-ingested, which lost everything on a crash in between.
        Explicit id/timestamp/revision/embedding are stored verbatim.
        """
        prepared = []
        missing = [r for r in records if r.get("embedding") is None]
        if missing:
            vecs = self.embedder.embed_batch([str(r.get("text", "")) for r in missing])
            for r, v in zip(missing, vecs):
                r["embedding"] = v
        for r in records:
            meta = dict(r.get("metadata") or {})
            rec_id = str(r.get("id") or meta.get("id") or "")
            meta["id"] = rec_id
            prepared.append({
                "id": rec_id,
                "text": r.get("text", ""),
                "source": r.get("source", "user_input"),
                "metadata": meta,
                "timestamp": float(r.get("timestamp", time.time())),
                "revision": int(r.get("revision", 1)),
                "embedding": r["embedding"],
            })
        return self.engine.replace_all(prepared)

    def compact(self, recluster: bool = False) -> Dict[str, Any]:
        """
        Coalesce under-filled blocks into full ones. With `recluster=True` the
        rows are first re-ordered by a global spherical k-means, which is what
        makes an unordered corpus routable; it changes the stored order and
        therefore the order `get_all_records` returns.
        """
        return self.engine.compact(recluster=recluster)

    def flush(self):
        """Force flush pending memtable facts into continuous container blocks."""
        self.engine.flush()

    def stats(self) -> Dict[str, Any]:
        """Return memory statistics, document count, and working set footprint."""
        return self.engine.stats()

    def inspect(self) -> Dict[str, Any]:
        """
        Inspect the contents, sources, and metadata tags stored in this vault.
        Returns a summary of all document sources, metadata keys, and unique values.
        """
        all_recs = self.get_all_records(include_embeddings=False)
        sources: Dict[str, int] = {}
        metadata_summary: Dict[str, Dict[str, int]] = {}

        for r in all_recs:
            src = str(r.get("source", "unknown"))
            sources[src] = sources.get(src, 0) + 1

            meta = r.get("metadata") or {}
            for k, v in meta.items():
                if k in ["is_code", "is_chunked", "timestamp", "start_line", "end_line"]:
                    continue
                if k not in metadata_summary:
                    metadata_summary[k] = {}
                val_str = str(v)
                metadata_summary[k][val_str] = metadata_summary[k].get(val_str, 0) + 1

        return {
            "file_path": self.path,
            "total_documents": len(all_recs),
            "sources": sources,
            "metadata_summary": metadata_summary
        }

    def close(self):
        """Flush pending facts and release the engine. Idempotent."""
        self.flush()
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
