"""
nanomem.vault
~~~~~~~~~~~~~
Primary public API for the nanomem continuous memory engine.
Provides a clean, intuitive 4-method interface: add(), search(), chat(), and ask().
"""

import os
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
    ``scratch/refound/scale_results_v3r3.json``): 22.0 MB on disk, 0.344 ms p50
    search, evidence recall@4 70.4% -- identical to an exhaustive scan (0.272 ms)
    and to FAISS IndexFlatIP (0.391 ms) on the same data. At 71,433 paragraphs:
    155.9 MB, 1.809 ms p50, 60.6% -- again identical to both. The index is 0.609x
    of (raw UTF-8 text + fp32 vectors), but roughly half of that saving is
    precision, not format: against raw text + fp16 vectors, the shape nanomem
    actually stores, the same index is 1.056x
    (``scratch/refound/headtohead_v3.json`` -> ``index_size``). Search is exact and
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
        on_torn_tail: str = "warn"
    ):
        self.path = os.path.abspath(path)
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
                                  on_torn_tail=on_torn_tail)
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
        is_cjk = any(ord(c) > 0x2E80 for c in clean)

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
            p_cjk = any(ord(c) > 0x2E80 for c in p)
            if (not p_cjk and len(p_words) > max_words) or (p_cjk and len(p) > cls.MAX_FACT_CHARS_CJK):
                sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？\n])\s+", p) if s.strip()]
                raw_units.extend(sentences if sentences else [p])
            else:
                raw_units.append(p)

        units = []
        for u in raw_units:
            u_words = u.split()
            u_cjk = any(ord(c) > 0x2E80 for c in u)
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

        meta = dict(metadata or {})
        import hashlib
        doc_id = str(id or meta.get("id") or f"doc_{hashlib.md5(clean_text.encode('utf-8')).hexdigest()[:10]}")
        meta["id"] = doc_id
        if metadata is not None and isinstance(metadata, dict):
            metadata["id"] = doc_id
        self.last_id = doc_id

        chunks = self.split_large_text(clean_text, max_words=self.MAX_FACT_WORDS, overlap_words=self.OVERLAP_WORDS)
        if len(chunks) <= 1:
            vec = self.embedder.embed(clean_text)
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
        vecs = self.embedder.embed_batch(chunks)

        for idx, ch in enumerate(chunks):
            chunk_meta = dict(meta)
            chunk_id = f"{parent_id}_chunk_{idx+1}"
            chunk_meta["id"] = chunk_id
            chunk_meta["chunk_index"] = idx + 1
            chunk_meta["chunk_total"] = total_chunks
            chunk_meta["parent_id"] = parent_id
            chunk_meta["is_chunked"] = True

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
        stride = max(1, lines_per_chunk - overlap_lines)
        code_exts = {
            ".py", ".ts", ".js", ".tsx", ".jsx", ".rs", ".go", ".cpp", ".c", ".h", ".hpp",
            ".java", ".cs", ".rb", ".php", ".swift", ".kt", ".sh", ".bash", ".sql", ".html", ".css"
        }
        is_code = ext in code_exts

        for start_idx in range(0, len(raw_lines), stride):
            slice_lines = raw_lines[start_idx : start_idx + lines_per_chunk]
            if not slice_lines:
                break
            chunk_text = "".join(slice_lines).strip()
            if chunk_text:
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

                chunk_source = source or f"{basename}:{start_line}-{end_line}"
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": chunk_meta,
                    "source": chunk_source
                })

        if chunks:
            self.add_batch(chunks, batch_size=32)
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
        ignored = set(ignore_dirs or [
            ".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".idea",
            ".vscode", "dist", "build", "target", ".next", ".nuxt", "coverage", ".pytest_cache"
        ])

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
        # (scratch/refound/ingest_ram_results.json, ``directory_estimate``):
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
        batch_size: int = 64
    ) -> int:
        """
        High-throughput batch ingestion for books, datasets, and corpora.
        If embeddings are already present (e.g. during vault merge or split),
        they are transferred directly, without re-calling the embedding model.
        """
        if not records:
            return 0

        total = len(records)
        # ARENA SIZING. A bulk caller knows how many rows are coming; the engine
        # does not, and nothing in the library used to tell it. Measured over
        # a 71,433-document ingest of 768-d vectors, peak ru_maxrss of the whole
        # build-then-serve process, two runs each
        # (scratch/refound/ingest_ram_results.json, summary + replicate),
        # median of four runs: 820.5 MB through the capacity doubling this
        # engine shipped with, 313.7 MB through this method today -- 15 MB of
        # which is the caller's own list of 71,433 record dicts, not the
        # vault's.
        # It is only a hint -- a wrong one costs nothing but pages nobody
        # touches.
        self.engine.reserve_additional_rows(total)
        for i in range(0, total, batch_size):
            chunk = records[i : i + batch_size]
            texts = [str(r.get("text", "")).strip() for r in chunk]
            has_all_vecs = all("embedding" in r and r["embedding"] is not None for r in chunk)
            if has_all_vecs:
                vecs = [r["embedding"] for r in chunk]
            else:
                vecs = self.embedder.embed_batch(texts)

            for j, r in enumerate(chunk):
                if not texts[j]:
                    continue
                meta = dict(r.get("metadata", {}))
                import hashlib
                doc_id = str(r.get("id") or meta.get("id") or f"doc_{hashlib.md5(f'{texts[j]}_{time.time()}_{j}'.encode()).hexdigest()[:10]}")
                meta["id"] = doc_id
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
        return total

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
        vaults in ``scratch/refound/temporal_as_of_results.json``.

        Search is an exact, LINEAR scan: latency grows with the corpus and the
        measured p50 is published per size in
        ``scratch/refound/scale_results_v3r3.json`` -- there is no fixed
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
                beam_width=beam_width
            )


        safe_top_k = max(1, min(50, int(top_k))) if top_k is not None else 3
        words = str(query).strip().split()
        clean_query = " ".join(words[:100]) if len(words) > 100 else str(query).strip()
        if not clean_query:
            return []

        sub_queries = self.decompose_query(clean_query) if decompose else [clean_query]

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

        for d in range(max_depth):
            for sq, hits in sub_hits:
                if d < len(hits):
                    h = hits[d]
                    if h["text"] not in seen_texts:
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
        bridge: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Two-pass retrieval for bridge ("A -> B -> answer") questions. Opt-in.

        Pass 1 is an ordinary search. Pass 2 re-embeds the question together with
        the text of the hop-1 winner -- `f"{question}\n{hop1_text}"` -- and
        searches with that vector, then merges: the top ceil(k/2) single-pass
        hits plus the top floor(k/2) bridge hits, ranked by cosine and
        de-duplicated by id, so the budget is the same `k` documents.

        MEASURED, equal budget, evidence recall@4, on a re-opened vault
        (scratch/refound/multihop_texthop_v3r2_1190.json, 1,190 paragraphs, 120
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
        safe_top_k = max(1, min(50, int(top_k))) if top_k is not None else 3
        words = str(query).strip().split()
        clean_query = " ".join(words[:100]) if len(words) > 100 else str(query).strip()
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
            temporal_direction=temporal_direction
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
            temporal_direction=temporal_direction
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

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an exact memory record, fact, or chunk by its unique ID.
        Returns the record dictionary if found, or None if not present.
        """
        clean_id = str(id).strip()
        if not clean_id:
            return None
        self.flush()
        rec = self.engine.get(clean_id)
        if rec is None:
            return None
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
            return False

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
            "timestamp": time.time(),
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
            self.add_batch(to_insert, batch_size=64)

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
        Returns the number of deleted records.

        COST. There are no tombstones in 3.0 (DECISIONS #11), so a delete is a
        full atomic rewrite of the container with pre-computed vectors: measured
        6.6 ms at 1,000 records and 65.2 ms at 10,000 (768-d), and about 1 s at
        71k -- linear in vault size, not "sub-millisecond execution" as this
        docstring claimed through 3.0.2. The rewrite holds the container's
        exclusive lock for its whole duration, so concurrent appenders block.
        Batch your deletions: one call with a list of ids costs one rewrite.
        See scratch/refound/rewrite_cost_v3r4.json.
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

        kept = []
        deleted_count = 0
        for r in all_recs:
            drop = False
            rec_id = str(r.get("id") or (r.get("metadata") or {}).get("id") or "")
            meta = r.get("metadata") or {}

            if target_ids and rec_id in target_ids:
                drop = True
            elif text_exact is not None and r.get("text", "").strip() == text_exact.strip():
                drop = True
            elif text_contains is not None and text_contains.lower() in r.get("text", "").lower():
                drop = True
            elif source is not None:
                src_str = str(r.get("source", ""))
                fn_str = str(meta.get("filename", ""))
                fp_str = str(meta.get("file_path", ""))
                clean_src = str(source).strip()
                if (clean_src == src_str or
                    clean_src == fn_str or
                    src_str.startswith(f"{clean_src}:") or
                    clean_src in src_str or
                    fp_str.endswith(clean_src)):
                    drop = True
            elif where is not None:
                if self.engine._matches_filter(meta, where):
                    drop = True

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
            target.add_batch(matching, batch_size=64)
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
                target.add_batch(matching, batch_size=64)
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
            return target.add_batch(matching, batch_size=64)

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
            return target.add_batch(matching, batch_size=64)

    def split_by_key(self, metadata_key: str, output_dir: str) -> Dict[str, int]:
        """
        Automatically partitions this vault into multiple separate .dat files based on the values of a metadata key.
        E.g. split_by_key("category", "./categories") creates "./categories/<val>.dat".
        """
        import re
        os.makedirs(output_dir, exist_ok=True)
        all_recs = self.get_all_records()
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for r in all_recs:
            val = (r.get("metadata") or {}).get(metadata_key)
            if val is not None:
                safe_name = re.sub(r"[^\w\-_]", "_", str(val)).strip("_")
                if safe_name:
                    groups.setdefault(safe_name, []).append(r)

        results = {}
        for grp_name, recs in groups.items():
            target_path = os.path.join(output_dir, f"{grp_name}.dat")
            with self._sibling(target_path) as target:
                count = target.add_batch(recs, batch_size=64)
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

        Returns ``{"groups", "deleted", "kept", "ids"}``. With ``dry_run=True``
        nothing is written and ``ids`` is what would go -- worth using first on
        a vault you care about, because deletion here is a full rewrite and
        there is no undo.
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
            if len(recs) < max(2, int(min_revisions)):
                continue
            # newest LAST, by the same order `history` reports
            recs.sort(key=lambda r: (int(r.get("revision", 1)),
                                     float(r.get("timestamp", 0.0))))
            older = recs[:-keep]               # everything but the newest `keep`
            if cutoff is not None:
                older = [r for r in older
                         if float(r.get("timestamp", 0.0)) < cutoff]
            if older:
                touched += 1
                doomed.extend(r["id"] for r in older)

        out = {"groups": touched, "deleted": 0 if dry_run else len(doomed),
               "kept": keep, "ids": doomed}
        if doomed and not dry_run:
            self.delete(ids=doomed)
        return out

    def prune(self, target_freed_bytes: Optional[int] = None, older_than_days: Optional[int] = None) -> int:
        """Reclaim disk space by removing the OLDEST records, by age alone.

        READ THIS BEFORE USING IT ON A MEMORY VAULT. The selection is age and
        nothing else: it does not know a revision from a current value, so
        ``prune(older_than_days=365)`` deletes a fact that has been true and
        unchanged for two years exactly as readily as a superseded one. Worse
        than losing it, the query that used to answer it then returns the
        NEAREST OTHER FACT rather than nothing -- a wrong answer where there
        used to be a right one.

        It is the right call for a document corpus you are ageing out. For
        dropping old values of facts that changed, use
        :meth:`forget_superseded`, which keeps the current value by
        construction.
        """
        self.flush()
        if not os.path.exists(self.path):
            return 0
        current_sz = os.path.getsize(self.path)
        records = self.get_all_records()
        if not records:
            return 0

        now = time.time()
        kept_records = []
        for r in records:
            if older_than_days is not None:
                age_days = (now - r.get("timestamp", now)) / 86400.0
                if age_days > older_than_days:
                    continue
            kept_records.append(r)

        if target_freed_bytes and target_freed_bytes > 0:
            bytes_to_free = min(current_sz, target_freed_bytes)
            avg_rec_sz = max(100, current_sz // max(1, len(records)))
            recs_to_drop = min(len(kept_records), max(1, bytes_to_free // avg_rec_sz))
            kept_records = kept_records[recs_to_drop:]

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
