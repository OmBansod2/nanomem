"""
nanomem.embed
~~~~~~~~~~~~~
Embedding adapter. Talks to any Ollama- or OpenAI-compatible embeddings
endpoint and adopts whatever vector width that model returns; falls back to a
deterministic LEXICAL encoder when nothing answers. There are no bundled neural
weights -- see `EmbeddingProvider` and `_offline_encode_batch`.
"""

import os
import json
import time
import urllib.error
import urllib.request
import warnings
import numpy as np
from typing import Dict, List, Optional

from .errors import EmbeddingWidthError

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/embed"
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")

#: Width of the deterministic offline encoder. It is a property of THAT
#: encoder, not a limit of the store: the container holds whatever width it
#: is given, and an existing vault's header width always wins.
OFFLINE_DIM = 768


class EmbeddingProvider:
    """Embeds text with a model you choose, or a lexical encoder if none answers.

    Talks to any Ollama- or OpenAI-compatible embeddings endpoint (``base_url``
    or ``NANOMEM_EMBED_URL``) and adopts whatever width that model returns.

    There are NO bundled neural weights. Through 0.6.2 this docstring claimed
    there were, and a 139 MB GGUF of nomic-embed-text-v1.5 sat at
    ``assets/model.bin`` in both macOS bundles -- never opened, because reading
    GGUF means llama.cpp or writing BERT inference in numpy, and either costs
    the only-dependency-is-numpy property. It was removed at 0.6.3 along with the
    two attributes that stat'd it. For offline embeddings run a local daemon
    (``ollama pull nomic-embed-text``), which is the same model. The fallback
    when nothing answers is :meth:`_offline_encode_batch`, which is lexical
    hashing and is spelled out there.
    """

    #: Which encoder served the most recent batch: "model" or "offline-lexical".
    last_backend: str = "model"
    #: One warning per process, not one per call.
    _fallback_announced: bool = False
    #: Endpoints that did not answer -> the monotonic time to dial them again.
    #: CLASS level, so one outage is learned once for the whole process rather
    #: than re-discovered by every Vault, every call.
    _down_until: Dict[str, float] = {}
    #: How long to stop dialling an endpoint that refused or timed out.
    #:
    #: WHY THIS EXISTS. `embed_batch` dialled the endpoint on EVERY call and
    #: remembered nothing, so an install with no daemon paid the full connect
    #: cost per add() and per search(). Where a refused connection is instant
    #: -- Linux, macOS -- that is invisible, and it stayed invisible for as
    #: long as nobody measured a platform where it is not. On the Windows CI
    #: runner the SYN is dropped rather than refused and each attempt costs
    #: ~4 s: 2,267 attempts in one test run, 2 h 35 m of wall clock against
    #: under 4 minutes everywhere else, and the same ~4 s on every single
    #: add() and search() for any Windows user without a daemon. Measured in
    #: evidence/windows_embed_stall.json.
    DOWN_COOLDOWN = 30.0


    def __init__(self, model: str = DEFAULT_MODEL, base_url: Optional[str] = None,
                 dim: Optional[int] = None):
        self.model = model
        self.base_url = base_url or os.getenv("NANOMEM_EMBED_URL", DEFAULT_OLLAMA_URL)
        self._dim = int(dim) if dim else None

    @property
    def dim(self) -> int:
        """The model's ACTUAL output width, probed once from the server.

        This was hard-coded to 768 through 0.6.1, which made `embed_model=` a
        parameter you could pass and not a model you could use: `Vault` sizes
        its engine from this number, so naming any other model built a 768-d
        vault that then rejected every write --

            Vault(embed_model="all-minilm")
            -> ValueError: embedding has 384 dims, vault has 768

        The container was never the limitation; it stores whatever width it is
        given and an existing file's header always wins. Only this constant was.

        Probed lazily, once, with one short request, so constructing a provider
        costs nothing until something actually needs the width. If the server
        cannot be reached the deterministic offline encoder answers instead, and
        that encoder emits ``OFFLINE_DIM`` unless a width was declared with
        ``dim=``, which it honours. Pass ``dim=`` to skip the probe --
        the right thing to do for a provider that will only ever be handed a
        lookup table, and for an air-gapped install.
        """
        if self._dim is None:
            self._dim = self._probe_dim()
        return self._dim

    def _probe_dim(self) -> int:
        try:
            vecs = self._remote_embed_batch(["nanomem dimension probe"])
            if vecs is not None and vecs.ndim == 2 and vecs.shape[1] > 0:
                return int(vecs.shape[1])
        except Exception:
            pass
        return OFFLINE_DIM

    #: Texts per HTTP request. A whole document used to go in ONE request against
    #: a fixed 5 s timeout, so a 20,000-word document (40 chunks) timed out and
    #: EVERY chunk was silently stored with the lexical fallback -- measured
    #: 40/40, 120/120 and 240/240 chunks at 20k, 60k and 120k words. The document
    #: was then unsearchable by meaning, with nothing to say so.
    REMOTE_BATCH = 16
    #: Base seconds, plus `PER_TEXT_TIMEOUT` for each text in the request.
    BASE_TIMEOUT = 5.0
    PER_TEXT_TIMEOUT = 1.0
    MAX_TIMEOUT = 120.0

    @classmethod
    def forget_unreachable_endpoints(cls) -> None:
        """Dial every endpoint again on the next call, cooldown or not.

        An endpoint that stops answering is not tried again for
        `DOWN_COOLDOWN` seconds. That is the right default for a process that
        would otherwise pay the connect cost forever, and the wrong one the
        moment you have just started the daemon yourself and want this run to
        use it. Call this then, rather than waiting the cooldown out.
        """
        cls._down_until.clear()

    def _remote_embed_batch(self, texts: List[str]):
        """The daemon call alone. ``None`` when it does not answer usefully.

        Splits into `REMOTE_BATCH`-sized requests and is ALL-OR-NOTHING: if any
        request fails the whole call fails, so a single document can never end up
        with some chunks embedded by the model and some by the lexical fallback.
        Those vectors are not comparable, and a half-and-half document would be
        worse than a wholly lexical one because the failure would be invisible.
        """
        if time.monotonic() < EmbeddingProvider._down_until.get(self.base_url, 0.0):
            return None                 # still cooling off; do not dial again
        if len(texts) > self.REMOTE_BATCH:
            out = []
            for i in range(0, len(texts), self.REMOTE_BATCH):
                part = self._remote_embed_batch(texts[i:i + self.REMOTE_BATCH])
                if part is None:
                    return None
                out.append(part)
            return np.vstack(out)
        timeout = min(self.MAX_TIMEOUT,
                      self.BASE_TIMEOUT + self.PER_TEXT_TIMEOUT * len(texts))
        req_data = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url, data=req_data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError:
            # The endpoint ANSWERED -- wrong model, bad request, whatever. That
            # is not an outage, it costs nothing to ask again, and tripping the
            # breaker on it would hide a daemon that is plainly alive.
            raise
        except OSError as exc:
            # Refused, unroutable, timed out, DNS gone. `urllib.error.URLError`
            # and `socket.timeout` are both OSError, so this is the whole
            # family of "nothing answered".
            EmbeddingProvider._down_until[self.base_url] = (
                time.monotonic() + self.DOWN_COOLDOWN)
            raise exc
        EmbeddingProvider._down_until.pop(self.base_url, None)
        if "embeddings" not in data:
            return None
        vecs = np.array(data["embeddings"], dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != len(texts):
            return None
        if self._dim and int(vecs.shape[1]) != int(self._dim):
            # A DECLARED WIDTH THAT THE ENDPOINT CONTRADICTS IS AN ERROR.
            # Reporting one width from `.dim` and returning another is how a
            # caller ends up sizing a vault, a cache or a matrix wrongly with
            # nothing to tell them. `dim=` skips the probe; it does not override
            # the model.
            raise EmbeddingWidthError(
                f"{self.model} at {self.base_url} returns {vecs.shape[1]}-d "
                f"vectors, but dim={self._dim} was declared. Pass the model's "
                f"real width, or omit dim= and let it be probed.")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        return vecs / norms

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        # 1. Try local daemon (Ollama / LocalAI)
        try:
            vecs = self._remote_embed_batch(texts)
            if vecs is not None:
                if self._dim is None:
                    self._dim = int(vecs.shape[1])
                self.last_backend = "model"
                return vecs
        except EmbeddingWidthError:
            # A WIDTH CONFLICT IS NOT A DAEMON OUTAGE. The blanket `except` below
            # exists so an unreachable endpoint degrades to the lexical encoder
            # instead of failing. Letting it swallow this would be worse than the
            # bug it replaced: the caller declared a width the model contradicts,
            # and they would silently get lexical hashing at the declared width
            # while believing they were using the model.
            raise
        except Exception:
            pass

        # 2. Air-gapped deterministic semantic fallback (Zero network egress)
        #
        # SAY SO. Falling back is right for an air-gapped install, but in a vault
        # whose other rows were embedded by the model the two are not comparable,
        # and the row written during an outage becomes unfindable -- measured: a
        # fact added while the daemon was down did not appear in the results for
        # its own question, and nothing warned. A memory product that quietly
        # stores something it can never retrieve is worse than one that fails.
        if not EmbeddingProvider._fallback_announced:
            EmbeddingProvider._fallback_announced = True
            warnings.warn(
                "nanomem: the embedding endpoint did not answer, so this text was "
                "encoded with the LEXICAL fallback. It does not model meaning, and "
                "these vectors are not comparable with rows embedded by the model -- "
                "a record written now may not be findable later. Check the daemon "
                "and re-add anything written during the outage; rows carry "
                "metadata['embed_backend'] so you can find them.",
                RuntimeWarning, stacklevel=2)
        self.last_backend = "offline-lexical"
        return self._offline_encode_batch(texts)

    def _offline_encode_batch(self, texts: List[str]) -> np.ndarray:
        """A deterministic LEXICAL encoder. It does not model meaning.

        Each word is hashed to a sine frequency and summed with positional
        decay, and character trigrams bump fixed dimensions. Two texts therefore
        score highly when they SHARE WORDS AND CHARACTERS, which is not the same
        thing as agreeing. Measured on this build:

        ==========================================  ======
        pair                                        cosine
        ==========================================  ======
        "the server is up" / "the server is down"    0.783
        "my dog is black" / "my car is black"        0.740
        "I drive a car" / "I own an automobile"      0.286
        ==========================================  ======

        Opposites score near-identical and synonyms score unrelated. Treat it as
        fuzzy string matching that keeps the pipeline running when no model
        answers -- good enough for a smoke test or an exact-phrase lookup, and
        wrong for anything that depends on semantics. Anything measured or
        demonstrated with it is measuring string overlap.

        It is deterministic across processes and sessions (hashlib, not the
        salted builtin ``hash``), and emits ``OFFLINE_DIM`` columns unless a
        width was declared with ``dim=``, which it honours.
        """
        import hashlib
        # A DECLARED WIDTH IS HONOURED HERE, NOT JUST REPORTED.
        # This always emitted OFFLINE_DIM, so `EmbeddingProvider(dim=384)` --
        # the documented air-gapped case, where this encoder IS the model --
        # reported 384 and returned 768. The scheme hashes into as many columns
        # as it is given, so there was never a reason it could not honour one.
        width = int(self._dim) if self._dim else OFFLINE_DIM
        batch_vecs = np.zeros((len(texts), width), dtype=np.float32)
        
        for idx, text in enumerate(texts):
            clean = text.lower().strip()
            words = clean.split()
            vec = np.zeros(width, dtype=np.float32)
            
            for w_idx, w in enumerate(words):
                h = int.from_bytes(hashlib.md5(w.encode("utf-8")).digest()[:4], "little")
                w_vec = np.sin(np.arange(width) * (h % 1000 + 1) * 0.01)
                decay = 1.0 / (1.0 + 0.05 * w_idx)
                vec += w_vec * decay

            if len(clean) >= 3:
                for i in range(len(clean) - 2):
                    tri = clean[i:i+3].encode("utf-8")
                    th = int.from_bytes(hashlib.md5(tri).digest()[:4], "little") % width
                    vec[th] += 0.35

            norm = np.linalg.norm(vec) + 1e-8
            batch_vecs[idx] = vec / norm

        return batch_vecs
