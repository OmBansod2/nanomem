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
import urllib.request
import numpy as np
from typing import List, Optional

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
        that encoder emits ``OFFLINE_DIM``. Pass ``dim=`` to skip the probe --
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

    def _remote_embed_batch(self, texts: List[str]):
        """The daemon call alone. ``None`` when it does not answer usefully."""
        req_data = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url, data=req_data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
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
