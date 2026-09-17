"""
nanomem.classifier
~~~~~~~~~~~~~~~~~~
Write classifier v2 -- the "should this utterance be stored?" gate.

What it is
----------
A small **trained linear (logistic) head** over

    [ L2-normalised 768-d embedding ;  23 generic surface features ]

plus four small, documented rule layers (empty input, explicit "forget that",
explicit "remember this", and a short self-declarative fact).  Nothing else.  The
weights live in ``assets/write_classifier.npz`` and are produced by
``scratch/refound/train_write_classifier.py``, which also writes
``scratch/refound/write_classifier_v2_results.json`` with the provenance and every
number quoted below.

Measured (write_classifier_v2_results.json, 2026-09-16)
-------------------------------------------------------
* leave-one-persona-out on the 3-persona training chat set: **94.9% accuracy,
  92.8 F1** (surface-only head 92.3% / 89.4; embedding-only ablation 90.3% / 85.3).
* held out, 2 unseen personas, 231 turns, evaluated once with the frozen model:
  **90.5% accuracy, 87.2 F1** (precision 89.3, recall 85.2); surface-only head
  87.0% / 82.8.  The v1 classifier this replaces scored 76.2% / 69.6 on the same
  turns (clean_chat_results_heldout_baseline.json).
* decision latency: the head itself is 0.06 ms (0.29 ms including the embedder
  provenance check); end to end it is one embedding call, measured p50 11.9-23.6 ms
  depending on the load on the local daemon.  With no daemon reachable: p50 0.05 ms.

Where the decision point comes from (0.5.0 moved it: 0.60 -> 0.05)
------------------------------------------------------------------
``train_write_classifier.py`` chose the full head's threshold by leave-one-
persona-out **accuracy/F1**, which prices a false positive and a false negative
the same.  Deployment does not.  A false negative destroys a question
permanently -- the fact is never written, so no amount of retrieval quality can
recover it -- while a false positive costs about 1.9 kB of vault and no
measurable query time.  At 0.60 the gate ran at **100.0% precision / 64.6%
recall** over a 14-persona, 1,652-turn conversational benchmark: it admitted
zero of the 1,064 noise turns and refused 208 of the 588 real facts, which made
**135 of 420 questions (32.1%) unanswerable before retrieval ever ran.**

``scratch/refound/write_policy_results.json`` measured the sweep under a
pre-registered, hash-verified protocol: every point chosen on a 120-question dev
split, the frozen choice scored once on a disjoint 300-question test split.
Moving only this constant, with the rule layers untouched:

    end-to-end top-1   35.7% -> 49.0%   (+13.3 pt, paired bootstrap 95% CI
                                         [+8.7, +18.3]; persona-clustered CI
                                         [+9.7, +16.7])
    gate recall        64.5% -> 94.0%   precision 100.0% -> 92.9%
    documents stored     271 -> 425     (+57%), 0.492 -> 0.764 MiB / 10 personas
    p50 retrieval      0.313 -> 0.377 ms on those vaults

Storing *everything* scores 49.7% (+14.0 pt) -- 0.7 pt more for 2.8x the rows
and 5.6x the added bytes -- so 0.05 is the efficient point, not the extreme one.
The gate's own F1 is better here (0.935) than at 0.60 (0.784), so 0.60 was not
optimal even on the objective it was selected for.  Both curves are flat from
0.10 to 0.02: **0.05 is a region, not a tuned constant.**

Two things this did NOT change, and both matter:

* the **surface** head's threshold is still the trained 0.60.  That study handed
  a real embedding to every turn, so every decision it measured went through the
  full head; the no-embedder path was never in the experiment and is not moved on
  its evidence.
* the **rule layers** are untouched, and they are a second, independent write-time
  loss: with the learned head fully off (threshold 0.00) the rules still drop 52
  of 1,180 turns, 14 of them answers to questions, capping gate recall at 95.0%.
  Worth about 3.4 pt, and it needs its own study.

``WriteClassifier(threshold=0.60)`` restores the previous behaviour exactly, and
the trained value is still reported as
``inspect()["model_info"]["threshold_trained_full"]``.

Two heads are shipped:

* ``full``    -- embedding + surface features.  Used when a real embedding is
                 available (Ollama / an OpenAI-compatible endpoint).
* ``surface`` -- surface features only.  Used when no embedder is reachable, so
                 the gate degrades to a measured fallback instead of silently
                 classifying hash vectors.  Its coefficients are also compiled
                 into this file (``_SURFACE_FALLBACK_*``) so the gate still works
                 if the asset is missing from an installation.

Honesty notes
-------------
* ``nanomem.embed.EmbeddingProvider`` silently falls back to a deterministic md5
  n-gram encoder when no embedding daemon answers.  Those vectors are *not*
  comparable with the vectors the head was trained on, so this module detects the
  fallback (``_looks_like_offline_fallback``) and routes to the surface head
  instead.  ``inspect()["embedder"]`` always says which path ran.
* The two chat corpora are made of long, discursive turns -- only 1 of the 106
  training turns of <=20 words is a keeper -- so "short" is nearly a perfect proxy
  for "chaff" in them and neither set can tell you whether the gate keeps a one-line
  personal fact.  See the note on FEATURE_NAMES for what this module does about it,
  and `short_utterance_probe` in the results file for what it buys (12/15 short facts
  kept, 16/16 short chaff rejected; a head allowed to see raw length keeps 0/15).
* No vocabulary in this file is copied from, or tuned on, any benchmark fixture.
  Every word list here is a closed class of English function words / discourse
  markers (interrogatives, request verbs, acknowledgement tokens, month and
  weekday names, measurement units).  The v1 rule layers, which hard-coded
  phrases lifted verbatim from a chat benchmark, are gone, as are the v1
  ``manifold_prototypes.npz`` archetypes (which paraphrased the same benchmark).
"""

import os
import re
import time
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .embed import EmbeddingProvider

__all__ = [
    "WriteClassifier",
    "CognitiveManifoldClassifier",
    "get_classifier",
    "extract_surface_features",
    "FEATURE_NAMES",
    "EMBED_DIM",
    "DEPLOYMENT_THRESHOLD_FULL",
]

#: P(store) at or above which the FULL (embedding) head writes a turn.
#:
#: This is a DEPLOYMENT constant, deliberately not the value the trainer picked.
#: The trainer optimised accuracy/F1, where the two error types cost the same;
#: here a dropped fact is unrecoverable and a kept noise turn costs ~1.9 kB.
#: Measured end to end, +13.3 pt of top-1 [CI +8.7, +18.3] against the trained
#: 0.60 -- see this module's docstring and
#: ``scratch/refound/write_policy_results.json``.  The asset's own threshold is
#: still loaded and reported (``model_info["threshold_trained_full"]``); pass
#: ``WriteClassifier(threshold=...)`` to override either one.
DEPLOYMENT_THRESHOLD_FULL = 0.05

EMBED_DIM = 768

# ---------------------------------------------------------------------------
# Generic lexical resources.
#
# These are closed-class English function words and discourse markers plus the
# calendar/measurement vocabulary every English text shares.  They are NOT
# derived from any evaluation set: removing any single entry changes a feature
# value, never a hard-coded decision.
# ---------------------------------------------------------------------------

# Interrogative / auxiliary-fronted sentence openers.
_INTERROGATIVE_OPENERS = frozenset("""
who whom whose what which when where why how
is are was were am do does did done
can could will would shall should may might must
have has had
""".split())

# Imperative verbs people address to an assistant ("write me...", "explain...").
_REQUEST_VERBS = frozenset("""
tell write give explain help make show find check translate summarise summarize
draft list suggest recommend compare fix calculate convert describe rewrite
generate create send search look teach define rank rate pick choose remind
""".split())

# Acknowledgement / greeting / interjection openers.
_ACK_TOKENS = frozenset("""
ok okay kk yes yeah yep yup no nope nah sure right true exactly agreed fine
alright indeed thanks thank thx ta cheers please sorry congrats congratulations
haha hahaha hehe heh lol lmao rofl wow oh ah ahh ugh hmm hm huh eh
hi hello hey yo bye goodbye night morning evening afternoon
nice cool great awesome lovely perfect amazing brilliant
well anyway honestly seriously really exactly god lord
""".split())

_MONTHS = frozenset("""
january february march april may june july august september october november
december jan feb mar apr jun jul aug sep sept oct nov dec
""".split())

_WEEKDAYS = frozenset("""
monday tuesday wednesday thursday friday saturday sunday
mon tue tues wed thu thur thurs fri sat sun
today tomorrow yesterday tonight
""".split())

# Explicit user directives.  These are the only phrases that can override the
# trained head, and both directions are represented.  They match imperative /
# second-person framing only: a bare mention of the verb ("I can't remember
# whether I gave him the dose") is a narrative, not an instruction, so the
# patterns are anchored to a sentence start, to "please", or to an explicit
# object ("remember that ...").
_RE_MEMORY_DIRECTIVE = re.compile(
    r"(?:^|[.!?;]\s+|\bplease\s+)"
    r"(?:remember|note|keep in mind|make a note|take note|write (?:this|that) down|"
    r"save (?:this|that)|store (?:this|that))\b"
    r"|\bremember (?:that|this|to)\b"
    r"|\bnote that\b"
    r"|\bkeep (?:this|that|it) in mind\b"
    r"|\bfor the record\b"
    r"|\b(?:don'?t|do not) forget\b",
    re.I,
)
_RE_FORGET_DIRECTIVE = re.compile(
    r"\bforget (?:that|it|this|what)\b"
    r"|\b(?:don'?t|do not) (?:remember|save|store|record|keep) (?:that|this|it)\b"
    r"|\b(?:delete|ignore|disregard|scratch) (?:that|this|it)\b"
    r"|\bnever ?mind\b",
    re.I,
)

# Short self-declarative facts.  The training corpora are chat logs in which almost
# every keeper is a long, discursive turn, so the head has no evidence at all about
# short declaratives ("My blood type is O negative") -- the regime a memory library
# most needs to get right.  This rule floors that regime with structure only:
# possessive-copular ("my X is ...") or a first-person stative verb, in a short,
# non-interrogative, non-request sentence.  Measured cost on the two chat sets: it
# fires on 3 of 580 turns (2 keepers, 1 false positive).
_RE_SHORT_FACT = re.compile(
    r"\bmy [a-z][\w'\-]*(?:[ \-][a-z][\w'\-]*)? (?:is|are|was|were)\b"
    r"|\bi(?:'m| am) (?:allergic|intolerant) to\b"
    r"|\bi(?:'m| am) (?:based in|originally from|from) [A-Z]"
    r"|\bi (?:live|work|study|drive|own|rent|speak|moved|graduated)\b"
    r"|\bi (?:don'?t|do not|never|always) (?:eat|drink|take|use|drive|work)\b",
    re.I,
)
#: an intention or a mood is not a fact
_RE_NOT_A_FACT = re.compile(r"\bi (?:have|need|want|had) to\b", re.I)
#: upper bound (in word tokens) for "short" in the rule above
SHORT_FACT_MAX_WORDS = 20

# Structural patterns.
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_RE_URL = re.compile(r"https?://\S+|\bwww\.\S+\b")
_RE_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_RE_LONG_NUM = re.compile(r"\d[\d\s.\-()]{5,}\d")          # phone / account / id runs
_RE_DIGIT = re.compile(r"\d")
_RE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_RE_NUMERIC_DATE = re.compile(r"\b\d{1,2}[/.\-]\d{1,2}(?:[/.\-]\d{2,4})?\b")
_RE_CLOCK = re.compile(r"\b\d{1,2}\s?[:h]\s?\d{2}\b|\b\d{1,2}\s?(?:am|pm)\b", re.I)
_RE_CURRENCY = re.compile(r"[$€£¥₦₹₽₩₪₫₱฿]|\b(?:usd|eur|gbp|jpy|ngn|brl|inr|cad|aud|chf|cny|krw|mxn)\b", re.I)
_RE_UNIT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:mg|mcg|kg|g|ml|cl|l|km|cm|mm|m|kb|mb|gb|tb|hz|%|"
    r"units?|hrs?|hours?|mins?|minutes?|secs?|seconds?|days?|weeks?|months?|years?)\b",
    re.I,
)
_RE_SELF_PREDICATE = re.compile(
    r"\b(?:i am|i'm|im|i was|i work|i live|i have|i've|i use|i take|i need|i moved|"
    r"we are|we're|we have|we've|we moved)\b|\bmy \w+(?:'s)? (?:is|are|was|were|will be|became)\b|\bis my\b|\bare my\b",
    re.I,
)
_RE_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_RE_TOKEN = re.compile(r"\S+")
_RE_CAP_WORD = re.compile(r"\b[A-Z][a-z]{1,}\b")
_RE_SENT_START = re.compile(r"(?:^|[.!?]\s+|\n)\s*")

_FIRST_PERSON = frozenset("i i'm im me my mine myself we us our ours we're we've i've i'll".split())
_SECOND_PERSON = frozenset("you your yours yourself you're youre u ur".split())

# NOTE on the two length features.  Chat corpora make length a near-perfect shortcut
# for "durable" -- in the 3-persona training set only 1 of the 106 turns of <=20 words
# is a keeper -- and a head that sees raw length learns "long = store" and then refuses
# every short fact a memory library exists to keep ("My blood type is O negative":
# measured 0/15 kept).  They are kept as features but their clip floor (`_FEATURE_LO`,
# fitted at the 25th percentile of training length) means the head can use length as
# positive evidence and cannot use shortness as negative evidence below that floor.
FEATURE_NAMES = (
    "log_chars",            # 0  log1p(len(text)) / 5, clipped at the training p25 floor
    "log_tokens",           # 1  log1p(#whitespace tokens) / 5, same floor
    "avg_token_len",        # 2  mean token length / 10
    "has_question_mark",    # 3  '?' present
    "ends_question",        # 4  text ends with '?'
    "opens_interrogative",  # 5  first word is a wh-word or fronted auxiliary
    "request_cue",          # 6  imperative request verb opener, or "please"
    "second_person_ratio",  # 7  you/your tokens / #word tokens
    "first_person_ratio",   # 8  i/me/we/our tokens / #word tokens
    "possessive_my_ratio",  # 9  "my"/"our" tokens / #word tokens
    "self_predicate",       # 10 "i am ..." / "my X is ..." / "... is my ..."
    "digit_ratio",          # 11 digits / characters
    "has_long_number",      # 12 >=7-char digit run (phone, account, id)
    "has_email",            # 13
    "has_url_or_ip",        # 14
    "has_year_or_date",     # 15 4-digit year or numeric date
    "has_clock_time",       # 16 12:30 / 14h00 / 3pm
    "has_month_or_weekday", # 17 calendar word
    "has_currency",         # 18 currency symbol or ISO code
    "has_measure_unit",     # 19 number + measurement unit
    "proper_noun_ratio",    # 20 capitalised non-sentence-initial words / word tokens
    "ack_opener",           # 21 opens with an acknowledgement/interjection token
    "exclamation",          # 22 '!' present
)
N_FEATURES = len(FEATURE_NAMES)

#: length features whose clip floor is fitted at a percentile of the training
#: distribution rather than at the observed minimum (see the note on FEATURE_NAMES)
LENGTH_FEATURES = ("log_chars", "log_tokens")

#: features that are 0/1 indicators; their clip range is always [0, 1], even for an
#: indicator that never fired in training (otherwise the observed max would pin it to 0).
INDICATOR_FEATURES = tuple(
    FEATURE_NAMES.index(n) for n in (
        "has_question_mark", "ends_question", "opens_interrogative", "request_cue",
        "self_predicate", "has_long_number", "has_email", "has_url_or_ip",
        "has_year_or_date", "has_clock_time", "has_month_or_weekday", "has_currency",
        "has_measure_unit", "ack_opener", "exclamation",
    )
)


def extract_surface_features(text: str) -> np.ndarray:
    """Return the ``N_FEATURES`` generic surface features for ``text``.

    Pure stdlib + numpy, no network, no model.  Order matches ``FEATURE_NAMES``
    and is part of the on-disk contract of ``assets/write_classifier.npz``.
    """
    t = (text or "").strip()
    f = np.zeros(N_FEATURES, dtype=np.float32)
    if not t:
        return f
    low = t.lower()
    tokens = _RE_TOKEN.findall(t)
    words = _RE_WORD.findall(low)
    n_words = max(len(words), 1)

    f[0] = math.log1p(len(t)) / 5.0
    f[1] = math.log1p(len(tokens)) / 5.0
    f[2] = (sum(len(x) for x in tokens) / max(len(tokens), 1)) / 10.0

    f[3] = 1.0 if "?" in t else 0.0
    f[4] = 1.0 if t.endswith("?") else 0.0

    first = words[0] if words else ""
    f[5] = 1.0 if first in _INTERROGATIVE_OPENERS else 0.0
    f[6] = 1.0 if (first in _REQUEST_VERBS or "please" in low) else 0.0

    second = sum(1 for w in words if w in _SECOND_PERSON)
    # contractions lose their apostrophe in \w+ tokenisation; count them on the raw text
    second += low.count("you're") + low.count("youre")
    firstp = sum(1 for w in words if w in _FIRST_PERSON)
    firstp += low.count("i'm") + low.count("i've") + low.count("i'll")
    my = sum(1 for w in words if w in ("my", "our", "mine", "ours"))
    f[7] = second / n_words
    f[8] = firstp / n_words
    f[9] = my / n_words
    f[10] = 1.0 if _RE_SELF_PREDICATE.search(t) else 0.0

    n_digits = len(_RE_DIGIT.findall(t))
    f[11] = n_digits / len(t)
    f[12] = 1.0 if _RE_LONG_NUM.search(t) else 0.0
    f[13] = 1.0 if _RE_EMAIL.search(t) else 0.0
    f[14] = 1.0 if (_RE_URL.search(t) or _RE_IP.search(t)) else 0.0
    f[15] = 1.0 if (_RE_YEAR.search(t) or _RE_NUMERIC_DATE.search(t)) else 0.0
    f[16] = 1.0 if _RE_CLOCK.search(t) else 0.0
    f[17] = 1.0 if any(w in _MONTHS or w in _WEEKDAYS for w in words) else 0.0
    f[18] = 1.0 if _RE_CURRENCY.search(t) else 0.0
    f[19] = 1.0 if _RE_UNIT.search(t) else 0.0

    # Capitalised words that do not start a sentence -> proper-noun-ish.
    sent_starts = {m.end() for m in _RE_SENT_START.finditer(t)}
    caps = sum(1 for m in _RE_CAP_WORD.finditer(t) if m.start() not in sent_starts)
    f[20] = caps / n_words

    f[21] = 1.0 if first in _ACK_TOKENS else 0.0
    f[22] = 1.0 if "!" in t else 0.0
    return f


# ---------------------------------------------------------------------------
# Compiled surface-only fallback head.
#
# Emitted by scratch/refound/train_write_classifier.py --emit-fallback so the
# gate keeps its measured behaviour even if assets/write_classifier.npz is not
# installed.  Raw feature space (standardisation already folded in).
# ---------------------------------------------------------------------------
_SURFACE_FALLBACK_W = np.array([
     7.06385994,  # log_chars
     6.79860258,  # log_tokens
     3.26087260,  # avg_token_len
    -1.50079107,  # has_question_mark
    -0.11409295,  # ends_question
    -1.39925086,  # opens_interrogative
    -2.03127432,  # request_cue
    -0.16632578,  # second_person_ratio
    -2.51185846,  # first_person_ratio
     7.04880714,  # possessive_my_ratio
     0.67084736,  # self_predicate
     18.76884270,  # digit_ratio
     1.12765396,  # has_long_number
     2.90146565,  # has_email
     0.00000000,  # has_url_or_ip
     0.93030894,  # has_year_or_date
     0.51543438,  # has_clock_time
     0.04740350,  # has_month_or_weekday
     1.07983160,  # has_currency
     1.60502267,  # has_measure_unit
     13.18102932,  # proper_noun_ratio
    -0.90943617,  # ack_opener
     0.57321066,  # exclamation
], dtype=np.float32)
_SURFACE_FALLBACK_B = -14.15579913
_SURFACE_FALLBACK_THRESHOLD = 0.6
_SURFACE_FALLBACK_TRAINED = True
# Per-feature range seen in training.  Features are clipped to it before scoring, so the
# linear head can never extrapolate a large weight off the end of a feature it only saw in
# a narrow band (e.g. an all-capitalised sentence driving proper_noun_ratio to 1.0).
_FEATURE_LO = np.array([0.923024, 0.599146, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000], dtype=np.float32)
_FEATURE_HI = np.array([1.242922, 0.890869, 0.750000, 1.000000, 1.000000, 1.000000, 1.000000, 0.222222, 0.250000, 0.125000, 1.000000, 0.139535, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.424242, 1.000000, 1.000000], dtype=np.float32)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class WriteClassifier:
    """Decides whether a conversational utterance is worth persisting.

    >>> c = WriteClassifier()
    >>> c.should_store("My blood type is O negative, in case it ever matters.")
    True
    >>> c.should_store("haha that is brilliant")
    False
    """

    #: how long (seconds) an embedder probe result is trusted before re-probing
    PROBE_TTL = 60.0
    #: the offline-fallback check re-encodes the text, which costs ~1 ms per 100 words;
    #: above this length the classifier probes the daemon with a short string instead
    VERIFY_MAX_TOKENS = 120
    #: the string used for that probe
    PROBE_TEXT = "nanomem embedder availability probe"

    def __init__(
        self,
        cache_path: Optional[str] = None,
        embedder: Optional[EmbeddingProvider] = None,
        threshold: Optional[float] = None,
        verify_embedder: bool = True,
    ):
        self.embedder = embedder or EmbeddingProvider()
        if cache_path is None:
            cache_path = os.path.join(os.path.dirname(__file__), "assets", "write_classifier.npz")
        self.cache_path = cache_path
        self.verify_embedder = verify_embedder

        self.w_emb: Optional[np.ndarray] = None
        self.w_surf_full: Optional[np.ndarray] = None
        self.b_full: float = 0.0
        # Deployment decision point, NOT the trainer's. `_load_weights` records
        # the asset's own value in `threshold_trained_full` and leaves this one
        # alone; see DEPLOYMENT_THRESHOLD_FULL.
        self.threshold_full: float = float(DEPLOYMENT_THRESHOLD_FULL)
        self.threshold_trained_full: Optional[float] = None
        self.feat_lo: np.ndarray = _FEATURE_LO.copy()
        self.feat_hi: np.ndarray = _FEATURE_HI.copy()
        self.w_surf: np.ndarray = _SURFACE_FALLBACK_W.copy()
        self.b_surf: float = float(_SURFACE_FALLBACK_B)
        self.threshold_surf: float = float(_SURFACE_FALLBACK_THRESHOLD)
        self.model_info: Dict[str, Any] = {
            "source": "builtin_fallback",
            "has_embedding_head": False,
            "trained": bool(_SURFACE_FALLBACK_TRAINED),
        }
        self._load_weights()
        if threshold is not None:
            self.threshold_full = float(threshold)
            self.threshold_surf = float(threshold)
            self.model_info["threshold_full"] = float(threshold)
            self.model_info["threshold_source"] = "caller"
        self.model_info.setdefault("threshold_full", float(self.threshold_full))
        self.model_info.setdefault("threshold_surface", float(self.threshold_surf))

        self._emb_ok: Optional[bool] = None
        self._emb_probe_at: float = 0.0

    # -- weights ------------------------------------------------------------
    def _load_weights(self) -> None:
        try:
            if not os.path.exists(self.cache_path):
                return
            d = np.load(self.cache_path, allow_pickle=False)
            names = [str(x) for x in d["feature_names"]] if "feature_names" in d else list(FEATURE_NAMES)
            if tuple(names) != FEATURE_NAMES:
                # Asset was built against a different feature contract: ignore it.
                return
            if "feat_lo" in d and d["feat_lo"].size == N_FEATURES:
                self.feat_lo = d["feat_lo"].astype(np.float32).reshape(N_FEATURES)
                self.feat_hi = d["feat_hi"].astype(np.float32).reshape(N_FEATURES)
            self.w_surf = d["w_surface"].astype(np.float32).reshape(N_FEATURES)
            self.b_surf = float(d["b_surface"])
            self.threshold_surf = float(d["threshold_surface"]) if "threshold_surface" in d else 0.5
            if "w_emb" in d and d["w_emb"].size == EMBED_DIM:
                self.w_emb = d["w_emb"].astype(np.float32).reshape(EMBED_DIM)
                self.w_surf_full = d["w_surface_full"].astype(np.float32).reshape(N_FEATURES)
                self.b_full = float(d["b_full"])
                # The asset's threshold is READ but not applied: it was selected
                # on symmetric accuracy/F1 and deployment's errors are not
                # symmetric.  Kept so `inspect()` can still show it and so a
                # caller can reproduce the trainer's own operating point.
                self.threshold_trained_full = (
                    float(d["threshold_full"]) if "threshold_full" in d else None)
            self.model_info = {
                "source": os.path.basename(self.cache_path),
                "has_embedding_head": self.w_emb is not None,
                "trained": True,
                "threshold_full": float(self.threshold_full),
                "threshold_trained_full": self.threshold_trained_full,
                "threshold_source": "nanomem.classifier.DEPLOYMENT_THRESHOLD_FULL",
                "trained_on": str(d["trained_on"]) if "trained_on" in d else "",
                "version": str(d["version"]) if "version" in d else "",
            }
        except Exception:
            # Corrupt or unreadable asset -> keep the compiled fallback head.
            pass

    # -- embedder handling --------------------------------------------------
    def _looks_like_offline_fallback(self, text: str, vec: np.ndarray) -> bool:
        """True if ``vec`` is EmbeddingProvider's md5 n-gram fallback, not a real
        embedding.  The provider swallows connection errors silently, so this is
        the only way to know which encoder actually ran."""
        try:
            off = self.embedder._offline_encode_batch([text])[0]
        except Exception:
            return False
        if off.shape != vec.shape:
            return False
        return float(np.dot(off, vec)) > 0.999

    def _probe_embedder(self) -> bool:
        """Cached liveness+provenance probe, used when the text is too long to verify
        directly.  Costs one short embedding call at most once per ``PROBE_TTL``."""
        now = time.monotonic()
        if self._emb_ok is not None and (now - self._emb_probe_at) < self.PROBE_TTL:
            return self._emb_ok
        try:
            v = np.asarray(self.embedder.embed(self.PROBE_TEXT), dtype=np.float32).reshape(-1)
            ok = bool(v.size == EMBED_DIM
                      and not self._looks_like_offline_fallback(self.PROBE_TEXT, v))
        except Exception:
            ok = False
        self._emb_ok, self._emb_probe_at = ok, now
        return ok

    def _verify_vector(self, text: str, vec: np.ndarray) -> bool:
        """True if ``vec`` is a real embedding rather than the md5 fallback."""
        if not self.verify_embedder:
            return True
        if len(_RE_TOKEN.findall(text)) <= self.VERIFY_MAX_TOKENS:
            ok = not self._looks_like_offline_fallback(text, vec)
            self._emb_ok, self._emb_probe_at = ok, time.monotonic()
            return ok
        return self._probe_embedder()

    def _embedding_for(self, text: str) -> Optional[np.ndarray]:
        """Return an L2-normalised real embedding, or None if unavailable."""
        if self.w_emb is None:
            return None
        now = time.monotonic()
        if self._emb_ok is False and (now - self._emb_probe_at) < self.PROBE_TTL:
            return None                      # daemon was down a moment ago: do not re-dial
        try:
            v = np.asarray(self.embedder.embed(text), dtype=np.float32).reshape(-1)
        except Exception:
            self._emb_ok, self._emb_probe_at = False, now
            return None
        if v.size != EMBED_DIM or not self._verify_vector(text, v):
            self._emb_ok, self._emb_probe_at = False, time.monotonic()
            return None
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else None

    # -- rules --------------------------------------------------------------
    @staticmethod
    def _rule_layer(text: str) -> Optional[Dict[str, Any]]:
        """Three generic rules that pre-empt the head.  Everything else is learned."""
        t = (text or "").strip()
        if len(t) < 3 or not any(c.isalnum() for c in t):
            return {"should_store": False, "reason": "empty_or_too_short"}
        if _RE_FORGET_DIRECTIVE.search(t):
            return {"should_store": False, "reason": "explicit_forget_directive"}
        if _RE_MEMORY_DIRECTIVE.search(t):
            return {"should_store": True, "reason": "explicit_memory_directive"}
        words = _RE_WORD.findall(t)
        if (3 <= len(words) <= SHORT_FACT_MAX_WORDS and "?" not in t
                and words[0].lower() not in _ACK_TOKENS
                and words[0].lower() not in _REQUEST_VERBS
                and "please" not in t.lower()
                and not _RE_NOT_A_FACT.search(t)
                and _RE_SHORT_FACT.search(t)):
            return {"should_store": True, "reason": "short_self_declarative_fact"}
        return None

    # -- public API ---------------------------------------------------------
    def classify(
        self,
        text: str,
        embedding: Optional[Sequence[float]] = None,
        use_rules: bool = True,
    ) -> Dict[str, Any]:
        """Classify ``text``; ``embedding`` may be a vector the caller already has.

        Returns a dict with ``should_store`` (bool), ``prob`` (calibrated
        probability of "store"), ``margin`` (logit), ``reason``, ``model`` and
        ``embedder`` (which encoder actually ran).
        """
        rule = self._rule_layer(text) if use_rules else None
        feats = np.clip(extract_surface_features(text), self.feat_lo, self.feat_hi)

        vec: Optional[np.ndarray] = None
        src = "none"
        if self.w_emb is not None:
            if embedding is not None:
                v = np.asarray(embedding, dtype=np.float32).reshape(-1)
                if v.size == EMBED_DIM and self._verify_vector(text, v):
                    n = float(np.linalg.norm(v))
                    if n > 1e-9:
                        vec, src = v / n, "caller"
                else:
                    src = "offline_fallback"
            elif rule is None:
                vec = self._embedding_for(text)
                src = "provider" if vec is not None else "offline_fallback"

        if vec is not None:
            logit = float(np.dot(self.w_emb, vec) + np.dot(self.w_surf_full, feats) + self.b_full)
            thr, model = self.threshold_full, "logistic_v2_full"
        else:
            logit = float(np.dot(self.w_surf, feats) + self.b_surf)
            thr, model = self.threshold_surf, "logistic_v2_surface"

        prob = _sigmoid(logit)
        out: Dict[str, Any] = {
            "should_store": bool(prob >= thr),
            "prob": prob,
            "margin": float(logit),
            "threshold": float(thr),
            "reason": "learned_head",
            "model": model,
            "embedder": src,
        }
        if rule is not None:
            out["should_store"] = rule["should_store"]
            out["reason"] = rule["reason"]
        return out

    def should_store(self, text: str, embedding: Optional[Sequence[float]] = None) -> bool:
        """True if ``text`` should be written to the vault."""
        return bool(self.classify(text, embedding=embedding)["should_store"])

    def inspect(self, text: str, embedding: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        """Full diagnostics for ``text`` (adds the surface-feature breakdown)."""
        res = self.classify(text, embedding=embedding)
        raw = extract_surface_features(text)
        res["features"] = {k: float(v) for k, v in zip(FEATURE_NAMES, raw)}
        res["features_clipped"] = {k: float(v) for k, v in zip(FEATURE_NAMES, np.clip(raw, self.feat_lo, self.feat_hi))}
        res["model_info"] = dict(self.model_info)
        return res

    # Alias: Vault.inspect_memory() historically called .inspect().
    inspect_memory = inspect


# Backwards-compatible name.  v1 shipped a prototype-similarity "cognitive
# manifold" classifier; the class is gone, the import path is not.
CognitiveManifoldClassifier = WriteClassifier


_GLOBAL_CLASSIFIER: Optional[WriteClassifier] = None


def get_classifier() -> WriteClassifier:
    """Process-wide singleton (weights are loaded once)."""
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None:
        _GLOBAL_CLASSIFIER = WriteClassifier()
    return _GLOBAL_CLASSIFIER
