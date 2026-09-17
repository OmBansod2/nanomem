"""
nanomem.entities
~~~~~~~~~~~~~~~~
Generic entity tagging, query-intent detection and temporal (revision)
resolution. Pure functions over NFKC-normalised lower-cased text; no proper
nouns, brands, breeds, product names or benchmark vocabulary appear anywhere in
this module, and there is deliberately no "capitalised word" fallback (that was
the source of junk revision groups such as ``Update`` in v2).

Two layers produce an entity:

1. **structural classes** -- an ordered table of patterns for things that have a
   recognisable shape or a fixed name (an email address, a phone number, a
   credential, a birthday, ...). First match wins.
2. **slot entities** -- generic possessive / ownership extraction
   (``my <slot> is ...``, ``my <slot>'s name is ...``, ``I use/drive/own a
   <slot>``, ``every <period>``), normalised and then mapped through a synonym
   table so that ``mobile``, ``cell`` and ``phone number`` all land on
   ``phone_number``.

Revision groups are scoped by ``(user_id, project, entity)`` -- the same triple
``Vault.merge`` reconciles on.
"""

import functools
import re
import unicodedata

import numpy as np

# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
_ALIASES = {"favourite": "favorite", "neighbourhood": "neighborhood"}
_MAX_ENTITY_LEN = 48


def normalize_entity(s) -> str:
    """NFKC + lower + non-alphanumerics to ``_``, collapsed, spelling-aliased."""
    if s is None:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).strip().lower()
    for a, b in _ALIASES.items():
        t = t.replace(a, b)
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    t = re.sub(r"_+", "_", t)
    return t[:_MAX_ENTITY_LEN]


def _norm_text(text) -> str:
    return unicodedata.normalize("NFKC", str(text or ""))


# ---------------------------------------------------------------------------
# structural classes (ordered; first match wins)
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://\S+")
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
# The hyphen is inside the character class, so a bare \w boundary let a SERIAL
# number match: a serial like "frame number XYZ-2211-4407" starts its digit run
# right after a hyphen, has eight digits, and sits next to the word "number", so
# it was tagged `phone_number` (and, next to somebody else's possessive,
# `other_phone_number`).
# Excluding an adjacent word character OR hyphen means a digit run welded to a
# letter prefix is not a phone number, while a real number preceded by a space,
# a bracket or "+" still is.
PHONE_RE = re.compile(r"(?<![\w-])\+?\d[\d\s().-]{6,}\d(?![\w-])")
PHONE_WORD_RE = re.compile(r"\b(phone|mobile|cell|cellphone|telephone|whatsapp|tel|number|line|sim)\b")

_STRUCTURAL = (
    ("emergency_contact", re.compile(r"\bemergency contact\b")),
    ("credential", re.compile(
        r"\b(api[\s_-]?key|secret[\s_-]?key|password|passcode|passphrase|"
        r"access[\s_-]?token|auth[\s_-]?token|bearer token|jwt|pin code|private key)\b")),
    ("email_address", EMAIL_RE),
    ("url", URL_RE),
    ("ip_address", IP_RE),
    ("phone_number", None),          # handled specially (needs digits + a cue)
    ("birthday", re.compile(r"\b(birthday|born on|date of birth|dob)\b")),
    ("allergy", re.compile(r"\b(allergic|allerg(?:y|ies))\b")),
    # NOTE: the verb frames below require a first-person subject. Without that
    # guard, "<someone else>'s visits are moving to Saturdays" was tagged
    # `location` and joined the user's address revision group.
    ("location", re.compile(
        r"\b(?:i|we)(?:'m|'ve|'d| am| have| has| are| was| were| will| had)?\s+"
        r"(?:just |finally |officially |recently |now )*"
        r"(?:live|living|lives|stay|staying|moved|moving|move|relocated|relocating)\b"
        r"|\bmy (?:home |new |current |old |previous )?address\b"
        r"|\b(?:new|current|old|previous) address\b"
        r"|\b(?:based in|hometown|neighborhood)\b")),
    ("career", re.compile(
        r"\b(?:i|we)(?:'m|'ve| am| have| are| was| were)?\s+(?:just |finally |now )*"
        r"(?:work|works|working|joined|hired|promoted)\b"
        r"|\bjob title\b|\bmy (?:new |current |old )?(?:title|role|position|employer|company)\b"
        r"|\b(?:my )?(?:role|title|position) is\b|\bpromoted to\b|\bprofession\b")),
    ("name", re.compile(r"\b(my name is|call me|i go by|everyone calls me)\b")),
)

# Two kinds of structural evidence, and they are not equally strong.
#
# A VALUE-SHAPE class recognises the value itself in the text -- an "@", a run of
# phone digits, a credential word next to a token. If it fires, the sentence
# really does carry a value of that kind.
#
# A FRAME class recognises a VERB or PHRASE frame instead ("I moved ...", "I
# work ...", "my name is ..."). Those frames are about actions, and an action
# verb takes any object: "I moved my ACCOUNTS to <bank>", "WE MOVED; my OFFICE
# is on the seventh floor now" and "I moved to <address>" all match the
# `location` frame, but only the last one is about a location. When a frame
# class fires AND the sentence also names its attribute in a possessive slot,
# the SLOT is the better answer -- it is the part of the sentence that says
# WHICH attribute the value belongs to. Measured: 3 of the 5 remaining generic
# revision-probe failures and 1 of 2 remaining round-4 dev-persona failures were
# a frame class overriding an explicit slot
# (scratch/refound/ranking_dev_r4_frame.json).
FRAME_CLASSES = frozenset({"location", "career", "name", "allergy", "birthday",
                           "routine", "emergency_contact"})


def _has_phone_digits(text: str) -> bool:
    for m in PHONE_RE.finditer(text):
        if sum(c.isdigit() for c in m.group(0)) >= 8:
            return True
    return False


def _structural_class(text: str, lowered: str):
    for name, pat in _STRUCTURAL:
        if name == "phone_number":
            if _has_phone_digits(text) and (PHONE_WORD_RE.search(lowered) or re.search(r"\bmy\b", lowered)):
                return "phone_number"
            continue
        if pat.search(text if name in ("email_address", "url", "ip_address") else lowered):
            return name
    return None


# ---------------------------------------------------------------------------
# slot entities
# ---------------------------------------------------------------------------
# WHOSE attribute is this? A possessive is the only structural evidence a
# sentence gives, so the slot patterns below accept ANY possessive and let
# `is_third_party` decide whose it is. 3.0.2 hard-coded "my", so "his phone
# number is ..." and "their address is ..." produced NO entity at all and then
# inherited the user's own entity by anaphora -- a contact's number was stored
# as revision 2 of the user's own phone number and returned as the answer to
# "what is my phone number?" (measured: scratch/refound/third_party_v3r4.json).
# The contraction stems are excluded so "it's name is ..." is not read as a
# possessive.
_CONTRACTION_STEMS = r"it|he|she|that|there|what|who|here|let|one|how|where|when|why"
# NAMED, because WHICH possessive matched is evidence: it is the only thing in
# the sentence that says whose attribute the slot names. Every pattern below
# uses `_POSS` at most once, so the group name is unambiguous per pattern.
_POSS = (r"(?P<poss>\b(?:my|our|his|her|their)\b|"
         r"\b(?!(?:" + _CONTRACTION_STEMS + r")'s\b)[a-z\u00c0-\u024f]+'s)")
# The possessives that make an attribute the VAULT OWNER's. "your" is listed
# because an assistant writing back to the user ("your mobile number is ...")
# restates the USER's own fact; it is not part of `_POSS` itself, which only
# needs to recognise slots, and adding it there would change slot extraction on
# text nobody has measured.
FIRST_PERSON_POSSESSIVES = frozenset({"my", "our", "your"})

# "as" is in the terminator set because "<verb> my <slot> AS <value>" is the
# ordinary English ASSIGNMENT frame -- reissue, list, register, record, rename --
# and it is a copula in exactly the way "is" is. Without it, "they reissued my
# building badge number as <value>" named no attribute at all, the record fell
# through to the anaphora path and inherited the entity of whatever the previous
# chat turn was about (`membership`), and the badge's two values never formed a
# revision group (scratch/refound/temporal_bench_results.json -> by_subtype ->
# current/badge_id).
P1 = re.compile(_POSS + r"\s+(?P<slot>[a-z][a-z'\- ]{1,64}?)\s+(?:is|are|was|were|will be|has been|as|=|:)\s")
P2 = re.compile(_POSS + r"\s+(?P<slot>[a-z][a-z'\- ]{1,40}?)'s name is")
P3 = re.compile(
    r"\bi (?:use|drive|own|have|play|prefer|take|wear|ride)\s+(?:a|an|the|my)?\s*"
    r"(?P<slot>[a-z][a-z'\- ]{1,40}?)(?:\s+(?:named|called|is|that|which)|[,.]|$)")
P4 = re.compile(r"\b(?:every|each) (?:morning|evening|night|day|week|weekend|"
                r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b")
# "changed my cell", "switched my address", "got a new job" -- a change verb taking
# a first-person possessive. Purely structural: the verb list is a closed set of
# replacement verbs, the slot is whatever noun follows.
P5 = re.compile(r"\b(?:changed|change|switched|switch|updated|update|renewed|replaced|"
                r"got|getting|moved|cancelled|canceled)\s+" + _POSS + r"\s+"
                r"(?P<slot>[a-z][a-z'\- ]{1,30}?)(?:\s+(?:is|are|was|were|to|from|and)|[,.;:]|$)")
# "New address: ...", "Current flat, ...", "old number was ..." -- a recency
# determiner introducing the attribute it revises.
P6 = re.compile(r"\b(?:new|current|old|previous|latest|former)\s+"
                r"(?P<slot>[a-z][a-z'\- ]{1,30}?)(?:\s+(?:is|are|was|were|as of)|[:,.\-]|$)")

# Two kinds of modifier, and telling them apart is what stops the ranking layer
# from calling two DIFFERENT attributes one fact.
#
# REVISION_MODIFIERS mark a restatement of the SAME attribute in time ("my NEW
# phone number"), so they are stripped and the restatement lands in the same
# revision group as the original.
#
# DISTINGUISHING_MODIFIERS name a different attribute with the same head noun
# ("my PRIMARY email" vs "my BACKUP email", "my FIRST language" vs "my SECOND
# language"). 3.0.1 stripped these too, so both records were tagged
# ``email_address``, joined one revision group, and the later one was promoted
# over the answer. Measured on the generic adjacent-attribute probes:
# ``scratch/refound/adjacent_attributes_v3r3.json``.
REVISION_MODIFIERS = {"new", "old", "current", "latest", "previous", "original",
                      "former", "updated", "recent", "own", "daily", "usual", "go_to"}
DISTINGUISHING_MODIFIERS = {"first", "second", "third", "primary", "secondary",
                            "main", "backup", "spare", "other", "little", "big",
                            "elder", "younger", "left", "right", "weekday",
                            "weekend", "emergency"}
MODIFIER_STOP = REVISION_MODIFIERS       # what `normalize_slot` strips
# `OTHER_PREFIX` ("other_") is reserved for third-party attributes, so the
# distinguishing modifier "other" is renamed rather than emitted verbatim.
_RESERVED_LEAD = "other"
_RESERVED_LEAD_ALT = "alternate"
MODIFIER_KEEP = {"favorite", "work", "personal", "home"} | DISTINGUISHING_MODIFIERS
GENERIC_HEADS = {"day", "time", "life", "week", "morning", "evening", "thing",
                 "question", "plan", "idea", "point", "turn", "problem", "guess",
                 "one", "stuff", "rule", "update", "news", "way", "bit", "lot",
                 "part", "kind", "sort", "note", "reason", "story", "self"}

SYNONYM_CLASSES = {
    "phone_number": {"phone", "mobile", "cell", "cellphone", "telephone", "number",
                     "whatsapp", "phone_number", "mobile_number", "cell_number",
                     "contact_number", "tel"},
    "email_address": {"email", "e_mail", "mail", "email_address"},
    "career": {"job", "work", "title", "job_title", "role", "position", "employer",
               "company", "career", "profession", "workplace"},
    "location": {"home", "address", "city", "location", "apartment", "house", "flat",
                 "neighborhood", "hometown", "residence", "place", "postcode", "street"},
    "pet": {"dog", "cat", "puppy", "kitten", "pet", "animal", "hamster", "rabbit",
            "parrot", "bird", "fish"},
    "allergy": {"allergy", "allergic", "allergies"},
    "birthday": {"birthday", "birthdate", "dob", "date_of_birth"},
    "credential": {"api_key", "token", "password", "secret", "credential",
                   "passphrase", "pin"},
    "name": {"name", "preferred_name", "nickname", "full_name"},
    "emergency_contact": {"emergency_contact"},
    "routine": {"routine", "schedule"},
    "url": {"url", "website", "site", "link"},
    "ip_address": {"ip", "ip_address"},
}
_SYNONYM_LOOKUP = {}
for _cls, _members in SYNONYM_CLASSES.items():
    _SYNONYM_LOOKUP[_cls] = _cls
    for _m in _members:
        _SYNONYM_LOOKUP[_m] = _cls

SINGLE_VALUED = {"phone_number", "email_address", "location", "career", "credential",
                 "emergency_contact", "birthday", "name", "url", "ip_address", "routine"}


FUNCTION_TOKENS = {"the", "a", "an", "of", "from", "for", "to", "in", "on", "at",
                   "with", "and", "or", "by", "as", "that", "this", "some", "any"}
# An attribute name is a NOUN PHRASE. These are adverbs -- deictic time words and
# the common focus/degree adverbs -- and an adverb is never part of one, so a
# slot that swallowed one has its tail trimmed the same way a trailing function
# word is. Without this, "my personal account JUST before this one" produced the
# entity `personal_account_just`, which matched no record at all and left the
# whole layer silent on that question (measured: boost +0.000,
# scratch/refound/ranking_dev_r5_*.json).
TAIL_ADVERBS = {"just", "only", "really", "actually", "ever", "back", "then",
                "now", "today", "currently", "again", "still", "anymore",
                "recently", "originally", "previously", "formerly", "initially",
                "before", "earlier", "lately", "nowadays", "yet", "already"}


def _singularise(tok: str) -> str:
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


# An INNER possessive inside a slot names whose attribute it is: in "my
# daughter's school" the school is the daughter's, and in "my colleague
# <Firstname> <Lastname>'s desk phone" the desk phone is the colleague's -- the
# leading "my" binds the intervening noun, not the attribute. Keeping
# <possessor> + <attribute> gives one entity per possessor, which both names the
# right attribute and keeps two people's copies of it apart (`daughter_school`
# vs `son_school`).
#
# WHICH possessor. "my colleague <Firstname> <Lastname>'s" is a ROLE NOUN with a
# proper name in APPOSITION: the role is what is grammatically bound to "my", and
# the name merely identifies which colleague. So the possessor is the FIRST token
# of the run -- a common noun taken from the sentence -- and the apposed name is
# dropped. 3.0.3 took the token immediately before the ``'s`` instead, which put
# a PROPER NOUN in an entity name (`<surname>_desk_phone`), something this
# module's docstring promises never happens, and made the two sides ASYMMETRIC:
# the statement kept the possessor while the QUESTION's slot pattern hit its
# 40-character limit, fell back to consuming `<Lastname>'s` as the possessive
# marker, and produced a bare `desk_phone_number` -- so the two never matched and
# the user's OWN number answered every question about the colleague's. Measured:
# 0/10 on that phrasing against 10/10 for a plain cosine scan
# (``scratch/refound/temporal_bench_results.json`` -> ``own_fact_leak``).
_INNER_POSSESSIVE = re.compile(
    r"^(?P<lead>.*?)\b(?P<owner>[a-z\u00c0-\u024f]+)'s\s+(?P<attr>\S.*)$")


def normalize_slot(slot: str):
    """Normalise a raw possessive slot phrase into an entity, or ``None``."""
    raw = unicodedata.normalize("NFKC", str(slot or "")).strip().lower()
    m = _INNER_POSSESSIVE.search(raw)
    if m:
        run = re.findall(r"[a-z\u00c0-\u024f]+", m.group("lead")) or [m.group("owner")]
        slot = f"{run[0]} {m.group('attr')}"
    norm = normalize_entity(slot)
    if not norm:
        return None
    toks = [t for t in norm.split("_") if t]
    while toks and toks[0] in MODIFIER_STOP and toks[0] not in MODIFIER_KEEP:
        toks = toks[1:]
    while toks and (toks[-1] in FUNCTION_TOKENS or toks[-1] in TAIL_ADVERBS):
        toks = toks[:-1]
    if not toks:
        return None
    toks = toks[:3]
    while toks and (toks[-1] in FUNCTION_TOKENS or toks[-1] in TAIL_ADVERBS):
        toks = toks[:-1]
    if not toks:
        return None
    toks = list(toks)
    if toks[0] == _RESERVED_LEAD:
        # "my OTHER phone number" is a DISTINGUISHING_MODIFIER and must stay in
        # the slot -- but `other_phone_number` is exactly the namespace
        # `OTHER_PREFIX` reserves for SOMEBODY ELSE's phone number, so the user's
        # own second number was being tagged (and, through `query_intent`'s
        # slot-first path, asked for) as a third party's. Renaming the modifier
        # keeps the distinction and vacates the namespace.
        toks[0] = _RESERVED_LEAD_ALT
    toks[-1] = _singularise(toks[-1])
    if toks[-1] in GENERIC_HEADS or toks[0] in GENERIC_HEADS:
        return None
    key = "_".join(toks)
    if key in _SYNONYM_LOOKUP:
        return _SYNONYM_LOOKUP[key]
    head = toks[-1]
    if head in _SYNONYM_LOOKUP and len(toks) == 1:
        return _SYNONYM_LOOKUP[head]
    return key


# P2/P1/P5 all require a POSSESSIVE ("<somebody>'s <slot> is ...", "changed my
# <slot>"), so the slot they yield is explicitly attached to an owner. P6 (a bare
# recency determiner: "old <slot> was ...") and P3 ("I use/drive/own a <slot>")
# are weaker -- P6 in particular will happily read "I live in the OLD QUARTER
# near the bakery" as the slot `quarter_near`. Only the possessive group is
# allowed to demote a structural FRAME class.
_POSSESSIVE_PATTERNS = (P2, P1, P5)
_WEAK_SLOT_PATTERNS = (P6, P3)


# A capitalised word that is not opening a sentence is a PROPER NAME. This
# module's docstring promises that no proper noun ever becomes an entity, and it
# is enforced here -- the only place case is looked at, and it is looked at only
# to REJECT, never to invent (v2's "capitalised word" fallback invented junk
# groups such as `Update`, which is why there is no positive rule).
#
# An attribute is named by a common-noun phrase; a phrase containing a proper name
# is identifying a PARTICULAR THING, which is a VALUE, not a kind of fact. "I
# drive a bronze <Make> <Model> roadster" is the value of the car attribute, and
# 3.0.3 tagged it `bronze_<make>_<model>` -- so every car the user has ever owned
# became its OWN attribute, the three statements never formed a revision group,
# and "what car do I drive?" was decided by raw cosine. (The vocabulary-free
# cosine window would have grouped them, but it refuses candidates the tagger has
# positively called different attributes, and here the tagger had.) Measured:
# ``scratch/refound/temporal_bench_results.json`` -> ``by_subtype`` ->
# ``current/vehicle``.
_SENTENCE_BREAK = ".!?;:\n\u2014-\"'\u201c(["
_NAME_TOKEN = re.compile(r"\b[A-Z][A-Za-z\u00c0-\u024f'\u2019-]*")


@functools.lru_cache(maxsize=1024)
def proper_name_tokens(text) -> frozenset:
    """Lower-cased tokens that appear CAPITALISED MID-SENTENCE in ``text``."""
    t = _norm_text(text)
    if t == t.lower():
        # No capitals at all -- the common case for a typed question, and two C
        # string ops instead of a regex pass over every write.
        return frozenset()
    out = set()
    for m in _NAME_TOKEN.finditer(t):
        w = m.group(0)
        if w.lower() == "i":
            continue
        # Walk back over whitespace rather than slicing: slicing the prefix once
        # per capitalised token is quadratic in the length of an ingested
        # paragraph, and this runs on every write.
        j = m.start() - 1
        while j >= 0 and t[j].isspace():
            j -= 1
        if j < 0 or t[j] in _SENTENCE_BREAK:
            continue
        out.add(w.lower())
    return frozenset(out)


def _slot_entity(lowered: str, possessive_only: bool = False,
                 names: frozenset = frozenset()):
    pats = _POSSESSIVE_PATTERNS if possessive_only \
        else _POSSESSIVE_PATTERNS + _WEAK_SLOT_PATTERNS
    for pat in pats:
        m = pat.search(lowered)
        if m:
            ent = normalize_slot(m.group("slot"))
            if ent and names and (set(ent.split("_")) & names):
                continue          # names a VALUE, not an attribute -- see above
            if ent:
                return ent
    if not possessive_only and P4.search(lowered):
        return "routine"
    return None


@functools.lru_cache(maxsize=1024)
def detect_entity(text):
    """Best-effort generic entity for a statement, or ``None``.

    Structural class first, then a possessive/ownership slot. Never guesses from
    capitalisation, so an ordinary sentence with a proper noun in it yields
    ``None`` rather than a junk group.
    """
    if not text:
        return None
    t = _norm_text(text)
    low = t.lower()
    names = proper_name_tokens(t)
    ent = _structural_class(t, low)
    slot = _slot_entity(low, names=names)
    if ent is None:
        ent = slot
    elif ent in FRAME_CLASSES and _frame_loses_to_slot(low, ent, names):
        # A frame class lost to an explicit POSSESSIVE slot: see FRAME_CLASSES.
        ent = _slot_entity(low, possessive_only=True, names=names)
    elif slot and slot != ent and _specialises(slot, ent):
        # The structural class recognises the SHAPE of the value ("...@...", a
        # password-ish token) and is therefore coarse: "my primary email" and "my
        # backup email" both come out `email_address`, "my wifi password" and "my
        # router admin password" both come out `credential`. When the sentence
        # also names the attribute in a possessive slot AND that slot is the same
        # kind of thing, the slot is the better answer: it is the only part of
        # the sentence that says WHICH email or WHICH password.
        # 3.0.1 took the class and stopped, so those pairs shared one revision
        # group and the later one was promoted over the answer -- 3 of the 4
        # remaining failures on the generic adjacent-attribute probes
        # (scratch/refound/adjacent_attributes_v3r3.json).
        ent = slot
    if ent and is_third_party(t):
        return OTHER_PREFIX + ent
    return ent


def _frame_loses_to_slot(lowered: str, cls: str,
                        names: frozenset = frozenset()) -> bool:
    """True when a FRAME class should yield to the sentence's possessive slot."""
    poss = _slot_entity(lowered, possessive_only=True, names=names)
    return bool(poss) and poss != cls and not entities_match(poss, cls)


def _specialises(slot: str, cls: str) -> bool:
    """True when ``slot`` names the same KIND of attribute as the structural class
    ``cls`` but says more about it (``wifi_password`` under ``credential``).

    "Says more" means it carries a QUALIFIER over the same class head
    (:func:`split_qualified`). 3.0.3 looked at the last token alone, so "backup
    email address" was read as the *location* word "address" and did not
    specialise ``email_address`` at all -- both the primary and the backup record
    stayed bare ``email_address``, shared one revision group, and the later one
    was promoted over the answer (``scratch/refound/temporal_bench_results.json``
    -> ``by_subtype`` -> ``adjacent/backup_email_vs_primary_email``, 4/10).
    Reading the longest trailing class run instead gets "email address" right and
    leaves the qualifier "backup" to distinguish the two.
    """
    st = [x for x in normalize_entity(slot).split("_") if x]
    if len(st) < 2:
        return False
    quals, head_cls = split_qualified(slot)
    if head_cls == cls:
        return bool(quals)
    head = _singularise(st[-1])
    return cls == "credential" and head in _CRED_TOKENS


# The reserved namespace for somebody else's attributes. `normalize_slot` can
# never emit an entity that starts with it (see the "other" -> "alternate"
# remap there), so `other_<x>` unambiguously means "not the user's".
OTHER_PREFIX = "other_"
_FIRST_PERSON = re.compile(r"\b(my|mine|i|i'm|im|i've|me|we|our|us)\b")
# "it's" / "that's" / "what's" are contractions, not possessives, so the stems of
# the common English contractions are excluded (see `_CONTRACTION_STEMS` above).
_THIRD_PARTY_POSSESSIVE = re.compile(
    r"\b(?!my\b|our\b|your\b)(?!(?:" + _CONTRACTION_STEMS + r")'s\b)"
    r"[a-z\u00c0-\u024f]+'s\b|\b(?:his|her|their)\b")


def is_first_person(text) -> bool:
    """True when a statement or question carries a first-person marker."""
    return bool(_FIRST_PERSON.search(_norm_text(text).lower()))


def strip_other(entity) -> str:
    """``entity`` without the third-party prefix."""
    e = normalize_entity(entity)
    return e[len(OTHER_PREFIX):] if e.startswith(OTHER_PREFIX) else e


def as_other(entity) -> str:
    """``entity`` in the third-party namespace (idempotent)."""
    e = strip_other(entity)
    return (OTHER_PREFIX + e) if e else ""


def inherit_entity(previous, text):
    """The entity an anaphoric follow-up ("his is ...", "it's now ...") inherits.

    3.0.2 inherited ``previous`` verbatim, which crossed the third-party boundary
    the rest of this module enforces: ``detect_entity`` tags somebody else's
    attribute ``other_<x>`` and ``entities_match`` refuses to merge across the
    prefix, but the anaphora path in ``VaultEngine.add_fact`` never asked whose
    fact the follow-up was. Two chat turns -- the user's own number, then a
    contact's -- put the contact's number in the user's own revision group as
    revision 2, and it was returned at rank 1 for "what is my phone number?"
    despite the LOWER cosine (measured: scratch/refound/third_party_v3r4.json).

    Ownership is taken from the FOLLOW-UP when it states one, and carried over
    from the previous subject when it does not.
    """
    ent = normalize_entity(previous)
    if not ent:
        return None
    if is_third_party(text):
        return as_other(ent)
    if is_first_person(text):
        return strip_other(ent)
    return ent


def _binding_owner(lowered):
    """Is the attribute named by this text somebody ELSE's? ``None`` if unknown.

    THE POSSESSIVE THAT BINDS THE SLOT, not "is there a first-person word
    anywhere in the sentence". The slot pattern that fired already located the
    possessive the attribute hangs off; "my" / "our" / "your" mean the vault's
    owner and anything else ("<somebody>'s", "his", "her", "their") means
    somebody else. A first-person word ELSEWHERE in the sentence is irrelevant:
    "I met <name> today -- her mobile number is ..." carries "I", so the
    whole-sentence test called a contact's number the user's own and filed it as
    a revision of the user's own phone number.

    Returns ``None`` when no slot pattern applies at all, so that
    :func:`is_third_party` falls back to the whole-sentence test rather than
    guess.

    A possessive CHAIN ("my colleague <name>'s desk phone") is not decided here:
    the outer possessive is the user's, and the possessor of the attribute is
    carried in the entity name itself by :func:`normalize_slot`
    (`colleague_desk_phone`), which keeps it distinct from the user's own
    `desk_phone` without leaving the first-person namespace.
    """
    for pat in (P2, P1, P5, Q_SLOT_1, Q_SLOT_2, Q_SLOT_3):
        m = pat.search(lowered)
        if not m:
            continue
        poss = (m.groupdict().get("poss") or "").strip()
        if not poss:
            continue
        return poss not in FIRST_PERSON_POSSESSIVES
    return None


def is_third_party(text) -> bool:
    """True when a statement or question is about somebody else's attribute.

    Structural test only: whose possessive BINDS the attribute
    (:func:`_binding_owner`), falling back -- when no slot pattern applies -- to
    the whole-sentence test 3.0.3 used, a non-first-person possessive with no
    first-person marker anywhere. Such records are tagged ``other_<class>`` so
    that they form their own revision group and can never be returned as the
    answer to "what is MY ...".

    The whole-sentence fallback ALONE was wrong for a possessive chain: "my
    colleague <name>'s desk phone is ..." carries "my", so 3.0.3 called it the
    user's own fact and the user's own number then answered questions about the
    colleague's. Measured: 0/10 on that phrasing against 10/10 for a plain cosine
    scan (``scratch/refound/temporal_bench_results.json``).
    """
    low = _norm_text(text).lower()
    # FAST PATH, and exact: with no non-first-person possessive anywhere, every
    # branch below returns False, and this runs on every write.
    if not _THIRD_PARTY_POSSESSIVE.search(low):
        return False
    owned = _binding_owner(low)
    if owned is not None:
        return owned
    return not _FIRST_PERSON.search(low)


PRONOUN_LED = re.compile(r"^(he|she|it|they|his|her|its|their|that)\b", re.IGNORECASE)


_OWN_POSSESSIVE = re.compile(r"\b(?:my|our)\b")


def is_pronoun_led(text) -> bool:
    """True when a statement opens with a pronoun and needs the session's subject.

    NOT when the sentence already carries a first-person possessive of its own.
    "THEY reissued MY building badge number as ..." opens with a pronoun, but the
    pronoun is an impersonal subject and the sentence says perfectly well whose
    attribute it is about; inheriting the previous turn's subject filed it under
    an unrelated attribute. Anaphora is a last resort for a statement that names
    no owner, and a possessive is exactly the thing that names one.
    """
    t = _norm_text(text).strip()
    if not PRONOUN_LED.match(t):
        return False
    return not _OWN_POSSESSIVE.search(t.lower())


# ---------------------------------------------------------------------------
# query intent
# ---------------------------------------------------------------------------
_INTENT_PHRASES = (
    ("emergency_contact", ("emergency contact",)),
    ("credential", ("api key", "access token", "auth token", "secret key", "private key")),
    ("birthday", ("date of birth",)),
    ("name", ("my name", "who am i", "what do i go by", "call me")),
)
_INTENT_WORDS = (
    ("phone_number", {"phone", "mobile", "cell", "cellphone", "telephone", "number", "whatsapp"}),
    ("email_address", {"email", "mail"}),
    ("allergy", {"allergic", "allergy", "allergies"}),
    ("birthday", {"birthday", "born", "birthdate", "dob"}),
    ("pet", {"pet", "pets", "dog", "dogs", "cat", "cats", "animal", "animals"}),
    ("credential", {"credential", "credentials", "password", "token", "passphrase"}),
    ("location", {"live", "living", "address", "city", "neighborhood", "hometown", "reside"}),
    ("career", {"work", "job", "career", "company", "employer", "employed", "profession",
                "title", "role", "position"}),
    ("routine", {"routine", "schedule"}),
    ("url", {"website", "url"}),
)
# The slot bound is 64, not 40: a POSSESSIVE CHAIN ("my colleague <Firstname>
# <Lastname>'s desk phone number") is longer than a bare attribute phrase, and
# when the bound cut it short the engine backtracked to the INNER possessive and
# silently produced a different, unqualified entity from the one the STATEMENT
# carries. The pattern is non-greedy and still anchored by a possessive and a
# terminator, so a longer bound only ever admits a slot that was being rejected
# outright; `normalize_slot` truncates to three tokens and drops
# :data:`GENERIC_HEADS` either way.
Q_SLOT_1 = re.compile(
    r"\b(?:what|which|who|where|when|how)\b.*?" + _POSS + r"\s+"
    r"(?P<slot>[a-z][a-z'\- ]{1,64}?)(?:\?|$|\s+(?:is|was|are|do|does|did|now|currently|"
    r"originally|these days|again))")
# "Which <A> should go on my <B> now?" -- the QUESTIONED attribute is the noun
# right after the interrogative, not the possessive later in the sentence. 3.0.2
# resolved the possessive first and answered `<B>`.
Q_SLOT_0 = re.compile(
    r"\b(?:what|which)\s+(?P<slot>[a-z][a-z'\- ]{1,40}?)\s+"
    r"(?:should|do|does|did|am|are|is|was|were|will|would|have|has)\b")
Q_SLOT_2 = re.compile(r"\b(?:about|of)\s+" + _POSS + r"\s+(?P<slot>[a-z][a-z'\- ]{1,40}?)(?:\?|$|[,.])")
Q_SLOT_3 = re.compile(r"\bwhat (?P<slot>[a-z][a-z'\- ]{1,40}?) "
                      r"(?:do i|does (?:he|she|they|[a-z\u00c0-\u024f]+)) "
                      r"(?:use|have|has|drive|drives|ride|rides|play|plays|own|owns)")


# FIRST and second person only. 3.0.0 also counted "his"/"her"/"their", which is
# ordinary third-person factoid wording, so document questions were classed as
# personal-memory questions. That is no longer what keeps a document corpus
# exact -- the record-level rule in `VaultEngine._is_revisable` does, and it
# holds whatever the wording (measured: 0/120 either way,
# scratch/refound/exactness_v3r2.json) -- but a question about somebody else is
# still not a question about the user, so the set stays tight. "we"/"us" are out
# too: "us" collides with the lower-cased country abbreviation.
PERSONAL_MARKERS = {"my", "mine", "myself", "i", "me", "im", "our", "ours",
                    "your", "yours"}
_PERSONAL_PHRASES = ("emergency contact", "who am i", "remind me", "do i ", "did i ",
                     "am i ", "was i ", "have i ", "i'm", "i've")


def is_personal_query(query) -> bool:
    """True when a question is about a remembered personal attribute.

    This is half the gate on the entity/temporal layer (the other half is
    :func:`temporal_question`'s ``personal_corpus`` argument: a vault with no
    tagged records has no revision groups at all). Without it a generic document
    query like "how does tokenization work" would pick up the ``career`` intent
    from the bare word "work" and hand a +0.25 boost to unrelated records.
    """
    if not query:
        return False
    low = _norm_text(query).lower()
    if any(p in low for p in _PERSONAL_PHRASES):
        return True
    return bool(set(re.findall(r"[a-z']+", low)) & PERSONAL_MARKERS)


# The value side of the `name` frame: what follows it is the user's own name.
_NAME_VALUE = re.compile(
    r"\b(?:my name is|call me|i go by|everyone calls me)\s+(?P<value>[^.,;!?]{1,60})")
_NAME_STOP = {"the", "a", "an", "just", "only", "actually", "really", "still",
              "mr", "mrs", "ms", "dr", "prof", "sir", "madam"}


def self_name_tokens(text) -> frozenset:
    """Tokens of the name a ``name``-tagged record states, or an empty set.

    Used to answer "is this question about the vault's owner?" from the VAULT's
    own data rather than from any list of names. See
    :meth:`nanomem.engine.VaultEngine._self_name_tokens`.
    """
    m = _NAME_VALUE.search(_norm_text(text).lower())
    if not m:
        return frozenset()
    toks = re.findall(r"[a-z\u00c0-\u024f]{3,}", m.group("value"))
    return frozenset(t for t in toks if t not in _NAME_STOP)


@functools.lru_cache(maxsize=512)
def query_intent(query, self_named: bool = False):
    """The entity a question is asking about, or ``None``.

    Returns ``None`` for questions that are not about a remembered attribute
    (see :func:`is_personal_query`), so document corpora are never re-ranked by
    the personal-attribute logic.

    A question in the THIRD person ("what is <somebody>'s phone number?", "what
    is her address?") resolves into the ``other_`` namespace, which is where
    :func:`detect_entity` puts the statements that answer it. 3.0.2 returned
    ``None`` for every such question -- ``is_personal_query`` was the only gate
    and it looks for first-person markers -- so the whole entity/temporal layer
    was silent, the measured boost was exactly +0.000 and search degenerated to
    raw cosine. Measured: scratch/refound/third_party_v3r4.json.

    A question that names the VAULT'S OWNER ("what is <owner> allergic to?",
    "what keyboard does <owner> use?") is a first-person question written in the
    third person, and ``self_named=True`` says so: the personal gate is skipped
    and the answer stays OUT of the ``other_`` namespace. The engine decides that
    flag from the names its own ``name``-tagged records state
    (:meth:`nanomem.engine.VaultEngine._self_name_tokens`), so no list of names
    appears here and a document corpus -- which has no ``name`` record -- can
    never trigger it.
    """
    if not query:
        return None
    third = is_third_party(query) and not self_named
    if not (third or self_named or is_personal_query(query)):
        return None
    low = _norm_text(query).lower()
    # The POSSESSIVE SLOT comes first, so that a question resolves to the same
    # granularity `detect_entity` gives the statement that answers it. 3.0.1 ran
    # the coarse phrase/word cues first: "what is my WORK address?" matched the
    # word "address" and returned `location`, which is the entity the HOME
    # address record carries, so the home record collected intent_boost +
    # group_hoist (+0.50) and the work record collected nothing. Measured: it was
    # the last adjacent-attribute failure left after the window fix
    # (scratch/refound/adjacent_attributes_v3r3.json).
    out = query_intents(query, self_named)
    return out[0] if out else None


@functools.lru_cache(maxsize=512)
def query_intents(query, self_named: bool = False):
    """Every entity a question could be asking about, best guess first.

    A question often names two attributes -- "which <A> should go on my <B> now?"
    names both -- and only one of them is the one being asked for.
    :func:`query_intent` returns the first guess; the engine calls THIS and picks
    the first candidate its own entity table actually knows, which resolves the
    ambiguity from the vault's data instead of from a rule
    (:meth:`nanomem.engine.VaultEngine._resolve_intent`). 3.0.2 resolved the
    possessive slot unconditionally and answered `invoice`.
    """
    if not query:
        return ()
    third = is_third_party(query) and not self_named
    if not (third or self_named or is_personal_query(query)):
        return ()
    low = _norm_text(query).lower()
    out = []

    def add(ent):
        if not ent:
            return
        e = as_other(ent) if third else ent
        if e and e not in out:
            out.append(e)

    # The POSSESSIVE SLOT comes first, so that a question resolves to the same
    # granularity `detect_entity` gives the statement that answers it. 3.0.1 ran
    # the coarse phrase/word cues first: "what is my WORK address?" matched the
    # word "address" and returned `location`, which is the entity the HOME
    # address record carries, so the home record collected intent_boost +
    # group_hoist (+0.50) and the work record collected nothing. Measured: it was
    # the last adjacent-attribute failure left after the window fix
    # (scratch/refound/adjacent_attributes_v3r3.json).
    for pat in (Q_SLOT_3, Q_SLOT_2, Q_SLOT_1, Q_SLOT_0):
        m = pat.search(low)
        if m:
            add(normalize_slot(m.group("slot")))
    for ent, phrases in _INTENT_PHRASES:
        if any(p in low for p in phrases):
            add(ent)
    words = set(re.findall(r"[a-z]+", low))
    for ent, cues in _INTENT_WORDS:
        if words & cues:
            add(ent)
    if low.strip().startswith("where"):
        add("location")
    return tuple(out)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
def entity_class(entity):
    """Map an entity onto its synonym class, or return the normalised entity.

    The third-party prefix is carried through rather than folded away, so
    ``other_mobile`` and ``other_phone_number`` are the same class while
    ``other_phone_number`` and ``phone_number`` stay different facts.
    """
    e = normalize_entity(entity)
    if not e:
        return None
    if e.startswith(OTHER_PREFIX):
        inner = entity_class(e[len(OTHER_PREFIX):])
        return (OTHER_PREFIX + inner) if inner else e
    if e in _SYNONYM_LOOKUP:
        return _SYNONYM_LOOKUP[e]
    toks = [t for t in e.split("_") if t]
    if toks and toks[-1] in _SYNONYM_LOOKUP and len(toks) == 1:
        return _SYNONYM_LOOKUP[toks[-1]]
    return e


_CRED_TOKENS = {"key", "secret", "token", "auth", "password", "credential"}


@functools.lru_cache(maxsize=1024)
def split_qualified(entity):
    """``(frozenset(qualifiers), class or None)`` for an entity name.

    An English attribute name is a (possibly empty) run of QUALIFIER words
    followed by a HEAD that names the KIND of attribute: "desk phone number" is
    the qualifier "desk" over the head "phone number", "backup email address" is
    "backup" over "email address". :data:`SYNONYM_CLASSES` is the table of heads,
    so the decomposition is: take the LONGEST TRAILING token run that names a
    class, everything before it is the qualifier.

    Two kinds of qualifier are then dropped, because neither names a different
    attribute:

    * a :data:`REVISION_MODIFIERS` token ("my NEW phone number"), which marks a
      restatement in time -- this is what :func:`entities_match` already did; and
    * a token that belongs to THE SAME CLASS AS THE HEAD. "home address" is the
      location word "home" over the location word "address": it re-states the
      class rather than narrowing it, so it denotes the same attribute as a bare
      "address" -- whereas "OFFICE address" narrows it, because "office" is not a
      location word. This is read off the synonym table, not from a list of
      redundant pairs, so it holds for every class at once ("mobile phone
      number" -> ``phone_number``, "home phone number" -> ``home`` + phone
      class, which stays distinct).

    Why it exists: the same attribute is named at DIFFERENT GRANULARITY by a
    statement and by the question that asks for it -- a chat log says "my desk
    phone at the office is ..." and the user asks "what is my desk phone
    NUMBER?". 3.0.3 compared the raw token sets, so ``desk_phone`` and
    ``desk_phone_number`` were different facts; the question then fell through to
    its coarser second candidate ``phone_number`` and the user's MOBILE collected
    the boost. Measured: 0/20 on that sibling pair, and 2/20 on the shared-head
    email pair (``scratch/refound/temporal_bench_results.json`` ->
    ``by_subtype``).
    """
    e = normalize_entity(entity)
    if not e:
        return frozenset(), None
    if e.startswith(OTHER_PREFIX):
        e = e[len(OTHER_PREFIX):]
    toks = [t for t in e.split("_") if t]
    if not toks:
        return frozenset(), None
    cls, quals = None, toks
    for ln in range(len(toks), 0, -1):
        cand = _SYNONYM_LOOKUP.get("_".join(toks[-ln:]))
        if cand:
            cls, quals = cand, toks[:-ln]
            break
    keep = [t for t in quals
            if t not in MODIFIER_STOP
            and not (cls is not None and _SYNONYM_LOOKUP.get(t) == cls)]
    return frozenset(keep), cls


def entities_match(a, b) -> bool:
    """True when two entity names denote the same attribute.

    Equality, the same synonym class, or the credential rule. Both of the looser
    rules 3.0.1 also used were measurably wrong, and both were wrong the same
    way -- they let a QUALIFIED attribute match a differently qualified one:

    * bare token-set containment made ``account`` match ``savings_account``, so
      "where is my savings account?" returned the current account;
    * the credential rule fired on the shared token ``password`` alone, so "what
      is my wifi password?" returned the router admin password.

    A qualifier is now only ignored when it is a :data:`REVISION_MODIFIERS` token
    ("my NEW phone number" is still the same fact as "my phone number"); those
    are stripped by :func:`normalize_slot` before an entity is ever stored, so
    what is left has to agree. Measured: the last 2 of 40 generic
    adjacent-attribute probes (``scratch/refound/adjacent_attributes_v3r3.json``).
    """
    na, nb = normalize_entity(a), normalize_entity(b)
    if not na or not nb:
        return False
    if na.startswith(OTHER_PREFIX) != nb.startswith(OTHER_PREFIX):
        return False              # somebody else's attribute is a different fact
    if na == nb:
        return True
    ca, cb = entity_class(na), entity_class(nb)
    if ca and ca == cb:
        return True
    ta = {t for t in na.split("_") if t and t not in MODIFIER_STOP}
    tb = {t for t in nb.split("_") if t and t not in MODIFIER_STOP}
    if ta == tb:
        return True
    # SAME QUALIFIERS OVER THE SAME CLASS is the same attribute, however much of
    # the head noun each side happened to spell out (see :func:`split_qualified`).
    qa, ka = split_qualified(na)
    qb, kb = split_qualified(nb)
    if ka is not None and ka == kb and qa == qb:
        return True
    # Credentials are named inconsistently ("api key" / "access token"), so two
    # credential names still match -- but only when at least one of them is BARE,
    # i.e. carries no qualifier of its own to contradict the other.
    if (ta & _CRED_TOKENS) and (tb & _CRED_TOKENS):
        if not (ta - _CRED_TOKENS) or not (tb - _CRED_TOKENS):
            return True
        if (ta - _CRED_TOKENS) & (tb - _CRED_TOKENS):
            return True
    return False


@functools.lru_cache(maxsize=512)
def _matching_ids_cached(entity, names_key, names):
    return tuple(i for i, nm in enumerate(names) if entities_match(entity, nm))


def matching_ids(entity, entity_names) -> np.ndarray:
    """int32 ids of interned entity names that match ``entity``."""
    if not entity or not entity_names:
        return np.zeros(0, dtype=np.int32)
    ids = _matching_ids_cached(normalize_entity(entity), len(entity_names), tuple(entity_names))
    return np.asarray(ids, dtype=np.int32)


def is_single_valued(entity) -> bool:
    """True for attributes that have exactly one current value at a time.

    Never true in the third-party namespace: nanomem tags somebody else's phone
    number ``other_phone_number`` without knowing WHICH somebody, so two
    contacts' numbers share one entity and the newer one must not be read as
    superseding the older.
    """
    c = entity_class(entity)
    if not c or c.startswith(OTHER_PREFIX):
        return False
    return c in SINGLE_VALUED or c.startswith("favorite_")


TEMPORAL_CUES = ("current", "currently", "now", "latest", "newest", "updated", "new",
                 "today", "original", "originally", "first", "previous", "previously",
                 "old", "before", "used to", "earlier", "recent", "recently",
                 "these days", "at the moment", "right now", "nowadays", "anymore",
                 "no longer", "instead of", "switched", "since then", "back then",
                 "still", "again")


def has_temporal_cue(query) -> bool:
    """True when the question explicitly asks about a point in an attribute's history."""
    low = _norm_text(query).lower()
    words = set(re.findall(r"[a-z]+", low))
    for cue in TEMPORAL_CUES:
        if " " in cue:
            if cue in low:
                return True
        elif cue in words:
            return True
    return False


# ---------------------------------------------------------------------------
# which point in the history? (the DIRECTION a question asks for)
# ---------------------------------------------------------------------------
# :data:`TEMPORAL_CUES` says a question is about a point in an attribute's
# history. It does NOT say WHICH point -- and 3.0.3 threw that away, because
# ``temporal_direction`` was a caller argument defaulting to "current". A
# question worded "what was my ORIGINAL address?" therefore engaged the ranking
# layer and ordered the group NEWEST-FIRST, which is the exact opposite of what
# it asked for. Measured on the temporal benchmark: 9.0% top-1 on historical
# questions against 22.0% for a plain cosine scan -- the layer was worse than
# doing nothing (``scratch/refound/temporal_bench_results.json`` ->
# ``arms.nanomem_default.by_type``). The shipped ``chat.py`` and ``cli.py`` never
# set ``temporal_direction``, so that WAS the out-of-the-box behaviour.
#
# English marks the direction on the question itself, with a closed class of
# deictic adverbs and adjectives. They split three ways:
#
#   PRESENT       "current", "now", "these days", "latest"   -> the newest value
#   ANTERIOR      "used to", "former", "before", "old"       -> a superseded one
#     of which
#     IMMEDIATE   "previous", "just before", "the one before"-> the one BEFORE
#                                                               the current one
#     INITIAL     "original", "originally", "at first"       -> the FIRST one
#
# Two guards keep this from firing on ordinary wording:
#
# * the WEAK cues ("first", "old", "last", "past", "earlier") are also ordinary
#   attributive adjectives -- "my FIRST name", "my OLD school friend" -- so they
#   count only when the question ALSO carries PAST TENSE on its main verb
#   ("what WAS", "where DID I", "what HAD I"). Reference to a superseded value is
#   marked by past tense in English; reference to a current one is not. The
#   STRONG cues ("original", "previously", "used to", "formerly") are inherently
#   anterior and need no such support.
# * the whole function only ever runs behind :func:`temporal_question` and the
#   engine's personal-corpus gate, so a document corpus never reaches it.
DIRECTION_CURRENT = "current"
DIRECTION_PREVIOUS = "previous"
DIRECTION_OLDEST = "oldest"

# An explicit request from the CALLER. "historical" is 3.0.x's spelling and keeps
# its meaning (the engine's ``historical_mode`` picks the member); "oldest" and
# "previous" name the member directly.
EXPLICIT_HISTORY = frozenset({"historical", "oldest", "previous"})
# ... and the spellings that mean "the newest value, whatever the wording says".
EXPLICIT_PRESENT = frozenset({"present", "newest", "latest", "now"})

# Past tense on the question's own verb. "use to" is deliberately NOT here: it
# is the cue the guard is meant to VALIDATE, so listing it would let the cue
# satisfy its own guard.
_PAST_FRAME = re.compile(r"\b(?:was|were|did|had|used to)\b")
_ANTERIOR_STRONG = re.compile(
    r"\b(?:original|originally|previous|previously|former|formerly|initially|"
    r"prior|preceding|beforehand)\b|\bused to\b|\bback then\b|\bat first\b|"
    r"\bto begin with\b|\bto start with\b|\bno longer\b")
# "use to" is the bare form English takes after "did" ("what did I USE TO
# drive?"); it is weak because the same two words are an ordinary present-tense
# verb plus infinitive ("what do I use to open it?"), which the past-tense guard
# separates.
_ANTERIOR_WEAK = re.compile(
    r"\b(?:first|old|older|last|past|earlier|before)\b|\buse to\b")
# IMMEDIATE anteriority is comparative: it names a value RELATIVE to another one.
# "just/immediately/right before", "the one before", "previous", "prior".
_IMMEDIATE = re.compile(
    r"\b(?:previous|previously|prior|preceding)\b"
    r"|\b(?:just|immediately|right|directly)\s+(?:before|prior)\b"
    r"|\bthe one (?:just |immediately |right )?before\b"
    r"|\bbefore (?:the |this |that |my )?(?:one|current|present|latest|newest|last)\b")
# INITIAL anteriority is superlative/ordinal: the START of the series.
_INITIAL_STRONG = re.compile(
    r"\b(?:original|originally|initially)\b|\bat first\b"
    r"|\bto begin with\b|\bto start with\b|\bvery first\b")
_INITIAL_WEAK = re.compile(r"\bfirst\b|\bto start with\b|\bstarted (?:out |off )?with\b")


def query_direction(query, default: str = DIRECTION_CURRENT):
    """Which point in an attribute's history a question asks for.

    Returns ``"current"``, ``"previous"`` or ``"oldest"``; ``default`` is
    returned when the question carries no anterior cue at all, and
    ``"historical"`` is returned for an anterior question whose cue says nothing
    about WHICH earlier value (so the engine's ``historical_mode`` decides, which
    is what 3.0.3 did for every historical question).

    A PRESENT cue does not beat an ANTERIOR one: in "what was my number just
    before the one I have NOW?" the present cue is the anchor of the comparison,
    not the value being asked for. Anteriority therefore wins whenever both
    appear -- which is also why this is decided by cue CLASS rather than by
    whichever cue happens to come first in the sentence.
    """
    low = _norm_text(query).lower()
    if not low:
        return default
    past = bool(_PAST_FRAME.search(low))
    strong = bool(_ANTERIOR_STRONG.search(low))
    if not strong and not (past and _ANTERIOR_WEAK.search(low)):
        return default
    if _IMMEDIATE.search(low):
        return DIRECTION_PREVIOUS
    if _INITIAL_STRONG.search(low) or (past and _INITIAL_WEAK.search(low)):
        return DIRECTION_OLDEST
    return "historical"


def resolve_direction(temporal_direction, query_text, historical_mode="oldest"):
    """``(historical, mode)`` for one search.

    ``temporal_direction`` is the caller's word, and only two of its values are
    an instruction:

    * anything in :data:`EXPLICIT_HISTORY` -- the caller asked for a superseded
      value by name, and ``"oldest"`` / ``"previous"`` additionally say which;
    * anything in :data:`EXPLICIT_PRESENT` -- the caller wants the newest value
      whatever the question says.

    ``"current"`` (the signature default, so also every caller that never passes
    the argument at all -- ``chat.py``, ``cli.py``, ``Vault.search``) means "not
    specified", and the direction is then read off the QUESTION
    (:func:`query_direction`). That is the fix for 3.0.3 answering "what was my
    original address?" with the current one.
    """
    raw = str(temporal_direction or DIRECTION_CURRENT).strip().lower()
    mode = str(historical_mode or "oldest")
    if raw in EXPLICIT_HISTORY:
        return True, (raw if raw in (DIRECTION_OLDEST, DIRECTION_PREVIOUS) else mode)
    if raw in EXPLICIT_PRESENT:
        return False, mode
    want = query_direction(query_text)
    if want == DIRECTION_CURRENT:
        return False, mode
    if want in (DIRECTION_OLDEST, DIRECTION_PREVIOUS):
        return True, want
    return True, mode


def is_explicit_history(temporal_direction) -> bool:
    """True when the CALLER asked for a superseded value by name."""
    return str(temporal_direction or "").strip().lower() in EXPLICIT_HISTORY


# Sources whose records nanomem treats as entries in a personal log: the entity
# tagger runs on them, and only they (or any record that carries an entity tag,
# whatever its source) may be treated as revisions of one another. A document
# ingested by `Vault.ingest_file` carries the file name as its source and is
# therefore never permuted against another paragraph -- which is what keeps
# document retrieval identical to an exhaustive cosine scan.
PERSONAL_SOURCES = frozenset({"chat_session", "chat", "user_input", "cli",
                              "proxy", "api", "rest_api",
                              # `mcp.py` writes with source="mcp_client" by
                              # default, and until 0.6.6 that was not in this
                              # set -- so the tagger never ran on the MCP path,
                              # no revision group ever formed, and FOUR of the
                              # seven tools that server exposes were degraded or
                              # dead: `nanomem_history` reported a real chain as
                              # "has one value and has never changed", and
                              # `nanomem_volatility` could only ever return
                              # nothing, because `volatility()` excludes records
                              # with no entity. An MCP client writing a user's
                              # facts IS a chat session; it just has an agent in
                              # the middle.
                              "mcp_client", "mcp"})

GROUP_SEP = "\x1f"


def make_group_key(user_id, project, entity) -> str:
    """The revision-group key: ``"<user_id>\\x1f<project>\\x1f<normalised entity>"``.

    Returns ``""`` when there is no entity (such records never auto-revision).
    """
    ent = normalize_entity(entity)
    if not ent:
        return ""
    uid = str(user_id).strip() if user_id is not None else ""
    proj = str(project).strip() if project is not None else ""
    return f"{uid}{GROUP_SEP}{proj}{GROUP_SEP}{ent}"


# ---------------------------------------------------------------------------
# temporal resolution
# ---------------------------------------------------------------------------
INTENT_BOOST = 0.25         # a record tagged with the entity the question named
GROUP_HOIST = 0.25          # ... and in the revision group the question named
REVISION_LEAD = 0.20        # ... and the member of that group the question asked for
# INTENT_BOOST + GROUP_HOIST + REVISION_LEAD is the LARGEST amount any boost can
# add to a score (engine.VaultEngine._max_boost, published as
# stats()['max_boost'] = 0.70). EVERY entity term is ADDITIVE, so for a
# cosine-scored vault `0 <= hit['score'] - hit['cosine'] <= stats()['max_boost']`
# holds for every hit -- which is the score contract DECISIONS #8 makes binding.
# 3.0.1 instead PERMUTED a group's scores into temporal order, handing one record
# another record's score: measured excess +0.6402 against a published cap of
# 0.50, and scores that fell BELOW their own cosine. Measured excess under this
# build over the same probes: 0.6113, inside the 0.70 cap
# (scratch/refound/adjacent_attributes_v3r3.json).
#
# REVISION_LEAD is a CAP, not a tuned weight: `apply_revision_lead` raises the
# wanted revision to a hair above the best score already held inside its own
# group and no further. Because the engine first levels the entity boosts across
# the group, the spread it has to close is just the cosine spread, which is at
# most WINDOW_DELTA -- so 0.20 covers it with margin and is never the binding
# constraint. Measured: 0.20 and 0.30 score identically on every set.
#
# GROUPING WINDOWS. These two are UNCHANGED from 3.0.1 (0.06 / 0.16). The
# adjacent-attribute regression 3.0.1 shipped was not a threshold problem and
# re-tuning does not fix it: at WINDOW_DELTA = 0.05 the adjacent probes do reach
# 40/40, but the revision probes fall 12 -> 9, the 3-persona chat set falls
# 31 -> 21 of 36 and the golden chat vault falls 12 -> 10 of 12
# (adjacent_attributes_v3r3.json -> threshold_grid). What fixes it is refusing to
# group records the tagger has already called different attributes -- see
# `VaultEngine._cosine_window`, `_specialises` and `entities_match` above.
GROUP_COS_DELTA = 0.06      # a TAGGED revision must be within this cosine of its group's best
WINDOW_DELTA = 0.16         # UNTAGGED competing statements: cosine window below the leader
WINDOW_MAX = 4              # ... capped at this many candidates
WINDOW_SIM = 0.60           # ... and they must look like each other by this much
# The cosine window used for a candidate that explicitly ANNOUNCES a revision
# ("I moved to ...", "my new ...", "changed to ..."). Widening it for marked
# candidates alone is SAFE -- 15 of 16 generic revision probes carry a marker on
# the newer statement and 0 of 40 adjacent-attribute probes carry one at all
# (scratch/refound/marker_separation_v3r4.json) -- but it is not
# USEFUL: measured at 0.18 / 0.20 / 0.22 / 0.30 it wins nothing on the revision
# probes and costs up to 2 of 36 on the 3-persona chat set and 1 of 24 on the
# round-4 dev personas (scratch/refound/ranking_dev_H_wide*.json,
# ranking_dev_r4_current.json). It therefore ships EQUAL to WINDOW_DELTA, i.e.
# off, and remains a constructor knob so the ablation is reproducible.
WINDOW_DELTA_MARKED = WINDOW_DELTA
# WINDOW_SIM is also unchanged. Round 3 briefly moved it to 0.55 because 0.55 and
# 0.60 tied on the 3-persona selection set and 0.55 won one generic revision
# probe; measured once on the 2-persona HELD-OUT set, that cost two of 24 answers
# (15/24 at 0.55 vs 17/24 at 0.60). A tie on the tuning set is not a reason to
# move a threshold.
#
# All four were re-swept in round 4 over FIVE sets at once -- 40 generic
# adjacent-attribute probes, 16 generic revision probes, the 3-persona
# out-of-sample selection chat set, a NEW 3-persona dev chat set written for
# round 4, and the migrated golden chat vault -- and the shipped point is the
# best on all five simultaneously: adjacent-attribute top-1 40/40 (exactly plain
# cosine; 3.0.1 scored 30/40), revision top-1 14/16 (plain cosine 3/16; 3.0.2
# scored 11/16), historical 16/16, selection chat 33/36 top-1 and 36/36 top-3,
# dev chat 23/24 top-1 and 24/24 top-3, golden 12/12. Grid, ablations and
# per-case failures: scratch/refound/ranking_dev_r4_*.json.


def apply_intent_boost(cos, entity_id, intent_ids, mask=None, weight=INTENT_BOOST):
    """Additive bonus for candidates whose stored entity matches the question's intent."""
    cos = np.asarray(cos, dtype=np.float32)
    boost = np.zeros(cos.shape[0], dtype=np.float32)
    if weight and intent_ids is not None and len(intent_ids):
        boost[np.isin(np.asarray(entity_id, dtype=np.int32),
                      np.asarray(intent_ids, dtype=np.int32))] += float(weight)
    return boost


def resolve_top_entity(scored, entity_id, entity_names, intent=None):
    """The entity the answer is about: the question's intent, else the best hit's."""
    if intent:
        return intent
    if scored.size == 0:
        return None
    best = int(np.argmax(scored))
    eid = int(np.asarray(entity_id)[best])
    return entity_names[eid] if 0 <= eid < len(entity_names) else None


def temporal_applies(top_entity, query_text, temporal_direction, intent=None) -> bool:
    """Whether the revision ordering should decide which member of a group surfaces.

    True when the caller asked for history explicitly, when the question carries a
    temporal cue, when the attribute is single-valued by class, or when the
    question is a personal-memory question at all -- in a personal memory two
    statements of the same fact ARE revisions of it, and the most recent one is
    what "now" means.

    This function looks only at the wording. Whether the layer runs at all is
    decided by :func:`temporal_question`, which additionally requires the vault to
    hold entity-tagged records.
    """
    if is_explicit_history(temporal_direction):
        return True
    if has_temporal_cue(query_text):
        return True
    if is_single_valued(top_entity):
        return True
    return bool(intent) and is_personal_query(query_text)


def temporal_question(query_text, temporal_direction, intent=None,
                      personal_corpus: bool = True) -> bool:
    """Gate on the whole entity/temporal layer: is this a question about *when*?

    Three conditions, and all of the automatic ones must hold:

    * an explicit ``temporal_direction="historical"`` always engages the layer --
      the caller asked for history by name;
    * otherwise the vault must actually hold entity-tagged records
      (``personal_corpus``), because a corpus of documents has no revisions to
      resolve; and
    * the question must be about the user (:func:`is_personal_query`).

    Wording alone is NOT enough, which is what 3.0.0 got wrong: "first", "new",
    "before", "still", "again" and "original" are ordinary factoid wording, and on
    the 1,190-paragraph validation corpus with its 120 real questions the layer
    fired on 18/120 and re-ordered their top-4 (14/120 at rank 1). The decisive
    guard is the DATA, not the question: ``personal_corpus`` here, and inside the
    engine the rule that only a revisable record (entity-tagged, or written by a
    chat session) may join a revision group. Measured after both:
    0/120 (``scratch/refound/exactness_v3r2.json``).
    """
    if is_explicit_history(temporal_direction):
        return True
    if not personal_corpus:
        return False
    return (intent is not None or is_personal_query(query_text)
            or has_temporal_cue(query_text))


REVISION_MARKERS = ("changed", "change", "changing", "switched", "switch", "switching",
                    "updated", "update", "moved", "moving", "renamed", "replaced",
                    "replacing", "cancelled", "canceled", "now", "new", "instead",
                    "as of", "from now on", "these days", "no longer", "latest",
                    "finally", "again", "started",
                    # Round 4: the same closed class of English replacement verbs,
                    # extended after the generic revision probes showed three
                    # restatements that announce a change in words this list did
                    # not contain. No corpus-specific vocabulary; the separation
                    # this list buys is measured in
                    # scratch/refound/marker_separation_v3r4.json.
                    "rescheduled", "reschedule", "upgraded", "upgrade",
                    "downgraded", "swapped", "swap", "transferred", "reassigned",
                    "relocated", "renewed", "superseded", "supersedes")
_MARKER_WORDS = frozenset(m for m in REVISION_MARKERS if " " not in m)
_MARKER_PHRASES = tuple(m for m in REVISION_MARKERS if " " in m)


def has_revision_marker(text) -> bool:
    """True when a statement announces that it supersedes an earlier one.

    A closed class of English change/recency markers ("changed", "switched",
    "now", "new", "as of", "no longer", ...) -- the same kind of list as
    :data:`TEMPORAL_CUES`, and no more corpus-specific. It is the strongest
    signal available for "which of these two statements is the current value",
    because a chat log rarely repeats a fact without flagging the change.
    """
    low = _norm_text(text).lower()
    if any(p in low for p in _MARKER_PHRASES):
        return True
    return bool(set(re.findall(r"[a-z]+", low)) & _MARKER_WORDS)


def temporal_order(rev, ts, rows, historical: bool, marks=None,
                   historical_mode: str = "previous") -> np.ndarray:
    """``rows`` ordered by which revision the caller asked for.

    "current": records that announce a change first, then by ``(revision,
    timestamp)`` descending -- newest wins.

    "historical": ``historical_mode="previous"`` returns the revision
    immediately BEFORE the newest first (what "my previous address" / "the one
    before I switched" means), with the newest pushed to the back;
    ``"oldest"`` returns strict ascending order instead.
    """
    """``rows`` re-ordered by preference: newest-first, or oldest-first for history.

    Ordering is lexicographic on ``(revision, timestamp)``, which is why v2's junk
    groups -- whose revisions all tie at 1 -- still resolve correctly by time.
    """
    rows = np.asarray(rows)
    rv = np.asarray(rev)[rows]
    tv = np.asarray(ts)[rows]
    if marks is None:
        keys = (tv, rv)
    else:
        keys = (tv, rv, np.asarray(marks, dtype=np.int8)[rows])
    order = np.lexsort(keys)                  # ascending = oldest / unmarked first
    if not historical:
        order = order[::-1]
    elif str(historical_mode) == "previous":
        desc = order[::-1]                    # newest first
        order = np.concatenate([desc[1:], desc[:1]])
    return rows[order]


def group_relevance_floor(rows, cos, delta: float = GROUP_COS_DELTA) -> np.ndarray:
    """Keep only the group members that are plausible answers to THIS question.

    Two statements of the same fact sit at a comparable cosine from a question
    about that fact. A member far below the group's best is not a revision of the
    answer, it merely carries the same tag -- promoting it (which an unrestricted
    permutation would do) is how a commute note outranked an actual address.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size < 2 or delta is None or delta <= 0:
        return rows
    c = np.asarray(cos)[rows]
    return rows[c >= (c.max() - float(delta))]


LEAD_EPS = 1e-6


def apply_revision_lead(final: np.ndarray, rows, rev, ts, historical: bool,
                        marks=None, historical_mode: str = "previous",
                        weight: float = REVISION_LEAD) -> np.ndarray:
    """ADD just enough to make the requested revision lead its own group.

    The group is ordered by :func:`temporal_order`; the wanted member is raised
    to a hair above the best score anyone in the group already holds, and nobody
    else is touched. The lift is clamped at ``weight``, so:

    * it is an ADDITION -- ``score >= cosine`` and
      ``score - cosine <= stats()['max_boost']`` for every hit, which is the
      contract DECISIONS #8 makes binding. 3.0.1 PERMUTED the group's scores
      instead, handing one record another record's score: measured excess
      +0.6402 against a published cap of 0.50, and scores that fell BELOW their
      own cosine (``scratch/refound/adjacent_attributes_v3r3.json`` -> "v3.0.1").
    * it is MINIMAL -- the winner is lifted to the top of its own group and no
      further, so it can never displace a record that already outscores the
      whole group. A flat ``+weight`` could.

    The lift needed is bounded by the group's internal score spread. The engine
    LEVELS the entity boosts across the group first, so what is left to close is
    just the cosine spread, at most ``window_delta`` (0.16) -- which is why the
    shipped ``weight`` (``REVISION_LEAD`` = 0.20) is a hard cap rather than a
    tuned value and is not the binding constraint. Measured over every lift this
    release's five ranking sets apply: 72 calls, maximum lift 0.1563, cap reached
    0 times (``scratch/refound/ranking_dev_r4_shipped.json`` ->
    ``revision_lead_applied``). 3.0.2's docstring quoted 0.3928 against a 0.20
    clamp, which is arithmetically impossible -- stale round-2 text for a
    ``weight`` of 0.50 that never shipped.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size < 2 or not weight:
        return final
    order = temporal_order(rev, ts, rows, historical, marks, historical_mode)
    win = int(order[0])
    need = float(np.max(final[rows])) - float(final[win]) + LEAD_EPS
    if need > 0:
        final[win] += min(need, float(weight))
    return final


def permute_group_scores(final: np.ndarray, rows, rev, ts, historical: bool,
                         marks=None, historical_mode: str = "previous") -> np.ndarray:
    """DEPRECATED (3.0.1 behaviour). Re-assign a group's own scores by rank.

    A permutation hands one record another record's score, so the result is no
    longer ``cosine + bounded boosts``: measured ``score - cosine`` reached
    +0.6402 against a published cap of 0.50, and some records scored BELOW their
    own cosine. The engine uses :func:`apply_revision_lead` instead; this is kept
    for one release so the regression is reproducible from the library itself
    (``tests/test_adjacent_attributes.py`` re-creates the old ranking with it).
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size < 2:
        return final
    scores = np.sort(final[rows])[::-1]
    for slot, r in enumerate(temporal_order(rev, ts, rows, historical, marks, historical_mode)):
        final[int(r)] = scores[slot]
    return final


def apply_temporal_boosts(cos, entity_id, rev, ts, entity_names, query_text,
                          temporal_direction="current", mask=None, intent_ids=None,
                          intent=None):
    """Back-compatible wrapper: intent boost plus the entity-group revision order.

    Returns ``(boost, top_entity)`` where ``boost`` is what must be ADDED to the
    base score. Callers that can supply candidate vectors should instead use
    :func:`apply_intent_boost` + :func:`permute_group_scores`, which also catches
    revision groups that the tagger never labelled.
    """
    cos = np.asarray(cos, dtype=np.float32)
    n = cos.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32), None
    mask = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    boost = apply_intent_boost(cos, entity_id, intent_ids, mask)
    if not mask.any():
        return boost, None
    scored = np.where(mask, cos + boost, -np.inf)
    top_entity = resolve_top_entity(scored, entity_id, entity_names, intent)
    if not top_entity:
        return boost, None
    explicit = is_explicit_history(temporal_direction)
    if intent is None and not explicit and not is_personal_query(query_text):
        return boost, top_entity
    if not temporal_applies(top_entity, query_text, temporal_direction, intent):
        return boost, top_entity
    ids = matching_ids(top_entity, entity_names)
    if ids.size == 0:
        return boost, top_entity
    group = np.flatnonzero(mask & np.isin(np.asarray(entity_id, dtype=np.int32), ids))
    if group.size < 2:
        return boost, top_entity
    final = scored.copy()
    permute_group_scores(final, group, rev, ts, explicit)
    return (final - np.where(mask, cos, 0.0)).astype(np.float32), top_entity


# ---------------------------------------------------------------------------
# migration repair
# ---------------------------------------------------------------------------
_CLASSLIKE = re.compile(r"^[a-z0-9_]+$")


def derive_entity_for_migrated(meta, text):
    """Entity for a record coming out of a v2 file.

    A v2 ``metadata['entity']`` is kept only when it already looks like a class
    tag (``^[a-z0-9_]+$``). Anything else -- v2's capitalised-phrase guesses --
    is discarded and re-derived from the text with :func:`detect_entity`, so the
    group column, ``Vault.merge`` and the temporal resolver all agree. The
    original value is preserved by the caller as ``metadata['entity_v2']``.
    """
    raw = (meta or {}).get("entity")
    if raw is not None:
        s = str(raw).strip()
        if s and _CLASSLIKE.match(s):
            return normalize_entity(s)
    return detect_entity(text)


__all__ = [
    "normalize_entity", "normalize_slot", "detect_entity", "is_pronoun_led",
    "query_intent", "query_intents", "entity_class", "entities_match",
    "matching_ids",
    "is_single_valued", "has_temporal_cue", "make_group_key",
    "query_direction", "resolve_direction", "is_explicit_history",
    "split_qualified", "DIRECTION_CURRENT", "DIRECTION_PREVIOUS",
    "DIRECTION_OLDEST", "EXPLICIT_HISTORY", "EXPLICIT_PRESENT",
    "FIRST_PERSON_POSSESSIVES",
    "apply_temporal_boosts", "apply_intent_boost", "resolve_top_entity",
    "temporal_applies", "temporal_question", "temporal_order", "permute_group_scores",
    "derive_entity_for_migrated", "is_personal_query",
    "FRAME_CLASSES", "is_third_party", "is_first_person", "inherit_entity", "strip_other",
    "as_other", "self_name_tokens", "OTHER_PREFIX", "group_relevance_floor", "GROUP_COS_DELTA",
    "REVISION_MODIFIERS", "DISTINGUISHING_MODIFIERS", "WINDOW_DELTA_MARKED",
    "WINDOW_DELTA", "WINDOW_MAX", "WINDOW_SIM",
    "has_revision_marker", "REVISION_MARKERS",
    "SYNONYM_CLASSES", "SINGLE_VALUED", "TEMPORAL_CUES", "GROUP_SEP",
    "PERSONAL_SOURCES",
    "INTENT_BOOST", "GROUP_HOIST", "REVISION_LEAD", "apply_revision_lead",
    "PERSONAL_MARKERS",
]
