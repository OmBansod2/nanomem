"""Generic entity tagging, query intent, revision ordering -- and no fixture vocabulary.

WHY THIS FILE MUST NEVER READ ``scratch/refound`` AGAIN
-------------------------------------------------------
The fixture-contamination guard at the bottom of this file used to build its
forbidden vocabulary by OPENING the chat fixtures -- including
``clean_chat_benchmark_heldout.json`` -- every time it ran. Two things were
wrong with that, and both bit:

1. QUARANTINE. Held-out and scored-once fixtures are supposed to stay pristine,
   and ``python3 -m pytest`` is run dozens of times a day, by people and by
   agents, for reasons that have nothing to do with this guard. So the cheapest
   possible action in the repository silently touched the material that experiments
   depend on staying untouched. Worse, several agents in one session stated "I
   did not open the held-out set" in good faith while having run the suite; the
   claim was false and nobody could tell, because a test suite is not where
   anyone looks for fixture access. Note that an access timestamp does not settle
   it either -- reads do not reliably bump ``atime`` on this filesystem. The only
   proof of non-access that holds is BY CONSTRUCTION: the code never names the
   path. That is what this file now does, and
   ``test_this_module_cannot_open_a_quarantined_fixture`` checks it mechanically
   rather than asking to be believed.

2. PACKAGING. ``scratch/refound`` is not part of the distributed package, so in a
   clean checkout the guard skipped itself entirely and the strongest audit in
   the suite silently became a no-op.

The vocabulary is therefore extracted ONCE, offline, by
``tests/data/fixture_vocab_source.py`` into ``tests/data/fixture_vocab.txt``,
which is committed, tiny, and holds keyed digests rather than words -- so the
asset cannot leak the persona vocabulary it exists to police, and cannot
contaminate a reader. The guard hashes the tokens it finds in library source and
looks them up. It skips only when that asset is missing.

If you are adding a fixture: run the extractor by hand and commit the regenerated
table. Do not point a test at the fixture directory.
"""

import hashlib
import os
import re

import numpy as np
import pytest

from nanomem import entities as E

LIB_FILES = ["engine.py", "entities.py", "container.py", "arena.py", "routing.py",
             "crypto.py", "legacy_v2.py", "vault.py", "errors.py", "screen.py"]
PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nanomem")

# Names that only ever appeared in a benchmark fixture. None of them may be
# reachable from library code -- the v2 engine hard-coded several of them.
FIXTURE_TOKENS = ["barnaby", "ergodox", "disco elysium", "catan", "golden retriever",
                  "terraforming mars", "keychron", "royal enfield", "simba",
                  "koregaon", "verdant grid", "nidhi", "rohan mehta"]


def test_no_fixture_vocabulary_in_library():
    hits = []
    for name in LIB_FILES:
        path = os.path.join(PKG, name)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read().lower()
        for token in FIXTURE_TOKENS:
            if token in text:
                hits.append((name, token))
    assert not hits, hits


@pytest.mark.parametrize("text,want", [
    ("My phone number is 020-4455-7788.", "phone_number"),
    ("Update: my new phone number is 1234567890", "phone_number"),
    ("my mobile is +49 176 4471 0298", "phone_number"),
    ("You can reach me at someone@example.org", "email_address"),
    ("The portfolio lives at https://example.com/work", "url"),
    ("The router sits on 192.168.1.14", "ip_address"),
    ("My emergency contact is my sister, 98220-11223.", "emergency_contact"),
    ("The API key is sk-not-a-real-key-000", "credential"),
    ("I'm allergic to shellfish.", "allergy"),
    ("My birthday is 14 March 1993.", "birthday"),
    ("I live in the old quarter near the bakery.", "location"),
    ("I moved to a flat on the north side.", "location"),
    ("New address: 187 Some Street, second floor left", "location"),
    ("Current flat, for the record: 61 Another Street", "location"),
    ("I work as a data engineer.", "career"),
    ("I got promoted to lead engineer.", "career"),
    ("My name is Alex.", "name"),
    ("My dog is a beagle called Bo.", "pet"),
    ("My favourite board game is the one with trains.", "favorite_board_game"),
    ("Changed my cell. New one is 902-580-2247", "phone_number"),
    ("I walk every evening at seven.", "routine"),
])
def test_detect_entity_positive(text, want):
    assert E.detect_entity(text) == want


@pytest.mark.parametrize("text", [
    "my day was great",
    "New rule from the doctor: no coffee after four",
    "I bought a second-hand motorcycle in matte green.",
    "Thanks, that was helpful.",
    "The meeting ran long again.",
    "Gate code changed again, it's a new one now.",
])
def test_detect_entity_negative(text):
    assert E.detect_entity(text) is None


def test_no_capitalised_name_fallback():
    """v2 turned any capitalised phrase into an entity; that is gone."""
    assert E.detect_entity("Anna Karenina is a long novel.") is None
    assert E.detect_entity("Update: nothing much happened.") is None


def test_third_party_scoping():
    mine = E.detect_entity("If you need it, my cell is 902-471-8836.")
    theirs = E.detect_entity("Chloe's cell, for the trip: 902-403-7715.")
    assert mine == "phone_number"
    assert theirs == "other_phone_number"
    assert not E.entities_match(mine, theirs)


@pytest.mark.parametrize("a,b", [("phone", "phone_number"), ("number", "phone_number"),
                                 ("key", "api_key"), ("address", "location"),
                                 ("mobile", "cell"), ("dog", "pet")])
def test_entities_match_table(a, b):
    assert E.entities_match(a, b)


def test_entities_do_not_match():
    assert not E.entities_match("phone_number", "email_address")
    assert not E.entities_match("location", "career")
    assert not E.entities_match("", "phone_number")


def test_normalize_entity():
    assert E.normalize_entity(" Favourite Video-Game ") == "favorite_video_game"
    assert E.normalize_entity(None) == ""
    assert len(E.normalize_entity("x" * 100)) == 48


@pytest.mark.parametrize("query,want", [
    ("What is my current phone number?", "phone_number"),
    ("Where do I live now?", "location"),
    ("What is my job title?", "career"),
    ("Who is my emergency contact?", "emergency_contact"),
    ("What am I allergic to?", "allergy"),
    ("When is my birthday?", "birthday"),
    ("Tell me about my dog", "pet"),
    ("What keyboard do I use?", "keyboard"),
])
def test_query_intent(query, want):
    assert E.query_intent(query) == want


def test_query_intent_ignores_document_questions():
    """A corpus question must not pick up a personal intent from a bare word."""
    for q in ["how does tokenization work", "what is a transformer attention head",
              "evaluating language models", "context window limits"]:
        assert E.query_intent(q) is None
        assert E.is_personal_query(q) is False


def test_group_key_and_single_valued():
    k = E.make_group_key("u", "p", "Phone Number")
    assert k == "u\x1fp\x1fphone_number"
    assert E.make_group_key("u", "p", None) == ""
    assert E.is_single_valued("phone_number") and E.is_single_valued("favorite_book")
    assert not E.is_single_valued("pet")


@pytest.mark.parametrize("a,b", [
    ("account", "savings_account"),          # 3.0.1 matched these by containment
    ("wifi_password", "router_admin_password"),   # ... and these by the credential rule
    ("email_address", "backup_email"),
    ("location", "work_address"),
    ("laptop", "work_laptop"),
])
def test_entities_that_must_not_match(a, b):
    """A qualifier the other name does not carry means a DIFFERENT attribute.

    Deliberate narrowing, and it costs something: "my game" no longer matches
    "my favourite video game" either. The evidence is in
    ``scratch/refound/adjacent_attributes_v3r3.json`` -- every one of these pairs
    produced a measured wrong answer at rank 1 under 3.0.1 ("where is my savings
    account?" returned the current account; "what is my wifi password?" returned
    the router admin password).
    """
    assert not E.entities_match(a, b)
    assert not E.entities_match(b, a)


def test_revision_modifiers_are_still_stripped():
    """The other half of the rule: a RESTATEMENT must keep matching the original."""
    for a, b in [("new_phone_number", "phone_number"), ("old_address", "address"),
                 ("current_job", "job"), ("latest_laptop", "laptop")]:
        assert E.normalize_slot(a.replace("_", " ")) is not None
        assert E.entities_match(E.normalize_slot(a.replace("_", " ")),
                                E.normalize_slot(b.replace("_", " ")))


def test_distinguishing_modifiers_are_kept():
    assert E.normalize_slot("primary email") != E.normalize_slot("backup email")
    assert E.normalize_slot("first language") != E.normalize_slot("second language")
    assert E.normalize_slot("new phone number") == E.normalize_slot("phone number")


def test_specific_slot_beats_the_structural_class():
    """The value's SHAPE is coarse; the sentence's own slot says which one it is."""
    assert E.detect_entity("My primary email is a.b@example.test.") == "primary_email"
    assert E.detect_entity("My backup email is c.d@example.test.") == "backup_email"
    assert E.detect_entity("My email is e.f@example.test.") == "email_address"
    assert E.detect_entity("My wifi password is thistle-moor-41.") == "wifi_password"
    # ... but a restatement of the same attribute still lands on the same entity
    assert E.detect_entity("My new phone number is 0300 555 4820.") == "phone_number"


def test_query_intent_matches_statement_granularity():
    """A question resolves at the same granularity as the statement answering it."""
    assert E.query_intent("What is my work address?") == "work_address"
    assert E.detect_entity("My work address is 44 Steelmill Lane.") == "work_address"
    assert not E.entities_match(E.query_intent("What is my work address?"), "location")


def test_revision_lead_is_additive_and_bounded():
    """`apply_revision_lead` ADDS; it never hands a record another record's score."""
    rev = np.array([1, 1], dtype=np.int32)
    ts = np.array([10.0, 20.0], dtype=np.float64)
    rows = np.array([0, 1], dtype=np.int64)
    final = np.array([0.90, 0.80])
    E.apply_revision_lead(final, rows, rev, ts, historical=False,
                          historical_mode="oldest", weight=0.20)
    assert final[1] > final[0]                       # the later statement leads
    assert final[0] == 0.90                          # nobody else is touched
    assert 0.0 < final[1] - 0.80 <= 0.20             # bounded, and minimal
    # The cap binds rather than overshooting.
    final = np.array([0.90, 0.10])
    E.apply_revision_lead(final, rows, rev, ts, historical=False,
                          historical_mode="oldest", weight=0.20)
    assert final[1] == pytest.approx(0.30)
    assert final[1] < final[0]                       # a bounded lead cannot leap 0.80


def test_temporal_order_and_permutation():
    rev = np.array([1, 1, 2], dtype=np.int32)
    ts = np.array([10.0, 20.0, 5.0], dtype=np.float64)
    rows = np.array([0, 1, 2], dtype=np.int64)
    assert list(E.temporal_order(rev, ts, rows, historical=False,
                                 historical_mode="oldest")) == [2, 1, 0]
    assert list(E.temporal_order(rev, ts, rows, historical=True,
                                 historical_mode="oldest")) == [0, 1, 2]
    final = np.array([0.9, 0.5, 0.1])
    E.permute_group_scores(final, rows, rev, ts, historical=False,
                           historical_mode="oldest")
    assert list(final) == [0.1, 0.5, 0.9]                # ranks kept, order swapped


def test_group_relevance_floor():
    cos = np.array([0.60, 0.56, 0.40], dtype=np.float32)
    rows = np.array([0, 1, 2], dtype=np.int64)
    kept = E.group_relevance_floor(rows, cos, 0.06)
    assert list(kept) == [0, 1]


def test_has_temporal_cue():
    assert E.has_temporal_cue("what is my current address")
    assert E.has_temporal_cue("what was it before")
    assert E.has_temporal_cue("what do I use these days")
    assert not E.has_temporal_cue("what is my address")


def test_derive_entity_for_migrated():
    assert E.derive_entity_for_migrated({"entity": "phone_number"}, "anything") == "phone_number"
    assert E.derive_entity_for_migrated({"entity": "Update"},
                                        "Update: my new phone number is 1234567890") == "phone_number"
    assert E.derive_entity_for_migrated({"entity": "Some Proper Noun"},
                                        "I bought a bicycle yesterday.") is None


# --- mechanical version of the audit above ---------------------------------
# The hard-coded list is a floor, not a guarantee: round 3 shipped two tokens in
# a test fixture that also appear in the 3-persona chat set, and round 4 shipped
# three comments whose example wording came from the HELD-OUT chat set. So the
# forbidden set is derived from the fixtures rather than written down by hand --
# but it is derived ONCE, offline, into the digest table next to this file, and
# NEVER at run time. See the module docstring for why that distinction is the
# whole point of this section.

_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "data", "fixture_vocab.txt")
_VOCAB_PERSON = b"nanomemv"        # must match tests/data/fixture_vocab_source.py
_MIN_VOCAB = 300                   # the table shipped with 467; a short one is broken

# Ordinary modern English and technical terms that some fixture happens to
# capitalise or to use as a slug component, and that this library needs for its
# own reasons. Every entry must be a COMMON noun or a TECHNICAL term; a name
# never belongs here, and adding one defeats the test.
#   whatsapp/maps  - brand names that became the ordinary word for a category of
#                    personal attribute, the way "biro" or "hoover" did; they are
#                    synonyms in `entities.SYNONYM_CLASSES` on their own merits.
#   okay/robotics  - ordinary modern English, simply absent from the 1913-derived
#                    /usr/share/dict/words the harvest subtracted.
#   memtable       - nanomem's own storage vocabulary, which reached a golden
#                    migration fixture as a slug component (`engine.py` x18).
_ALLOW = {"json", "email", "wifi", "maps", "steps", "sim", "savings", "locks",
          "whatsapp", "saturdays", "benchmark", "info", "html", "toml", "yaml",
          "nov", "fri", "wed", "mar", "numpy", "numeric", "capitalised",
          "licence", "substring", "okay", "robotics", "memtable"}

# KNOWN OVERLAP, not an exemption. `temporal_bench.json` and the committed
# `tests/data/adjacent_attributes_source.py` were written from the same pool of
# invented surnames, so two of them are in both. That is precisely the round-3
# leak class, caught here for real -- but the file it is in is test DATA, not
# library code, and renaming the names means regenerating
# `tests/data/adjacent_attributes.npz` alongside. Until its owner does that, the
# overlap is pinned to that one file and to those two tokens: the test below
# still fails if either token reaches any library or application file, and fails
# if a THIRD name joins them. Delete this set when the names are changed.
_TEST_DATA_OVERLAP_FILE = os.path.join("tests", "data", "adjacent_attributes_source.py")
_TEST_DATA_OVERLAP = {"calder", "fenwick"}


def _digest(token):
    return hashlib.blake2s(token.encode("utf-8"), digest_size=8,
                           person=_VOCAB_PERSON).hexdigest()


def _fixture_digests():
    """The baked table, or ``None`` when the asset is not checked out."""
    if not os.path.exists(_VOCAB):
        return None
    with open(_VOCAB, encoding="utf-8") as fh:
        rows = [ln.strip() for ln in fh]
    return {r for r in rows if r and not r.startswith("#")} or None


def _tokens(text):
    """Every word-shaped run in ``text``, lower-cased.

    Splitting on non-alphanumerics means a name is found inside an identifier or
    a slug (``royal_enfield``, ``desk-phone``), not only between spaces -- the
    ``\\b`` search this test used before could not see either, because ``_`` is a
    word character to ``re``. Both splits are unioned so an accented name is
    matched whole AND in the truncated form a byte-oriented split would produce.
    """
    low = text.lower()
    out = set(re.split(r"[^0-9a-zÀ-ɏ]+", low))
    out |= set(re.split(r"[^0-9a-z]+", low))
    return {t for t in out if len(t) >= 4}


def _scan(text, forbidden, extra_allow=()):
    """``[(line number, token)]`` for every fixture proper noun in ``text``."""
    allow = _ALLOW | set(extra_allow)
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for t in sorted(_tokens(line) - allow):
            if _digest(t) in forbidden:
                hits.append((i, t))
    return hits


def _guard_targets():
    """Everything the fixture vocabulary must stay out of: CODE AND PROSE ALIKE."""
    root = os.path.dirname(PKG)
    paths = [os.path.join(PKG, f) for f in sorted(os.listdir(PKG)) if f.endswith(".py")]
    paths += [os.path.join(root, n) for n in ("server.py", "chat.py", "demo.py")]
    paths.append(os.path.join(root, _TEST_DATA_OVERLAP_FILE))
    return [p for p in paths if os.path.exists(p)]


def test_fixture_vocabulary_asset_is_well_formed():
    forbidden = _fixture_digests()
    if forbidden is None:
        pytest.skip("tests/data/fixture_vocab.txt is not checked out")
    assert len(forbidden) >= _MIN_VOCAB, len(forbidden)
    assert all(re.fullmatch(r"[0-9a-f]{16}", d) for d in forbidden), "not a digest table"
    header = [ln for ln in open(_VOCAB, encoding="utf-8") if ln.startswith("#")]
    assert any("excluded (quarantined" in ln for ln in header), header
    assert any(ln.startswith("# source:") for ln in header), header
    # The table really is fixture vocabulary and not, say, an empty file with a
    # header: names this repository has ALREADY committed in plain text, in
    # `FIXTURE_TOKENS` above, must be in it.
    for token in ("keychron", "simba", "koregaon", "nidhi", "mehta", "enfield"):
        assert _digest(token) in forbidden, token
    # ... and it is not so broad that it forbids ordinary English.
    for token in ("phone", "address", "number", "search", "vault", "memory"):
        assert _digest(token) not in forbidden, token


def test_no_chat_fixture_proper_nouns_in_library():
    forbidden = _fixture_digests()
    if forbidden is None:
        pytest.skip("tests/data/fixture_vocab.txt is not checked out")
    hits = []
    for path in _guard_targets():
        rel = os.path.relpath(path, os.path.dirname(PKG))
        allow = _TEST_DATA_OVERLAP if rel == _TEST_DATA_OVERLAP_FILE else ()
        text = open(path, encoding="utf-8", errors="ignore").read()
        hits += ["%s:%d [%s]" % (rel, n, t) for n, t in _scan(text, forbidden, allow)]
    assert hits == [], "fixture vocabulary in shipped source:\n" + "\n".join(hits)


def test_the_pinned_test_data_overlap_has_not_spread():
    """The two pinned names may live in that one test-data file and nowhere else."""
    forbidden = _fixture_digests()
    if forbidden is None:
        pytest.skip("tests/data/fixture_vocab.txt is not checked out")
    assert all(_digest(t) in forbidden for t in _TEST_DATA_OVERLAP), _TEST_DATA_OVERLAP
    spread = []
    for path in _guard_targets():
        rel = os.path.relpath(path, os.path.dirname(PKG))
        if rel == _TEST_DATA_OVERLAP_FILE:
            continue
        found = _tokens(open(path, encoding="utf-8", errors="ignore").read())
        spread += [(rel, t) for t in sorted(found & _TEST_DATA_OVERLAP)]
    assert spread == [], spread


def test_the_guard_still_fails_on_a_reintroduced_fixture_name():
    """POSITIVE CONTROL. A guard that cannot fail is not a guard.

    Every probe is a name this repository already committed in plain text, in
    ``FIXTURE_TOKENS`` at the top of this file, so the control adds no fixture
    vocabulary of its own. Each shape below is one the old ``\\b``-anchored
    search could miss.
    """
    forbidden = _fixture_digests()
    if forbidden is None:
        pytest.skip("tests/data/fixture_vocab.txt is not checked out")
    probes = sorted({w for tok in FIXTURE_TOKENS for w in tok.split()
                     if _digest(w) in forbidden})
    assert len(probes) >= 4, probes
    for p in probes:
        shapes = [
            "It replaced the %s I had before, they said." % p.capitalize(),  # mid-sentence capital
            'ENTITY = "%s_desk_phone"' % p,                                  # slug, underscores
            'SLOT = "%s-desk-phone"' % p,                                    # slug, hyphens
            "ENTITY_%s_PHONE = 1" % p.upper(),                               # identifier, shouting
            "# the %s case, see the dev fixtures" % p,                       # a comment
            "return cfg.%s.value" % p,                                       # an attribute path
        ]
        for line in shapes:
            assert _scan(line, forbidden) == [(1, p)], (p, line)
    # ... and it stays quiet on text that carries no fixture vocabulary.
    assert _scan("the quick brown fox jumps over the lazy dog\n"
                 "phone number, work address, emergency contact", forbidden) == []


def test_this_module_cannot_open_a_quarantined_fixture():
    """MECHANICAL AUDIT of the rule in the module docstring, not a promise.

    Parses this file and checks that no *executable* string constant names a
    fixture, a JSON file or the scratch tree -- docstrings and comments are
    exempt, because they are the only places the ban may be discussed. If this
    fails, someone has wired a path back in and the whole suite is reading
    quarantined material again.
    """
    import ast
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    tree = ast.parse(src)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node) is not None:
            docstrings.add(id(node.body[0].value))
    # This function's own body is exempt: the needles it searches FOR are string
    # constants, so it would otherwise always flag itself.
    me = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "test_this_module_cannot_open_a_quarantined_fixture")
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings \
                and not (me.lineno <= node.lineno <= me.end_lineno):
            low = node.value.lower()
            if ".json" in low or "refound" in low or "scratch" in low:
                bad.append((node.lineno, node.value[:60]))
    assert bad == [], bad
    # The exemption is one function wide, and the audit still has teeth: a path
    # added anywhere else in the module is caught.
    assert len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and me.lineno <= n.lineno <= me.end_lineno]) == 1
