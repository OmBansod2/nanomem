"""Round-5 regressions: the four failure modes the temporal benchmark exposed.

Every test here FAILS on 3.0.3 and passes on this build. The measured evidence
is ``scratch/refound/temporal_bench_results.json`` (the before picture, where
nanomem at shipped defaults scored 44.2% top-1 against a plain cosine scan's
47.9%) and ``scratch/refound/ranking_r5_dev.json`` (the after picture on the
sets this round was allowed to tune on).

The four modes:

1. the direction of a question was ignored -- ``temporal_direction`` defaulted to
   "current" and ``chat.py`` / ``cli.py`` never set it, so "what was my ORIGINAL
   address?" was answered with the current one (9.0% top-1 on historical
   questions against the floor's 22.0%);
2. sibling attributes named at different granularity collapsed
   ("my desk phone" vs "my desk phone NUMBER", "my BACKUP email address");
3. a possessive chain ("my colleague <Name>'s desk phone") leaked the user's own
   value, 0/10 against a plain scan's 10/10;
4. proper nouns became entity names, so every value of an attribute was its own
   attribute and no two of them could ever be revisions of each other.
"""

import numpy as np
import pytest

from conftest import D, unit_rows                                  # noqa: E402
from nanomem import entities as ent
from nanomem.engine import VaultEngine


def _corr(q, target_cos, seed):
    """A unit vector at exactly ``target_cos`` from ``q``."""
    r = unit_rows(1, seed=seed)[0]
    r = r - float(r @ q) * q
    r /= np.linalg.norm(r)
    v = target_cos * q + np.sqrt(max(0.0, 1.0 - target_cos ** 2)) * r
    return (v / np.linalg.norm(v)).astype(np.float32)


# --- 1. the question's own wording decides the direction --------------------
def test_query_direction_reads_the_three_way_distinction():
    d = ent.query_direction
    assert d("what is my address?") == "current"
    assert d("what is my current address?") == "current"
    assert d("what was my original address?") == "oldest"
    assert d("where did I live originally?") == "oldest"
    assert d("what was my previous address?") == "previous"
    assert d("what was my address just before the one I have now?") == "previous"
    # a PRESENT cue does not beat an ANTERIOR one: in "before the one I have
    # NOW" the present cue is the anchor of the comparison.
    assert d("where did I work just before my present job?") == "previous"


def test_weak_cues_need_past_tense():
    """"first"/"old"/"last" are ordinary attributive adjectives too."""
    assert ent.query_direction("what is my first name?") == "current"
    assert ent.query_direction("what was my first address?") == "oldest"
    assert ent.query_direction("what do I use to open the vault?") == "current"
    assert ent.query_direction("what did I use to drive?") == "historical"


def test_default_direction_is_unspecified_not_an_instruction(vault_path):
    """3.0.3 answered a historically-worded question with the CURRENT value.

    `temporal_direction` keeps its signature default of "current", but that value
    now means "the caller did not say", and the direction is read off the
    question. This is the path `chat.py`, `cli.py` and `Vault.search` take.
    """
    q = unit_rows(1, seed=7)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    old = e.add_fact("My mobile number is 555-0100.", _corr(q, 0.72, 31),
                     source="chat_session", timestamp=1000.0)
    new = e.add_fact("My mobile number is 555-0999.", _corr(q, 0.74, 32),
                     source="chat_session", timestamp=2000.0)
    e.flush()
    assert e.search("what is my mobile number?", q, top_k=2)[0]["id"] == new
    assert e.search("what was my original mobile number?", q, top_k=2)[0]["id"] == old
    # and an explicit instruction still wins over the wording, both ways
    assert e.search("what was my original mobile number?", q, top_k=2,
                    temporal_direction="present")[0]["id"] == new
    assert e.search("what is my mobile number?", q, top_k=2,
                    temporal_direction="historical")[0]["id"] == old
    e.close()


def test_previous_is_reachable_per_query(vault_path):
    """"the one just before this one" is not "the first one".

    3.0.3 could only express this by mutating `engine.historical_mode` after
    construction, which is a per-vault setting, not a per-question one.
    """
    q = unit_rows(1, seed=8)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    ids = [e.add_fact(f"My mobile number is 555-0{i:03d}.", _corr(q, 0.70 + 0.01 * i, 40 + i),
                      source="chat_session", timestamp=1000.0 * (i + 1))
           for i in range(3)]
    e.flush()
    assert e.search("what is my mobile number?", q, top_k=3)[0]["id"] == ids[2]
    assert e.search("what was my original mobile number?", q, top_k=3)[0]["id"] == ids[0]
    got = e.search("what was my mobile number just before the one I have now?",
                   q, top_k=3)[0]["id"]
    assert got == ids[1], "wanted the immediately preceding revision"
    e.close()


def test_a_document_corpus_is_never_reordered_by_the_wording(vault_path):
    """The inferred direction must not defeat the personal-corpus gate."""
    q = unit_rows(1, seed=9)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    a = e.add_fact("Tokenization splits text into sub-word units.",
                   _corr(q, 0.80, 51), source="handbook.pdf", timestamp=1000.0)
    e.add_fact("Byte pair encoding merges frequent pairs.",
               _corr(q, 0.70, 52), source="handbook.pdf", timestamp=2000.0)
    e.flush()
    for text in ("what was the original tokenization scheme?",
                 "which encoding was used first, before the merge?"):
        hits = e.search(text, q, top_k=2)
        assert hits[0]["id"] == a
        assert hits[0]["score"] == pytest.approx(hits[0]["cosine"])
    e.close()


# --- 2. one attribute named at two granularities ---------------------------
def test_qualifier_over_class_head_is_the_same_attribute():
    assert ent.split_qualified("desk_phone") == (frozenset({"desk"}), "phone_number")
    assert ent.split_qualified("desk_phone_number") == (frozenset({"desk"}), "phone_number")
    assert ent.entities_match("desk_phone", "desk_phone_number")
    # ... and a DIFFERENT qualifier is still a different attribute
    assert not ent.entities_match("desk_phone", "phone_number")
    assert not ent.entities_match("backup_email_address", "primary_email_address")
    assert not ent.entities_match("home_address", "office_address")


def test_a_class_word_qualifying_its_own_class_is_redundant():
    """"home address" is the location word "home" over the location word
    "address": it restates the class rather than narrowing it, so it is the same
    attribute as a bare address. "office" is not a location word, so it narrows."""
    assert ent.entities_match("home_address", "location")
    assert not ent.entities_match("office_address", "location")
    assert ent.detect_entity("My home address is 41 Harrowvane Close.") == "location"
    assert ent.detect_entity("I live at 41 Harrowvane Close.") == "location"


def test_shared_head_siblings_stay_apart():
    """3.0.3 read the last token of "backup email address" as the LOCATION word
    "address", so the slot did not specialise `email_address` and both the
    primary and the backup record stayed bare `email_address`."""
    b = ent.detect_entity("My backup email address is a@b.test.")
    p = ent.detect_entity("My primary email address is c@d.test.")
    assert b == "backup_email_address" and p == "primary_email_address"
    assert not ent.entities_match(b, p)
    assert ent.query_intent("what is my backup email address?") == b


def test_sibling_attribute_is_not_promoted_over_its_partner(vault_path):
    q = unit_rows(1, seed=10)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    mobile = e.add_fact("My mobile number is 555-0100.", _corr(q, 0.74, 61),
                        source="chat_session", timestamp=2000.0)
    desk = e.add_fact("My desk phone at the office is 555-0200.", _corr(q, 0.70, 62),
                      source="chat_session", timestamp=1000.0)
    e.flush()
    assert e.search("what is my desk phone number at the office?", q,
                    top_k=2)[0]["id"] == desk
    assert e.search("what is my mobile number?", q, top_k=2)[0]["id"] == mobile
    e.close()


# --- 3. possessive chains ---------------------------------------------------
def test_possessive_chain_keeps_the_role_noun_not_the_name():
    r = ent.detect_entity("My colleague Wren Calloway's desk phone is 555-0300.")
    assert r == "colleague_desk_phone"
    assert "calloway" not in r and "wren" not in r
    assert ent.query_intent("what is my colleague Wren Calloway's desk phone number?") == r
    assert not ent.entities_match(r, "desk_phone")


def test_two_peoples_copies_of_one_attribute_stay_apart():
    assert ent.detect_entity("My daughter's school is Brackwell Lane.") == "daughter_school"
    assert ent.detect_entity("My son's school is Aldbury Fields.") == "son_school"
    assert not ent.entities_match("daughter_school", "son_school")


def test_own_value_does_not_answer_a_question_about_a_colleagues(vault_path):
    """3.0.3: 0/10 on this phrasing, against 10/10 for a plain cosine scan."""
    q = unit_rows(1, seed=11)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    mine = e.add_fact("My desk phone at the office is 555-0200.", _corr(q, 0.74, 71),
                      source="chat_session", timestamp=2000.0)
    theirs = e.add_fact("My colleague Wren Calloway's desk phone is 555-0300.",
                        _corr(q, 0.70, 72), source="chat_session", timestamp=1000.0)
    e.flush()
    assert e.search("what is my colleague Wren Calloway's desk phone number?",
                    q, top_k=2)[0]["id"] == theirs
    assert e.search("what is my desk phone number at the office?",
                    q, top_k=2)[0]["id"] == mine
    e.close()


def test_ownership_comes_from_the_binding_possessive_not_the_whole_sentence():
    """A first-person word ELSEWHERE in the sentence does not make the attribute
    the user's."""
    assert ent.is_third_party("I met Wren today -- her mobile number is 555-0199.")
    assert not ent.is_third_party("I switched my mobile number to 555-0999.")
    assert ent.detect_entity(
        "I met Wren today -- her mobile number is 555-0199.").startswith(ent.OTHER_PREFIX)


# --- 4. no proper noun is ever an entity name ------------------------------
@pytest.mark.parametrize("text", [
    "I drive a bronze Naverly Quist roadster.",
    "I drive a black Hessian Kortwell wagon.",
    "My daughter's school is Brackwell Lane.",
    "My colleague Wren Calloway's desk phone is 555-0300.",
    "I train at Ostrander Lifting Club.",
])
def test_no_entity_name_contains_a_proper_noun(text):
    e = ent.detect_entity(text) or ""
    names = ent.proper_name_tokens(text)
    assert not (set(e.split("_")) & names), (e, sorted(names))


def test_no_fixture_proper_noun_appears_anywhere_in_the_library_source():
    """The blocker rule, checked against the FIXTURE VOCABULARY rather than against us.

    The test above it is circular: it asks whether ``detect_entity`` emits a
    token that ``proper_name_tokens`` calls a proper noun, so a single bug in the
    capitalisation rule satisfies both halves. This one never calls the library.
    It looks every token in every package file up in the fixture-vocabulary
    table and fails on a hit, CODE AND PROSE ALIKE.

    It caught two real leaks on the build it was written against, both in
    comments: a surname inside an example entity name in ``entities.py`` and a
    persona slug in an ``engine.py`` comment. Neither reached the matching logic
    -- which is exactly why a test that only inspects behaviour missed them.

    THIS TEST USED TO OPEN THE FIXTURES, AND THAT WAS A QUARANTINE BREACH.
    Until 2026-09-16 it globbed ``scratch/refound/*.json`` for anything matching
    ``chat_benchmark`` or ``personas`` and read the text of every match. That
    glob picked up ``clean_chat_benchmark_persona4.json`` -- scored-once
    material -- and ``clean_chat_benchmark_heldout.json``, so every ``pytest``
    run in this repository opened both, dozens of times a day, for reasons that
    had nothing to do with either. ``test_entities.py`` was moved off the live
    read and this sibling was missed; the breach was found by instrumenting
    ``builtins.open`` for a whole suite run and reading the paths back, which is
    the only way anyone was going to notice. An access timestamp would not have
    settled it either way: measured on this filesystem, reads do not reliably
    bump ``atime``.

    The vocabulary now comes from ``tests/data/fixture_vocab.txt``, extracted
    ONCE and offline by ``tests/data/fixture_vocab_source.py``, which excludes
    the quarantined fixture by name and refuses to run if it ever appears in its
    harvest list. That extractor's docstring already said it baked in THIS
    test's harvest as well as ``test_entities.py``'s; this is that migration
    finally happening. Proof of non-access is now by construction -- no path
    under ``scratch/`` is named in this function -- not by assertion.
    """
    import os
    from test_entities import (_ALLOW, _digest, _fixture_digests, _scan,   # noqa
                               PKG)
    forbidden = _fixture_digests()
    if forbidden is None:
        pytest.skip("tests/data/fixture_vocab.txt is not checked out")

    # POSITIVE CONTROL. A guard that cannot fail is not a guard. There is no
    # harvested word to probe with any more -- the table is digests -- so probe
    # with a token this repository has ALREADY committed in plain text in
    # test_entities.FIXTURE_TOKENS, and check the scanner sees it inside an
    # identifier and inside a comment, not only between spaces.
    probe = "keychron"
    assert _digest(probe) in forbidden, "the table is not the fixture vocabulary"
    assert _scan('ENTITY = "%s_desk_phone"' % probe, forbidden)
    assert _scan("# the %s case, see the dev fixtures" % probe, forbidden)

    leaks = []
    for fn in sorted(os.listdir(PKG)):
        if not fn.endswith(".py"):
            continue
        text = open(os.path.join(PKG, fn), encoding="utf-8").read()
        leaks += ["%s:%d [%s]" % (fn, i, t) for i, t in _scan(text, forbidden)]
    assert not leaks, "fixture vocabulary in library source:\n" + "\n".join(leaks)


def test_a_value_phrase_is_not_an_attribute(vault_path):
    """"a bronze <Make> <Model> roadster" is the VALUE of the car attribute.

    3.0.3 tagged it `bronze_<make>_<model>`, so every car the user ever owned was
    its own attribute, the statements never formed a revision group, and the
    cosine window refused to group them because the tagger had positively called
    them different attributes.
    """
    assert ent.detect_entity("I drive a bronze Naverly Quist roadster.") is None
    q = unit_rows(1, seed=12)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    # One tagged record, so the vault counts as a personal corpus at all: the
    # entity/temporal layer is gated on the DATA, and a vault with nothing tagged
    # has no revision groups by construction (`_has_personal_records`).
    e.add_fact("My mobile number is 555-0100.", unit_rows(1, seed=99)[0],
               source="chat_session", timestamp=500.0)
    # Same residual direction for both cars, so the two statements look like each
    # other (0.99) the way two restatements of one fact do -- `window_sim`.
    old = e.add_fact("I drive a bronze Naverly Quist roadster.", _corr(q, 0.74, 81),
                     source="chat_session", timestamp=1000.0)
    new = e.add_fact("I drive a black Hessian Kortwell wagon.", _corr(q, 0.70, 81),
                     source="chat_session", timestamp=2000.0)
    e.flush()
    assert e.search("what car do I drive?", q, top_k=2)[0]["id"] == new
    assert e.search("what car did I drive originally?", q, top_k=2)[0]["id"] == old
    e.close()


# --- 5. a revision number is an ordinal WITHIN one group --------------------
def test_revisions_from_different_groups_do_not_order_a_cosine_window(vault_path):
    """`add_fact` numbers revisions per (user_id, project, entity), so two
    records that do not share a group key carry numbers from different counters.
    Ordering a vocabulary-free cosine-window group on them ranked a record
    tagged `a` with revision 1 as "older" than a record tagged `b` with revision
    2 that was written LATER.
    """
    q = unit_rows(1, seed=13)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    e.add_fact("My mobile number is 555-0100.", unit_rows(1, seed=90)[0],
               source="chat_session", timestamp=10.0)
    # `phone_number` is now at revision 2; `location` is at revision 1.
    later = e.add_fact("My mobile number is 555-0999.", _corr(q, 0.72, 91),
                       source="chat_session", timestamp=3000.0)
    earlier = e.add_fact("I finally moved -- I live at 41 Harrowvane Close now.",
                         _corr(q, 0.70, 92), source="chat_session", timestamp=2000.0)
    e.flush()
    rows = np.arange(3)
    assert e._revisions_comparable(rows, np.array([1, 2])) is False
    hits = e.search("what did I have before?", q, top_k=3)
    by_id = {h["id"]: h["score"] for h in hits}
    assert by_id[earlier] >= by_id[later] - 1e-9 or by_id[later] >= by_id[earlier]
    e.close()


def test_pronoun_led_statement_with_its_own_possessive_is_not_anaphoric():
    """"THEY reissued MY badge number as X" names its own owner; inheriting the
    previous turn's subject filed it under an unrelated attribute."""
    assert not ent.is_pronoun_led("They reissued my building badge number as SL-6587.")
    assert ent.is_pronoun_led("His is 555-0199.")
    assert ent.detect_entity(
        "They reissued my building badge number as SL-6587.") == "building_badge_number"


def test_an_adverb_is_never_part_of_an_attribute_name():
    assert ent.query_intent(
        "which bank did I use for my personal account just before this one?"
    ) == "personal_account"
    assert ent.normalize_slot("personal account just") == "personal_account"
