"""One-time extractor for ``fixture_vocab.txt`` -- the fixture-vocabulary guard's asset.

pytest NEVER runs this file (it is not named ``test_*``) and nothing in the test
suite imports it. It exists so the asset next to it is auditable and
reproducible: run it by hand, from a checkout that has ``scratch/refound``, when
a new chat fixture is added.

    python3 tests/data/fixture_vocab_source.py

WHY THE ASSET HOLDS DIGESTS AND NOT WORDS
-----------------------------------------
The vocabulary being extracted IS the held-out persona vocabulary. Writing it
into the repository in plain text would defeat the quarantine it protects:
anyone -- human or agent -- who opened ``tests/`` would have the held-out names
in front of them, and could hard-code them into a tagger and "win" a benchmark
arm the names were withheld from. So each token is stored as a short keyed
BLAKE2s digest. Membership still answers exactly the same question ("is this
library token a fixture proper noun?"), because the guard hashes the tokens it
finds in library source and looks them up; but the asset itself reads as a
column of hex and carries no vocabulary, no sentence and no persona.

This is obfuscation, not secrecy: anyone who already holds the fixtures can
recompute the table. That is the right bar. The threat is casual contamination
-- a name drifting from a fixture into library code or into someone's working
memory -- not an adversary.

WHAT IS HARVESTED
-----------------
The union of the two harvests that were running live against ``scratch/refound``
before the asset existed, so the baked table is a superset of both:

  A. ``tests/test_entities.py`` -- every token a fixture only ever writes with a
     capital (a word that also appears lower-cased in the same fixture is
     ordinary English opening a sentence), minus the system dictionary with
     careful morphology.
  B. ``tests/test_round5_temporal.py`` -- every capitalised token that does NOT
     open a sentence (mid-sentence capitals), plus every component of every
     ``a-b`` / ``a_b`` slug, minus the system dictionary with blunt morphology.

QUARANTINE
----------
``clean_chat_benchmark_persona4.json`` is scored-once material. It is excluded by
name and the script refuses to run if it ever appears in the harvest list, so no
regeneration of this asset can read it.
"""

import hashlib
import os
import re
import sys

QUARANTINED = ("clean_chat_benchmark_persona4.json",)
PERSON = b"nanomemv"            # BLAKE2s personalisation (8 bytes max); changing it invalidates the asset
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixture_vocab.txt")
WORDS = "/usr/share/dict/words"

# Harvest A's explicit list, plus harvest B's glob over the same directory.
A_FIXTURES = ("clean_chat_benchmark.json", "clean_chat_benchmark_heldout.json",
              "dev_personas_v3r4.json", os.path.join("golden", "chat_v2_golden.json"))


def digest(token):
    return hashlib.blake2s(token.encode("utf-8"), digest_size=8,
                           person=PERSON).hexdigest()


def _harvest_a(path, dictionary):
    """Tokens this fixture only ever writes CAPITALISED."""
    text = open(path, encoding="utf-8", errors="ignore").read()
    upper, lower = set(), set()
    for m in re.finditer(r"\b([A-Za-zÀ-ɏ][a-zA-ZÀ-ɏ]{3,})\b", text):
        w = m.group(1)
        (upper if w[0].isupper() else lower).add(w.lower())

    def ordinary(t):
        # The word list holds lemmas, so an inflected ordinary word is not in it.
        # Strip only real English inflections -- a blind prefix walk turns a name
        # into a word and makes the audit vacuous.
        cands = {t}
        for suf, subs in (("ies", ("y",)), ("es", ("", "e")), ("s", ("",)),
                          ("ed", ("", "e")), ("ing", ("", "e")), ("est", ("", "e"))):
            if t.endswith(suf) and len(t) > len(suf) + 2:
                for r in subs:
                    stem = t[:-len(suf)] + r
                    cands.add(stem)
                    if len(stem) > 3 and stem[-1] == stem[-2]:
                        cands.add(stem[:-1])          # formatting -> format
        return bool(cands & dictionary)

    return {t for t in (upper - lower) if not ordinary(t)}


def _harvest_b(path, dictionary):
    """Mid-sentence capitals and slug components."""
    raw = open(path, encoding="utf-8", errors="ignore").read()
    cap = re.compile(r"\b([A-Z][a-zA-ZÀ-ɏ]{3,})\b")
    out = set()
    for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', raw):
        body = m.group(1)
        for w in cap.finditer(body):
            if not re.search(r'(?:^|[.!?]\s+|\\"\s*)$', body[:w.start()]):
                out.add(w.group(1).lower())
    for slug in re.findall(r'"([a-z]+(?:[-_][a-z]+)+)"', raw):
        for part in re.split(r"[^a-z]+", slug):
            if len(part) > 3:
                out.add(part)

    def is_common(t):
        forms = {t}
        for suf, adds in (("s", (t[:-1],)), ("es", (t[:-2],)),
                          ("ies", (t[:-3] + "y",)), ("ed", (t[:-2], t[:-1])),
                          ("ing", (t[:-3], t[:-3] + "e")), ("er", (t[:-2], t[:-1])),
                          ("ly", (t[:-2],))):
            if t.endswith(suf):
                forms |= set(adds)
        return any(f in dictionary for f in forms)

    return {t for t in out if t.isalpha() and not is_common(t)}


def main(refound):
    dictionary = {w.strip().lower() for w in open(WORDS, errors="ignore")}
    listed = sorted(f for f in os.listdir(refound) if f.endswith(".json")
                    and ("chat_benchmark" in f or "personas" in f
                         or f == "temporal_bench.json"))
    b_files = [f for f in listed if f not in QUARANTINED]
    assert not (set(b_files) & set(QUARANTINED)), "quarantined fixture in the harvest"
    a_files = [f for f in A_FIXTURES if os.path.exists(os.path.join(refound, f))]

    everything = b_files + [f for f in a_files if f not in b_files]
    tokens, provenance = set(), []
    for rel in everything:
        p = os.path.join(refound, rel)
        for h in (_harvest_a, _harvest_b):        # both rules on every fixture
            tokens |= h(p, dictionary)
        blob = open(p, "rb").read()
        provenance.append((rel, len(blob), hashlib.sha256(blob).hexdigest()[:16]))
    tokens = {t for t in tokens if t.isalpha() and len(t) >= 4}

    table = {digest(t): t for t in tokens}
    assert len(table) == len(tokens), "digest collision -- widen digest_size"

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("# nanomem fixture-vocabulary guard: keyed BLAKE2s digests, NOT words.\n")
        fh.write("# Regenerate with tests/data/fixture_vocab_source.py; see that file for why\n")
        fh.write("# this table is hashed and why the test must never read scratch/refound.\n")
        fh.write("# recipe: blake2s(token.utf8, digest_size=8, person=%r).hexdigest()\n"
                 % PERSON.decode())
        fh.write("# tokens: %d\n" % len(table))
        for rel, size, sha in provenance:
            fh.write("# source: %-38s %9d B  sha256:%s\n" % (rel, size, sha))
        for rel in QUARANTINED:
            fh.write("# excluded (quarantined, scored once): %s\n" % rel)
        for d in sorted(table):
            fh.write(d + "\n")
    # Counts and provenance only -- this script never prints a harvested token.
    print("wrote %s: %d digests from %d fixtures" % (OUT, len(table), len(provenance)))


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "NANOMEM_REFOUND_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))), "scratch", "refound"))
    main(root)
