"""
nanomem - The Stale Fact Demo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run it:
    python demo_stale.py
    python demo_stale.py --brief      # same computed output, prose trimmed
    NANOMEM_DEMO_PAUSE=0.35 python demo_stale.py --brief   # paced, for recording

`--brief` and NANOMEM_DEMO_PAUSE change only what is PRINTED around the
results and how fast. They do not touch a single computed value: the same
search runs, the same labels come back. Colour is emitted only to a tty, so
redirected output is byte-identical to what it always was.

Nobody ever tells an assistant "I have changed teams". The fact arrives as
scenery inside a question about something else, months apart -- and then turns
up in output you are about to send.

Everything this prints comes from running nanomem. Nothing is hard-coded, and
the vault is a temporary file that is deleted at the end. Change the SESSIONS
below and re-run it; the labels are computed, not written.

What it does NOT claim: that nanomem always ranks the current value first. It
does not, and the run below shows it not doing so. What it guarantees is that a
superseded fact is never handed over unlabelled.
"""

import datetime
import os
import shutil
import sys
import tempfile
import time

from nanomem import Vault
from nanomem.vault import staleness_label

DAY = 86400
NOW = time.time()

# ---- presentation only -------------------------------------------------
BRIEF = "--brief" in sys.argv
PAUSE = float(os.environ.get("NANOMEM_DEMO_PAUSE", "0") or 0)
_TTY = sys.stdout.isatty()
BOLD = "\033[1m" if _TTY else ""
DIM = "\033[2m" if _TTY else ""
RED = "\033[31m" if _TTY else ""
GREEN = "\033[32m" if _TTY else ""
YELLOW = "\033[33m" if _TTY else ""
CYAN = "\033[36m" if _TTY else ""
OFF = "\033[0m" if _TTY else ""


def beat(n=1.0):
    """Pace the output for a screen recording. Zero by default."""
    if PAUSE:
        time.sleep(PAUSE * n)


def prose(text):
    """Explanatory paragraphs. Suppressed by --brief; results never are."""
    if not BRIEF:
        print(text)


def when(days_ago):
    return datetime.date.fromtimestamp(NOW - days_ago * DAY).strftime("%b %d")


# (days ago, what the user actually typed, the attribute the app recognised)
#
# The app tags `entity` on the two turns that carry a fact about the user's
# team. That is what README's "Tell it what an attribute is" section asks for,
# and it is the difference between this working and not: with nothing declared,
# the lexical tagger groups 0 of 100 narratively-phrased chains
# (evidence/temporal_drift_results.json), and none of these turns would group.
SESSIONS = [
    (250, "I'm on the payments team, can you help me debug this webhook retry loop?", "team"),
    (240, "our oncall rotation is brutal this quarter, three pages last night", None),
    (200, "can you review this stripe idempotency key handling before I ship it", None),
    (120, "moved to infra last month, can you explain how our k8s ingress is set up?", "team"),
    (100, "we run argocd for deploys now, not the old jenkins pipeline", None),
    (40, "drafting the runbook for the ingress failover, what should it cover", None),
]

QUERY = "what has this person worked on, payments and webhooks?"


def main():
    workdir = tempfile.mkdtemp(prefix="nanomem_demo_")
    try:
        vault = Vault(os.path.join(workdir, "memory.dat"))

        print("%sEight months of ordinary work. Nobody ever announces a "
              "change.%s\n" % (BOLD, OFF))
        beat(2)
        for days, text, entity in SESSIONS:
            vault.add(text,
                      metadata={"entity": entity} if entity else None,
                      timestamp=NOW - days * DAY)
            print("  %s%-8s%s %s" % (DIM, when(days), OFF, text))
            if entity:
                print("  %s%-8s %s^ a fact about their team arrived here, inside a "
                      "question about something else%s" % (DIM, "", CYAN, OFF))
                beat(1.5)
            beat(0.6)
        vault.flush()

        if not BRIEF:
            print("\n" + "=" * 78)
        print("\n%sToday the assistant is writing their conference bio. It "
              "retrieves:%s\n" % (BOLD, OFF))
        print('   query: %s"%s"%s\n' % (CYAN, QUERY, OFF))
        beat(2)
        hits = vault.search(QUERY, top_k=3, decompose=False)

        print("  A similarity-only store returns, in this order:\n")
        beat(1)
        for i, h in enumerate(hits):
            print("    [%d] %s" % (i + 1, h["text"]))
            beat(0.5)
        prose("\n      The stale one is FIRST, because the question is worded like"
              "\n      the old job. Both facts are in there. Nothing says one replaced"
              "\n      the other, so the model writes \'works on the payments team\'"
              "\n      into a bio that is about to be sent.")
        print("")
        beat(3)

        print("  %snanomem returns the same records, and says which is still "
              "true:%s\n" % (BOLD, OFF))
        beat(1)
        for i, h in enumerate(hits):
            print("    [%d] %s" % (i + 1, h["text"]))
            label = staleness_label(h)
            if label:
                beat(1.5)
                print("        %s`-- %s%s" % (RED, label, OFF))
            beat(0.6)
        prose("\n      It did NOT re-rank this one. The old record really is the"
              "\n      closer match for that question, and pretending otherwise would"
              "\n      be the worse lie. It is handed over LABELLED instead, so the"
              "\n      model cannot use it as a fact.")
        prose("")
        beat(3)

        if not BRIEF:
            print("=" * 78)
        print("\n%sAsk what it has been, and the chain is there:%s\n" % (BOLD, OFF))
        beat(1)
        for c in vault.history("what team am I on"):
            text = c["text"]
            if BRIEF and len(text) > 58:
                text = text[:57] + "\u2026"
            live = not c["superseded"]
            print("    rev%d  %s%-11s%s %s"
                  % (c["revision"],
                     GREEN if live else YELLOW,
                     "CURRENT" if live else "superseded",
                     OFF,
                     text))
            beat(0.8)

        prose("\n    A fact that never changed is marked with nothing at all:")
        vault.add("my postgres connection pool is capped at 20",
                  metadata={"entity": "pool_size"}, timestamp=NOW - 3000 * DAY)
        vault.flush()
        h = vault.search("what is my connection pool capped at", top_k=1,
                         decompose=False)[0]
        if not BRIEF:
            print("      superseded=%-5s (written %d years ago)  %s"
                  % (h["superseded"], (NOW - h["timestamp"]) / 365.25 / DAY,
                     h["text"]))
        beat(3)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
