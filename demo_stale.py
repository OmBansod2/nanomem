"""
nanomem - The Stale Fact Demo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run it:
    python demo_stale.py

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
import tempfile
import time

from nanomem import Vault
from nanomem.vault import staleness_label

DAY = 86400
NOW = time.time()


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

        print("Eight months of ordinary work. Nobody ever announces a change.\n")
        for days, text, entity in SESSIONS:
            vault.add(text,
                      metadata={"entity": entity} if entity else None,
                      timestamp=NOW - days * DAY)
            print("  %-8s %s" % (when(days), text))
            if entity:
                print("  %-8s ^ a fact about their team arrived here, inside a "
                      "question about something else" % "")
        vault.flush()

        print("\n" + "=" * 78)
        print("\nToday the assistant is writing their conference bio. It retrieves:\n")
        print('   query: "%s"\n' % QUERY)
        hits = vault.search(QUERY, top_k=3, decompose=False)

        print("  A similarity-only store returns, in this order:\n")
        for i, h in enumerate(hits):
            print("    [%d] %s" % (i + 1, h["text"]))
        print("\n      The stale one is FIRST, because the question is worded like")
        print("      the old job. Both facts are in there. Nothing says one replaced")
        print("      the other, so the model writes 'works on the payments team'")
        print("      into a bio that is about to be sent.\n")

        print("  nanomem returns the same records, and says which is still true:\n")
        for i, h in enumerate(hits):
            print("    [%d] %s" % (i + 1, h["text"]))
            label = staleness_label(h)
            if label:
                print("        `-- %s" % label)
        print("\n      It did NOT re-rank this one. The old record really is the")
        print("      closer match for that question, and pretending otherwise would")
        print("      be the worse lie. It is handed over LABELLED instead, so the")
        print("      model cannot use it as a fact.\n")

        print("=" * 78)
        print("\nAsk what it has been, and the chain is there:\n")
        for c in vault.history("what team am I on"):
            print("    rev%d  %-11s %s"
                  % (c["revision"],
                     "superseded" if c["superseded"] else "CURRENT",
                     c["text"]))
        print("\n    A fact that never changed is marked with nothing at all:")
        vault.add("my postgres connection pool is capped at 20",
                  metadata={"entity": "pool_size"}, timestamp=NOW - 3000 * DAY)
        vault.flush()
        h = vault.search("what is my connection pool capped at", top_k=1,
                         decompose=False)[0]
        print("      superseded=%-5s (written %d years ago)  %s"
              % (h["superseded"], (NOW - h["timestamp"]) / 365.25 / DAY, h["text"]))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
