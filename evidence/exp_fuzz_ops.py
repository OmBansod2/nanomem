"""Randomised operation fuzzer with invariants checked after every step.

Enumerated cases test the states I thought of. This churns a vault through
random sequences of add / add_batch / update / delete / prune /
forget_superseded / compact / flush / reopen / merge and asserts, after EVERY
operation, properties that must hold in all of them.

The invariants are deliberately ones that do not depend on the embedder:
retrieval is pinned with `filter=`, so what is under test is the temporal and
structural logic, not which sentence a model happens to put closest.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quarantine_guard                        # noqa: E402
quarantine_guard.enforce()

import json, random, shutil, tempfile, traceback   # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "..", "nanomem_standalone"))
from nanomem.vault import Vault                 # noqa: E402

DAY = 86400.0
T0 = 1_600_000_000.0
# TWO DISJOINT POOLS. A chain holding both a one-line fact and a 20,000-word
# policy has no single right answer to "what is the current X": the relevance
# floor is supposed to prefer the record that ANSWERS the question over a
# boilerplate paragraph that merely is newer, which is the `doc32` lesson from
# 0.7.5. Mixing them tests the ranker's judgement, not an invariant, so the
# pools are kept apart and each gets the assertion that is actually true of it.
FACT_ENTITIES = ["home_address", "phone", "employer", "car", "locker_code"]
# The revision group key is (user_id, project, entity), so the same entity name
# under two users is two chains. Nothing in this fuzzer exercised that until
# now: every earlier seed wrote as a single anonymous user.
# Every write is OWNED. An anonymous write plus a `{"entity": ...}` filter is
# an UNCONSTRAINED query -- it legitimately sees every user's rows, which is
# recorded as use case C2 and is not a leak. Asserting isolation against it was
# the harness being wrong, so the fuzzer now always names an owner and the
# isolation invariant means what it says.
USERS = ["alice", "bob", "carol"]
DOC_ENTITIES = ["refund_policy", "privacy_policy", "sla_policy"]
ENTITIES = FACT_ENTITIES + DOC_ENTITIES
FILLER = (" Customers are advised that the terms described herein apply to all "
          "orders placed through any channel operated by the company. ")


class Model:
    """What the vault should hold.

    A CHUNKED record is one value spread over several rows that share a
    parent_id, a timestamp and an entity. The first version of this model
    treated each chunk as its own value, so `current` picked an arbitrary
    paragraph of boilerplate and every long write looked like a failure. That
    was the model being wrong, not the vault.
    """
    def __init__(self):
        self.chains = {}        # entity -> {version_key: (ts, [texts])}

    def current(self, e):
        """The set of texts making up the newest version, or None.

        TIME FIRST, THE ARRIVAL COUNTER BREAKING TIES -- the rule the engine
        documents and `test_the_arrival_counter_still_breaks_a_timestamp_tie`
        pins. This used to break ties on the document id, which is arbitrary,
        so whenever two versions of one chain landed on the identical timestamp
        the model and the vault disagreed by coin flip. Seed 49 hit it after 46
        operations: two rows at ts=1765888000, revisions 4 and 6, and the engine
        correctly took revision 6.
        """
        vers = self.chains.get(e) or {}
        if not vers:
            return None
        key = max(vers, key=lambda k: (vers[k][0], vers[k][2]))
        return frozenset(vers[key][1])

    def resync(self, v):
        self.chains = {}
        for r in v.engine.iter_records():
            meta = r.get("metadata") or {}
            e = meta.get("entity")
            if not e:
                continue
            e = (meta.get("user_id"), e)          # the real chain identity
            key = meta.get("parent_id") or r.get("id")
            ts = float(r.get("timestamp", 0.0))
            rev = int(r.get("revision", 1))
            slot = self.chains.setdefault(e, {}).setdefault(key, (ts, [], rev))
            slot[1].append(r["text"])
            if rev > slot[2]:                     # a version's revision is its max
                self.chains[e][key] = (slot[0], slot[1], rev)


class Fail(Exception):
    pass


def check(v, model, step, op):
    """Invariants that must hold after every operation."""
    recs = list(v.engine.iter_records())          # I1: iterates without error
    def brief(x):
        if x is None:
            return None
        return sorted(t[:60] for t in x)[:2] if isinstance(x, frozenset) else x[:60]

    for chain_key, vers in model.chains.items():
        uid, e = chain_key
        if not vers:
            continue
        filt = {"entity": e, "user_id": uid}
        if e in DOC_ENTITIES:
            # STRUCTURAL only: every chunk of the newest version is present.
            # Which chunk ranks first is a relevance question, not an invariant.
            want_doc = model.current(chain_key)
            have = {r["text"] for r in recs}
            missing = [t for t in want_doc if t not in have]
            if missing:
                raise Fail(f"step {step} after {op}: doc {chain_key!r} lost "
                           f"{len(missing)} of {len(want_doc)} chunks of its "
                           f"current version")
            continue
        want = model.current(chain_key)            # a frozenset of chunk texts
        q = f"what is the current {e.replace('_', ' ')}"
        hit = v.search(q, top_k=1, filter=filt)
        # ISOLATION: a filtered search must never cross a user boundary.
        for h in v.search(q, top_k=5, filter=filt):
            if (h.get("metadata") or {}).get("user_id") != uid:
                raise Fail(f"step {step} after {op}: user {uid!r} filter "
                           f"returned a row owned by "
                           f"{(h.get('metadata') or {}).get('user_id')!r}")
        got = hit[0]["text"] if hit else None
        if got not in want:                        # I2: newest-by-time wins
            raise Fail(f"step {step} after {op}: chain {chain_key!r} current\n"
                       f"   got  {brief(got)!r}\n   want one of {brief(want)!r}")
        newest_ts = max(ts for ts, _t, _r in vers.values())
        ah = v.search(q, top_k=1, filter=filt, as_of=newest_ts)
        ag = ah[0]["text"] if ah else None
        if ag not in want:                         # I3: as_of(newest) == current
            raise Fail(f"step {step} after {op}: chain {chain_key!r} as_of(newest)\n"
                       f"   got  {brief(ag)!r}\n   want one of {brief(want)!r}")
    return len(recs)


def run_one(seed, n_steps=120):
    rng = random.Random(seed)
    tmp = tempfile.mkdtemp(prefix=f"fuzz_{seed}_")
    path = os.path.join(tmp, "f.dat")
    enc = rng.random() < 0.3
    pw = "fuzz password" if enc else None
    v = Vault(path, password=pw) if enc else Vault(path)
    model = Model()
    log = []
    counter = [0]

    def new_text(e):
        counter[0] += 1
        return f"My {e.replace('_', ' ')} is value-{counter[0]}."

    try:
        for step in range(n_steps):
            op = rng.choice(["add", "add", "add", "add_batch", "add_long",
                             "update", "delete", "flush", "reopen",
                             "prune", "forget_superseded", "compact",
                             "merge", "export"])
            e = (rng.choice(DOC_ENTITIES) if op == "add_long"
                 else rng.choice(FACT_ENTITIES))
            ts = T0 + rng.randint(0, 2000) * DAY      # deliberately out of order
            uid = rng.choice(USERS)

            def meta_for(ent):
                return {"entity": ent, "user_id": uid}
            if op == "add":
                v.add(new_text(e), metadata=meta_for(e),
                      source=rng.choice(["user_input", "config_audit", "chat"]),
                      timestamp=ts)
            elif op == "add_batch":
                rows = [{"text": new_text(e), "metadata": meta_for(e),
                         "source": "config_audit",
                         "timestamp": T0 + rng.randint(0, 2000) * DAY}
                        for _ in range(rng.randint(1, 4))]
                v.add_batch(rows)
            elif op == "add_long":
                v.add(new_text(e) + FILLER * rng.randint(20, 80),
                      metadata=meta_for(e), source="policy_doc", timestamp=ts)
            elif op == "update":
                recs = [r for r in v.engine.iter_records()
                        if (r.get("metadata") or {}).get("entity")]
                if recs:
                    r = rng.choice(recs)
                    v.update(r["id"], text=r["text"] + " (revised)")
            elif op == "delete":
                recs = list(v.engine.iter_records())
                if recs:
                    v.delete(id=rng.choice(recs)["id"])
            elif op == "flush":
                v.flush()
            elif op == "reopen":
                v.flush(); v.close()
                v = Vault(path, password=pw) if enc else Vault(path)
            elif op == "prune":
                v.flush()
                before = {e2: model.current(e2) for e2 in model.chains}
                v.prune(older_than_days=rng.choice([365, 900, 1500]))
                model.resync(v)
                for e2, want in before.items():
                    if (want is not None and e2 in model.chains
                            and e2[1] in FACT_ENTITIES
                            and model.current(e2) != want):
                        raise Fail(f"step {step} prune: {e2!r} current value "
                                   f"changed {sorted(want)!r} -> "
                                   f"{sorted(model.current(e2))!r}")
            elif op == "forget_superseded":
                v.flush()
                before = {e2: model.current(e2) for e2 in model.chains}
                v.forget_superseded(keep=rng.choice([1, 2, 3]))
                model.resync(v)
                for e2, want in before.items():
                    if (want is not None and e2 in model.chains
                            and e2[1] in FACT_ENTITIES
                            and model.current(e2) != want):
                        raise Fail(f"step {step} forget_superseded: {e2!r} "
                                   f"current {sorted(want)!r} -> "
                                   f"{sorted(model.current(e2))!r}")
            elif op == "merge":
                # build a small side vault and fold it in
                v.flush()
                side = os.path.join(tmp, f"side_{step}.dat")
                sv = Vault(side, password=pw) if enc else Vault(side)
                for _ in range(rng.randint(1, 3)):
                    se = rng.choice(FACT_ENTITIES)
                    sv.add(new_text(se), metadata=meta_for(se),
                           timestamp=T0 + rng.randint(0, 2000) * DAY)
                sv.flush(); sv.close()
                v.merge(side, reconcile_revisions=rng.random() < 0.5)
                v.flush()
            elif op == "export":
                v.flush()
                dst = os.path.join(tmp, f"exp_{step}.dat")
                n_out = v.export(dst)
                ev = Vault(dst, password=pw) if enc else Vault(dst)
                n_in = len(list(ev.engine.iter_records()))
                ev.close()
                if n_in != n_out:
                    raise Fail(f"step {step} export: reported {n_out} records, "
                               f"the target holds {n_in}")
            elif op == "compact":
                v.flush()
                before = {e2: model.current(e2) for e2 in model.chains}
                v.compact()
                model.resync(v)
                if before != {e2: model.current(e2) for e2 in model.chains}:
                    raise Fail(f"step {step} compact changed an answer")
            v.flush()
            model.resync(v)
            n = check(v, model, step, op)
            log.append(op)
        # encryption must survive the whole churn
        leak = None
        if enc:
            v.flush(); v.close()
            with open(path, "rb") as fh:
                leak = b"is value-" in fh.read()
            v = Vault(path, password=pw)
        recs = len(list(v.engine.iter_records()))
        v.close(); shutil.rmtree(tmp, ignore_errors=True)
        return {"seed": seed, "ok": True, "encrypted": enc, "records": recs,
                "ops": len(log), "plaintext_leak": leak}
    except Fail as f:
        try: v.close()
        except Exception: pass
        return {"seed": seed, "ok": False, "encrypted": enc,
                "invariant": str(f), "ops_done": len(log)}
    except Exception:
        try: v.close()
        except Exception: pass
        return {"seed": seed, "ok": False, "encrypted": enc, "crash": True,
                "traceback": traceback.format_exc().strip().splitlines()[-3:],
                "ops_done": len(log)}


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    out = []
    for seed in range(n_seeds):
        r = run_one(seed)
        out.append(r)
        mark = "ok " if r["ok"] else "XX "
        extra = "" if r["ok"] else f"  {r.get('invariant') or r.get('traceback')}"
        print(f"  {mark}seed {seed:>3} enc={str(r['encrypted']):<5} "
              f"{'records=%d' % r['records'] if r['ok'] else 'FAILED at op %d' % r['ops_done']}"
              f"{extra}")
    bad = [r for r in out if not r["ok"]]
    leaks = [r for r in out if r.get("plaintext_leak")]
    print(f"\n{len(out)-len(bad)}/{len(out)} seeds clean, {len(leaks)} plaintext leaks")
    p = os.path.join(HERE, "fuzz_ops_results.json")
    json.dump({"seeds": n_seeds, "clean": len(out)-len(bad), "results": out},
              open(p, "w"), indent=2, default=str)
    print(f"wrote {p}")
