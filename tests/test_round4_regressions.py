"""Round-4 regressions: one test per defect the round-3 verifiers found.

Every test in this file FAILS on 3.0.2 and passes on 3.0.3. The numbered
comments name the defect; the measured evidence lives in
scratch/refound/ranking_dev_r4_shipped.json, third_party_v3r4.json and
router_persist_v3r4.json.
"""

import json
import multiprocessing as mp
import os
import warnings

import numpy as np
import pytest

from conftest import D, unit_rows                                  # noqa: E402
from nanomem import container as C
from nanomem import crypto
from nanomem import entities as ent
from nanomem.engine import VaultEngine
from nanomem.errors import (ClosedVaultError, CorruptContainerError,
                            NanomemError, NotEncryptedError)
from nanomem.vault import Vault, resolve_vault_path


def _corr(q, target_cos, seed):
    """A unit vector at exactly ``target_cos`` from ``q``."""
    r = unit_rows(1, seed=seed)[0]
    r = r - float(r @ q) * q
    r /= np.linalg.norm(r)
    v = target_cos * q + np.sqrt(max(0.0, 1.0 - target_cos ** 2)) * r
    return (v / np.linalg.norm(v)).astype(np.float32)


# --- 1. the blocker: anaphora crossed the third-party boundary --------------
def test_anaphora_does_not_inherit_the_users_own_entity_for_somebody_else(vault_path):
    """3.0.2 stored a contact's number as revision 2 of the USER's number.

    `detect_entity` tags somebody else's attribute `other_<x>` and
    `entities_match` refuses to merge across the prefix, but the anaphora path in
    `add_fact` inherited the previous subject verbatim and never asked whose fact
    the follow-up was. The contact's record then won rank 1 for "what is my phone
    number?" on the LOWER cosine.
    """
    q = unit_rows(1, seed=3)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    mine = e.add_fact("My phone number is 555-0143.", _corr(q, 0.80, 11),
                      source="chat_session", metadata={"user_id": "u"})
    theirs = e.add_fact("His phone number is 555-0199.", _corr(q, 0.78, 12),
                        source="chat_session", metadata={"user_id": "u"})
    e.flush()
    assert e.get(theirs)["metadata"]["entity"].startswith(ent.OTHER_PREFIX)
    assert e.get(mine)["metadata"]["entity"] == "phone_number"
    assert e.get(theirs)["revision"] == 1          # not revision 2 of the user's
    hits = e.search("what is my phone number?", q, top_k=2)
    assert hits[0]["id"] == mine, [(h["text"], h["score"]) for h in hits]
    e.close()


def test_anaphora_does_not_leak_one_rest_clients_entity_onto_another(vault_path):
    """`source="rest_api"` shares one `_last_entity` slot across every caller."""
    e = VaultEngine(vault_path, embed_dim=D)
    a = e.add_fact("My phone number is 555-0143.", unit_rows(1, seed=1)[0],
                   source="rest_api")
    b = e.add_fact("Their address is 91 Rowan Road.", unit_rows(1, seed=2)[0],
                   source="rest_api")
    assert e.get(a)["metadata"]["entity"] == "phone_number"
    got = e.get(b)["metadata"].get("entity")
    assert got != "phone_number"                   # 3.0.2 stored exactly this
    assert got.startswith(ent.OTHER_PREFIX)
    e.close()


def test_inherit_entity_takes_ownership_from_the_followup():
    assert ent.inherit_entity("phone_number", "his is 555-0199") == "other_phone_number"
    assert ent.inherit_entity("other_phone_number", "mine is 555-0143") == "phone_number"
    assert ent.inherit_entity("phone_number", "it is 555-0143") == "phone_number"


# --- 2. third-person questions got no boost at all --------------------------
def test_third_person_question_resolves_into_the_other_namespace():
    assert ent.query_intent("what is my phone number?") == "phone_number"
    assert ent.query_intent("what is Wren Calloway's phone number?") == "other_phone_number"
    assert ent.query_intent("what is his address?") == "other_location"
    assert ent.detect_entity("Wren's phone number is 555-0199.") == "other_phone_number"


def test_third_person_question_is_actually_boosted(vault_path):
    """3.0.2 measured score - cosine == +0.000 for every third-person question."""
    q = unit_rows(1, seed=5)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    own = e.add_fact("My phone number is 555-0143.", _corr(q, 0.70, 21),
                     source="chat_session")
    oth = e.add_fact("Wren's phone number is 555-0199.", _corr(q, 0.66, 22),
                     source="chat_session")
    e.flush()
    hits = e.search("what is Wren's phone number?", q, top_k=2)
    assert hits[0]["id"] == oth
    assert hits[0]["score"] - hits[0]["cosine"] > 0.0
    assert e.search("what is my phone number?", q, top_k=2)[0]["id"] == own
    e.close()


def test_the_users_own_other_attribute_stays_out_of_the_reserved_namespace():
    """`normalize_slot` must never emit `other_<x>`: that is OTHER_PREFIX."""
    slot = ent.normalize_slot("other phone number")
    assert not slot.startswith(ent.OTHER_PREFIX), slot
    assert ent.query_intent("what is my other phone number?") == slot
    assert not ent.is_single_valued("other_phone_number")
    assert ent.entity_class("other_mobile") == ent.entity_class("other_phone_number")
    assert ent.entity_class("other_phone_number") != ent.entity_class("phone_number")


# --- 3. a frame class must not outrank an explicit possessive slot ----------
def test_action_verb_frame_loses_to_the_possessive_slot():
    assert ent.detect_entity("We moved; my desk is on the third floor now.") == "desk"
    assert ent.detect_entity("I moved my orders to Fendrell Timber last month.") == "order"
    # ... but a frame with no possessive slot, and a WEAK slot pattern, do not.
    assert ent.detect_entity("I moved to 61 Bellamy Terrace last month.") == "location"
    assert ent.detect_entity("I live in the old quarter near the bakery.") == "location"


def test_a_revision_marker_overrides_a_tagger_disagreement(vault_path):
    """13 of 16 generic revisions carry a marker; 0 of 40 adjacent pairs do."""
    a, b = "My bank is Corvid Union.", "I moved my accounts to Thornfield Bank."
    assert not ent.entities_match(ent.detect_entity(a), ent.detect_entity(b))
    assert ent.has_revision_marker(b) and not ent.has_revision_marker(a)
    q = unit_rows(1, seed=7)[0]
    e = VaultEngine(vault_path, embed_dim=D)
    va = _corr(q, 0.63, 31)
    vb = va + 0.5 * unit_rows(1, seed=32)[0]
    vb = (vb / np.linalg.norm(vb)).astype(np.float32)
    # the geometry the rule is about: close enough to be one group, a real gap
    assert float(vb @ va) >= 0.60 and 0.0 < float(q @ va) - float(q @ vb) < 0.16
    old = e.add_fact(a, va, source="chat_session", timestamp=1000.0)
    new = e.add_fact(b, vb, source="chat_session", timestamp=2000.0)
    e.flush()
    assert e.search("Which bank do I use?", q, top_k=2)[0]["id"] == new
    off = VaultEngine(vault_path, embed_dim=D, marker_overrides_tag=False)
    assert off.search("Which bank do I use?", q, top_k=2)[0]["id"] == old
    off.close()
    e.close()


# --- 4. replace_all(order=) silently discarded the memtable -----------------
def test_replace_all_with_an_order_keeps_the_pending_records(vault_path):
    e = VaultEngine(vault_path, embed_dim=D, block_capacity=5)
    V = unit_rows(8, seed=9)
    for i in range(5):
        e.add_fact(f"flushed {i}", V[i], id=f"f{i}")
    e.flush()
    recs = list(e.iter_records())
    ids = [e.add_fact(f"pending {i}", V[5 + i], id=f"p{i}") for i in range(3)]
    assert e.count() == 8
    assert e.replace_all(recs, order=list(range(len(recs)))) == 8
    assert e.count() == 8
    for i in ids:
        assert e.get(i) is not None, i           # 3.0.2 lost all three
    e.close()


def test_replace_all_rejects_a_non_permutation_and_a_bad_revision(vault_path):
    e = VaultEngine(vault_path, embed_dim=D)
    V = unit_rows(3, seed=10)
    for i in range(3):
        e.add_fact(f"r{i}", V[i], id=f"r{i}")
    e.flush()
    recs = list(e.iter_records())
    with pytest.raises(ValueError):
        e.replace_all(recs, order=[0, 0, 1])
    for bad in (2 ** 40, -5):                    # 3.0.2: OverflowError / accepted
        recs[0]["revision"] = bad
        with pytest.raises(ValueError):
            e.replace_all(recs)
    recs[0]["revision"] = 1
    assert e.count() == 3
    assert sorted(r["id"] for r in e.iter_records()) == ["r0", "r1", "r2"]
    e.close()


# --- 5. router="auto" re-clustered the whole file on EVERY open -------------
def test_the_one_time_recluster_is_recorded_in_the_file(tmp_path):
    p = str(tmp_path / "r.dat")
    e = VaultEngine(p, embed_dim=D, block_capacity=200)
    V = unit_rows(600, seed=11)
    for i in range(600):
        e.add_fact(f"r{i}", V[i], id=f"r{i}")
    e.flush()
    e.close()
    assert not C.detect_version(p) == 2
    inos = []
    for _ in range(3):
        e = VaultEngine(p, embed_dim=D, n_exhaustive=100, router="auto")
        inos.append(os.stat(p).st_ino)
        assert e.count() == 600
        assert e.header.reclustered
        e.close()
    assert inos[0] == inos[1] == inos[2], inos   # 3.0.2 rewrote on every open


def _open_router(p, q):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nanomem.engine import VaultEngine as VE
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = VE(p, embed_dim=D, n_exhaustive=100, router="auto")
        q.put(("OK", e.count()))
        e.close()
    except Exception as exc:                     # pragma: no cover - failure path
        q.put(("FAIL", type(exc).__name__))


def test_concurrent_router_opens_do_not_raise_from_the_constructor(tmp_path):
    p = str(tmp_path / "rr.dat")
    e = VaultEngine(p, embed_dim=D, block_capacity=200)
    V = unit_rows(600, seed=12)
    for i in range(600):
        e.add_fact(f"r{i}", V[i], id=f"r{i}")
    e.flush()
    e.close()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_open_router, args=(p, q)) for _ in range(4)]
    for x in procs:
        x.start()
    for x in procs:
        x.join(120)
    got = [q.get() for _ in procs]
    assert got == [("OK", 600)] * 4, got         # 3.0.2: 3 of 4 ContainerReplacedError


# --- 6. a passphrase that was silently dropped or downgraded ----------------
def test_empty_password_is_an_error_not_a_plaintext_vault(tmp_path):
    with pytest.raises(ValueError):
        VaultEngine(str(tmp_path / "e.dat"), embed_dim=D, password="")
    assert not os.path.exists(str(tmp_path / "e.dat"))


def test_bytes_password_is_an_error_not_a_stringified_one(tmp_path):
    with pytest.raises(TypeError):
        VaultEngine(str(tmp_path / "b.dat"), embed_dim=D, password=b"secret")
    with pytest.raises(TypeError):
        crypto.normalize_password(b"secret")


def test_password_on_a_v2_readonly_open_is_refused(tmp_path):
    from conftest import golden
    import shutil
    src = golden("chat_v2.dat")
    p = str(tmp_path / "v2.dat")
    shutil.copy(src, p)
    with pytest.raises(NotEncryptedError):
        VaultEngine(p, embed_dim=D, password="pw", migrate=False)
    # ... and without a password the read-only v2 open still works.
    e = VaultEngine(p, embed_dim=D, migrate=False)
    assert e.count() > 0
    assert e.stats()["encrypted_at_rest"] is False
    e.close()


def test_explicit_password_none_is_not_overridden_by_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOMEM_PASSWORD", "from-the-environment")
    v = Vault(str(tmp_path / "plain.dat"), password=None)
    assert v.stats()["encrypted_at_rest"] is False
    v.close()
    v = Vault(str(tmp_path / "env.dat"))
    assert v.stats()["encrypted_at_rest"] is True
    v.close()


# --- 7. structural crypto invariants ---------------------------------------
def test_cleartext_into_an_encrypted_container_is_structurally_impossible():
    h = C.Container._new_header(D, "float16", 4, 0, "pw")
    with pytest.raises(NanomemError):
        C.build_block_blob(h, None, 0, unit_rows(1), None,
                           [0.0], [1], ["i"], [""], [{}])
    h2 = C.Container._new_header(D, "float16", 4, 0, None)
    keys = crypto.derive_keys("pw", os.urandom(16), 14, 8, 1)
    with pytest.raises(NanomemError):
        C.build_block_blob(h2, keys, 0, unit_rows(1), None,
                           [0.0], [1], ["i"], [""], [{}])
    keys.wipe()


def test_an_unimplemented_cipher_id_is_named_not_mis_decrypted(tmp_path):
    p = str(tmp_path / "c.dat")
    e = VaultEngine(p, embed_dim=D, password="pw")
    e.add_fact("x", unit_rows(1, seed=13)[0])
    e.flush()
    e.close()
    raw = bytearray(open(p, "rb").read())
    h = C.unpack_file_header(bytes(raw[:256]), check_version=False)
    h.flags = (h.flags & ~C.CIPHER_MASK) | (2 << C.CIPHER_SHIFT)
    raw[:256] = C.pack_file_header(h)            # CRC recomputed; auth now stale
    open(p, "wb").write(bytes(raw))
    with pytest.raises(CorruptContainerError) as exc:
        VaultEngine(p, embed_dim=D, password="pw")
    assert "cipher" in str(exc.value)


def test_migration_wipes_the_key_it_derives(tmp_path, monkeypatch):
    from conftest import golden
    import shutil
    from nanomem import legacy_v2
    src = golden("chat_v2.dat")
    p = str(tmp_path / "m.dat")
    shutil.copy(src, p)
    seen = []
    real = crypto.derive_keys

    def spy(*a, **k):
        km = real(*a, **k)
        seen.append(km)
        return km

    monkeypatch.setattr(crypto, "derive_keys", spy)
    legacy_v2.migrate_v2_to_v3(p, password="pw", quiet=True)
    assert seen
    assert all(km._wiped for km in seen), [km._wiped for km in seen]


# --- 8. lifecycle / validation ---------------------------------------------
def test_the_engine_context_manager_flushes(vault_path):
    with VaultEngine(vault_path, embed_dim=D) as e:
        did = e.add_fact("important secret", unit_rows(1, seed=14)[0])
    e2 = VaultEngine(vault_path, embed_dim=D)
    assert e2.count() == 1                       # 3.0.2: 0
    assert e2.get(did) is not None
    e2.close()


def test_a_non_finite_embedding_or_timestamp_is_refused(vault_path):
    e = VaultEngine(vault_path, embed_dim=D)
    with pytest.raises(ValueError):
        e.add_fact("poisoned", np.full(D, np.nan, dtype=np.float32))
    with pytest.raises(ValueError):
        e.add_fact("poisoned", np.zeros(D, dtype=np.float32))
    with pytest.raises(ValueError):
        e.add_fact("poisoned", unit_rows(1, seed=15)[0], timestamp=float("nan"))
    assert e.count() == 0
    e.flush()
    assert json.dumps(e.search("q", unit_rows(1, seed=15)[0])) == "[]"
    e.close()


def test_a_closed_engine_still_refuses_every_write(vault_path):
    e = VaultEngine(vault_path, embed_dim=D)
    e.close()
    with pytest.raises(ClosedVaultError):
        e.add_fact("x", unit_rows(1, seed=16)[0])


# --- 9. network endpoints must not be an arbitrary-path primitive ----------
@pytest.mark.parametrize("bad", [
    "/etc/nanomem.dat", "~/x.dat", "../escape.dat", "a/../../escape.dat",
    "notadat.txt", "",
])
def test_resolve_vault_path_refuses_anything_outside_the_root(tmp_path, bad):
    with pytest.raises(ValueError):
        resolve_vault_path(bad, str(tmp_path))


def test_resolve_vault_path_accepts_a_name_inside_the_root(tmp_path):
    got = resolve_vault_path("sub/ok.dat", str(tmp_path))
    assert got == os.path.join(os.path.realpath(str(tmp_path)), "sub", "ok.dat")


def test_the_proxy_never_opens_a_vault_without_the_operators_password():
    """3.0.2's /v1/vault/init called a bare `Vault(name)` and silently downgraded."""
    import inspect
    from nanomem import proxy
    src = inspect.getsource(proxy.MemoryProxyHandler._handle_vault_init)
    assert "resolve_vault_path" in src
    assert src.count("password=pw") == 2
    assert "Vault(path)" not in src
    assert inspect.signature(proxy.run_proxy).parameters["host"].default == "127.0.0.1"


def test_the_proxy_tests_its_vault_with_is_not_none(tmp_path):
    """`Vault.__len__` makes a bare truth test False for an EMPTY vault."""
    import inspect
    from nanomem import proxy
    src = inspect.getsource(proxy)
    body = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    assert "if self.vault:" not in body
    v = Vault(str(tmp_path / "empty.dat"))
    assert len(v) == 0 and bool(v) is False      # the trap itself
    v.close()


# --- 10. the published score contract --------------------------------------
def test_the_search_docstring_bound_is_the_one_stats_publishes(vault_path):
    e = VaultEngine(vault_path, embed_dim=D)
    e.add_fact("My phone number is 555-0143.", unit_rows(1, seed=17)[0],
               source="chat_session")
    e.flush()
    mb = e.stats()["max_boost"]
    doc = VaultEngine.search.__doc__
    assert f"+{mb:.2f}" in doc, doc[:400]
    assert "stats()['max_boost']" in doc
    e.close()
