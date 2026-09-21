"""An empty criterion must never WIDEN a destructive operation.

Why this file is written over the SURFACE rather than over a call
-----------------------------------------------------------------
0.7.20 fixed `delete(where={})`, wrote "a validation rule cannot hold on one of
four doors" into its own changelog, and then shipped that release's new guard on
one of six. The seventh black-box round measured the other five taking a
ten-record vault to zero:

    delete(where={})                raised ValueError      (the one that was fixed)
    delete(text_contains='')        10 -> 0
    delete(source='')               10 -> 0
    export(where={}, purge=True)    10 -> 0
    split({}, purge=True)           10 -> 0
    split_by_doc('', purge=True)    10 -> 0
    unmerge('')                     10 -> 0   <- no `purge` flag to decline

Writing the generalisation as prose is what failed. So this test enumerates every
criterion-shaped parameter on the public `Vault` surface by INTROSPECTION and
requires each one to carry a declared verdict below. A method added later, or a
new parameter on an existing one, fails `test_every_criterion_parameter_is_declared`
until a human states which way it goes -- which is the only mechanism that has
actually caught this class instead of describing it.

The rule is DIRECTION, not emptiness. `delete(ids=[])` means "remove these zero
records" and must stay a no-op: a loop over an empty list is ordinary code, and
raising there would be hostile. `delete(where={})` means "remove records
constrained by nothing", which is never what a caller typed on purpose.
"""

import inspect
import os

import pytest

from nanomem import Vault


# A parameter is criterion-shaped if its name is one a caller uses to SELECT
# records. Names here are matched exactly; the completeness test below is what
# forces this set to keep up with the code.
CRITERION_PARAMS = frozenset({
    "where", "filter_dict", "source", "source_doc", "text_contains",
    "text_exact", "filename", "vault_name_or_project", "metadata_key",
    "source_name", "query", "ids", "id",
})

REFUSES = "refuses"          # empty widens a DESTRUCTIVE op -> must raise ValueError
NARROWS = "narrows"          # empty selects nothing -> no-op, vault intact
READS = "reads"              # non-destructive; empty may mean everything
LABEL = "label"              # not a selector: a value written onto the record


def _declare(verdict, invoke=None):
    return (verdict, invoke)


# (method, parameter) -> (verdict, how to drive it with an EMPTY value)
#
# `invoke` receives (vault, tmpdir) and calls the method with that one parameter
# empty and everything else valid.
DECLARED = {
    # --- destructive: must refuse -------------------------------------------
    ("delete", "where"):
        _declare(REFUSES, lambda v, d: v.delete(where={})),
    ("delete", "text_contains"):
        _declare(REFUSES, lambda v, d: v.delete(text_contains="")),
    ("delete", "source"):
        _declare(REFUSES, lambda v, d: v.delete(source="")),
    ("export", "where"):
        _declare(REFUSES, lambda v, d: v.export(os.path.join(d, "e.dat"), where={}, purge=True)),
    ("export", "source_doc"):
        _declare(REFUSES, lambda v, d: v.export(os.path.join(d, "e2.dat"), source_doc="", purge=True)),
    ("split", "filter_dict"):
        _declare(REFUSES, lambda v, d: v.split({}, os.path.join(d, "s.dat"), purge=True)),
    ("split_by_doc", "filename"):
        _declare(REFUSES, lambda v, d: v.split_by_doc("", os.path.join(d, "s2.dat"), purge=True)),
    ("unmerge", "vault_name_or_project"):
        _declare(REFUSES, lambda v, d: v.unmerge("")),

    # --- narrowing: empty selects nothing, and that is well defined ----------
    ("delete", "ids"):
        _declare(NARROWS, lambda v, d: v.delete(ids=[])),
    ("delete", "id"):
        _declare(NARROWS, lambda v, d: v.delete(id="")),
    ("delete", "text_exact"):
        _declare(NARROWS, lambda v, d: v.delete(text_exact="")),
    ("forget", "query"):
        _declare(NARROWS, lambda v, d: v.forget("")),
    ("get", "id"):
        _declare(NARROWS, lambda v, d: v.get("")),
    ("exists", "id"):
        _declare(NARROWS, lambda v, d: v.exists("")),

    # --- non-destructive: empty may legitimately mean "everything" -----------
    # `export` WITHOUT purge is a backup, and backing everything up is a real
    # thing to want. Pinned here so that staying permissive is a decision.
    ("get_all_records", "where"):
        _declare(READS, lambda v, d: v.get_all_records(where={})),
    ("search", "query"):
        _declare(READS, lambda v, d: v.search("")),
    ("search_multihop", "query"):
        _declare(READS, lambda v, d: v.search_multihop("")),
    ("history", "query"):
        _declare(READS, lambda v, d: v.history("")),
    ("decompose_query", "query"):
        _declare(READS, lambda v, d: v.decompose_query("")),
    ("split_by_source", "source_name"):
        _declare(READS, lambda v, d: v.split_by_source("", os.path.join(d, "s3.dat"))),
    ("split_by_key", "metadata_key"):
        _declare(READS, lambda v, d: v.split_by_key("", os.path.join(d, "out"))),

    # --- not selectors: a value written ONTO a record ------------------------
    ("add", "source"): _declare(LABEL),
    ("add", "id"): _declare(LABEL),
    ("ingest_file", "source"): _declare(LABEL),
    ("update", "source"): _declare(LABEL),
    ("update", "id"): _declare(NARROWS, lambda v, d: v.update("", "replacement text")),
}


def _public_criterion_parameters():
    found = set()
    for name, fn in inspect.getmembers(Vault, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        for param in sig.parameters:
            if param in CRITERION_PARAMS:
                found.add((name, param))
    return found


@pytest.fixture()
def populated(tmp_path):
    v = Vault(str(tmp_path / "v.dat"))
    for i in range(10):
        v.add("fact number %d about topic %d" % (i, i % 3), metadata={"k": i})
    assert len(v.get_all_records()) == 10
    return v, str(tmp_path)


def test_every_criterion_parameter_is_declared():
    """A new selector must be classified before it can ship.

    This is the whole point of the file. If it fails, do not add the pair to
    DECLARED to make it green -- drive it with an empty value first and find out
    which way it actually goes.
    """
    undeclared = sorted(_public_criterion_parameters() - set(DECLARED))
    assert not undeclared, (
        "undeclared criterion parameter(s): %s. Each must be driven with an "
        "EMPTY value and declared REFUSES / NARROWS / READS / LABEL in "
        "tests/test_empty_criterion_safety.py." % (undeclared,))


def test_no_declaration_is_stale():
    """A declaration for a parameter that no longer exists hides real coverage."""
    live = _public_criterion_parameters()
    stale = sorted(k for k in DECLARED if k not in live)
    assert not stale, "DECLARED names parameter(s) that no longer exist: %s" % (stale,)


@pytest.mark.parametrize(
    "method,param",
    sorted(k for k, val in DECLARED.items() if val[0] == REFUSES),
)
def test_an_empty_widening_criterion_is_refused(populated, method, param, tmp_path):
    v, d = populated
    _, invoke = DECLARED[(method, param)]
    with pytest.raises(ValueError) as excinfo:
        invoke(v, d)
    # The message must name the parameter the CALLER wrote, not the one it was
    # forwarded as -- `split(filter_dict={})` reaching `export` said "where".
    assert param in str(excinfo.value), str(excinfo.value)
    assert len(v.get_all_records()) == 10, "%s.%s destroyed records while raising" % (method, param)


@pytest.mark.parametrize(
    "method,param",
    sorted(k for k, val in DECLARED.items() if val[0] in (NARROWS, READS)),
)
def test_an_empty_non_widening_criterion_leaves_the_vault_intact(populated, method, param):
    v, d = populated
    _, invoke = DECLARED[(method, param)]
    invoke(v, d)
    assert len(v.get_all_records()) == 10, (
        "%s(%s=<empty>) is declared non-destructive but removed records" % (method, param))


def test_a_narrowing_criterion_selects_nothing(populated):
    """NARROWS is stronger than "did not destroy": it must match zero records."""
    v, _ = populated
    assert v.delete(ids=[]) == 0
    assert v.delete(id="") == 0
    assert v.delete(text_exact="") == 0
    assert v.get("") is None
    assert v.exists("") is False
    assert v.forget("") == []
    assert len(v.get_all_records()) == 10


def test_a_real_criterion_still_works_on_every_guarded_surface(tmp_path):
    """The guard must not have bought safety by breaking the feature."""
    v = Vault(str(tmp_path / "w.dat"))
    for i in range(6):
        v.add("payload %d" % i, metadata={"bucket": "a" if i < 3 else "b"})

    assert v.delete(where={"bucket": "a"}) == 3
    assert len(v.get_all_records()) == 3

    assert v.export(str(tmp_path / "out.dat"), where={"bucket": "b"}, purge=True) == 3
    assert len(v.get_all_records()) == 0
    assert len(Vault(str(tmp_path / "out.dat")).get_all_records()) == 3


def test_export_without_purge_still_backs_up_everything(tmp_path):
    """`export(where={})` with no purge is a FULL BACKUP, and stays allowed.

    Pinned deliberately: the guard is on the destructive direction only, so that
    decision is visible here rather than implied by its absence.
    """
    v = Vault(str(tmp_path / "b.dat"))
    for i in range(4):
        v.add("row %d" % i)
    assert v.export(str(tmp_path / "backup.dat"), where={}) == 4
    assert len(v.get_all_records()) == 4
    assert len(Vault(str(tmp_path / "backup.dat")).get_all_records()) == 4
