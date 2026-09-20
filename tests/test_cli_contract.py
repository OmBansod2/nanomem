"""The CLI's side of the bargain: say what happened, on the right stream, with
an exit code a script can branch on.

Four failure paths printed an error to STDOUT and fell off the end of the
function, which exits 0 -- so `nanomem ingest missing.txt && next_step` ran
`next_step`, and redirecting stdout put the error in the output file with
nothing on stderr.
"""
import os
import subprocess
import sys

import pytest

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(tmp_path, *args):
    r = subprocess.run([sys.executable, "-m", "nanomem.cli"] + list(args),
                       capture_output=True, text=True, timeout=240,
                       cwd=str(tmp_path), env=dict(os.environ, PYTHONPATH=PKG_ROOT))
    return r.returncode, r.stdout, r.stderr


def test_reading_a_vault_that_does_not_exist_is_an_error(tmp_path):
    """It printed "No matching memories found" and exited 0, so a mistyped path
    was indistinguishable from an empty vault. Opening a vault CREATES it, so
    this also has to not leave one behind."""
    missing = tmp_path / "typo.dat"
    rc, out, err = _run(tmp_path, "search", "anything", "--vault", str(missing))
    assert rc != 0
    assert "Error" in err and "no vault" in err
    assert not missing.exists(), "a read created the vault it was asked to read"


def test_writing_still_creates_a_vault_on_demand(tmp_path):
    """The control that must NOT flip."""
    v = tmp_path / "fresh.dat"
    rc, out, err = _run(tmp_path, "add", "a real fact", "--vault", str(v))
    assert rc == 0, err
    assert v.exists()
    rc, out, err = _run(tmp_path, "search", "real", "--vault", str(v))
    assert rc == 0 and "fact" in out


@pytest.mark.parametrize("args,needle", [
    (("ingest", "definitely_missing.txt"), "not found"),
    (("user", "delete", "ghost_user_that_is_not_there"), "does not exist"),
], ids=["ingest-missing-file", "delete-absent-user"])
def test_a_failure_exits_non_zero_and_reports_on_stderr(tmp_path, args, needle):
    rc, out, err = _run(tmp_path, *args)
    assert rc != 0, "exited 0 on a failure: %r" % (out or err)
    assert needle in err, "error went to stdout, not stderr: out=%r err=%r" % (out[:120], err[:120])


def test_adding_empty_text_is_not_reported_as_stored(tmp_path):
    """`add()` returns "" for whitespace-only input, and the CLI printed
    "Stored fact  in 1.7 µs ... -> '   '" -- an empty id and a success exit for a
    fact that was never stored."""
    rc, out, err = _run(tmp_path, "add", "   ", "--vault", str(tmp_path / "w.dat"))
    assert rc != 0
    assert "Stored fact" not in out
    assert "empty" in err or "whitespace" in err


def test_a_reversed_date_range_is_a_message_not_a_traceback(tmp_path):
    v = tmp_path / "c.dat"
    _run(tmp_path, "add", "a fact", "--vault", str(v))
    rc, out, err = _run(tmp_path, "changes", "--since", "2030-01-01",
                        "--until", "2020-01-01", "--vault", str(v))
    assert rc != 0
    assert "Traceback" not in (out + err), "dumped a stack at the user"
    assert "--until" in err and "--since" in err
