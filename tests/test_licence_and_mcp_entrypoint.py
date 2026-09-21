"""The licence files, and the MCP server's front door.

nanomem moved from AGPL-3.0-or-later to Apache-2.0 at 0.7.23. Copies WERE
distributed under the old terms and their recipients keep them, so the
superseded texts must keep shipping -- a licence someone received and can no
longer read is the failure this guards.
"""

import os
import subprocess
import sys

import pytest

import nanomem
from nanomem import mcp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


def test_the_licence_is_apache_2_0_including_the_patent_grant():
    lic = _read("LICENSE")
    assert "Apache License" in lic
    assert "Version 2.0, January 2004" in lic
    # Section 3 is the reason Apache was chosen over MIT, and the reason the
    # choice needed thinking about. It must actually be present.
    assert "Grant of Patent License" in lic
    assert "APPENDIX: How to apply the Apache License" in lic


def test_a_notice_file_ships():
    """Apache-2.0 section 4(d): NOTICE travels with redistributions."""
    notice = _read("NOTICE")
    assert "nanomem" in notice
    assert "Apache License" in notice


@pytest.mark.parametrize("name", ["LICENSE.agpl-3.0-or-later.md",
                                  "LICENSE.preview-v1.0.md"])
def test_superseded_licences_are_still_readable(name):
    text = _read(name)
    assert "SUPERSEDED" in text or "NO LONGER" in text
    assert os.path.getsize(os.path.join(HERE, name)) > 500


def test_the_agpl_text_itself_is_preserved_not_just_a_pointer():
    """Retaining a heading and dropping the terms would be worse than nothing."""
    text = _read("LICENSE.agpl-3.0-or-later.md")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text
    assert "Version 3, 19 November 2007" in text


def test_pyproject_declares_apache_and_nothing_else():
    """Only ACTIVE declarations count -- the file keeps comments about past ones.

    pyproject quotes the old `license = "AGPL-3.0-or-later"` line inside a
    comment explaining a packaging bug it caused on Python 3.8. That prose is
    worth keeping, so this reads the declarations rather than the file.
    """
    live = [ln.strip() for ln in _read("pyproject.toml").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    decls = [ln for ln in live if ln.startswith("license =")]
    assert decls == ['license = "Apache-2.0"'], decls
    assert not [ln for ln in live
                if "AGPL" in ln and "LICENSE.agpl-3.0-or-later.md" not in ln]


# --------------------------------------------------------------------------
# the MCP entry point
# --------------------------------------------------------------------------
def test_the_console_script_is_declared():
    tml = _read("pyproject.toml")
    assert 'nanomem-mcp = "nanomem.mcp:main"' in tml


def test_vault_path_resolution_order(monkeypatch, tmp_path):
    """--vault beats NANOMEM_VAULT beats the default under the home directory."""
    monkeypatch.delenv("NANOMEM_VAULT", raising=False)
    assert mcp.resolve_vault_path() == mcp.DEFAULT_VAULT

    monkeypatch.setenv("NANOMEM_VAULT", str(tmp_path / "from_env.dat"))
    assert mcp.resolve_vault_path() == str(tmp_path / "from_env.dat")
    assert mcp.resolve_vault_path(str(tmp_path / "flag.dat")) == str(tmp_path / "flag.dat")

    # An empty env var is not a path. It used to be swallowed elsewhere in this
    # library and created a plaintext vault; here it must simply not win.
    monkeypatch.setenv("NANOMEM_VAULT", "   ")
    assert mcp.resolve_vault_path() == mcp.DEFAULT_VAULT


def test_the_default_vault_is_not_relative_to_the_working_directory():
    """An MCP server does not choose its cwd -- the client launches it."""
    assert os.path.isabs(mcp.DEFAULT_VAULT)


def test_the_entry_point_reports_its_version(capsys):
    assert mcp.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert nanomem.__version__ in out
    assert nanomem.ENGINE_VERSION in out


def test_the_server_module_still_runs_as_a_module(tmp_path):
    """`python -m nanomem.mcp` is in shipped documentation and must keep working."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=HERE)
    env.pop("NANOMEM_VAULT", None)
    r = subprocess.run([sys.executable, "-m", "nanomem.mcp", "--version"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert nanomem.__version__ in r.stdout
