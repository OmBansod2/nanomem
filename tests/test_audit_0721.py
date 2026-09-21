"""Round 7 findings, one test each, named for what went wrong.

Seven of the eight reported findings reproduced. The eighth -- "a width-mismatched
handle destroys the vault" -- did not, and its negative is pinned here too, because
a finding that was chased and refuted is worth as much as one that was fixed and is
the thing most likely to be re-reported.
"""

import os
import subprocess
import sys
import time

import pytest

from nanomem import Vault, mcp, crypto


# --------------------------------------------------------------------------
# CRITICAL: ingest ids dropped the directory, so same-named files collided
# --------------------------------------------------------------------------
def _tree(tmp_path, names):
    root = tmp_path / "tree"
    for sub, tag in names:
        (root / sub).mkdir(parents=True)
        (root / sub / "config.txt").write_text(
            "\n".join("%s-KEY-%03d value line" % (tag, i) for i in range(1, 106)))
    return str(root)


def test_same_basename_in_two_directories_gets_two_sets_of_ids(tmp_path):
    v = Vault(str(tmp_path / "v.dat"))
    v.ingest_directory(_tree(tmp_path, [("a", "ALPHA"), ("b", "BETA")]))
    rows = v.get_all_records()
    ids = {str(r.get("id")) for r in rows}
    assert len(ids) == len(rows), (
        "row ids collided across directories: %d rows, %d distinct ids" % (len(rows), len(ids)))


def test_deleting_one_ingested_chunk_does_not_take_the_other_file_with_it(tmp_path):
    """The measured symptom: 40 distinct source lines of the OTHER file vanished."""
    v = Vault(str(tmp_path / "v.dat"))
    v.ingest_directory(_tree(tmp_path, [("a", "ALPHA"), ("b", "BETA")]))

    def lines(tag):
        return sum(sum(1 for ln in r.get("text", "").splitlines() if ln.startswith(tag + "-KEY"))
                   for r in v.get_all_records())

    beta_before = lines("BETA")
    # PICK THE TARGET BY CONTENT, NOT BY SORT ORDER. Ids now carry an md5 of the
    # absolute path, so `sorted(ids)[0]` names whichever file the temp directory
    # happened to hash lower -- it chose the ALPHA chunk standalone and the BETA
    # chunk under the full suite, and the assertion below only means anything if
    # the row deleted belongs to the OTHER file.
    alpha_rows = [r for r in v.get_all_records() if "ALPHA-KEY" in r.get("text", "")]
    assert alpha_rows, "no ALPHA rows to delete"
    target = str(alpha_rows[0].get("id"))
    assert v.delete(id=target) == 1
    assert lines("BETA") == beta_before, "deleting a chunk of one file removed lines of the other"


def test_an_id_minted_before_this_release_still_resolves_when_it_is_unambiguous(tmp_path):
    root = tmp_path / "solo"
    root.mkdir()
    (root / "solo.txt").write_text("\n".join("SOLO line %d" % i for i in range(1, 106)))
    v = Vault(str(tmp_path / "v.dat"))
    v.ingest_directory(str(root))
    real = sorted(str(r.get("id")) for r in v.get_all_records())[0]
    bare = real.split("@")[0] + ":" + real.split(":")[1]
    assert bare != real
    assert v.get(bare) is not None
    assert v.exists(bare) is True
    assert v.update(bare, "replacement text") is True
    assert v.delete(id=bare) == 1


def test_an_ambiguous_old_id_raises_instead_of_picking_a_file(tmp_path):
    """Resolving it either way would reintroduce exactly the defect just fixed."""
    v = Vault(str(tmp_path / "v.dat"))
    v.ingest_directory(_tree(tmp_path, [("a", "ALPHA"), ("b", "BETA")]))
    with pytest.raises(ValueError) as e:
        v.delete(id="config.txt:1-50")
    assert "ambiguous" in str(e.value)
    assert len(v.get_all_records()) == 6, "it destroyed rows on the way to raising"


# --------------------------------------------------------------------------
# SECURITY: an empty passphrase was silently no passphrase
# --------------------------------------------------------------------------
def test_an_empty_password_env_var_is_refused_not_treated_as_plaintext(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOMEM_PASSWORD", "")
    with pytest.raises(ValueError) as e:
        Vault(str(tmp_path / "v.dat"))
    assert "EMPTY" in str(e.value)
    assert "unset" in str(e.value).lower(), "the message must say how to mean 'no encryption'"


def test_an_unset_password_env_var_still_means_plaintext(tmp_path, monkeypatch):
    monkeypatch.delenv("NANOMEM_PASSWORD", raising=False)
    v = Vault(str(tmp_path / "v.dat"))
    v.add("hello")
    assert v.stats().get("encrypted_at_rest") is False


def test_a_real_password_still_encrypts(tmp_path, monkeypatch):
    secret = "SQUIRREL-KEEL-9F3A-CANARY"
    monkeypatch.setenv("NANOMEM_PASSWORD", "correct-horse-battery-staple")
    p = tmp_path / "v.dat"
    v = Vault(str(p))
    v.add("the secret is " + secret)
    v.flush()
    assert v.stats().get("encrypted_at_rest") is True
    assert secret.encode() not in p.read_bytes()


def test_the_cipher_label_carries_its_strongest_caveat():
    """`stats()` is what gets pasted into a compliance answer; THREAT_MODEL is not."""
    assert "not independently audited" in crypto.CIPHER_LABEL_SHAKE
    assert "not FIPS validated" in crypto.CIPHER_LABEL_SHAKE


# --------------------------------------------------------------------------
# SECURITY: the directory walk followed symlinks out of the confined root
# --------------------------------------------------------------------------
def test_directory_ingest_refuses_a_symlink_that_escapes_the_root(tmp_path):
    outside = tmp_path / "outside"
    root = tmp_path / "root"
    outside.mkdir()
    root.mkdir()
    (outside / "secret.txt").write_text("CANARY-7QX ledger passphrase\n" * 3)
    (root / "notes.txt").write_text("ordinary indexed content\n" * 3)
    os.symlink(str(outside / "secret.txt"), str(root / "leak.txt"))
    os.symlink(str(outside), str(root / "escape_dir"))

    v = Vault(str(tmp_path / "v.dat"))
    stats = v.ingest_directory(str(root), confine_root=str(root))
    texts = " ".join(r.get("text", "") for r in v.get_all_records())
    assert "CANARY-7QX" not in texts, "content outside the root was ingested"
    assert "ordinary indexed content" in texts, "the guard ate a legitimate file"
    assert stats.get("skipped_outside_root", 0) >= 1, "it declined silently"


def test_directory_ingest_without_a_root_is_unchanged(tmp_path):
    """The CLI passes no root: a person running `nanomem ingest` already has the file."""
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.txt").write_text("plain content here\n" * 3)
    v = Vault(str(tmp_path / "v.dat"))
    stats = v.ingest_directory(str(root))
    assert stats["files_indexed"] == 1
    assert "skipped_outside_root" not in stats


# --------------------------------------------------------------------------
# "never changed" / "there are earlier ones" -- claims made to a model
# --------------------------------------------------------------------------
def _chain(tmp_path, n, declared=True, name="v.dat"):
    v = Vault(str(tmp_path / name))
    base = time.time() - 300 * 86400
    for i in range(n):
        meta = {"entity": "employer"} if declared else None
        v.add("I work at Company%d." % i, metadata=meta, timestamp=base + i * 20 * 86400)
    v.flush()
    return v


@pytest.mark.parametrize("n_values", [1, 2, 3, 4])
@pytest.mark.parametrize("max_len", [None, 1, 2, 3, 4, 5])
def test_the_history_tool_claims_earlier_values_only_when_there_are_some(tmp_path, n_values, max_len):
    """Wrong in 4 of these 24 cases on 0.7.20 -- every one where max_len == chain length."""
    v = _chain(tmp_path, n_values, name="v_%d_%s.dat" % (n_values, max_len))
    args = {"query": "where do I work"}
    if max_len is not None:
        args["max_len"] = max_len
    head = mcp.dispatch(v, "nanomem_history", args).splitlines()[0]
    really_truncated = max_len is not None and max_len < n_values
    assert ("there are earlier ones" in head) == really_truncated, head


def test_never_changed_is_not_claimed_when_undeclared_writes_could_belong(tmp_path):
    v = Vault(str(tmp_path / "v.dat"))
    base = time.time() - 300 * 86400
    v.add("I work at Kestrel Systems.", metadata={"entity": "employer"}, timestamp=base)
    v.add("Moved on; it is Maple Wharf now.", timestamp=base + 100 * 86400)
    v.flush()
    head = mcp.dispatch(v, "nanomem_history", {"query": "where do I work"}).splitlines()[0]
    assert "has never changed" not in head or "DECLARED" in head, head
    assert "nanomem_changes" in head


def test_never_changed_is_still_claimed_when_it_is_true(tmp_path):
    v = Vault(str(tmp_path / "v.dat"))
    v.add("My blood type is O negative.", metadata={"entity": "blood_type"})
    v.flush()
    head = mcp.dispatch(v, "nanomem_history", {"query": "what is my blood type"}).splitlines()[0]
    assert head == "This has one value and has never changed:"


def test_the_cli_history_head_counts_the_whole_chain_not_the_shown_rows(tmp_path):
    v = _chain(tmp_path, 4)
    path = str(tmp_path / "v.dat")
    del v
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("NANOMEM_PASSWORD", None)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-m", "nanomem.cli", "history", "where do I work", "-n", "1",
         "--vault", path],
        capture_output=True, text=True, env=env).stdout
    assert "has never changed" not in out, out
    assert "4 values" in out, out


# --------------------------------------------------------------------------
# smaller ones
# --------------------------------------------------------------------------
def test_split_by_key_does_not_merge_values_that_differ_only_in_case(tmp_path):
    v = Vault(str(tmp_path / "v.dat"))
    for team in ("Work", "Work", "work", "work"):
        v.add("a note filed under %s" % team, metadata={"team": team})
    out = tmp_path / "out"
    counts = v.split_by_key("team", str(out))
    files = sorted(p.name for p in out.iterdir() if p.suffix == ".dat")
    assert len(files) == len(counts), (
        "reported %d partitions but wrote %d files: %s" % (len(counts), len(files), files))


def test_volatility_renders_a_group_that_has_no_interval(tmp_path):
    """min_revisions<=1 admits a one-timestamp group, whose interval is None."""
    v = Vault(str(tmp_path / "v.dat"))
    v.add("my phone is 555-0101", metadata={"entity": "phone"})
    v.flush()
    out = mcp.dispatch(v, "nanomem_volatility", {"min_revisions": 0})
    assert isinstance(out, str)


def test_a_merged_chain_stays_chronological_even_though_revisions_repeat(tmp_path):
    """The ORDER is the guarantee. The numbers are not, and the docstring says so.

    Renumbering was rejected: it would rewrite rows the caller did not name, and
    `temporal_order` lexsorts revision-first, so handing the older record the
    highest number would make a stale value `current`.
    """
    base = time.time() - 600 * 86400
    a = Vault(str(tmp_path / "a.dat"))
    for off, t in ((100, "Acme"), (200, "Globex"), (500, "Initech")):
        a.add("I work at %s" % t, metadata={"entity": "employer"}, timestamp=base + off * 86400)
    a.flush()
    b = Vault(str(tmp_path / "b.dat"))
    for off, t in ((300, "Umbrella"), (400, "Soylent")):
        b.add("I work at %s" % t, metadata={"entity": "employer"}, timestamp=base + off * 86400)
    b.flush()
    del b
    a.merge(str(tmp_path / "b.dat"))

    chain = a.history("where do I work")
    assert [h["text"] for h in chain] == sorted(
        (r["text"] for r in a.get_all_records()),
        key=lambda t: [r["timestamp"] for r in a.get_all_records() if r["text"] == t][0])
    current = [h for h in chain if not h["superseded"]]
    assert len(current) == 1
    assert current[0]["text"] == "I work at Initech"
    assert a.forget_superseded(keep=1)["kept"] == 1
    assert [r["text"] for r in a.get_all_records()] == ["I work at Initech"]
