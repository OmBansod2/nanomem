#!/usr/bin/env python3
"""Refuse to publish nanomem while anything is provably wrong.

Run it before `twine upload`. It exits non-zero and prints every problem, so
"did we remember everything?" stops being something anyone has to hold in their
head.

Every check here exists because that exact thing was ALREADY WRONG at some point
in this project and shipped, or nearly did:

  * the wheel declared Apache-2.0 while bundling an All-Rights-Reserved LICENSE
  * README announced "Package 0.3.0 / engine 3.0.3" four releases late
  * it claimed "237 tests" when there were 485
  * `hello@nanomem.dev` and two URLs that do not resolve
  * `EmbeddingProvider.dim` hard-coded, so `embed_model=` did nothing
  * a 139 MB model that nothing opened
  * seven README links that resolve against pypi.org and 404 there
  * `*.npz` in .gitignore eating a shipped asset -- twice

Usage:  python3 release_preflight.py [--offline]
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = []
NOTES = []


def bad(msg):
    PROBLEMS.append(msg)


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


def check_versions():
    init = read("nanomem/__init__.py")
    ver = re.search(r'^__version__ = "([^"]+)"', init, re.M).group(1)
    eng = re.search(r'ENGINE_VERSION = "([^"]+)"', read("nanomem/engine.py")).group(1)
    NOTES.append(f"version {ver}, engine {eng}")

    for doc in ("README.md", "USER_MANUAL_DEVELOPER.md"):
        if not os.path.exists(os.path.join(HERE, doc)):
            continue
        text = read(doc)
        m = re.search(r"Package (\d+\.\d+\.\d+)", text)
        if m and m.group(1) != ver:
            bad(f"{doc} announces package {m.group(1)}, source is {ver}")
        for m in re.finditer(r"#\s*(\d+\.\d+\.\d+)\s+(\d+\.\d+\.\d+)", text):
            if m.group(1) != ver or m.group(2) != eng:
                bad(f"{doc} sample output says {m.group(1)} {m.group(2)}, "
                    f"source is {ver} {eng}")
    if f"## {ver}" not in read("CHANGELOG.md"):
        bad(f"CHANGELOG.md has no entry for {ver}")
    return ver


def check_test_count():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                       cwd=HERE, capture_output=True, text=True)
    m = re.search(r"(\d+) passed", r.stdout)
    if not m:
        bad("could not read a pass count from pytest")
        return
    passed = int(m.group(1))
    sk = re.search(r"(\d+) skipped", r.stdout)
    skipped = int(sk.group(1)) if sk else 0
    # The docs claim a SUITE SIZE, so compare against one. A clone of the public
    # mirror skips 16 tests that want research artifacts under scratch/refound/,
    # which do not ship -- 498 pass there and 514 here, off the same suite. Held
    # to the pass count alone, the claim could only be true in one of the two.
    n = passed + skipped
    failed = re.search(r"(\d+) failed", r.stdout)
    NOTES.append(f"{n} tests" + (f" ({passed} pass, {skipped} skip here)"
                                 if skipped else " pass"))
    if failed:
        bad(f"{failed.group(1)} tests FAIL")
    for doc in ("README.md", "USER_MANUAL_DEVELOPER.md"):
        if not os.path.exists(os.path.join(HERE, doc)):
            continue
        for m2 in re.finditer(r"\*{0,2}(\d{3})\*{0,2} tests", read(doc)):
            if int(m2.group(1)) != n:
                bad(f"{doc} claims {m2.group(1)} tests, the suite has {n}")


def check_metadata(offline):
    """Checks the PARSED metadata, never the raw text.

    The first version of this grepped pyproject.toml and flagged the comment
    that explains why the dead URLs were removed. A check that fires on its own
    documentation trains you to ignore it, which is worse than not having it.
    """
    tml = read("pyproject.toml")
    try:
        import tomllib
        proj = tomllib.loads(tml).get("project", {})
    except Exception as e:
        bad(f"pyproject.toml does not parse: {e}")
        proj = {}

    for a in proj.get("authors", []):
        for field, pats, why in (("email", ("hello@nanomem.dev",), "a placeholder email"),
                                 ("name", ("nanomem maintainers",), "a placeholder name")):
            v = (a.get(field) or "").strip()
            if not v:
                bad(f"project.authors is missing {field}")
            elif any(p in v.lower() for p in pats):
                bad(f"project.authors {field} is {why}: {v!r}")
    if not proj.get("authors"):
        bad("project.authors is empty -- PyPI will show no maintainer")

    for label, url in (proj.get("urls") or {}).items():
        if "nanomem-ai/nanomem" in url or url.rstrip("/") == "https://nanomem.dev":
            bad(f"project.urls {label} is a known-dead placeholder: {url}")
        elif not offline:
            try:
                import urllib.request
                urllib.request.urlopen(url, timeout=10)
            except Exception:
                bad(f"project.urls {label} is not reachable: {url}")
    if re.search(r"^\s*(TODO|FIXME)", tml, re.M):
        bad("a TODO/FIXME is left in packaging metadata")
    if 'license = "AGPL-3.0-or-later"' not in tml:
        bad("pyproject.toml does not declare AGPL-3.0-or-later")
    lic = read("LICENSE")
    if "GNU AFFERO GENERAL PUBLIC LICENSE" not in lic:
        bad("LICENSE is not the AGPL text")
    if "All Rights Reserved" in lic:
        bad("LICENSE still says All Rights Reserved -- this contradicted the "
            "wheel metadata once already")
    if not offline:
        for m in re.finditer(r'=\s*"(https://[^"]+)"', tml):
            url = m.group(1)
            try:
                import urllib.request
                urllib.request.urlopen(url, timeout=10)
            except Exception:
                bad(f"declared URL is not reachable: {url}")


def check_claims():
    """The class of bug that shipped three times: advertising what is absent."""
    src = "\n".join(read(os.path.join("nanomem", f))
                    for f in sorted(os.listdir(os.path.join(HERE, "nanomem")))
                    if f.endswith(".py"))
    # The AFFIRMATIVE claim only. `embed.py` now carries a docstring explaining
    # that there are NO bundled weights, and the first version of this check
    # flagged that explanation.
    for m in re.finditer(r"[^.]*bundled[^.]*neural weights[^.]*\.", src, re.I):
        sentence = m.group(0)
        if re.search(r"\b(no|not|never|without)\b", sentence, re.I):
            continue
        bad(f"source advertises bundled neural weights: {sentence.strip()[:90]!r}")
    assets = os.path.join(HERE, "nanomem", "assets")
    referenced = set(re.findall(r'assets["\']?,\s*["\']([\w.\-]+)["\']', src))
    for name in referenced:
        if not os.path.exists(os.path.join(assets, name)):
            bad(f"code references assets/{name}, which is not present")
    if not os.path.exists(os.path.join(assets, "write_classifier.npz")):
        bad("assets/write_classifier.npz is missing -- a clone without it fails "
            "3 write-gate tests")
    # Not just the classifier: EVERY .npz the package or the suite loads. The
    # blanket `*.npz` rule in .gitignore has now eaten two of them -- the
    # classifier head, where 3 write-gate tests FAILED loudly, and the probe
    # fixture, where `_fixture()` calls pytest.skip and a clone reports green
    # with the coverage gone. The silent one went unnoticed far longer.
    for root in ("nanomem", "tests"):
        for dirpath, _dirs, files in os.walk(os.path.join(HERE, root)):
            if "__pycache__" in dirpath:
                continue
            for f in sorted(files):
                if not f.endswith(".npz"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, f), HERE)
                r = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                                   cwd=HERE, capture_output=True)
                if r.returncode != 0:
                    bad(f"{rel} is NOT tracked by git -- clones get it missing")


def _artifacts(pattern, ver):
    """Current build artifacts, complaining about any stale one beside them.

    `twine upload dist/*` publishes everything it finds, so a leftover from the
    previous version in that directory is not clutter -- it is an accidental
    release.
    """
    import glob
    found = sorted(set(glob.glob(os.path.join(HERE, "dist", pattern))
                       + glob.glob(os.path.join(HERE, pattern))))
    current = []
    for f in found:
        if ver in os.path.basename(f):
            current.append(f)
        else:
            bad(f"stale artifact {os.path.relpath(f, HERE)} (source is {ver}) "
                f"-- `twine upload dist/*` would publish it too")
    return current


def check_wheel(ver):
    whls = _artifacts("nanomem-*.whl", ver)
    if not whls:
        NOTES.append("no wheel built yet (python3 -m build)")
        return
    import zipfile
    for w in whls:
        with zipfile.ZipFile(w) as z:
            names = z.namelist()
            meta = next((n for n in names if n.endswith("METADATA")), None)
            text = z.read(meta).decode() if meta else ""
            if "AGPL-3.0-or-later" not in text:
                bad(f"{os.path.basename(w)} does not declare AGPL-3.0-or-later")
            if not any("assets/write_classifier.npz" in n for n in names):
                bad(f"{os.path.basename(w)} is missing the classifier asset")
            if any(n.endswith("model.bin") for n in names):
                bad(f"{os.path.basename(w)} ships model.bin (139 MB, never opened)")


def check_sdist(ver):
    """The sdist is the AGPL "corresponding source" and what conda-forge,
    Debian and Homebrew actually build from -- they run the suite out of the
    tarball. There was none at all until 0.6.3, and the setuptools default
    (`packages = ["nanomem"]` and nothing else) would have shipped no tests,
    no CHANGELOG and none of the documents README links to: a tarball nobody
    downstream can verify. MANIFEST.in is what puts them there.
    """
    import tarfile
    sds = _artifacts("nanomem-*.tar.gz", ver)
    if not sds:
        NOTES.append("no sdist built yet (python3 -m build --sdist)")
        return
    for sd in sds:
        base = os.path.basename(sd)
        with tarfile.open(sd) as t:
            members = t.getmembers()
            names = {m.name.split("/", 1)[-1] for m in members}
            pkg = next((m for m in members if m.name.endswith("PKG-INFO")), None)
            text = t.extractfile(pkg).read().decode() if pkg else ""
            if "AGPL-3.0-or-later" not in text:
                bad(f"{base} does not declare AGPL-3.0-or-later")
            for need in ("LICENSE", "CHANGELOG.md", "MANIFEST.in",
                         "nanomem/assets/write_classifier.npz",
                         "tests/data/adjacent_attributes.npz"):
                if need not in names:
                    bad(f"{base} is missing {need}")
            if not any(n.startswith("tests/test_") for n in names):
                bad(f"{base} ships no tests -- nothing downstream can verify it")
            for n in sorted(names):
                if n.endswith((".dat", ".whl", ".pyc")) or "__pycache__" in n:
                    bad(f"{base} ships junk from this machine: {n}")
        NOTES.append(f"{base} carries the suite and both assets")


def check_readme_links():
    """PyPI renders README as the project page but does NOT resolve relative
    links: `](USER_MANUAL.md)` resolves against pypi.org and 404s. Seven of
    those were in the METADATA of the wheel built before this check existed.
    """
    for target in sorted(set(re.findall(r"\]\((?!https?:|#|mailto:)([^)]+)\)",
                                        read("README.md")))):
        bad(f"README links {target!r} relatively -- dead on the PyPI project "
            f"page, which does not resolve relative paths; use the full URL")


def check_readme_sample():
    """RUN the README's opening sample and diff it against its own comments.

    It is the first thing anyone sees after `pip install nanomem`, so it is the
    one piece of code in the project most likely to be tried and least likely to
    be tested. The first draft of it raised IndexError on the `as_of` line --
    the facts were all written at `now`, so "200 days ago" preceded every one of
    them -- and every printed line was a guess at what it would say.

    Convention: each `print(...)` is followed by `# <exactly what it prints>`.
    """
    import subprocess as sp
    import tempfile
    m = re.search(r"```python\n(.*?)```", read("README.md"), re.S)
    if not m:
        bad("README has no python sample to check")
        return
    block = m.group(1)
    want = [l[2:] for l in block.splitlines() if l.startswith("# ")]
    if not want:
        bad("the README sample prints nothing it commits to")
        return
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "s.py"), "w").write(block)
        env = dict(os.environ, PYTHONPATH=HERE)
        r = sp.run([sys.executable, "s.py"], cwd=d, env=env,
                   capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        bad("the README sample does not run: "
            + (r.stderr.strip().splitlines() or ["?"])[-1])
        return
    got = r.stdout.splitlines()
    if got != want:
        bad(f"the README sample prints {got} but its comments claim {want}")
    else:
        NOTES.append(f"the README sample runs and prints its {len(want)} "
                     f"claimed lines")


def main():
    offline = "--offline" in sys.argv
    ver = check_versions()
    check_test_count()
    check_metadata(offline)
    check_readme_links()
    check_readme_sample()
    check_claims()
    check_wheel(ver)
    check_sdist(ver)

    for n in NOTES:
        print(f"  ok    {n}")
    if not PROBLEMS:
        print("\nPREFLIGHT PASSED — safe to publish.")
        return 0
    print(f"\n{len(PROBLEMS)} PROBLEM(S) — DO NOT PUBLISH:\n")
    for p in PROBLEMS:
        print(f"  x  {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
