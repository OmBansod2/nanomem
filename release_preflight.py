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
    if 'license = "Apache-2.0"' not in tml:
        bad("pyproject.toml does not declare Apache-2.0")
    lic = read("LICENSE")
    if "Apache License" not in lic or "Grant of Patent License" not in lic:
        bad("LICENSE is not the Apache-2.0 text")
    # The superseded texts must keep shipping: copies were distributed under
    # them and their recipients keep those terms.
    for superseded in ("LICENSE.agpl-3.0-or-later.md", "LICENSE.preview-v1.0.md"):
        if not os.path.exists(os.path.join(HERE, superseded)):
            bad(f"{superseded} is missing -- a licence copies were shipped "
                f"under must stay readable")
    if not os.path.exists(os.path.join(HERE, "NOTICE")):
        bad("NOTICE is missing -- Apache-2.0 section 4(d) requires it to travel "
            "with redistributions")
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


def check_assets_and_absent_claims():
    """The class of bug that shipped three times: advertising what is absent.

    RENAMED in 0.7.16. This was called `check_claims`, and so is the function
    370 lines below it, which Python resolves by keeping the LAST definition --
    so from the day the second one was written this one never ran, including the
    untracked-`.npz` rule its own comment says has silently eaten two fixtures.
    A preflight check that does not run is worse than one that was never
    written, because the passing line in the report says it did.
    """
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
            if "Apache-2.0" not in text:
                bad(f"{os.path.basename(w)} does not declare Apache-2.0")
            if not any("assets/write_classifier.npz" in n for n in names):
                bad(f"{os.path.basename(w)} is missing the classifier asset")
            if any(n.endswith("model.bin") for n in names):
                bad(f"{os.path.basename(w)} ships model.bin (139 MB, never opened)")
        # SAY SO WHEN IT PASSES. Until 0.7.16 this check reported nothing on
        # success, so a verified wheel and a wheel nobody looked at printed the
        # same thing: nothing. That is the same failure as a check that prints a
        # passing line while doing nothing -- which this file had, for 370 lines,
        # in `check_claims`. A report you cannot read the absence of is not a
        # report.
        NOTES.append("%s declares Apache-2.0, carries the classifier "
                     "asset and ships no model.bin" % os.path.basename(w))


def check_sdist(ver):
    """The sdist is the complete source and what conda-forge,
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
            if "Apache-2.0" not in text:
                bad(f"{base} does not declare Apache-2.0")
            for need in ("LICENSE", "NOTICE", "CHANGELOG.md", "MANIFEST.in",
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


#: What the shipped documents must still contain. A number here is a FLOOR, in
#: bytes, set well below the current size -- it is there to catch a document
#: losing a third of itself, not to freeze its length.
_DOC_FLOORS = {
    "README.md": 18000,
    "BENCHMARKS.md": 10000,
    "USER_MANUAL.md": 14000,
    "USER_MANUAL_DEVELOPER.md": 30000,
    "USER_MANUAL_PERSONAL.md": 8000,
    "SERVICES_AND_API_SPECIFICATION.md": 14000,
    "MULTIHOP_REASONING_AND_TOPOLOGY_GUIDE.md": 9000,
    "CHANGELOG.md": 60000,
}

#: Sections the README promises. Losing one is losing a feature's documentation.
_README_SECTIONS = (
    "## Quickstart",
    "## Or use it from Python",
    "## It tells you when the answer is cut short",
    "## Tell it what an attribute is",
    "## Run the demo",
    "## Run the tests",
    "## Use it from your own script",
    "## What makes it different from a vector store",
    "## Building RAG on it",
    "## CLI",
    "## What it does, measured",
    "## Documentation",
)


def check_docs_are_intact():
    """A document must not quietly lose most of itself.

    THIS CHECK EXISTS BECAUSE THE GATE MISSED IT. While 0.7.23 was being
    prepared, an unbounded string replacement deleted roughly 380 lines of
    README -- "Run the demo", "Run the tests", the CLI section and more -- and
    `release_preflight.py` printed PREFLIGHT PASSED on the result. Every check
    it had was about whether the claims still RESOLVED, and the surviving ones
    did; nothing asked whether the rest was still there. It was caught by reading
    a diffstat, which is not a gate.

    Two cheap guards, both of which would have caught it: a byte floor per
    shipped document, set well under its current size, and the list of sections
    the README is supposed to have.
    """
    for name, floor in sorted(_DOC_FLOORS.items()):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            bad(f"{name} is missing")
            continue
        size = os.path.getsize(path)
        if size < floor:
            bad(f"{name} is {size} bytes, below its floor of {floor} -- a "
                f"document does not normally lose that much; check the diff "
                f"before lowering this number")
    readme = read("README.md")
    missing = [h for h in _README_SECTIONS if h not in readme]
    if missing:
        bad("README.md has lost section(s): " + ", ".join(missing))
    if not missing:
        NOTES.append("all %d shipped documents are above their size floor and "
                     "the README still has its %d sections"
                     % (len(_DOC_FLOORS), len(_README_SECTIONS)))


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


def _pool_numbers(o, acc=None):
    """Every number in a results file, as the strings a doc would quote."""
    acc = set() if acc is None else acc
    if isinstance(o, dict):
        for v in o.values():
            _pool_numbers(v, acc)
    elif isinstance(o, list):
        for v in o:
            _pool_numbers(v, acc)
    elif isinstance(o, bool):
        pass
    elif isinstance(o, (int, float)):
        acc.add(str(o))
        acc.add(f"{float(o):.2f}".rstrip("0").rstrip("."))
        acc.add(f"{float(o):.1f}")
    elif isinstance(o, str):
        import re as _r
        acc |= set(_r.findall(r"\d+\.\d+", o))
    return acc


def check_claims():
    """EVERY CLAIM IN THE SHIPPED DOCS MUST BE CHECKED BY SOMETHING.

    Five documented claims were false at once in 0.7.11 -- `embedder=`,
    `vector_dtype=` and `group_floor_sim=` did not exist, `prune()` returned
    something other than what a reader would assume, and `history` was called
    authoritative when it was not -- and no test asserted any of them. They were
    found by an outside reviewer reading the docs, one at a time, by luck.

    `evidence/claims_registry.json` enumerates the checkable claims in the README
    and the public docstrings. Each must name a test that would fail if the claim
    became false, or a results file that measures it, or carry an explicit
    waiver saying why it is not a behavioural claim. A claim with none of those
    is a claim nothing is defending, and it blocks the release.
    """
    import json as _j
    reg_path = os.path.join(HERE, "evidence", "claims_registry.json")
    if not os.path.exists(reg_path):
        bad("evidence/claims_registry.json is missing; nothing is checking the "
            "documentation's claims")
        return
    try:
        claims = _j.load(open(reg_path))["claims"]
    except Exception as e:
        bad(f"claims registry unreadable: {e}")
        return
    import re as _r
    names = set()
    tdir = os.path.join(HERE, "tests")
    for f in sorted(os.listdir(tdir)):
        if f.startswith("test_") and f.endswith(".py"):
            names |= set(_r.findall(r"^def (test_[a-z0-9_]+)\(",
                                    read(os.path.join("tests", f)), _r.M))
    unbacked, dangling = [], []
    for c in claims:
        t, w = c.get("test"), c.get("waived")
        if not t and not w:
            unbacked.append(c["id"])
        elif t and t.startswith("measured:"):
            if not os.path.exists(os.path.join(HERE, t.split(":", 1)[1])):
                dangling.append(f"{c['id']} -> {t}")
        elif t and t not in names:
            dangling.append(f"{c['id']} -> {t}")
    if unbacked:
        bad(f"{len(unbacked)} documented claim(s) have neither a test nor a "
            f"waiver: " + ", ".join(unbacked[:10]))
    if dangling:
        bad(f"{len(dangling)} claim(s) name a test or measurement that does not "
            f"exist: " + ", ".join(dangling[:6]))
    # ...AND THE REGISTRY MUST STILL DESCRIBE TODAY'S DOCS.
    # Checking only the claims already listed would let a NEW claim be added to
    # the README and go unchecked, which is the exact hole this whole mechanism
    # exists to close. The extraction is repeated here, deliberately, so the
    # check needs nothing outside the package to run.
    _ASSERTS = _r.compile(
        r"\b(returns?|raises?|refuses?|never|always|must|cannot|will not|"
        r"does not|defaults? to|is the default|guarantee[sd]?|preserv\w+|"
        r"survives?|exempt\w*|ignored|silently|by default)\b", _r.I)
    _HISTORY = _r.compile(
        r"\b(through \d|until \d|at 0\.\d|in 0\.\d|was |were |used to|"
        r"previously|3\.0\.\d|measured|earlier|before this|reported|found by)\b",
        _r.I)

    def _sentences(text):
        text = _r.sub(r"```.*?```", " ", text, flags=_r.S)
        keep = []
        for ln in text.splitlines():
            t = ln.strip()
            keep.append("" if (t.startswith("|") or t.startswith("#")
                               or t.startswith("---")) else ln)
        text = _r.sub(r"\s+", " ", "\n".join(keep))
        return [x.strip() for x in _r.split(r"(?<=[.!?])\s+", text)
                if len(x.strip()) > 25]

    def _live(text):
        return {x[:400] for x in _sentences(text)
                if _ASSERTS.search(x) and not _HISTORY.search(x)}

    found = _live(read("README.md"))
    import inspect as _i
    try:
        sys.path.insert(0, HERE)
        from nanomem.vault import Vault as _V
        from nanomem.embed import EmbeddingProvider as _EP
        from nanomem import mcp as _mcp
        for cls in (_V, _EP):
            for nm, member in sorted(vars(cls).items()):
                if nm.startswith("_"):
                    continue
                if not (_i.isfunction(member) or _i.ismethod(member)
                        or isinstance(member, (property, staticmethod, classmethod))):
                    continue
                found |= _live(_i.getdoc(member) or "")
        for t in _mcp.TOOLS:
            found |= _live(t.get("description", ""))
            for _pn, _pv in (t.get("inputSchema", {}).get("properties") or {}).items():
                found |= _live(_pv.get("description", ""))
    except Exception as e:
        bad(f"could not re-extract claims to check the registry is current: {e}")
        found = set()

    known = {c["claim"] for c in claims}
    new_claims = sorted(found - known)
    if new_claims:
        bad(f"{len(new_claims)} claim(s) in the docs are not in "
            f"evidence/claims_registry.json, so nothing is checking them: "
            + " | ".join(x[:90] for x in new_claims[:3]))
    if not unbacked and not dangling and not new_claims:
        n_t = sum(1 for c in claims if c.get("test"))
        NOTES.append(f"all {len(claims)} documented claims accounted for "
                     f"({n_t} by a test or measurement), and the registry "
                     f"matches today's docs")


def check_registry_counts():
    """A prose count of the registry must equal the registry.

    The CHANGELOG said the registry holds 47 claims, 33 of them with a test. It
    holds 45 and 31: two claims were removed and the sentence describing them was
    not. An outside reviewer opened the file and counted, which is not a release
    process. Found by the third black-box review (M1).

    This is the claims mechanism failing on its own description: `check_claims`
    verifies every claim in the docs is mapped, and the number of claims is
    itself a claim in the docs that nothing checked.
    """
    reg = json.loads(read(os.path.join("evidence", "claims_registry.json")))
    entries = reg.get("claims", reg.get("entries", []))
    summary = reg.get("summary", {})
    actual = {"total": len(entries),
              "with_test": summary.get("with_test"),
              "waived": summary.get("waived_not_a_claim")}
    if summary.get("total") != actual["total"]:
        bad("claims_registry.json summary says total=%s but carries %d entries"
            % (summary.get("total"), actual["total"]))
    # UNQUALIFIED counts only. A released CHANGELOG entry describes the registry
    # as it stood then, and the registry grows with the docs -- so a sentence
    # carrying an explicit "as of <version>" is history and is left alone, while
    # a bare count is a live claim about the shipped file and must match it.
    # Getting this wrong once already rewrote 0.7.15's entry to 0.7.16's numbers.
    pat = re.compile(r"enumerates the (\d+) checkable claims.*?false \((\d+)\).*?"
                     r"behavioural claim \((\d+)\)", re.S)
    for name in ("README.md", "CHANGELOG.md", "USER_MANUAL.md",
                 "USER_MANUAL_DEVELOPER.md"):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        for m in pat.finditer(read(name)):
            said = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            want = (actual["total"], actual["with_test"], actual["waived"])
            if said != want:
                bad("%s says the registry holds %s claims (%s with a test, %s "
                    "waived); it holds %s (%s, %s)"
                    % ((name,) + said + want))


def check_generated_blocks():
    """EVERY GENERATED BLOCK THE REFRESH KNOWS ABOUT MUST STILL EXIST.

    `refresh_benchmarks.py` rewrites named `<!-- GENERATED: x -->` regions from
    the results files. If a region is deleted, the refresh prints a `!!` line
    and carries on -- and in a job that takes a quarter of an hour, that line is
    not read. The `latency` block was dropped while another block was being
    added, and the page shipped one release later still saying "the two blocks
    above are generated" with one block above it.

    A warning nobody reads is not a guard, so a missing region fails here.
    """
    doc = os.path.join(HERE, "BENCHMARKS.md")
    ref = os.path.join(os.path.dirname(HERE), "scratch", "refound",
                       "refresh_benchmarks.py")
    if not (os.path.exists(doc) and os.path.exists(ref)):
        return
    import re as _r
    named = set(_r.findall(r'^\s*"([a-z_]+)":\s', read(os.path.relpath(ref, HERE)),
                           _r.M))
    # only the keys of the `blocks` dict, which are the ones rewrite_doc emits
    body = read(os.path.relpath(ref, HERE))
    m = _r.search(r"blocks = \{(.*?)\n    \}", body, _r.S)
    if not m:
        return
    named = set(_r.findall(r'"([a-z_]+)":', m.group(1)))
    text = read("BENCHMARKS.md")
    present = set(_r.findall(r"<!-- GENERATED: ([a-z_]+) ", text))
    missing = sorted(named - present)
    if missing:
        bad("BENCHMARKS.md is missing generated block(s) that "
            "refresh_benchmarks.py writes: " + ", ".join(missing)
            + " -- the page will silently stop showing those numbers")
    else:
        NOTES.append(f"all {len(named)} generated blocks present in BENCHMARKS.md")


def check_citations():
    """A CLAIM WHOSE EVIDENCE CANNOT BE OPENED IS NOT A CHECKED CLAIM.

    Through 0.7.13 the docs and source cited ~80 paths under `scratch/refound/`,
    a directory that has never been committed to either remote. A reader who
    followed one found nothing, so every claim resting on it was unverifiable in
    practice however true it was. The files are published under `evidence/` from
    0.7.14; this keeps them reachable, because the failure mode is silent -- a
    renamed or dropped results file breaks a citation with nothing to notice.
    """
    e = os.path.join(HERE, "evidence")
    if not os.path.isdir(e):
        bad("evidence/ is missing, but the documentation cites it")
        return
    import re as _r
    cited, bad_refs = set(), []
    srcs = [os.path.join(HERE, "nanomem", f)
            for f in sorted(os.listdir(os.path.join(HERE, "nanomem")))
            if f.endswith(".py")]
    srcs += [os.path.join(HERE, f) for f in sorted(os.listdir(HERE))
             if f.endswith(".md")]
    for path in srcs:
        for m in _r.findall(
                r"evidence/(\{[^}]*\}[A-Za-z0-9_./*-]*|[A-Za-z0-9_./*-]+)",
                read(os.path.relpath(path, HERE))):
            rel = m.rstrip("/.,;)`")
            cited.add(rel)
            if "{" in rel and "}" in rel:
                head, rest = rel.split("{", 1)
                body, tail = rest.split("}", 1)
                names = [head + x + tail for x in body.split(",")]
            elif "*" in rel:
                import glob as _g
                names = [os.path.relpath(x, e) for x in _g.glob(os.path.join(e, rel))]
                if not names:
                    bad_refs.append(rel + " (matches nothing)")
                    continue
            else:
                names = [rel]
            for n in names:
                if not os.path.exists(os.path.join(e, n)):
                    bad_refs.append(n)
    # NOTHING QUARANTINED, AND NO CONTACT-SHAPED STRINGS, MAY APPEAR HERE.
    # The first version of this check looked at FILENAMES in the top directory
    # only. It passed, and 44 dev-persona results files were published carrying
    # fabricated but realistic contact details -- an email at a real
    # university's domain and a block of phone numbers -- inside their recorded
    # query text. A name check cannot see that, so this reads the files.
    import re as _rx
    _CONTACT = _rx.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        r"|\b\d{4}[ -]\d{2}[ -]\d{2}[ -]\d{2}\b"
        r"|\+\d{1,3}[ -]?\d{7,}")
    # Reserved test domains are the point of reserved test domains.
    _SAFE = ("@mailbox.test", "@example.com", "@example.org", "@example.net",
             "@test.invalid", "@localhost")
    leaked, contacts = [], {}
    for root, _dirs, files in os.walk(e):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, e)
            if any(k in rel.lower() for k in ("persona4", "heldout", "embeds")):
                leaked.append(rel)
                continue
            if os.path.splitext(f)[1] not in (".json", ".md", ".py", ".txt"):
                continue
            try:
                txt = open(full, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            found = {m for m in _CONTACT.findall(txt)
                     if not any(d in m for d in _SAFE)}
            if found:
                contacts[rel] = sorted(found)[:3]
    if leaked:
        bad("evidence/ contains files derived from held-out fixtures: "
            + ", ".join(sorted(leaked)[:8]))
    if contacts:
        bad(f"{len(contacts)} published evidence file(s) contain email- or "
            f"phone-shaped strings, which do not belong in a public repository "
            f"even when fabricated: "
            + ", ".join(f"{k} {v}" for k, v in sorted(contacts.items())[:4]))
    if bad_refs:
        bad(f"{len(bad_refs)} citation(s) point at files that are not in "
            f"evidence/: " + ", ".join(sorted(set(bad_refs))[:10]))
    else:
        NOTES.append(f"all {len(cited)} evidence citations resolve")


def check_benchmarks():
    """Published evidence must describe the ENGINE being shipped.

    BENCHMARKS.md presents its numbers as current, and a reader who re-runs one
    and cannot reproduce it has been told something false. That happened here:
    the head-to-head table was measured on engine 3.0.4 and quoted months later
    against 3.4.1, across three ranking changes -- nothing in the file recorded
    which engine produced it, so nothing could notice.

    The discriminator is the ENGINE version, not the package version. Docs and
    packaging releases bump `__version__` and change no measured behaviour;
    anything that moves ranking or storage bumps `ENGINE_VERSION`, and that is
    exactly when every number on that page has to be re-measured or removed.
    """
    engine = re.search(r'ENGINE_VERSION = "([^"]+)"',
                       read("nanomem/engine.py")).group(1)
    d = os.path.join(HERE, "benchmarks")
    if not os.path.isdir(d):
        return
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    if not files:
        bad("benchmarks/ exists but holds no results files")
        return
    import json as _json
    stale, unstamped = [], []
    for f in files:
        try:
            obj = _json.load(open(os.path.join(d, f)))
        except Exception as e:
            bad(f"benchmarks/{f} is not readable JSON: {e}")
            continue
        m = (obj or {}).get("measured_on") if isinstance(obj, dict) else None
        if not m or not m.get("engine"):
            unstamped.append(f)
        elif str(m["engine"]) != engine:
            stale.append(f"{f} (engine {m['engine']})")
    for f in unstamped:
        bad(f"benchmarks/{f} records no measured_on.engine, so nothing can "
            f"tell whether its numbers still describe this build")
    if stale:
        bad(f"engine is {engine} but these results were measured on an "
            f"older one, and BENCHMARKS.md presents them as current: "
            + ", ".join(stale) + " -- re-run them or drop the claims")
    # ...AND THE SAME RULE IN `evidence/`, WHICH IS WHERE INDEX.md SAYS IT HOLDS.
    # This check scanned `benchmarks/` only, while `evidence/INDEX.md` states the
    # rule over "every results file" and names this guard as what enforces it.
    # `evidence/` keeps its own copies of some results files and the refresh
    # stages into `benchmarks/`, so two of them sat at engine 3.4.3 through a
    # release that shipped 3.4.4 -- one of them cited by the README for a live
    # claim. Found by the fourth black-box review (M1). A guard whose scope is
    # narrower than the rule it is named for is the rule not being enforced.
    edir = os.path.join(HERE, "evidence")
    e_stale = []
    for f in sorted(os.listdir(edir)):
        if not f.endswith(".json"):
            continue
        try:
            obj = _json.load(open(os.path.join(edir, f)))
        except Exception:
            continue
        m = (obj or {}).get("measured_on") if isinstance(obj, dict) else None
        if m and m.get("engine") and str(m["engine"]) != engine:
            e_stale.append(f"{f} (engine {m['engine']})")
    if e_stale:
        bad(f"engine is {engine} but these evidence/ files were measured on an "
            f"older one and are cited as current: " + ", ".join(e_stale)
            + " -- re-run them or drop the claims")
    if not stale and not unstamped and not e_stale:
        NOTES.append(f"{len(files)} benchmark results all measured on "
                     f"engine {engine}, and every stamped evidence/ file agrees")
    # AND every number the page quotes must be IN the file it cites. The
    # citation alone is not enough: one refresh after this guard was written,
    # BENCHMARKS.md still read "2.09 ms" while its own results file said 2.15 --
    # ordinary timing variance, but the page quotes two decimals and a reader
    # who opens the file to check finds a different number. Numbers get retyped;
    # retyping is where they drift.
    doc_p = os.path.join(HERE, "BENCHMARKS.md")
    if os.path.exists(doc_p):
        import re as _re2
        text = read(doc_p)
        pool = set()
        for f in files:
            try:
                pool |= _pool_numbers(_json.load(open(os.path.join(d, f))))
            except Exception:
                pass
        # decimals only: integers appear as counts, years and version parts and
        # would produce noise rather than signal
        quoted = set(_re2.findall(r"(?<![\w.])(\d+\.\d+)(?=\s*(?:ms|%|\s|\)|,|$))",
                                  text, _re2.M))
        # ROUNDING IS NOT DRIFT. The page rounds for readability -- 0.0149 s of
        # reopen is printed 0.015 -- so match numerically at the precision the
        # page itself used, not as strings. A check that cries wolf on rounding
        # is a check someone deletes.
        pool_f = []
        for x in pool:
            try:
                pool_f.append(float(x))
            except ValueError:
                pass
        unbacked = []
        for q in sorted(quoted):
            dp = len(q.split(".")[1])
            qf = float(q)
            tol = 0.5 * (10 ** -dp) + 1e-12      # "rounds to", half a unit in
            if not any(abs(v - qf) <= tol for v in pool_f):   # the last place
                unbacked.append(q)
        # version strings and figures quoted from the design specs, not results
        allow = {"0.7", "3.4", "1.8", "3.9", "3.11", "3.13", "3.12",
                 "19.0", "13.9"}
        unbacked = [q for q in unbacked if q not in allow]
        if unbacked:
            bad("BENCHMARKS.md quotes numbers that are in no results file: "
                + ", ".join(unbacked[:12])
                + " -- re-run refresh_benchmarks.py and update the prose")
        else:
            NOTES.append(f"every decimal BENCHMARKS.md quotes appears in a "
                         f"results file ({len(quoted)} checked)")

    # every results file BENCHMARKS.md cites must actually exist
    doc = os.path.join(HERE, "BENCHMARKS.md")
    if os.path.exists(doc):
        import re as _re
        cited = set(_re.findall(r"benchmarks/([A-Za-z0-9_]+\.json)", read(doc)))
        missing = sorted(c for c in cited if c not in files)
        if missing:
            bad("BENCHMARKS.md cites results files that are not published: "
                + ", ".join(missing))
        elif cited:
            NOTES.append(f"BENCHMARKS.md cites {len(cited)} results files, all present")


def main():
    offline = "--offline" in sys.argv
    ver = check_versions()
    check_test_count()
    check_metadata(offline)
    check_docs_are_intact()
    check_readme_links()
    check_readme_sample()
    check_wheel(ver)
    check_sdist(ver)
    check_assets_and_absent_claims()
    check_claims()
    check_registry_counts()
    check_generated_blocks()
    check_citations()
    check_benchmarks()

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
