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
    n, failed = int(m.group(1)), re.search(r"(\d+) failed", r.stdout)
    NOTES.append(f"{n} tests pass")
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
    wc = os.path.join(assets, "write_classifier.npz")
    if not os.path.exists(wc):
        bad("assets/write_classifier.npz is missing -- a clone without it fails "
            "3 write-gate tests")
    else:
        r = subprocess.run(["git", "ls-files", "--error-unmatch",
                            "nanomem/assets/write_classifier.npz"],
                           cwd=HERE, capture_output=True)
        if r.returncode != 0:
            bad("assets/write_classifier.npz is NOT tracked by git -- clones get "
                "a broken package (this is what *.npz in .gitignore did)")


def check_wheel(ver):
    import glob
    whls = glob.glob(os.path.join(HERE, "nanomem-*.whl"))
    if not whls:
        NOTES.append("no wheel beside the source yet (build one before upload)")
        return
    for w in whls:
        if ver not in os.path.basename(w):
            bad(f"stale wheel beside the source: {os.path.basename(w)} "
                f"(source is {ver})")
    import zipfile
    for w in whls:
        if ver not in os.path.basename(w):
            continue
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


def main():
    offline = "--offline" in sys.argv
    ver = check_versions()
    check_test_count()
    check_metadata(offline)
    check_claims()
    check_wheel(ver)

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
