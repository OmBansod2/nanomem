"""Enumerate the checkable claims in what `pip install nanomem` delivers.

A claim is a sentence asserting what the software DOES: a guarantee, a refusal,
a return value, a default, a limit. Explanation, history and rationale are not
claims and are not extracted -- the point is a list that can be closed, not a
list that is long.

Output is a draft registry. Every entry must end up either mapped to a test that
would fail if the claim became false, or deleted from the docs.
"""
import sys, os, re, json, inspect, pathlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quarantine_guard; quarantine_guard.enforce()
PKG = pathlib.Path(__file__).resolve().parents[2] / "nanomem_standalone"
sys.path.insert(0, str(PKG))

# Sentences that ASSERT behaviour.
ASSERTS = re.compile(
    r"\b(returns?|raises?|refuses?|never|always|must|cannot|will not|does not|"
    r"defaults? to|is the default|guarantee[sd]?|preserv\w+|survives?|"
    r"exempt\w*|ignored|silently|by default)\b", re.I)
# Sentences that are history or rationale, not a live claim.
HISTORY = re.compile(
    r"\b(through \d|until \d|at 0\.\d|in 0\.\d|was |were |used to|previously|"
    r"3\.0\.\d|measured|earlier|before this|reported|found by)\b", re.I)


def sentences(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    keep = []
    for l in text.splitlines():
        t = l.strip()
        if t.startswith("|") or t.startswith("#") or t.startswith("---"):
            keep.append("")          # a heading ENDS a sentence; it is not part of one
        else:
            keep.append(l)
    text = re.sub(r"\s+", " ", "\n".join(keep))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]


def claims_from(label, text):
    out = []
    for s in sentences(text):
        if not ASSERTS.search(s):
            continue
        out.append({"source": label, "claim": s[:400],
                    "historical": bool(HISTORY.search(s))})
    return out


def main():
    rows = claims_from("README.md", (PKG / "README.md").read_text())
    import nanomem
    from nanomem.vault import Vault
    from nanomem.embed import EmbeddingProvider
    from nanomem import mcp
    for cls in (Vault, EmbeddingProvider):
        for name, member in sorted(vars(cls).items()):
            if name.startswith("_"):
                continue
            # Only things that CARRY a nanomem docstring. A plain int attribute
            # inherits int.__doc__, and "Convert a number or string to an
            # integer" is CPython's claim, not this package's.
            if not (inspect.isfunction(member) or inspect.ismethod(member)
                    or isinstance(member, (property, staticmethod, classmethod))):
                continue
            doc = inspect.getdoc(member) or ""
            if doc:
                rows += claims_from(f"{cls.__name__}.{name}", doc)
    for t in mcp.TOOLS:
        rows += claims_from(f"mcp:{t['name']}", t.get("description", ""))
        for pname, p in (t.get("inputSchema", {}).get("properties") or {}).items():
            rows += claims_from(f"mcp:{t['name']}.{pname}", p.get("description", ""))

    live = [r for r in rows if not r["historical"]]

    # MERGE, DO NOT OVERWRITE. Re-extracting must not discard the mappings that
    # were confirmed by reading each test; it exists to surface claims the docs
    # GAINED. Existing entries are matched on their text, so a reworded claim
    # correctly appears as new and has to be re-confirmed.
    prev = {}
    out_path = pathlib.Path(__file__).with_name("claims_registry.json")
    if out_path.exists():
        for c in json.loads(out_path.read_text()).get("claims", []):
            prev[c["claim"]] = c
    kept = new = 0
    for i, r in enumerate(live):
        r["id"] = "C%03d" % (i + 1)
        r.pop("historical", None)
        old_entry = prev.get(r["claim"])
        if old_entry:
            r["id"] = old_entry.get("id", r["id"])
            r["test"] = old_entry.get("test")
            if old_entry.get("waived"):
                r["waived"] = old_entry["waived"]
            kept += 1
        else:
            r["test"] = None
            new += 1
    print(f"  carried over {kept} mapping(s); {new} claim(s) are NEW and unmapped")
    by_src = {}
    for r in live:
        by_src.setdefault(r["source"].split(".")[0], []).append(r)
    print(f"candidate sentences scanned : {len(rows) + 0}")
    print(f"historical/rationale dropped: {len(rows) - len(live)}")
    print(f"LIVE CLAIMS to account for  : {len(live)}")
    for k, v in sorted(by_src.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"    {k:<24} {len(v)}")
    n_t = sum(1 for c in live if c.get("test"))
    n_w = sum(1 for c in live if not c.get("test") and c.get("waived"))
    p = out_path
    p.write_text(json.dumps(
        {"summary": {"total": len(live), "with_test": n_t,
                     "waived_not_a_claim": n_w,
                     "still_unmapped": len(live) - n_t - n_w},
         "claims": live}, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
