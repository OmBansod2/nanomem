"""Train and evaluate nanomem's write classifier v2 (the "should this be stored?" gate).

Model:  logistic head over [ L2-normalised nomic-embed-text vector (768) ; 24 generic
        surface features from nanomem.classifier.extract_surface_features ].
        numpy only -- no sklearn, no torch.

Protocol (pre-registered, in this order):
  1. leave-one-persona-out (LOPO) on clean_chat_benchmark.json (3 personas) selects the
     L2 strength, the class weighting and the decision threshold.  Selection metric:
     mean LOPO accuracy, tie-broken by mean LOPO F1.
  2. the selected configuration is retrained on all 3 personas and written to
     nanomem/assets/write_classifier.npz.
  3. the frozen model is evaluated ONCE on clean_chat_benchmark_heldout.json
     (2 unseen personas) with --heldout.

Usage
-----
  python3 train_write_classifier.py                 # LOPO + train + save asset
  python3 train_write_classifier.py --heldout       # ... and the one held-out evaluation
  python3 train_write_classifier.py --emit-fallback # print the literals for classifier.py
  python3 train_write_classifier.py --no-save       # LOPO only, do not touch the asset

Embeddings are cached in clean_chat_embeds.npz / clean_chat_embeds_heldout.npz (nomic-embed-text
via the local Ollama daemon); the cache is spot-checked against a live re-embedding on every run.
Results are written to write_classifier_v2_results.json.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "4D llm", "nanomem_standalone")
if not os.path.isdir(PKG):
    PKG = "<repo>/nanomem_standalone"
sys.path.insert(0, PKG)

from nanomem.classifier import (  # noqa: E402
    extract_surface_features, FEATURE_NAMES, EMBED_DIM, WriteClassifier, INDICATOR_FEATURES,
    LENGTH_FEATURES,
)

TRAIN_JSON = os.path.join(HERE, "clean_chat_benchmark.json")
HELDOUT_JSON = os.path.join(HERE, "clean_chat_benchmark_heldout.json")
TRAIN_EMB = os.path.join(HERE, "clean_chat_embeds.npz")
HELDOUT_EMB = os.path.join(HERE, "clean_chat_embeds_heldout.npz")
ASSET = os.path.join(PKG, "nanomem", "assets", "write_classifier.npz")
RESULTS = os.path.join(HERE, "write_classifier_v2_results.json")

# A hand-written smoke probe of SHORT utterances.  Both chat benchmarks are made of long,
# discursive turns (only 1 of the 106 training turns of <=20 words is a keeper), so neither
# can tell you whether the gate keeps a one-line personal fact -- the thing a memory library
# is for.  Generic vocabulary, written here, used for reporting only: it never takes part in
# model selection.
PROBE_FACTS = [
    "My blood type is O negative.", "I'm allergic to penicillin.", "My sister's name is Ana.",
    "I moved to Lisbon in March.", "My rent is 1850 a month.", "I work night shifts.",
    "My passport expires in 2029.", "I don't eat pork.", "My gym is on Bell Street.",
    "I have a cat.", "My daughter starts school in September.", "I use a standing desk.",
    "My bank is Santander.", "I turned 34 last week.", "My office is on the third floor.",
]
PROBE_CHAFF = [
    "haha yes", "thanks!", "what time is it", "can you write that again", "ok sounds good",
    "no worries", "hmm maybe", "lol", "that's wild", "sure, go ahead", "good morning",
    "yeah exactly", "I am so tired today", "I'm going to bed", "I have to run", "I'm just venting",
]
PROBE_EMB = os.path.join(HERE, "short_probe_embeds.npz")

OLLAMA = os.getenv("NANOMEM_EMBED_URL", "http://localhost:11434/api/embed")
MODEL = "nomic-embed-text"
SEED = 0


# --------------------------------------------------------------------------- data
def embed_texts(texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        req = urllib.request.Request(
            OLLAMA,
            data=json.dumps({"model": MODEL, "input": chunk}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            out.append(np.array(json.loads(r.read())["embeddings"], dtype=np.float32))
    return np.vstack(out)


def load_set(path, emb_path, check=True):
    """Returns texts, y, persona, X (L2-normalised embeddings), F (surface features)."""
    B = json.load(open(path))
    turns = [(s["persona"], t) for s in B["sets"] for t in s["turns"]]
    texts = [t["text"] for _, t in turns]
    y = np.array([int(t["should_store"]) for _, t in turns])
    persona = np.array([p for p, _ in turns])
    if os.path.exists(emb_path):
        X = np.load(emb_path)["X"].astype(np.float32)
        if X.shape[0] != len(texts):
            X = embed_texts(texts)
            np.savez(emb_path, X=X)
        elif check:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(texts), size=min(4, len(texts)), replace=False)
            live = embed_texts([texts[i] for i in idx])
            live /= np.linalg.norm(live, axis=1, keepdims=True)
            cached = X[idx] / np.linalg.norm(X[idx], axis=1, keepdims=True)
            cos = float(np.min(np.sum(live * cached, axis=1)))
            assert cos > 0.999, f"embedding cache {emb_path} is stale (min cos {cos:.4f})"
    else:
        X = embed_texts(texts)
        np.savez(emb_path, X=X)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    F = np.stack([extract_surface_features(t) for t in texts]).astype(np.float32)
    return texts, y, persona, X, F


# --------------------------------------------------------------------------- model
def fit_logistic(Z, y, l2=1e-2, class_weight=1.0, max_iter=60, tol=1e-11,
                 intercept_scale=10.0):
    """L2-regularised logistic regression solved to convergence by damped Newton.

    numpy only.  The intercept is a constant column of `intercept_scale`, so its own
    penalty is l2 / intercept_scale**2 (i.e. effectively unpenalised).  When the design
    is wider than it is tall (the 768+24-d head on ~233 rows) the solve runs in the dual
    (representer theorem, n x n system) which is both faster and better conditioned;
    otherwise it runs in the primal.  Deterministic: no random init, no step schedule.
    """
    n, d = Z.shape
    Za = np.hstack([np.asarray(Z, dtype=np.float64), np.full((n, 1), float(intercept_scale))])
    yf = np.asarray(y, dtype=np.float64)
    sw = np.where(yf == 1, float(class_weight), 1.0)
    sw = sw / sw.mean()

    def obj_from_f(f, wsq):
        z = np.clip(f, -60, 60)
        ce = np.logaddexp(0.0, z) - yf * z
        return float(np.sum(sw * ce) / n + 0.5 * l2 * wsq)

    if d + 1 > n:                                   # --- dual (kernel) Newton
        K = Za @ Za.T
        a = np.zeros(n)
        f = np.zeros(n)
        J = obj_from_f(f, 0.0)
        I = np.eye(n)
        for _ in range(max_iter):
            p = 1.0 / (1.0 + np.exp(-np.clip(f, -60, 60)))
            g = sw * (p - yf) / n
            r = g + l2 * a                           # K^-1 * gradient
            if float(np.max(np.abs(r))) < tol:
                break
            W = sw * p * (1.0 - p) / n
            step = np.linalg.solve(W[:, None] * K + l2 * I, r)
            t = 1.0
            for _ls in range(40):
                a_new = a - t * step
                f_new = K @ a_new
                J_new = obj_from_f(f_new, float(a_new @ f_new))
                if J_new <= J + 1e-12:
                    break
                t *= 0.5
            if J_new > J + 1e-12:
                break
            a, f, J = a_new, f_new, J_new
        w_aug = Za.T @ a
    else:                                            # --- primal Newton
        w_aug = np.zeros(d + 1)
        f = np.zeros(n)
        J = obj_from_f(f, 0.0)
        I = np.eye(d + 1)
        for _ in range(max_iter):
            p = 1.0 / (1.0 + np.exp(-np.clip(f, -60, 60)))
            g = sw * (p - yf) / n
            grad = Za.T @ g + l2 * w_aug
            if float(np.max(np.abs(grad))) < tol:
                break
            W = sw * p * (1.0 - p) / n
            H = Za.T @ (W[:, None] * Za) + l2 * I
            step = np.linalg.solve(H, grad)
            t = 1.0
            for _ls in range(40):
                w_new = w_aug - t * step
                f_new = Za @ w_new
                J_new = obj_from_f(f_new, float(w_new @ w_new))
                if J_new <= J + 1e-12:
                    break
                t *= 0.5
            if J_new > J + 1e-12:
                break
            w_aug, f, J = w_new, f_new, J_new
    return w_aug[:d].copy(), float(w_aug[d] * intercept_scale)


LENGTH_FLOOR_PCT = 25.0   # see the FEATURE_NAMES note in nanomem/classifier.py


def clip_bounds(F, length_floor_pct=LENGTH_FLOOR_PCT):
    """Per-feature training range; features are clipped to it before scoring so the linear
    head never extrapolates off the end of a feature it only saw in a narrow band.

    The two length features get their floor at `length_floor_pct` of the training
    distribution instead of the observed minimum: chat corpora make "short" an almost
    perfect proxy for "chaff" (1 keeper among the 106 training turns of <=20 words), and a
    head that can see how short a short message is simply learns that proxy and then drops
    every one-line personal fact.  With the floor it can still use length as evidence *for*
    storing and cannot use shortness as evidence against."""
    lo = np.minimum(F.min(0), 0.0).astype(np.float32)
    hi = F.max(0).astype(np.float32)
    for i in INDICATOR_FEATURES:          # 0/1 indicators always keep the full [0, 1] range
        hi[i] = 1.0
    for n in LENGTH_FEATURES:
        i = FEATURE_NAMES.index(n)
        lo[i] = np.float32(np.percentile(F[:, i], length_floor_pct))
    return lo, hi


def standardise(F):
    mu = F.mean(0)
    sd = F.std(0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def apply_rules(pred, texts):
    """Overlay the three shipped rule layers on head predictions, so every number we
    report is the decision function that actually ships (head + rules)."""
    out = pred.copy()
    for i, t in enumerate(texts):
        r = WriteClassifier._rule_layer(t)
        if r is not None:
            out[i] = bool(r["should_store"])
    return out


def metrics(pred, y):
    pred = pred.astype(bool); yb = y.astype(bool)
    tp = int((pred & yb).sum()); fp = int((pred & ~yb).sum())
    fn = int((~pred & yb).sum()); tn = int((~pred & ~yb).sum())
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}


def build(Ztr, ytr, Zte, l2, cw, mu=None, sd=None):
    w, b = fit_logistic(Ztr, ytr, l2=l2, class_weight=cw)
    return (Zte @ w + b), (w, b)


# --------------------------------------------------------------------------- LOPO
def lopo(X, F, y, persona, use_emb, l2, cw, threshold, emb_scale=1.0, use_surf=True,
         texts=None):
    """The embedding block is L2-normalised, so each of its 768 dims has std ~1/sqrt(768);
    `emb_scale` puts it on a comparable footing with the standardised surface block, which
    otherwise dominates the shared L2 penalty.  It is folded back into w_emb when saving."""
    accs, f1s, per = [], [], {}
    for hold in sorted(set(persona.tolist())):
        tr, te = persona != hold, persona == hold
        lo, hi = clip_bounds(F[tr])
        Ftr_raw, Fte_raw = F[tr], np.clip(F[te], lo, hi)
        mu, sd = standardise(Ftr_raw)
        Ftr, Fte = (Ftr_raw - mu) / sd, (Fte_raw - mu) / sd
        btr = ([X[tr] * emb_scale] if use_emb else []) + ([Ftr] if use_surf else [])
        bte = ([X[te] * emb_scale] if use_emb else []) + ([Fte] if use_surf else [])
        Ztr, Zte = np.hstack(btr), np.hstack(bte)
        z, _ = build(Ztr, y[tr], Zte, l2, cw)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        pred = p >= threshold
        if texts is not None:
            pred = apply_rules(pred, [texts[i] for i in np.flatnonzero(te)])
        m = metrics(pred, y[te])
        accs.append(m["accuracy"]); f1s.append(m["f1"]); per[hold] = m
    return float(np.mean(accs)), float(np.mean(f1s)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", action="store_true", help="run the ONE held-out evaluation")
    ap.add_argument("--emit-fallback", action="store_true", help="print literals for classifier.py")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-check", action="store_true", help="skip the live embedding-cache spot check")
    ap.add_argument("--via-vault", action="store_true",
                    help="also score both sets through the shipped Vault.should_store() path "
                         "(live embedder, integration check + p50 decision latency)")
    args = ap.parse_args()

    texts, y, persona, X, F = load_set(TRAIN_JSON, TRAIN_EMB, check=not args.no_check)
    print(f"train: {len(y)} turns, {int(y.sum())} positive, personas {sorted(set(persona.tolist()))}")
    print(f"features: {len(FEATURE_NAMES)} surface + {EMBED_DIM} embedding dims")

    grid_l2 = [3e-3, 1e-2, 3e-2, 1e-1]
    grid_cw = [1.0, float((y == 0).sum() / max((y == 1).sum(), 1))]
    grid_thr = [0.4, 0.5, 0.6]
    grid_scale = [1.0, 5.0, float(np.sqrt(EMBED_DIM))]

    report = {"lopo": {}, "grid": []}
    best = {}
    for use_emb, name in ((True, "full"), (False, "surface")):
        rows = []
        for l2 in grid_l2:
            for cw in grid_cw:
                for thr in grid_thr:
                    for sc in (grid_scale if use_emb else [1.0]):
                        a, f1, per = lopo(X, F, y, persona, use_emb, l2, cw, thr, sc)
                        rows.append({"head": name, "l2": l2, "class_weight": round(cw, 3),
                                     "threshold": thr, "emb_scale": sc,
                                     "lopo_acc": a, "lopo_f1": f1,
                                     "per_persona_acc": {k: v["accuracy"] for k, v in per.items()}})
        rows.sort(key=lambda r: (-r["lopo_acc"], -r["lopo_f1"]))
        best[name] = rows[0]
        report["grid"].extend(rows)
        print(f"\n=== LOPO grid, {name} head (top 5 of {len(rows)}) ===")
        for r in rows[:5]:
            print(f"  l2={r['l2']:<6} cw={r['class_weight']:<6} thr={r['threshold']} "
                  f"scale={r['emb_scale']:<6.2f} acc {100*r['lopo_acc']:.1f}%  F1 {100*r['lopo_f1']:.1f}%")
        a, f1, per = lopo(X, F, y, persona, use_emb, best[name]["l2"],
                          best[name]["class_weight"], best[name]["threshold"],
                          best[name]["emb_scale"])
        report["lopo"][name] = {"selected": best[name], "mean_acc": a, "mean_f1": f1,
                                "per_persona": per}
        print(f"  SELECTED {name}: l2={best[name]['l2']} cw={best[name]['class_weight']} "
              f"thr={best[name]['threshold']} emb_scale={best[name]['emb_scale']:.2f} "
              f"-> LOPO acc {100*a:.1f}%  F1 {100*f1:.1f}%")
        for k, v in per.items():
            print(f"     {k:24s} acc {100*v['accuracy']:.1f}%  F1 {100*v['f1']:.1f}%")

    # ---- LOPO of the SHIPPED decision function (head + the 3 rule layers) --
    report["lopo_shipped_with_rules"] = {}
    print("\nLOPO of the shipped decision function (head + rule layers):")
    for name, use_emb in (("full", True), ("surface", False)):
        c = best[name]
        a, f1, per = lopo(X, F, y, persona, use_emb, c["l2"], c["class_weight"],
                          c["threshold"], c["emb_scale"], texts=texts)
        report["lopo_shipped_with_rules"][name] = {"mean_acc": a, "mean_f1": f1, "per_persona": per}
        print(f"  {name:8s} acc {100*a:.1f}%  F1 {100*f1:.1f}%")

    # ---- ablation: embedding block alone (reported, never shipped) --------
    abl = {}
    for l2 in grid_l2:
        for sc in grid_scale:
            a, f1, _ = lopo(X, F, y, persona, True, l2, 1.0, 0.5, sc, use_surf=False)
            abl[f"l2={l2},scale={sc:.2f}"] = {"lopo_acc": a, "lopo_f1": f1}
    bk = max(abl, key=lambda k: abl[k]["lopo_acc"])
    report["ablation_embedding_only"] = {"best_config": bk, **abl[bk], "all": abl}
    print(f"\nablation, embedding block only: best {bk} -> LOPO acc "
          f"{100*abl[bk]['lopo_acc']:.1f}%  F1 {100*abl[bk]['lopo_f1']:.1f}%")

    # ---- retrain the two selected heads on all 3 personas -----------------
    emb_scale = float(best["full"]["emb_scale"])
    feat_lo, feat_hi = clip_bounds(F)
    mu, sd = standardise(F)
    Fz = (F - mu) / sd
    w_full, b_full = fit_logistic(np.hstack([X * emb_scale, Fz]), y,
                                  l2=best["full"]["l2"], class_weight=best["full"]["class_weight"])
    w_surf_s, b_surf_s = fit_logistic(Fz, y,
                                      l2=best["surface"]["l2"], class_weight=best["surface"]["class_weight"])
    # fold standardisation back into raw-space weights so serving needs no stats
    w_emb = (w_full[:EMBED_DIM] * emb_scale).astype(np.float32)
    w_sf_full = (w_full[EMBED_DIM:] / sd).astype(np.float32)
    b_f = float(b_full - np.sum(w_full[EMBED_DIM:] * mu / sd))
    w_sf = (w_surf_s / sd).astype(np.float32)
    b_s = float(b_surf_s - np.sum(w_surf_s * mu / sd))

    def score_full(Xe, Fe):
        Fc = np.clip(Fe, feat_lo, feat_hi)
        return Xe @ w_emb + Fc @ w_sf_full + b_f

    def score_surf(Fe):
        return np.clip(Fe, feat_lo, feat_hi) @ w_sf + b_s

    ins_full = metrics(1 / (1 + np.exp(-score_full(X, F))) >= best["full"]["threshold"], y)
    ins_surf = metrics(1 / (1 + np.exp(-score_surf(F))) >= best["surface"]["threshold"], y)
    print(f"\nin-sample (3 personas, for sanity only): full acc {100*ins_full['accuracy']:.1f}%  "
          f"surface acc {100*ins_surf['accuracy']:.1f}%")
    report["in_sample_3_personas"] = {"full": ins_full, "surface": ins_surf}

    if not args.no_save:
        os.makedirs(os.path.dirname(ASSET), exist_ok=True)
        np.savez_compressed(
            ASSET,
            w_emb=w_emb, w_surface_full=w_sf_full, b_full=np.float32(b_f),
            threshold_full=np.float32(best["full"]["threshold"]),
            w_surface=w_sf, b_surface=np.float32(b_s),
            threshold_surface=np.float32(best["surface"]["threshold"]),
            feature_names=np.array(FEATURE_NAMES),
            feat_lo=feat_lo, feat_hi=feat_hi,
            version=np.array("write_classifier_v2"),
            trained_on=np.array("clean_chat_benchmark.json (3 personas, 349 turns); "
                                "nomic-embed-text; scratch/refound/train_write_classifier.py"),
            emb_scale=np.float32(emb_scale),
        )
        print(f"wrote {ASSET} ({os.path.getsize(ASSET)/1024:.1f} KB)")

    if args.emit_fallback:
        print("\n# ---- paste into nanomem/classifier.py ----")
        print("_SURFACE_FALLBACK_W = np.array([")
        for n, v in zip(FEATURE_NAMES, w_sf):
            print(f"    {v: .8f},  # {n}")
        print("], dtype=np.float32)")
        print(f"_SURFACE_FALLBACK_B = {b_s:.8f}")
        print(f"_SURFACE_FALLBACK_THRESHOLD = {best['surface']['threshold']}")
        print("_SURFACE_FALLBACK_TRAINED = True")
        print("_FEATURE_LO = np.array([" + ", ".join(f"{v:.6f}" for v in feat_lo) + "], dtype=np.float32)")
        print("_FEATURE_HI = np.array([" + ", ".join(f"{v:.6f}" for v in feat_hi) + "], dtype=np.float32)")

    # ---- the ONE held-out evaluation --------------------------------------
    if args.heldout:
        if not os.path.exists(HELDOUT_JSON):
            print(f"\nheld-out set not present at {HELDOUT_JSON}; run again once it exists:")
            print("  python3 train_write_classifier.py --heldout")
        else:
            ht, hy, hp, HX, HF = load_set(HELDOUT_JSON, HELDOUT_EMB, check=not args.no_check)
            pf = 1 / (1 + np.exp(-score_full(HX, HF))) >= best["full"]["threshold"]
            ps = 1 / (1 + np.exp(-score_surf(HF))) >= best["surface"]["threshold"]
            pf_r, ps_r = apply_rules(pf, ht), apply_rules(ps, ht)
            mf, ms = metrics(pf_r, hy), metrics(ps_r, hy)
            report_head_only = {"full": metrics(pf, hy), "surface": metrics(ps, hy)}
            per = {}
            for hold in sorted(set(hp.tolist())):
                sel = hp == hold
                per[hold] = {"full": metrics(pf_r[sel], hy[sel]), "surface": metrics(ps_r[sel], hy[sel])}
            report["heldout"] = {"n": int(len(hy)), "n_positive": int(hy.sum()),
                                 "note": "'full'/'surface' are the SHIPPED decision function "
                                         "(head + 3 rule layers); head_only is the bare head",
                                 "full": mf, "surface": ms, "head_only": report_head_only,
                                 "per_persona": per}
            print(f"\n=== HELD-OUT ({len(hy)} turns, {int(hy.sum())} positive, "
                  f"personas {sorted(set(hp.tolist()))}) -- evaluated once ===")
            print(f"  full head    acc {100*mf['accuracy']:.1f}%  P {100*mf['precision']:.1f}%  "
                  f"R {100*mf['recall']:.1f}%  F1 {100*mf['f1']:.1f}%")
            print(f"  surface head acc {100*ms['accuracy']:.1f}%  P {100*ms['precision']:.1f}%  "
                  f"R {100*ms['recall']:.1f}%  F1 {100*ms['f1']:.1f}%")
            for k, v in per.items():
                print(f"     {k:24s} full acc {100*v['full']['accuracy']:.1f}%  "
                      f"surface acc {100*v['surface']['accuracy']:.1f}%")

    if args.via_vault:
        from nanomem.vault import Vault
        report["via_vault"] = {}
        print("\n=== through the shipped Vault.should_store() path (live embedder) ===")
        for tag, path in (("train_3_personas", TRAIN_JSON), ("heldout_2_personas", HELDOUT_JSON)):
            if not os.path.exists(path):
                continue
            BB = json.load(open(path))
            tt = [t for ss in BB["sets"] for t in ss["turns"]]
            lat, pred = [], []
            for t in tt:
                t0 = time.perf_counter()
                pred.append(bool(Vault.should_store(t["text"])))
                lat.append((time.perf_counter() - t0) * 1000)
            mm = metrics(np.array(pred), np.array([int(t["should_store"]) for t in tt]))
            mm["p50_ms"] = float(np.percentile(lat, 50))
            mm["p90_ms"] = float(np.percentile(lat, 90))
            mm["in_sample"] = (tag == "train_3_personas")
            report["via_vault"][tag] = mm
            print(f"  {tag:20s} acc {100*mm['accuracy']:.1f}%  F1 {100*mm['f1']:.1f}%  "
                  f"p50 {mm['p50_ms']:.1f} ms  p90 {mm['p90_ms']:.1f} ms"
                  + ("   [IN-SAMPLE]" if mm["in_sample"] else ""))

    # ---- short-utterance smoke probe (reported, never used for selection) --
    probe_texts = PROBE_FACTS + PROBE_CHAFF
    probe_y = np.array([1] * len(PROBE_FACTS) + [0] * len(PROBE_CHAFF))
    if os.path.exists(PROBE_EMB) and np.load(PROBE_EMB)["X"].shape[0] == len(probe_texts):
        PX = np.load(PROBE_EMB)["X"]
    else:
        PX = embed_texts(probe_texts)
        np.savez(PROBE_EMB, X=PX)
    PX = PX / (np.linalg.norm(PX, axis=1, keepdims=True) + 1e-8)
    PF = np.stack([extract_surface_features(t) for t in probe_texts]).astype(np.float32)
    pf_p = apply_rules(1 / (1 + np.exp(-score_full(PX, PF))) >= best["full"]["threshold"], probe_texts)
    ps_p = apply_rules(1 / (1 + np.exp(-score_surf(PF))) >= best["surface"]["threshold"], probe_texts)
    mpf, mps = metrics(pf_p, probe_y), metrics(ps_p, probe_y)
    report["short_utterance_probe"] = {
        "note": "hand-written generic short sentences; reporting only, never used for selection",
        "n_facts": len(PROBE_FACTS), "n_chaff": len(PROBE_CHAFF),
        "full": mpf, "surface": mps,
        "facts_missed_full": [t for t, k in zip(PROBE_FACTS, pf_p[:len(PROBE_FACTS)]) if not k],
    }
    print(f"\nshort-utterance smoke probe: full head keeps {mpf['tp']}/{len(PROBE_FACTS)} short facts, "
          f"rejects {mpf['tn']}/{len(PROBE_CHAFF)} short chaff | surface head "
          f"{mps['tp']}/{len(PROBE_FACTS)}, {mps['tn']}/{len(PROBE_CHAFF)}")
    for t in report["short_utterance_probe"]["facts_missed_full"]:
        print(f"    missed: {t}")

    report["provenance"] = {
        "train_set": os.path.basename(TRAIN_JSON),
        "heldout_set": os.path.basename(HELDOUT_JSON) if args.heldout else None,
        "embedding_model": MODEL,
        "selection": "mean LOPO accuracy over 3 personas, tie-break mean LOPO F1",
        "seed": SEED,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json.dump(report, open(RESULTS, "w"), indent=1)
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
