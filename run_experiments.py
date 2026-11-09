"""Compare model approaches for predicting low blood glucose (< 80 mg/dL).

Two prediction settings, both scored per reading:
  shot_time : at injection time, predict whether BG at +h hours will be < 80.
  in_cycle  : same, but also using the latest earlier reading in this cycle.

Split: per cat, chronological 70/15/15 (train / val / test). Hyper-parameters and
the "best model" are chosen on val; the chosen config is refit on train+val
and evaluated once on test.

Usage:  .venv/bin/python run_experiments.py
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from fdm.clean import load_all
from fdm.features import (CATS, INCYCLE_FEATURES, LOW_BG, SHOT_FEATURES, chrono_split,
                          cycle_features, history_tensor, reading_table)
from fdm.models import MLP, GlucoseCurveNet, HistoryGRU, predict, train_one

OUT = Path(__file__).resolve().parent / "outputs"
SEL_SEEDS = [0, 1, 2]          # seeds used while tuning on val
FINAL_SEEDS = [0, 1, 2, 3, 4]  # seed ensemble for the final refit
GRID = list(product([32, 64], [1e-4, 1e-3]))  # (hidden, weight decay)
N_BOOT = 1000

# fixed categorical order (reference palette, light mode); baselines are gray
STYLE = {
    "Cat ID only (baseline)":      dict(color="#8a8984", ls="--"),
    "Clinical LR (pre-BG, dose, h)": dict(color="#2a78d6"),
    "Logistic regression":         dict(color="#eb6834"),
    "Gradient boosting":           dict(color="#1baf7a"),
    "PyTorch MLP":                 dict(color="#eda100"),
    "PyTorch history GRU":         dict(color="#e87ba4"),
    "PyTorch glucose-curve net":   dict(color="#008300"),
    "PyTorch ensemble":            dict(color="#6250d6"),
}


# ----------------------------------------------------------------------------- data
def build():
    readings, cycles = load_all()
    cf = cycle_features(cycles, readings)
    rt = reading_table(cf, readings)
    split = chrono_split(cf)
    rt["split"] = rt.cycle.map(split)
    hist = history_tensor(cf, readings)
    return rt, hist


def tensors(df, feats, scaler, hist):
    x = scaler.transform(df[feats].values).astype(np.float32)
    return {
        "x": torch.tensor(x),
        "cat": torch.tensor(df.cat.map({c: i for i, c in enumerate(CATS)}).values),
        "hist": torch.tensor(np.stack([hist[c] for c in df.cycle])),
        "y": torch.tensor(df.y.values, dtype=torch.float32),
        "log_bg": torch.tensor(np.log(df.bg.values), dtype=torch.float32),
    }


def onehot(df):
    return pd.get_dummies(df.cat).reindex(columns=CATS, fill_value=0).values.astype(float)


# ----------------------------------------------------------------------------- metrics
def boot_auc(y, p, groups, rng, n=N_BOOT):
    """95% CI for ROC AUC, resampling whole cycles (readings within a cycle correlate)."""
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        if 0 < y[ii].sum() < len(ii):
            vals.append(roc_auc_score(y[ii], p[ii]))
    return np.percentile(vals, [2.5, 97.5])


def safe_auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


# ----------------------------------------------------------------------------- models
def sk_models(feats, task):
    clin = ["log_pre", "dose", "hours"] + (["last_log_bg", "since_last"] if task == "in_cycle" else [])

    def cat_only(tr, te, _):
        m = LogisticRegression(max_iter=1000).fit(onehot(tr), tr.y)
        return m.predict_proba(onehot(te))[:, 1]

    def clinical(tr, te, _):
        f = lambda d: np.column_stack([d[clin].values, d.hours.values ** 2])
        s = StandardScaler().fit(f(tr))
        m = LogisticRegression(max_iter=2000).fit(s.transform(f(tr)), tr.y)
        return m.predict_proba(s.transform(f(te)))[:, 1]

    def logreg(tr, te, C):
        f = lambda d: np.column_stack([d[feats].values, d.hours.values ** 2, onehot(d)])
        s = StandardScaler().fit(f(tr))
        m = LogisticRegression(C=C, max_iter=5000).fit(s.transform(f(tr)), tr.y)
        return m.predict_proba(s.transform(f(te)))[:, 1]

    def gbm(tr, te, cfg):
        lr, depth = cfg
        f = lambda d: np.column_stack([d[feats].values, onehot(d)])
        m = HistGradientBoostingClassifier(learning_rate=lr, max_depth=depth, max_iter=300,
                                           l2_regularization=1.0, early_stopping=False,
                                           random_state=0).fit(f(tr), tr.y)
        return m.predict_proba(f(te))[:, 1]

    return {
        "Cat ID only (baseline)": (cat_only, [None]),
        "Clinical LR (pre-BG, dose, h)": (clinical, [None]),
        "Logistic regression": (logreg, [0.01, 0.1, 1.0, 10.0]),
        "Gradient boosting": (gbm, list(product([0.03, 0.1], [2, 3, 5]))),
    }


def torch_factories(feats):
    n = len(feats)
    step_dim = 3 + 2 * 11
    h_idx, p_idx = feats.index("hours"), feats.index("log_pre")
    return {
        "PyTorch MLP": lambda h: (lambda: MLP(n, len(CATS), hidden=h)),
        "PyTorch history GRU": lambda h: (lambda: HistoryGRU(n, len(CATS), step_dim, hidden=h)),
        "PyTorch glucose-curve net": lambda h: (
            lambda: GlucoseCurveNet(n, len(CATS), h_idx, p_idx, hidden=h)),
    }


def fit_torch(factory, scaler, feats, h_idx, p_idx, tr, va, seeds, wd, fixed_epochs=None):
    preds_models = []
    eps = []
    for s in seeds:
        def make():
            m = factory()
            if isinstance(m, GlucoseCurveNet):
                m.set_scale(scaler.mean_[h_idx], scaler.scale_[h_idx],
                            scaler.mean_[p_idx], scaler.scale_[p_idx])
            return m
        m, ep = train_one(make, tr, va, seed=s, wd=wd, fixed_epochs=fixed_epochs)
        preds_models.append(m)
        eps.append(ep)
    return preds_models, int(np.median(eps))


# ----------------------------------------------------------------------------- run
def run_task(rt, hist, task):
    feats = SHOT_FEATURES + (INCYCLE_FEATURES if task == "in_cycle" else [])
    tr, va, te = (rt[rt.split == s].reset_index(drop=True) for s in ["train", "val", "test"])
    trva = pd.concat([tr, va], ignore_index=True)
    print(f"\n=== {task}: train {len(tr)} ({tr.y.sum()} low), val {len(va)} ({va.y.sum()} low), "
          f"test {len(te)} ({te.y.sum()} low)")

    results, test_preds = [], {}

    # --- scikit-learn reference models
    for name, (fn, grid) in sk_models(feats, task).items():
        val_scores = [(safe_auc(va.y.values, fn(tr, va, g)), g) for g in grid]
        val_auc, g = max(val_scores, key=lambda t: t[0])
        test_preds[name] = fn(trva, te, g)
        results.append(dict(model=name, val_auc=val_auc, config=str(g)))
        print(f"  {name:32s} val AUC {val_auc:.3f}  cfg={g}")

    # --- PyTorch models
    h_idx, p_idx = feats.index("hours"), feats.index("log_pre")
    sc_tr = StandardScaler().fit(tr[feats].values)
    sc_trva = StandardScaler().fit(trva[feats].values)
    T_tr, T_va = tensors(tr, feats, sc_tr, hist), tensors(va, feats, sc_tr, hist)
    T_trva, T_te = tensors(trva, feats, sc_trva, hist), tensors(te, feats, sc_trva, hist)
    val_preds_torch = {}
    for name, fac in torch_factories(feats).items():
        best = None
        for hidden, wd in GRID:
            models, ep = fit_torch(fac(hidden), sc_tr, feats, h_idx, p_idx, T_tr, T_va,
                                   SEL_SEEDS, wd)
            pv = np.mean([predict(m, T_va) for m in models], 0)
            auc = safe_auc(va.y.values, pv)
            if best is None or auc > best[0]:
                best = (auc, hidden, wd, ep, pv)
        auc, hidden, wd, ep, pv = best
        val_preds_torch[name] = pv
        # refit on train+val for the same number of epochs, 5-seed ensemble
        models, _ = fit_torch(fac(hidden), sc_trva, feats, h_idx, p_idx, T_trva, None,
                              FINAL_SEEDS, wd, fixed_epochs=max(ep, 10))
        test_preds[name] = np.mean([predict(m, T_te) for m in models], 0)
        results.append(dict(model=name, val_auc=auc,
                            config=f"hidden={hidden}, wd={wd}, epochs={ep}"))
        print(f"  {name:32s} val AUC {auc:.3f}  hidden={hidden} wd={wd} epochs={ep}")

    ens_val = np.mean(list(val_preds_torch.values()), 0)
    test_preds["PyTorch ensemble"] = np.mean(
        [test_preds[k] for k in val_preds_torch], 0)
    results.append(dict(model="PyTorch ensemble", val_auc=safe_auc(va.y.values, ens_val),
                        config="mean of the 3 PyTorch models"))

    # --- test metrics
    rng = np.random.default_rng(0)
    y, groups = te.y.values, te.cycle.values
    m2 = (te.cat == "Cat_2").values
    for r in results:
        p = test_preds[r["model"]]
        lo, hi = boot_auc(y, p, groups, rng)
        lo2, hi2 = boot_auc(y[m2], p[m2], groups[m2], rng)
        r.update(
            test_auc=roc_auc_score(y, p), test_auc_lo=lo, test_auc_hi=hi,
            cat2_auc=roc_auc_score(y[m2], p[m2]), cat2_lo=lo2, cat2_hi=hi2,
            test_pr_auc=average_precision_score(y, p), test_brier=brier_score_loss(y, p),
        )
        for c in ["Cat_3", "Cat_4"]:
            mc = (te.cat == c).values
            r[f"{c.lower()}_auc"] = safe_auc(y[mc], p[mc])
    res = pd.DataFrame(results)
    res.insert(0, "task", task)
    counts = te.groupby("cat").y.agg(["size", "sum"]).rename(columns={"size": "n", "sum": "low"})
    return res, test_preds, te, counts


def plot_roc(task, te, preds, res, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    y = te.y.values
    m2 = (te.cat == "Cat_2").values
    panels = [("All cats (pooled)", np.ones_like(y, bool), "test_auc"),
              ("Cat_2 only (within-cat)", m2, "cat2_auc")]
    for ax, (sub, mask, col) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls=":", label="Chance (0.50)")
        order = res.sort_values(col, ascending=False)
        for _, r in order.iterrows():
            fpr, tpr, _ = roc_curve(y[mask], preds[r.model][mask])
            st = STYLE[r.model]
            ax.plot(fpr, tpr, lw=2, color=st["color"], ls=st.get("ls", "-"),
                    label=f"{r.model}  {r[col]:.3f}")
        ax.set_title(f"{sub}  ·  n={mask.sum()}, low={y[mask].sum()}", fontsize=11,
                     color="#0b0b0b", loc="left")
        ax.set_xlabel("False positive rate", color="#52514e")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.grid(color="#e8e7e2", lw=0.8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e")
        ax.legend(loc="lower right", fontsize=8.5, frameon=False, title="Test ROC AUC",
                  title_fontsize=9)
    axes[0].set_ylabel("True positive rate", color="#52514e")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color="#0b0b0b")
    fig.tight_layout()
    path = OUT / f"roc_{task}.png"
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def plot_summary(allres):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    tasks = [("shot_time", "At shot time"), ("in_cycle", "Mid-curve (with latest reading)")]
    names = list(STYLE)
    for ax, (task, label) in zip(axes, tasks):
        r = allres[allres.task == task].set_index("model").loc[names]
        ypos = np.arange(len(names))[::-1]
        for (col, lo, hi, off, mk, lab) in [("test_auc", "test_auc_lo", "test_auc_hi", 0.15, "o", "Pooled"),
                                            ("cat2_auc", "cat2_lo", "cat2_hi", -0.15, "s", "Cat_2 only")]:
            ax.errorbar(r[col], ypos + off, xerr=[r[col] - r[lo], r[hi] - r[col]], fmt=mk,
                        ms=7, color="#2a78d6" if mk == "o" else "#eb6834", ecolor="#c3c2b7",
                        elinewidth=2, capsize=0, label=lab, markeredgecolor="#fcfcfb",
                        markeredgewidth=1.5)
        ax.axvline(0.5, color="#c3c2b7", ls=":", lw=1)
        ax.set_yticks(ypos); ax.set_yticklabels(names, color="#0b0b0b")
        ax.set_xlim(0.45, 1.0)
        ax.set_title(label, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_xlabel("Test ROC AUC (95% CI, cycle bootstrap)", color="#52514e")
        ax.grid(axis="x", color="#e8e7e2", lw=0.8)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(colors="#52514e")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Predicting BG < 80 mg/dL: test ROC AUC by model", x=0.01, ha="left",
                 fontsize=13, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "auc_summary.png", dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    rt, hist = build()
    titles = {
        "shot_time": f"Will BG be < {LOW_BG:.0f} mg/dL at +h hours? Predicted at shot time",
        "in_cycle": f"Will BG be < {LOW_BG:.0f} mg/dL at +h hours? Predicted mid-curve, using the latest reading",
    }
    allres, counts_all = [], {}
    for task in ["shot_time", "in_cycle"]:
        res, preds, te, counts = run_task(rt, hist, task)
        plot_roc(task, te, preds, res, titles[task])
        allres.append(res)
        counts_all[task] = counts
        pd.DataFrame({"cycle": te.cycle, "cat": te.cat, "hours": te.hours, "bg": te.bg,
                      "y": te.y, **preds}).to_csv(OUT / f"test_predictions_{task}.csv", index=False)
    allres = pd.concat(allres, ignore_index=True)
    allres.to_csv(OUT / "model_comparison.csv", index=False)
    plot_summary(allres)

    best = {t: g.sort_values("val_auc", ascending=False).iloc[0].model
            for t, g in allres.groupby("task")}
    (OUT / "best_models.json").write_text(json.dumps(best, indent=2))
    cols = ["task", "model", "val_auc", "test_auc", "test_auc_lo", "test_auc_hi",
            "cat2_auc", "cat2_lo", "cat2_hi", "cat_3_auc", "cat_4_auc", "test_pr_auc", "test_brier"]
    print("\n", allres[cols].round(3).to_string(index=False))
    print("\nTest-set counts:", {k: v.to_dict() for k, v in counts_all.items()})
    print("Best by validation AUC:", best)


if __name__ == "__main__":
    main()
