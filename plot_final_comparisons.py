"""
plot_final_comparisons.py — Report-ready comparison figures for all completed experiments.

Reads from:
    results/metrics/*_best_epoch.json
    results/metrics/*_test_metrics.json   (primary)
    results/metrics/*_test_summary.csv    (fallback)
    results/metrics/*_test_auc_per_class.csv

Outputs to:
    results/figures/final_comparisons/

Usage:
    python plot_final_comparisons.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Paths ─────────────────────────────────────────────────────────────────────
METRICS_DIR = os.path.join("results", "metrics")
OUTPUT_DIR  = os.path.join("results", "figures", "final_comparisons")

# ── Experiment registry (logical order) ──────────────────────────────────────
EXPERIMENTS = [
    {"id": "01",  "label": "01 Baseline",   "name": "densenet121_bce"},
    {"id": "02",  "label": "02 CLAHE",       "name": "densenet121_clahe_bce"},
    {"id": "03a", "label": "03a CBAM-b4",    "name": "densenet121_clahe_cbam_block4_bce"},
    {"id": "03b", "label": "03b CBAM-b34",   "name": "densenet121_clahe_cbam_block34_bce"},
    {"id": "03c", "label": "03c CBAM-b1234", "name": "densenet121_clahe_cbam_block1234_bce"},
    {"id": "04",  "label": "04 ASL",         "name": "densenet121_clahe_cbam_block34_asl"},
    {"id": "04b", "label": "04b ASL-gn2",    "name": "densenet121_clahe_cbam_block34_asl_gn2_gp0"},
    {"id": "05",  "label": "05 FAAR",        "name": "densenet121_clahe_cbam_block34_asl_gn2_gp0_faar"},
    {"id": "05b", "label": "05b FAAR-cap3",  "name": "densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_cap3"},
    {"id": "06",  "label": "06 LR1e-3",      "name": "densenet121_bce_lr1e3_pat10"},
    {"id": "07",  "label": "07 Adam",        "name": "densenet121_cbam_block34_bce"},
    {"id": "08",  "label": "08 BCE+AdamW",   "name": "densenet121_cbam_block34_bce_adamw"},
    {"id": "09",  "label": "09 ASL+AdamW",   "name": "densenet121_cbam_block34_asl_gn2_gp0_adamw"},
    {"id": "10",  "label": "10 Focal+AdamW", "name": "densenet121_cbam_block34_focal_gamma2_adamw"},
]

CLASS_ORDER = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

# ── Color palette ─────────────────────────────────────────────────────────────
C_BLUE   = "#5B8DB8"
C_ORANGE = "#E07B54"
C_GREEN  = "#5BAF6E"
C_GREY   = "#A0A0A0"


# ── Data loading ──────────────────────────────────────────────────────────────

def _p(name, suffix):
    return os.path.join(METRICS_DIR, f"{name}{suffix}")


def load_experiment_data(exp):
    name   = exp["name"]
    result = {"id": exp["id"], "label": exp["label"], "name": name}

    # 1. best_epoch.json
    bej = _p(name, "_best_epoch.json")
    try:
        with open(bej) as f:
            d = json.load(f)
        result["best_val_auc"] = d["best_val_auc"]
        result["best_epoch"]   = d["best_epoch"]
    except FileNotFoundError:
        print(f"[WARNING] Missing: {bej} — skipping '{name}'")
        return None
    except KeyError as e:
        print(f"[WARNING] Key {e} missing in {bej} — skipping '{name}'")
        return None

    # 2. test_metrics.json (primary) or test_summary.csv (fallback)
    tmj = _p(name, "_test_metrics.json")
    tsc = _p(name, "_test_summary.csv")
    loaded = False
    if os.path.isfile(tmj):
        try:
            with open(tmj) as f:
                d = json.load(f)
            result["mean_test_auc"] = d["avg_auc"]
            result["precision"]     = d["precision"]
            result["recall"]        = d["recall"]
            result["f1"]            = d["f1"]
            result["threshold"]     = d.get("threshold", 0.5)
            result["per_class_auc"] = d.get("per_class_auc", {})
            loaded = True
        except (KeyError, json.JSONDecodeError) as e:
            print(f"[WARNING] Could not parse {tmj}: {e}")

    if not loaded:
        if os.path.isfile(tsc):
            try:
                df = pd.read_csv(tsc)
                result["mean_test_auc"] = float(df["avg_auc"].iloc[0])
                result["precision"]     = float(df["precision"].iloc[0])
                result["recall"]        = float(df["recall"].iloc[0])
                result["f1"]            = float(df["f1"].iloc[0])
                result["threshold"]     = float(df["threshold"].iloc[0])
                result["per_class_auc"] = {}
                loaded = True
            except (KeyError, IndexError) as e:
                print(f"[WARNING] Could not parse {tsc}: {e}")
        else:
            print(f"[WARNING] No test metrics file for '{name}' — values set to NaN")
        if not loaded:
            result.update({k: float("nan") for k in
                           ("mean_test_auc","precision","recall","f1","threshold")})
            result["per_class_auc"] = {}

    # 3. per-class CSV if not already from JSON
    if not result.get("per_class_auc"):
        pac = _p(name, "_test_auc_per_class.csv")
        if os.path.isfile(pac):
            try:
                df = pd.read_csv(pac)
                result["per_class_auc"] = dict(zip(df["class"], df["auc"].astype(float)))
            except Exception as e:
                print(f"[WARNING] Could not parse {pac}: {e}")

    return result


def load_all():
    records = []
    for exp in EXPERIMENTS:
        d = load_experiment_data(exp)
        if d is not None:
            records.append(d)
    return records


def _by_id(records, *ids):
    order = {eid: i for i, eid in enumerate(ids)}
    return sorted([r for r in records if r["id"] in set(ids)],
                  key=lambda r: order[r["id"]])


# ── Shared helpers ────────────────────────────────────────────────────────────

def _hbar_labels(ax, bars, fmt=".4f", fontsize=7.5, pad=0.0003):
    """Value labels right of each horizontal bar."""
    for bar in bars:
        w = bar.get_width()
        if not np.isnan(w):
            ax.text(w + pad, bar.get_y() + bar.get_height() / 2,
                    f"{w:{fmt}}", va="center", ha="left", fontsize=fontsize)


def _vbar_labels(ax, bars, fmt=".4f", fontsize=7.5, pad=0.0003):
    """Value labels above each vertical bar."""
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + pad,
                    f"{h:{fmt}}", ha="center", va="bottom", fontsize=fontsize)


def _savefig(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 1: All experiments — Mean Test AUC (horizontal bar) ───────────────

def plot_all_mean_test_auc(records):
    aucs   = [r["mean_test_auc"] for r in records]
    labels = [r["label"] for r in records]
    ids    = [r["id"] for r in records]

    best_auc = max((a for a in aucs if not np.isnan(a)), default=0)

    colors = [C_ORANGE if r["id"] == "08" else C_BLUE for r in records]

    y   = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(y, aucs, color=colors, edgecolor="white", linewidth=0.4, height=0.65)

    _hbar_labels(ax, bars, fmt=".4f", fontsize=7.5, pad=0.0002)

    # Annotate Exp08 best
    for i, r in enumerate(records):
        if r["id"] == "08":
            ax.text(aucs[i] + 0.0008, y[i] + 0.35, "★ Best AUC",
                    va="bottom", ha="left", fontsize=7.5, color=C_ORANGE, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean Test AUC", fontsize=11)
    ax.set_xlim(0.810, best_auc + 0.010)
    ax.set_title("Mean Test AUC — All Experiments", fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    _savefig(fig, "all_experiments_mean_test_auc.png")


# ── Figure 2: All experiments — Best Val AUC (horizontal bar) ────────────────

def plot_all_best_val_auc(records):
    aucs   = [r["best_val_auc"] for r in records]
    labels = [r["label"] for r in records]
    best_v = max((a for a in aucs if not np.isnan(a)), default=0)

    colors = [C_ORANGE if r["id"] == "08" else C_BLUE for r in records]

    y   = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(y, aucs, color=colors, edgecolor="white", linewidth=0.4, height=0.65)
    _hbar_labels(ax, bars, fmt=".4f", fontsize=7.5, pad=0.0002)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Best Validation AUC", fontsize=11)
    ax.set_xlim(0.818, best_v + 0.008)
    ax.set_title("Best Validation AUC — All Experiments", fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    _savefig(fig, "all_experiments_best_val_auc.png")


# ── Figure 3: All experiments — Precision / Recall / F1 (horizontal grouped) ─

def plot_all_precision_recall_f1(records):
    labels    = [r["label"] for r in records]
    precision = [r["precision"] for r in records]
    recall    = [r["recall"]    for r in records]
    f1        = [r["f1"]        for r in records]

    n     = len(labels)
    y     = np.arange(n)
    h     = 0.26
    fig, ax = plt.subplots(figsize=(12, 9))

    b1 = ax.barh(y + h,  precision, h, label="Precision", color=C_BLUE,   edgecolor="white")
    b2 = ax.barh(y,       recall,   h, label="Recall",    color=C_ORANGE, edgecolor="white")
    b3 = ax.barh(y - h,  f1,        h, label="F1 Score",  color=C_GREEN,  edgecolor="white")

    for bars in (b1, b2, b3):
        _hbar_labels(ax, bars, fmt=".3f", fontsize=6.5, pad=0.002)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_xlim(0, 0.60)
    ax.set_title("Precision / Recall / F1 Score — All Experiments", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.xaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    _savefig(fig, "all_experiments_precision_recall_f1.png")


# ── Figure 4: Scatter — Mean Test AUC vs F1 ──────────────────────────────────

def plot_auc_vs_f1(records):
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in records:
        x, y_val = r["mean_test_auc"], r["f1"]
        if np.isnan(x) or np.isnan(y_val):
            continue
        if r["id"] == "08":
            c, s, zorder = C_ORANGE, 140, 5
        elif r["id"] == "09":
            c, s, zorder = C_GREEN, 140, 5
        else:
            c, s, zorder = C_BLUE, 60, 3
        ax.scatter(x, y_val, color=c, s=s, zorder=zorder, edgecolors="white", linewidths=0.5)
        ax.annotate(r["label"], (x, y_val),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7, ha="left")

    ax.set_xlabel("Mean Test AUC", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Mean Test AUC vs F1 Score — All Experiments", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.35, linestyle="--")

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_ORANGE, markersize=9, label="08 BCE+AdamW (best AUC)"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_GREEN,  markersize=9, label="09 ASL+AdamW (best F1)"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_BLUE,   markersize=7, label="Other experiments"),
    ]
    ax.legend(handles=legend_els, fontsize=8)
    _savefig(fig, "all_experiments_auc_vs_f1_scatter.png")


# ── Figure 5: Scatter — Mean Test AUC vs Recall ───────────────────────────────

def plot_auc_vs_recall(records):
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in records:
        x, y_val = r["mean_test_auc"], r["recall"]
        if np.isnan(x) or np.isnan(y_val):
            continue
        if r["id"] == "08":
            c, s, zorder = C_ORANGE, 140, 5
        elif r["id"] == "09":
            c, s, zorder = C_GREEN, 140, 5
        else:
            c, s, zorder = C_BLUE, 60, 3
        ax.scatter(x, y_val, color=c, s=s, zorder=zorder, edgecolors="white", linewidths=0.5)
        ax.annotate(r["label"], (x, y_val),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7, ha="left")

    ax.set_xlabel("Mean Test AUC", fontsize=11)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("Mean Test AUC vs Recall — All Experiments", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.35, linestyle="--")

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_ORANGE, markersize=9, label="08 BCE+AdamW (best AUC)"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_GREEN,  markersize=9, label="09 ASL+AdamW (best Recall)"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_BLUE,   markersize=7, label="Other experiments"),
    ]
    ax.legend(handles=legend_els, fontsize=8)
    _savefig(fig, "all_experiments_auc_vs_recall_scatter.png")


# ── Figure 6: Summary heatmap (pure matplotlib, no seaborn) ──────────────────

def plot_summary_heatmap(records):
    metric_keys  = ["best_val_auc", "mean_test_auc", "precision", "recall", "f1"]
    metric_names = ["Best Val AUC", "Mean Test AUC", "Precision", "Recall", "F1"]
    labels = [r["label"] for r in records]

    matrix = np.array([[r.get(k, float("nan")) for k in metric_keys] for r in records],
                      dtype=float)

    # Per-column min-max normalization for coloring
    norm_mat = np.full_like(matrix, 0.5)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1:
            lo, hi = valid.min(), valid.max()
            if hi > lo:
                norm_mat[:, j] = (col - lo) / (hi - lo)
            else:
                norm_mat[:, j] = 0.5
        norm_mat[np.isnan(col), j] = 0.5

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(norm_mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Cell text
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                txt = "N/A"
            elif j < 2:    # AUC columns
                txt = f"{val:.4f}"
            else:
                txt = f"{val:.3f}"
            # Dark text on light cells, light text on dark cells
            brightness = norm_mat[i, j]
            color = "black" if 0.25 < brightness < 0.75 else ("black" if brightness >= 0.75 else "white")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Normalized score (per column)")
    ax.set_title("Experiment Summary Heatmap", fontsize=12, fontweight="bold", pad=20)
    _savefig(fig, "all_experiments_summary_heatmap.png")


# ── Figure 7: Per-class AUC — Exp08, 09, 10 ──────────────────────────────────

def plot_per_class_best_models(records):
    subset = _by_id(records, "08", "09", "10")
    if len(subset) < 3:
        print("[WARNING] Cannot plot Figure 7 — one or more of Exp08/09/10 not loaded.")
        return

    exps = subset
    pacs = [r.get("per_class_auc", {}) for r in exps]
    if any(not p for p in pacs):
        print("[WARNING] Per-class AUC missing for Figure 7.")
        return

    classes = [c for c in CLASS_ORDER if all(c in p for p in pacs)]
    vals    = [[p[c] for c in classes] for p in pacs]
    colors  = [C_BLUE, C_ORANGE, C_GREEN]

    x     = np.arange(len(classes))
    w     = 0.26
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (exp, v, c) in enumerate(zip(exps, vals, colors)):
        ax.bar(x + (i - 1) * w, v, w, label=exp["label"], color=c, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Test AUC", fontsize=11)
    ax.set_ylim(0.60, 1.00)
    ax.set_title("Per-Class Test AUC: Exp 08 vs Exp 09 vs Exp 10", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    _savefig(fig, "per_class_auc_best_models.png")


# ── Figure 8: Optimizer comparison — Exp07 (Adam) vs Exp08 (AdamW) ───────────

def plot_optimizer_comparison(records):
    subset = _by_id(records, "07", "08")
    if len(subset) < 2:
        print("[WARNING] Cannot plot Figure 8 — Exp07 or Exp08 not loaded.")
        return

    metrics = ["best_val_auc", "mean_test_auc", "precision", "recall", "f1"]
    mlabels = ["Best Val AUC", "Mean Test AUC", "Precision", "Recall", "F1"]
    exp07, exp08 = subset[0], subset[1]
    v07 = [exp07.get(m, float("nan")) for m in metrics]
    v08 = [exp08.get(m, float("nan")) for m in metrics]

    x   = np.arange(len(mlabels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, v07, w, label=exp07["label"], color=C_GREY,   edgecolor="white")
    b2 = ax.bar(x + w/2, v08, w, label=exp08["label"], color=C_BLUE,   edgecolor="white")
    for bars, fmt in [(b1, ".4f"), (b2, ".4f")]:
        _vbar_labels(ax, bars, fmt=fmt, fontsize=7.5, pad=0.003)

    ax.set_xticks(x)
    ax.set_xticklabels(mlabels, fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, max(max(v for v in v07 if not np.isnan(v)),
                       max(v for v in v08 if not np.isnan(v))) + 0.07)
    ax.set_title("Optimizer Comparison: Adam (Exp 07) vs AdamW (Exp 08)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    _savefig(fig, "optimizer_comparison.png")


# ── Figure 9: Loss comparison — Exp08 BCE, Exp09 ASL, Exp10 Focal ────────────

def plot_loss_comparison(records):
    subset = _by_id(records, "08", "09", "10")
    if len(subset) < 3:
        print("[WARNING] Cannot plot Figure 9 — one or more of Exp08/09/10 not loaded.")
        return

    metrics = ["mean_test_auc", "precision", "recall", "f1"]
    mlabels = ["Mean Test AUC", "Precision", "Recall", "F1"]
    colors  = [C_BLUE, C_ORANGE, C_GREEN]

    x   = np.arange(len(mlabels))
    w   = 0.26
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (exp, c) in enumerate(zip(subset, colors)):
        v = [exp.get(m, float("nan")) for m in metrics]
        bars = ax.bar(x + (i - 1) * w, v, w, label=exp["label"], color=c, edgecolor="white")
        _vbar_labels(ax, bars, fmt=".4f", fontsize=7, pad=0.003)

    ax.set_xticks(x)
    ax.set_xticklabels(mlabels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 0.65)
    ax.set_title("Loss Function Comparison (CBAM block34, no CLAHE, AdamW)\nExp 08 BCE vs Exp 09 ASL vs Exp 10 Focal",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    _savefig(fig, "loss_comparison_no_clahe_cbam_block34.png")


# ── Figure 10: CLAHE effect — Exp01 vs Exp02 ─────────────────────────────────

def plot_clahe_effect(records):
    subset = _by_id(records, "01", "02")
    if len(subset) < 2:
        print("[WARNING] Cannot plot Figure 10 — Exp01 or Exp02 not loaded.")
        return

    metrics = ["mean_test_auc", "precision", "recall", "f1"]
    mlabels = ["Mean Test AUC", "Precision", "Recall", "F1"]
    exp01, exp02 = subset[0], subset[1]
    v01 = [exp01.get(m, float("nan")) for m in metrics]
    v02 = [exp02.get(m, float("nan")) for m in metrics]

    x   = np.arange(len(mlabels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, v01, w, label=exp01["label"], color=C_GREY, edgecolor="white")
    b2 = ax.bar(x + w/2, v02, w, label=exp02["label"], color=C_BLUE, edgecolor="white")
    for bars in (b1, b2):
        _vbar_labels(ax, bars, fmt=".4f", fontsize=7.5, pad=0.003)

    ax.set_xticks(x)
    ax.set_xticklabels(mlabels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 0.70)
    ax.set_title("Effect of CLAHE Preprocessing\nExp 01 (Baseline) vs Exp 02 (+ CLAHE)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    _savefig(fig, "clahe_effect_comparison.png")


# ── Figure 11: CBAM placement — Exp03a / 03b / 03c ───────────────────────────

def plot_cbam_placement(records):
    subset = _by_id(records, "03a", "03b", "03c")
    if len(subset) < 3:
        print("[WARNING] Cannot plot Figure 11 — one or more of Exp03a/03b/03c not loaded.")
        return

    metrics = ["best_val_auc", "mean_test_auc"]
    mlabels = ["Best Val AUC", "Mean Test AUC"]
    colors  = [C_BLUE, C_ORANGE, C_GREY]  # 03b highlighted

    x   = np.arange(len(mlabels))
    w   = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (exp, c) in enumerate(zip(subset, colors)):
        v = [exp.get(m, float("nan")) for m in metrics]
        bars = ax.bar(x + (i - 1) * w, v, w, label=exp["label"], color=c, edgecolor="white")
        _vbar_labels(ax, bars, fmt=".4f", fontsize=7.5, pad=0.0003)

    ax.set_xticks(x)
    ax.set_xticklabels(mlabels, fontsize=10)
    ax.set_ylabel("AUC", fontsize=11)
    # Zoom y-axis
    all_vals = [subset[i].get(m, float("nan")) for i in range(3) for m in metrics]
    valid    = [v for v in all_vals if not np.isnan(v)]
    ax.set_ylim(min(valid) - 0.003, max(valid) + 0.006)
    ax.set_title("CBAM Placement Ablation\nExp 03a (block4) vs 03b (block34) vs 03c (block1234)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    _savefig(fig, "cbam_placement_comparison.png")


# ── Figure 12: Final best models — Exp08 vs Exp09 ────────────────────────────

def plot_final_best_models(records):
    subset = _by_id(records, "08", "09")
    if len(subset) < 2:
        print("[WARNING] Cannot plot Figure 12 — Exp08 or Exp09 not loaded.")
        return

    metrics = ["mean_test_auc", "precision", "recall", "f1"]
    mlabels = ["Mean Test AUC", "Precision", "Recall", "F1"]
    exp08, exp09 = subset[0], subset[1]
    v08 = [exp08.get(m, float("nan")) for m in metrics]
    v09 = [exp09.get(m, float("nan")) for m in metrics]

    x   = np.arange(len(mlabels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, v08, w, label=exp08["label"], color=C_BLUE,   edgecolor="white")
    b2 = ax.bar(x + w/2, v09, w, label=exp09["label"], color=C_ORANGE, edgecolor="white")
    for bars in (b1, b2):
        _vbar_labels(ax, bars, fmt=".4f", fontsize=7.5, pad=0.003)

    winner_text = []
    for v_a, v_b in zip(v08, v09):
        if np.isnan(v_a) or np.isnan(v_b) or v_a == v_b:
            winner_text.append("")
        elif v_a > v_b:
            winner_text.append(f"Higher: Exp {exp08['id']}")
        else:
            winner_text.append(f"Higher: Exp {exp09['id']}")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{m}\n{w}" if w else m for m, w in zip(mlabels, winner_text)],
        fontsize=8.5
    )
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, max(max(v for v in v08 + v09 if not np.isnan(v)), 0) + 0.07)
    ax.set_title("Best Models: Exp 08 (BCE+AdamW) vs Exp 09 (ASL+AdamW)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    _savefig(fig, "final_best_models_comparison.png")


# ── Consolidated CSV ──────────────────────────────────────────────────────────

def save_consolidated_csv(records):
    rows = [{
        "experiment_id":   r["id"],
        "display_label":   r["label"],
        "experiment_name": r["name"],
        "best_val_auc":    r.get("best_val_auc",  float("nan")),
        "best_epoch":      r.get("best_epoch",     ""),
        "mean_test_auc":   r.get("mean_test_auc",  float("nan")),
        "precision":       r.get("precision",      float("nan")),
        "recall":          r.get("recall",         float("nan")),
        "f1":              r.get("f1",             float("nan")),
        "threshold":       r.get("threshold",      0.5),
    } for r in records]
    df   = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "final_comparison_metrics_all.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"plot_final_comparisons.py")
    print(f"Metrics dir : {METRICS_DIR}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    print("Loading experiment data...")
    records = load_all()
    print(f"Loaded {len(records)}/{len(EXPERIMENTS)} experiments.\n")

    print("Saving consolidated CSV...")
    save_consolidated_csv(records)

    print("\nGenerating figures...")
    plot_all_mean_test_auc(records)           # Fig 1
    plot_all_best_val_auc(records)            # Fig 2
    plot_all_precision_recall_f1(records)     # Fig 3
    plot_auc_vs_f1(records)                   # Fig 4
    plot_auc_vs_recall(records)               # Fig 5
    plot_summary_heatmap(records)             # Fig 6
    plot_per_class_best_models(records)       # Fig 7
    plot_optimizer_comparison(records)        # Fig 8
    plot_loss_comparison(records)             # Fig 9
    plot_clahe_effect(records)                # Fig 10
    plot_cbam_placement(records)              # Fig 11
    plot_final_best_models(records)           # Fig 12

    print(f"\nDone. All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
