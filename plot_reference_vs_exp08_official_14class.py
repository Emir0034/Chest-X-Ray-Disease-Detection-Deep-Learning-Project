"""
plot_reference_vs_exp08_official_14class.py
Poster-friendly 14-class per-class AUC comparison:
Referenced Study vs. Exp08 Official Split.
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────

REFERENCED_AUC = {
    "Atelectasis":        0.7729,
    "Cardiomegaly":       0.8787,
    "Effusion":           0.8266,
    "Infiltration":       0.6655,
    "Mass":               0.8247,
    "Nodule":             0.7582,
    "Pneumonia":          0.7158,
    "Pneumothorax":       0.8639,
    "Consolidation":      0.7447,
    "Edema":              0.8403,
    "Emphysema":          0.9077,
    "Fibrosis":           0.8058,
    "Pleural_Thickening": 0.7670,
    "Hernia":             0.8911,
}

REF_OVERALL_AUC = float(np.mean(list(REFERENCED_AUC.values())))

DIFFICULT = {"Infiltration", "Nodule", "Pneumonia", "Pleural_Thickening", "Consolidation"}

CSV_PATH = os.path.join(
    "results", "metrics",
    "densenet121_cbam_block34_bce_adamw_official_split_test_auc_per_class.csv",
)

EXP08_AUC = {}
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        EXP08_AUC[row["class"]] = float(row["auc"])

EXP08_OVERALL_AUC = float(np.mean(list(EXP08_AUC.values())))

# ── Build comparison rows (sorted by Referenced AUC ascending) ────────────────

sorted_classes = sorted(REFERENCED_AUC, key=lambda c: REFERENCED_AUC[c])

rows = []
for c in sorted_classes:
    ref = REFERENCED_AUC[c]
    exp = EXP08_AUC[c]
    rows.append({
        "disease":                     c,
        "referenced_study_auc":        round(ref, 4),
        "exp08_official_split_auc":    round(exp, 4),
        "absolute_difference":         round(exp - ref, 4),
        "relative_difference_percent": round((exp - ref) / ref * 100, 2),
        "difficult_class":             c in DIFFICULT,
    })

n = len(rows)
X_all  = sum(1 for r in rows if r["exp08_official_split_auc"] > r["referenced_study_auc"])
X_diff = sum(1 for r in rows if r["difficult_class"] and r["exp08_official_split_auc"] > r["referenced_study_auc"])

# ── Colours ───────────────────────────────────────────────────────────────────

C_REF      = "#5B8DB8"
C_EXP      = "#E07B54"
C_POS      = "#228B22"
C_NEG      = "#CC3333"
C_BAND     = "#FFF3CC"
C_ANN_BG   = "#EAF3FB"
C_ANN_EDGE = "#5B8DB8"

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(n)
h = 0.35

for i, r in enumerate(rows):
    if r["difficult_class"]:
        ax.axhspan(i - 0.5, i + 0.5, color=C_BAND, alpha=0.50, zorder=0)

ref_vals = [r["referenced_study_auc"]     for r in rows]
exp_vals = [r["exp08_official_split_auc"] for r in rows]

ax.barh(y - h / 2, ref_vals, h, label="Referenced Study",     color=C_REF, alpha=0.88, edgecolor="white")
ax.barh(y + h / 2, exp_vals, h, label="Exp08 Official Split", color=C_EXP, alpha=0.88, edgecolor="white")

for i, r in enumerate(rows):
    rel   = r["relative_difference_percent"]
    color = C_POS if rel >= 0 else C_NEG
    lbl   = f"+{rel:.1f}%" if rel >= 0 else f"{rel:.1f}%"
    ax.text(r["exp08_official_split_auc"] + 0.004, y[i] + h / 2, lbl,
            va="center", ha="left", fontsize=7.5, color=color, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels([r["disease"] for r in rows], fontsize=10)
ax.set_xlabel("AUC", fontsize=11)
ax.set_xlim(0.60, 1.02)
ax.set_title("Official Split Check: Per-Class AUC",
              fontsize=12, fontweight="bold", pad=14)
ax.xaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

band_patch = mpatches.Patch(color=C_BAND, alpha=0.7, label="Difficult class")
handles, labels_ = ax.get_legend_handles_labels()
ax.legend(handles + [band_patch], labels_ + ["Difficult class"],
          fontsize=9.5, loc="lower right")

ann = (
    f"Referenced Study AUC: {REF_OVERALL_AUC:.4f}\n"
    f"Exp08 Official Split AUC: {EXP08_OVERALL_AUC:.4f}\n"
    f"Exp08 higher in {X_all}/{n} classes"
)
ax.annotate(
    ann,
    xy=(0.985, 0.305), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", facecolor=C_ANN_BG,
              edgecolor=C_ANN_EDGE, alpha=0.92),
)

ax.text(0.985, 0.985, "Reference comparison only; evaluation protocols may differ.",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, style="italic", color="#555555")

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────

OUT_DIR = os.path.join("results", "figures", "final_comparisons")
os.makedirs(OUT_DIR, exist_ok=True)
out_png = os.path.join(OUT_DIR, "reference_vs_exp08_official_14class_auc_notta.png")
out_csv = os.path.join(OUT_DIR, "reference_vs_exp08_official_14class_auc_notta.csv")

fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "disease", "referenced_study_auc", "exp08_official_split_auc",
        "absolute_difference", "relative_difference_percent", "difficult_class",
    ])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# ── Console summary ───────────────────────────────────────────────────────────

print(f"Saved figure: {out_png}")
print(f"Saved CSV: {out_csv}")
print()
print(f"{'Class':<20} {'Referenced':>11} {'Exp08 (Off.)':>13} {'AbsDiff':>9} {'RelDiff':>9} {'Difficult':>10}")
for r in rows:
    print(f"{r['disease']:<20} {r['referenced_study_auc']:>11.4f} "
          f"{r['exp08_official_split_auc']:>13.4f} {r['absolute_difference']:>+9.4f} "
          f"{r['relative_difference_percent']:>+8.1f}% {str(r['difficult_class']):>10}")
print()
print(f"Referenced Study average AUC : {REF_OVERALL_AUC:.4f}")
print(f"Exp08 Official Split avg AUC : {EXP08_OVERALL_AUC:.4f}")
print(f"Exp08 higher in {X_all}/{n} classes")
print(f"Exp08 higher in {X_diff}/{len(DIFFICULT)} difficult classes")
