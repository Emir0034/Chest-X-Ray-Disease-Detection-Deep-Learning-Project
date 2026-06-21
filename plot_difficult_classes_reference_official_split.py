"""
plot_difficult_classes_reference_official_split.py
Poster-friendly per-class AUC comparison for the 5 difficult disease classes:
Referenced Study vs. Exp08 Official Split.
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────

CLASSES = ["Infiltration", "Nodule", "Pneumonia", "Pleural_Thickening", "Consolidation"]

REFERENCED_AUC = {
    "Infiltration":       0.6831,
    "Nodule":             0.7697,
    "Pneumonia":          0.7042,
    "Pleural_Thickening": 0.7707,
    "Consolidation":      0.7458,
}

# Overall (14-class) Referenced Study AUC, as reported in the referenced study
REF_OVERALL_AUC = 0.8045

CSV_PATH = os.path.join(
    "results", "metrics",
    "densenet121_cbam_block34_bce_adamw_official_split_test_auc_per_class.csv",
)

EXP08_AUC = {}
all_aucs = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        auc = float(row["auc"])
        all_aucs.append(auc)
        if row["class"] in CLASSES:
            EXP08_AUC[row["class"]] = auc

# Overall (14-class) Exp08 Official Split mean AUC, computed from the CSV
EXP08_OVERALL_AUC = float(np.mean(all_aucs))

# ── Build comparison rows (sorted by Referenced AUC ascending) ────────────────

sorted_classes = sorted(CLASSES, key=lambda c: REFERENCED_AUC[c])

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
        "difficult_class":             True,
    })

X = sum(1 for r in rows if r["exp08_official_split_auc"] > r["referenced_study_auc"])

# ── Colours ───────────────────────────────────────────────────────────────────

C_REF      = "#5B8DB8"
C_EXP      = "#E07B54"
C_POS      = "#228B22"
C_NEG      = "#CC3333"
C_ANN_BG   = "#EAF3FB"
C_ANN_EDGE = "#5B8DB8"

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7.0, 3.6))
n = len(rows)
y = np.arange(n)
h = 0.35

ref_vals = [r["referenced_study_auc"]     for r in rows]
exp_vals = [r["exp08_official_split_auc"] for r in rows]

ax.barh(y - h / 2, ref_vals, h, label="Referenced Study",     color=C_REF, alpha=0.88, edgecolor="white")
ax.barh(y + h / 2, exp_vals, h, label="Exp08 Official Split", color=C_EXP, alpha=0.88, edgecolor="white")

for i, r in enumerate(rows):
    rel   = r["relative_difference_percent"]
    color = C_POS if rel >= 0 else C_NEG
    lbl   = f"+{rel:.1f}%" if rel >= 0 else f"{rel:.1f}%"
    ax.text(r["exp08_official_split_auc"] + 0.004, y[i] + h / 2, lbl,
            va="center", ha="left", fontsize=8, color=color, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels([r["disease"] for r in rows], fontsize=9.5)
ax.set_xlabel("AUC", fontsize=10)
ax.set_xlim(0.60, 0.88)
ax.set_title("Official Split Check: Difficult Classes",
              fontsize=11, fontweight="bold", pad=10)
ax.xaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
ax.legend(fontsize=8.5, loc="lower right")

ann = (
    f"Referenced Study AUC: {REF_OVERALL_AUC:.4f}\n"
    f"Exp08 Official Split AUC: {EXP08_OVERALL_AUC:.4f}\n"
    f"Exp08 higher in {X}/5 difficult classes"
)
ax.annotate(
    ann,
    xy=(0.015, 0.04), xycoords="axes fraction",
    ha="left", va="bottom", fontsize=8,
    bbox=dict(boxstyle="round,pad=0.4", facecolor=C_ANN_BG,
              edgecolor=C_ANN_EDGE, alpha=0.92),
)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────

OUT_DIR = os.path.join("results", "figures", "final_comparisons")
os.makedirs(OUT_DIR, exist_ok=True)
out_png = os.path.join(OUT_DIR, "difficult_classes_reference_vs_official_split.png")
out_csv = os.path.join(OUT_DIR, "difficult_classes_reference_vs_official_split.csv")

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
print(f"{'Class':<20} {'Referenced':>11} {'Exp08 (Off.)':>13} {'AbsDiff':>9} {'RelDiff':>9}")
for r in rows:
    print(f"{r['disease']:<20} {r['referenced_study_auc']:>11.4f} "
          f"{r['exp08_official_split_auc']:>13.4f} {r['absolute_difference']:>+9.4f} "
          f"{r['relative_difference_percent']:>+8.1f}%")
print()
print(f"Exp08 Official Split higher in {X}/5 difficult classes")
