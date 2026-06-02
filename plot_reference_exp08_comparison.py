"""
plot_reference_exp08_comparison.py
Generates a per-class AUC comparison between a referenced study and Exp08.
Reference comparison only; evaluation protocols may differ.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Data ─────────────────────────────────────────────────────────────────────

REFERENCED_AUC = {
    "Atelectasis":        0.7785,
    "Cardiomegaly":       0.8857,
    "Effusion":           0.8289,
    "Infiltration":       0.6848,
    "Mass":               0.8313,
    "Nodule":             0.7687,
    "Pneumonia":          0.7174,
    "Pneumothorax":       0.8670,
    "Consolidation":      0.7472,
    "Edema":              0.8436,
    "Emphysema":          0.9066,
    "Fibrosis":           0.8125,
    "Pleural_Thickening": 0.7670,
    "Hernia":             0.8952,
}

# Full-precision values from results/metrics/densenet121_cbam_block34_bce_adamw_test_metrics.json
EXP08_AUC = {
    "Atelectasis":        0.8169323477899486,
    "Cardiomegaly":       0.9026334296086965,
    "Effusion":           0.8877406069573278,
    "Infiltration":       0.7157038448553128,
    "Mass":               0.8451929604256563,
    "Nodule":             0.7494051718740216,
    "Pneumonia":          0.7594963218252228,
    "Pneumothorax":       0.8841078677289793,
    "Consolidation":      0.7998199216072868,
    "Edema":              0.9063432815050566,
    "Emphysema":          0.9213684323134864,
    "Fibrosis":           0.8102148743114294,
    "Pleural_Thickening": 0.7958926937992726,
    "Hernia":             0.9245970281356913,
}

DIFFICULT = {"Infiltration", "Nodule", "Pneumonia", "Pleural_Thickening", "Consolidation"}
REF_AVG   = 0.8096
EXP08_AVG = 0.8371034844812419

# ── Build DataFrame ───────────────────────────────────────────────────────────

rows = []
for d in REFERENCED_AUC:
    ref = REFERENCED_AUC[d]
    e08 = EXP08_AUC[d]
    rows.append({
        "disease":                     d,
        "referenced_study_auc":        round(ref, 4),
        "exp08_auc":                   round(e08, 4),
        "absolute_difference":         round(e08 - ref, 4),
        "relative_difference_percent": round((e08 - ref) / ref * 100, 2),
        "difficult_class":             d in DIFFICULT,
    })

df = pd.DataFrame(rows).sort_values("referenced_study_auc", ascending=True).reset_index(drop=True)

# ── Colours ───────────────────────────────────────────────────────────────────

C_REF      = "#5B8DB8"
C_EXP      = "#E07B54"
C_POS      = "#228B22"
C_NEG      = "#CC3333"
C_BAND     = "#FFF3CC"
C_ANN_BG   = "#EAF3FB"
C_ANN_EDGE = "#5B8DB8"

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 8))
n = len(df)
y = np.arange(n)
h = 0.35

# Background bands for difficult classes
for i, row in df.iterrows():
    if row["difficult_class"]:
        ax.axhspan(i - 0.5, i + 0.5, color=C_BAND, alpha=0.50, zorder=0)

# Bars
ax.barh(y - h / 2, df["referenced_study_auc"], h,
        label="Referenced Study", color=C_REF, alpha=0.88, edgecolor="white")
ax.barh(y + h / 2, df["exp08_auc"],             h,
        label="Exp08",            color=C_EXP, alpha=0.88, edgecolor="white")

# Relative-diff labels on Exp08 bars
for i, row in df.iterrows():
    rel   = row["relative_difference_percent"]
    color = C_POS if rel >= 0 else C_NEG
    lbl   = f"+{rel:.1f}%" if rel >= 0 else f"{rel:.1f}%"
    ax.text(row["exp08_auc"] + 0.004, i + h / 2, lbl,
            va="center", ha="left", fontsize=7.5, color=color, fontweight="bold")

# Axes
ax.set_yticks(y)
ax.set_yticklabels(df["disease"], fontsize=10)
ax.set_xlabel("AUC", fontsize=11)
ax.set_xlim(0.60, 1.02)
ax.set_title(
    "Per-Class AUC: Referenced Study vs. Exp08",
    fontsize=12, fontweight="bold", pad=14,
)
ax.xaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

# Legend (include difficult-class band)
band_patch = mpatches.Patch(color=C_BAND, alpha=0.7, label="Difficult class")
handles, labels_ = ax.get_legend_handles_labels()
ax.legend(handles + [band_patch], labels_ + ["Difficult class"],
          fontsize=9.5, loc="lower right")

# Average annotation box
abs_imp = EXP08_AVG - REF_AVG
rel_imp = abs_imp / REF_AVG * 100
ann = (
    f"Mean AUC\n"
    f"  Referenced Study : {REF_AVG:.4f}\n"
    f"  Exp08            : {EXP08_AVG:.4f}\n"
    f"  Δ : {abs_imp:+.4f}   ({rel_imp:+.2f}%)"
)
ax.annotate(
    ann,
    xy=(0.985, 0.02), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.5", facecolor=C_ANN_BG,
              edgecolor=C_ANN_EDGE, alpha=0.92),
)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────

OUT_DIR = os.path.join("results", "figures", "final_comparisons")
os.makedirs(OUT_DIR, exist_ok=True)
out_png = os.path.join(OUT_DIR, "reference_vs_exp08_per_class_auc.png")
out_csv = os.path.join(OUT_DIR, "reference_vs_exp08_per_class_auc.csv")

fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

df.to_csv(out_csv, index=False)

# ── Console summary ───────────────────────────────────────────────────────────

n_higher      = int((df["exp08_auc"] > df["referenced_study_auc"]).sum())
n_diff_higher = int((df["difficult_class"] & (df["exp08_auc"] > df["referenced_study_auc"])).sum())
n_diff_total  = int(df["difficult_class"].sum())

print(f"Figure saved : {out_png}")
print(f"CSV saved    : {out_csv}")
print()
print("=" * 52)
print(f"  Exp08 average AUC             : {EXP08_AVG:.4f}")
print(f"  Referenced Study average AUC  : {REF_AVG:.4f}")
print(f"  Absolute improvement          : {abs_imp:+.4f}")
print(f"  Relative improvement          : {rel_imp:+.2f}%")
print(f"  Classes where Exp08 higher    : {n_higher} / {n}")
print(f"  Difficult classes Exp08 higher: {n_diff_higher} / {n_diff_total}")
print("=" * 52)
