"""
plot_difficult_classes_reference_exp08.py
Compact poster-friendly comparison for the 5 difficult disease classes.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["Infiltration", "Nodule", "Pneumonia", "Pleural_Thickening", "Consolidation"]

REFERENCED = {
    "Infiltration":       0.6848,
    "Nodule":             0.7687,
    "Pneumonia":          0.7174,
    "Pleural_Thickening": 0.7670,
    "Consolidation":      0.7472,
}

# Full-precision from densenet121_cbam_block34_bce_adamw_test_metrics.json
EXP08 = {
    "Infiltration":       0.7157038448553128,
    "Nodule":             0.7494051718740216,
    "Pneumonia":          0.7594963218252228,
    "Pleural_Thickening": 0.7958926937992726,
    "Consolidation":      0.7998199216072868,
}

# Sort by Referenced AUC ascending (bottom = lowest)
sorted_classes = sorted(CLASSES, key=lambda c: REFERENCED[c])
ref_vals = [REFERENCED[c]                                      for c in sorted_classes]
exp_vals = [EXP08[c]                                           for c in sorted_classes]
rel_pcts = [(EXP08[c] - REFERENCED[c]) / REFERENCED[c] * 100  for c in sorted_classes]

C_REF = "#5B8DB8"
C_EXP = "#E07B54"
C_POS = "#228B22"
C_NEG = "#CC3333"

fig, ax = plt.subplots(figsize=(5.5, 3.0))
n = len(sorted_classes)
y = np.arange(n)
h = 0.32

ax.barh(y - h / 2, ref_vals, h, label="Referenced Study", color=C_REF, alpha=0.88, edgecolor="white")
ax.barh(y + h / 2, exp_vals, h, label="Exp08",            color=C_EXP, alpha=0.88, edgecolor="white")

for i, rel in enumerate(rel_pcts):
    color = C_POS if rel >= 0 else C_NEG
    lbl   = f"+{rel:.1f}%" if rel >= 0 else f"{rel:.1f}%"
    ax.text(exp_vals[i] + 0.003, y[i] + h / 2, lbl,
            va="center", ha="left", fontsize=7, color=color, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(sorted_classes, fontsize=8.5)
ax.set_xlabel("AUC", fontsize=8.5)
ax.set_title("Reference Comparison: Difficult Classes",
             fontsize=9.5, fontweight="bold", pad=8)
ax.set_xlim(0.60, 0.88)
ax.xaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

plt.tight_layout()

OUT_DIR = os.path.join("results", "figures", "final_comparisons")
os.makedirs(OUT_DIR, exist_ok=True)
out_png = os.path.join(OUT_DIR, "difficult_classes_reference_vs_exp08.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_png}")
