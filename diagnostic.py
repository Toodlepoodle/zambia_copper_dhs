"""
diagnostic_privatisation.py
===========================
Diagnose the pre‑ vs post‑privatisation figure (fig5_privatisation).
Checks the data and recreates the plot with debugging annotations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(r"C:\Users\Sayan\Desktop\ZAMBIA_V1")
OUT_DIR = BASE_DIR / "OUTPUT" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

priv_path = BASE_DIR / "OUTPUT" / "privatisation_split.csv"
if not priv_path.exists():
    raise FileNotFoundError(f"File not found: {priv_path}")

priv = pd.read_csv(priv_path)
print("=== privatisation_split.csv loaded ===")
print(f"Shape: {priv.shape}")
print("Columns:", priv.columns.tolist())
print("\nFirst few rows:")
print(priv.head())

# Check for extreme values (e.g., > 1e10) that could indicate missing data errors
extreme = priv[abs(priv["coef_s"]) > 1e10]
if not extreme.empty:
    print("\n⚠️ WARNING: Extreme coefficients found (likely due to missing data):")
    print(extreme[["outcome", "sample", "coef_s", "pval"]])
else:
    print("\nNo extreme coefficients detected.")

# Filter to the outcomes used in the figure
outcomes = ["delivery_facility", "delivery_skilled", "anc_4plus", "had_fever"]
samples = ["pre_priv", "post_priv"]

# Prepare data for plotting
data = {}
for out in outcomes:
    data[out] = {}
    for samp in samples:
        row = priv[(priv["outcome"] == out) & (priv["sample"] == samp)]
        if not row.empty:
            coef = row.iloc[0]["coef_s"]
            pval = row.iloc[0]["pval"]
        else:
            coef = np.nan
            pval = np.nan
        data[out][samp] = (coef, pval)

print("\n=== Extracted coefficients for figure ===")
for out in outcomes:
    print(f"{out:20s}: pre={data[out]['pre_priv'][0]:8.2f} (p={data[out]['pre_priv'][1]:.3f}), "
          f"post={data[out]['post_priv'][0]:8.2f} (p={data[out]['post_priv'][1]:.3f})")

# Recreate the figure
labels = ["Facility\nDelivery", "Skilled Birth\nAttendance",
          "ANC 4+\nVisits", "Child\nFever"]
x = np.arange(len(outcomes))
width = 0.35

pre_b = [data[out]["pre_priv"][0] for out in outcomes]
post_b = [data[out]["post_priv"][0] for out in outcomes]
pre_p = [data[out]["pre_priv"][1] for out in outcomes]
post_p = [data[out]["post_priv"][1] for out in outcomes]

# Determine y-limits with some padding
all_vals = [v for v in pre_b + post_b if not np.isnan(v)]
ymin = min(all_vals) - 4 if all_vals else -10
ymax = max(all_vals) + 5 if all_vals else 10

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_ylim(ymin, ymax)

# Bar plot
bars1 = ax.bar(x - width/2, pre_b, width, label="Pre‑privatisation (1992, 1996)",
               color="#555555", alpha=0.7, edgecolor="white", hatch="////")
bars2 = ax.bar(x + width/2, post_b, width, label="Post‑privatisation (2007–2024)",
               color="#111111", alpha=0.85, edgecolor="white")

# Add significance stars
def stars(p):
    if pd.isna(p): return ""
    if p < 0.01:   return "***"
    if p < 0.05:   return "**"
    if p < 0.10:   return "*"
    return ""

for i, (pb, pp, pob, pop) in enumerate(zip(pre_b, pre_p, post_b, post_p)):
    if not np.isnan(pb):
        y_offset = 1 if pb >= 0 else -3
        ax.text(i - width/2, pb + y_offset, stars(pp),
                ha="center", va="center", fontsize=11, fontweight="bold")
    if not np.isnan(pob):
        y_offset = 1 if pob >= 0 else -3
        ax.text(i + width/2, pob + y_offset, stars(pop),
                ha="center", va="center", fontsize=11, fontweight="bold")

# Reference line
ax.axhline(0, color="black", linewidth=0.8)

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Percentage Points")
ax.set_title("Pre vs. Post Privatisation\n"
             "Effect of copper boom on mining vs. non‑mining provinces\n"
             "Note: 2002 excluded from both samples to avoid boundary overlap")
ax.legend(frameon=False)
ax.text(0.98, 0.02, "*** p<0.01  ** p<0.05  * p<0.10",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="gray")

# Add annotations for extreme values (if any)
for i, (pb, pob) in enumerate(zip(pre_b, post_b)):
    if not np.isnan(pb) and (pb < ymin + 2 or pb > ymax - 2):
        ax.annotate(f"pre={pb:.1f}", xy=(i - width/2, pb), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8, color="darkred")
    if not np.isnan(pob) and (pob < ymin + 2 or pob > ymax - 2):
        ax.annotate(f"post={pob:.1f}", xy=(i + width/2, pob), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8, color="darkred")

# Save the diagnostic figure
out_path = OUT_DIR / "diagnostic_privatisation.png"
fig.tight_layout()
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nDiagnostic plot saved to: {out_path}")
print("\nCheck the plot for overlapping bars, extreme values, or misaligned labels.")
print("If the figure still looks wrong, inspect the printed coefficient values above.")