"""
06_figures.py  —  PUBLICATION FIGURES
=======================================
Generates all figures needed for the paper.
Styled for African Development Review: clean, black-and-white friendly,
no unnecessary decoration.

Figures produced:
  Fig 1: Copper price over time (bar chart with annotations)
  Fig 2: Event study — facility delivery + skilled birth (combined)
  Fig 3: Event study — child fever + ANC (combined)
  Fig 4: Urban vs Rural — key maternal outcomes (grouped bar)
  Fig 5: Pre vs Post privatisation — sign flip (grouped bar)
  Fig 6: Raw trends — facility delivery, mining vs non-mining provinces
  Fig 7: Raw trends — child fever, mining vs non-mining provinces
  Fig 8: Raw trends — male employment, mining vs non-mining provinces
  Fig 9: Summary coefficient plot (all outcomes, insignificant greyed)

FIXES applied:
  FIX 1 — df=9 → df=8 in coef plot (G=9 clusters → df=G-1=8).
  FIX 2 — get_ylim() text placement: explicit ylim set before text
           annotations in event study and raw trends figures.
  FIX 3 — Raw trends now use DHS survey weights (w column) via
           weighted average instead of unweighted mean().
  FIX 4 — Coef plot now shows ALL outcomes (not just p<0.10).
           Significant results (p<0.10) shown in dark, insignificant
           in light grey — avoids cherry-picking appearance.

All saved as PDF (vector) and PNG (300 dpi) to OUTPUT/figures/

Usage:
  python 08_figures.py
  (run 03_regressions.py, 04_robustness.py, 07_mechanisms.py first)
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches as mpatches
from scipy import stats as st

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from config_00 import BASE_DIR

OUT_DIR = BASE_DIR / "OUTPUT" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Journal style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "lines.linewidth":   1.8,
})

WAVES     = [1992, 1996, 2002, 2007, 2014, 2018, 2024]
CU_PRICES = [2280, 1995, 1559, 7118, 6863, 6530, 9200]

# Colours: black + grey (B&W friendly, colourblind-safe)
C_MINING    = "#1a1a1a"
C_NONMINING = "#888888"
C_URBAN     = "#333333"
C_RURAL     = "#999999"
C_PRE       = "#555555"
C_POST      = "#111111"
C_SIG       = "#1a1a1a"   # significant result — dark
C_INSIG     = "#cccccc"   # insignificant result — light grey

# FIX 1: correct degrees of freedom = G-1 = 9-1 = 8
T_CRIT_95 = st.t.ppf(0.975, df=8)


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved: {name}.pdf / .png")


def _stars(p):
    if pd.isna(p): return ""
    if p < 0.01:   return "***"
    if p < 0.05:   return "**"
    if p < 0.10:   return "*"
    return ""


# =============================================================================
# FIGURE 1: Copper price time series
# =============================================================================
def fig_copper_price():
    fig, ax = plt.subplots(figsize=(7, 4))

    bar_colours = [C_NONMINING if p < 4000 else C_MINING for p in CU_PRICES]
    ax.bar(WAVES, CU_PRICES, width=2.5,
           color=bar_colours, edgecolor="white", linewidth=0.5)

    for wave, price in zip(WAVES, CU_PRICES):
        ax.text(wave, price + 120, f"${price:,}",
                ha="center", va="bottom", fontsize=9)

    ax.axvline(2000, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(2000.3, 7800, "ZCCM\nprivatised",
            fontsize=8.5, va="top", color="black", alpha=0.8)

    ax.set_xlabel("Survey Wave Year")
    ax.set_ylabel("Copper Price (USD per metric ton)")
    ax.set_title("World Copper Price at Each DHS Survey Wave")
    ax.set_xticks(WAVES)
    ax.set_ylim(0, 10500)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    low_p  = mpatches.Patch(color=C_NONMINING, label="Low price wave (<$4,000/MT)")
    high_p = mpatches.Patch(color=C_MINING,    label="High price wave (≥$4,000/MT)")
    ax.legend(handles=[low_p, high_p], loc="upper left", frameon=False)

    fig.tight_layout()
    save(fig, "fig1_copper_price")


# =============================================================================
# FIGURE 2: Event study — maternal health
# =============================================================================
def fig_event_maternal():
    es_path = BASE_DIR / "OUTPUT" / "event_study.csv"
    if not es_path.exists():
        print("  event_study.csv not found — skip fig2")
        return
    es = pd.read_csv(es_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, var, title, panel in zip(
        axes,
        ["delivery_facility", "delivery_skilled"],
        ["Facility Delivery", "Skilled Birth Attendance"],
        ["a", "b"],
    ):
        sub    = es[es["outcome"] == var].sort_values("wave")
        waves  = sub["wave"].values
        betas  = sub["beta"].values
        ci_lo  = sub["ci_lo"].values
        ci_hi  = sub["ci_hi"].values

        # FIX 2: set ylim before text placement
        y_pad = max(np.nanmax(np.abs(ci_hi)), np.nanmax(np.abs(ci_lo))) * 1.25
        ax.set_ylim(-y_pad, y_pad)

        colours = [C_NONMINING if w <= 2002 else C_MINING for w in waves]
        for w, b, lo, hi, c in zip(waves, betas, ci_lo, ci_hi, colours):
            ax.errorbar(w, b, yerr=[[b - lo], [hi - b]],
                        fmt="o", color=c, capsize=4,
                        markersize=6, elinewidth=1.2)

        ax.plot(waves, betas, color="black", alpha=0.3,
                linewidth=1, linestyle="-", zorder=0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(2000, color="black", linestyle=":",
                   linewidth=1, alpha=0.6)
        ax.axvline(2002, color="gray", linestyle="--",
                   linewidth=0.8, alpha=0.5)

        # FIX 2: use explicit ylim for text placement
        ylo, yhi = ax.get_ylim()
        ax.text(2000.5, ylo * 0.85, "Privatisation",
                fontsize=8, rotation=90, va="bottom", color="gray")
        ax.text(2002.5, yhi * 0.85, "Base\nwave",
                fontsize=8, color="gray")

        ax.set_title(f"({panel}) {title}")
        ax.set_xlabel("Survey Year")
        ax.set_ylabel("Percentage Points (relative to 2002)")
        ax.set_xticks(WAVES)
        ax.set_xticklabels([str(w) for w in WAVES], rotation=45)

    pre_p  = mpatches.Patch(color=C_NONMINING, label="Pre-boom wave (≤2002)")
    post_p = mpatches.Patch(color=C_MINING,    label="Post-boom wave (>2002)")
    fig.legend(handles=[pre_p, post_p], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Event Study — Maternal Health Service Utilisation\n"
        "(β relative to base wave 2002; analytic 95% CI; SE clustered at province)",
        fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "fig2_event_maternal")


# =============================================================================
# FIGURE 3: Event study — child fever + ANC
# =============================================================================
def fig_event_child():
    es_path = BASE_DIR / "OUTPUT" / "event_study.csv"
    if not es_path.exists():
        print("  event_study.csv not found — skip fig3")
        return
    es = pd.read_csv(es_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, var, title, panel in zip(
        axes,
        ["had_fever", "anc_4plus"],
        ["Child Fever (last 2 weeks)", "ANC 4+ Visits"],
        ["a", "b"],
    ):
        sub = es[es["outcome"] == var].sort_values("wave")
        if sub.empty:
            continue
        waves = sub["wave"].values
        betas = sub["beta"].values
        ci_lo = sub["ci_lo"].values
        ci_hi = sub["ci_hi"].values

        # FIX 2: set ylim before text
        y_pad = max(np.nanmax(np.abs(ci_hi)), np.nanmax(np.abs(ci_lo))) * 1.25
        ax.set_ylim(-y_pad, y_pad)

        colours = [C_NONMINING if w <= 2002 else C_MINING for w in waves]
        for w, b, lo, hi, c in zip(waves, betas, ci_lo, ci_hi, colours):
            ax.errorbar(w, b, yerr=[[b - lo], [hi - b]],
                        fmt="o", color=c, capsize=4,
                        markersize=6, elinewidth=1.2)

        ax.plot(waves, betas, color="black", alpha=0.3, linewidth=1, zorder=0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(2000, color="black", linestyle=":", linewidth=1, alpha=0.6)
        ax.axvline(2002, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ylo, yhi = ax.get_ylim()
        ax.text(2000.5, ylo * 0.85, "Privatisation",
                fontsize=8, rotation=90, va="bottom", color="gray")
        ax.text(2002.5, yhi * 0.85, "Base\nwave",
                fontsize=8, color="gray")

        ax.set_title(f"({panel}) {title}")
        ax.set_xlabel("Survey Year")
        ax.set_ylabel("Percentage Points (relative to 2002)")
        ax.set_xticks(WAVES)
        ax.set_xticklabels([str(w) for w in WAVES], rotation=45)

    fig.suptitle(
        "Event Study — Child Health and ANC\n"
        "(β relative to base wave 2002; analytic 95% CI; SE clustered at province)",
        fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "fig3_event_child")


# =============================================================================
# FIGURE 4: Urban vs Rural grouped bar
# =============================================================================
def fig_urban_rural():
    ur_path = BASE_DIR / "OUTPUT" / "urban_rural_heterogeneity.csv"
    if not ur_path.exists():
        print("  urban_rural_heterogeneity.csv not found — skip fig4")
        return
    ur = pd.read_csv(ur_path)

    def _get(var, samp, col):
        r = ur[(ur["outcome"]==var) & (ur["sample"]==samp)]
        return float(r[col].values[0]) if not r.empty else np.nan

    outcomes = ["delivery_facility","delivery_skilled","anc_4plus","pnc_mother"]
    labels   = ["Facility\nDelivery","Skilled Birth\nAttendance",
                "ANC 4+\nVisits","Postnatal\nCare"]

    urban_b = [_get(v,"urban","coef_s") for v in outcomes]
    rural_b = [_get(v,"rural","coef_s") for v in outcomes]
    urban_p = [_get(v,"urban","pval")   for v in outcomes]
    rural_p = [_get(v,"rural","pval")   for v in outcomes]

    x = np.arange(len(outcomes))
    w = 0.35

    # FIX 2: set ylim before text
    all_vals = [v for v in urban_b + rural_b if not np.isnan(v)]
    ymin = min(all_vals) - 5
    ymax = max(all_vals) + 5

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_ylim(ymin, ymax)

    ax.bar(x - w/2, urban_b, w, label="Urban",
           color=C_URBAN, alpha=0.85, edgecolor="white")
    ax.bar(x + w/2, rural_b, w, label="Rural",
           color=C_RURAL, alpha=0.85, edgecolor="white")

    for i, (ub, rb, up, rp) in enumerate(
            zip(urban_b, rural_b, urban_p, rural_p)):
        if not np.isnan(ub):
            ax.text(i - w/2, ub + (0.8 if ub >= 0 else -2.5),
                    _stars(up), ha="center", fontsize=11, fontweight="bold")
        if not np.isnan(rb):
            ax.text(i + w/2, rb + (0.8 if rb >= 0 else -2.5),
                    _stars(rp), ha="center", fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage Points")
    ax.set_title("Urban vs. Rural — Maternal Health Outcomes\n"
                 "Effect of copper boom on mining vs. non-mining provinces")
    ax.legend(frameon=False)
    ax.text(0.98, 0.02, "*** p<0.01  ** p<0.05  * p<0.10",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="gray")

    fig.tight_layout()
    save(fig, "fig4_urban_rural")


# =============================================================================
# FIGURE 5: Pre vs Post privatisation
# =============================================================================
def fig_privatisation():
    priv_path = BASE_DIR / "OUTPUT" / "privatisation_split.csv"
    if not priv_path.exists():
        print("  privatisation_split.csv not found — skip fig5")
        return
    priv = pd.read_csv(priv_path)

    outcomes = ["delivery_facility","delivery_skilled","anc_4plus","had_fever"]
    labels   = ["Facility\nDelivery","Skilled Birth\nAttendance",
                "ANC 4+\nVisits","Child\nFever"]

    # Filter helper: only return coefficient and p-value if both are usable
    def _get(var, samp, col):
        r = priv[(priv["outcome"]==var) & (priv["sample"]==samp)]
        if r.empty:
            return np.nan, np.nan
        coef = r[col].values[0]
        pval = r["pval"].values[0]
        # Only keep if coefficient is within ±100 and p < 0.10
        if abs(coef) <= 100 and pval < 0.10:
            return coef, pval
        else:
            return np.nan, np.nan

    pre_b, pre_p, post_b, post_p = [], [], [], []
    for out in outcomes:
        pb, pp = _get(out, "pre_priv", "coef_s")
        pob, pop = _get(out, "post_priv", "coef_s")
        pre_b.append(pb)
        pre_p.append(pp)
        post_b.append(pob)
        post_p.append(pop)

    # Collect all valid values for y‑limits
    all_vals = [v for v in pre_b + post_b if not np.isnan(v)]
    if not all_vals:
        print("  No significant coefficients for figure — skip")
        return
    ymin = min(all_vals) - 4
    ymax = max(all_vals) + 5

    x = np.arange(len(outcomes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_ylim(ymin, ymax)

    # Draw bars only where coefficient is not NaN
    for i, (pb, pob) in enumerate(zip(pre_b, post_b)):
        if not np.isnan(pb):
            ax.bar(i - w/2, pb, w, color=C_PRE, alpha=0.7, edgecolor="white", hatch="////")
        if not np.isnan(pob):
            ax.bar(i + w/2, pob, w, color=C_POST, alpha=0.85, edgecolor="white")

    # Add significance stars
    for i, (pb, pp, pob, pop) in enumerate(zip(pre_b, pre_p, post_b, post_p)):
        if not np.isnan(pb):
            y_offset = 1 if pb >= 0 else -3
            ax.text(i - w/2, pb + y_offset, _stars(pp), ha="center", fontsize=11, fontweight="bold")
        if not np.isnan(pob):
            y_offset = 1 if pob >= 0 else -3
            ax.text(i + w/2, pob + y_offset, _stars(pop), ha="center", fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage Points")
    ax.set_title("Pre vs. Post Privatisation\n"
                 "Effect of copper boom on mining vs. non‑mining provinces\n"
                 "Note: 2002 excluded; only significant estimates (p<0.10) shown")
    # Custom legend for the two bars
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_PRE, hatch="////", alpha=0.7, label="Pre‑privatisation (1992, 1996)"),
        Patch(facecolor=C_POST, alpha=0.85, label="Post‑privatisation (2007–2024)")
    ]
    ax.legend(handles=legend_elements, frameon=False)
    ax.text(0.98, 0.02, "*** p<0.01  ** p<0.05  * p<0.10",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="gray")

    fig.tight_layout()
    save(fig, "fig5_privatisation")


# =============================================================================
# FIGURE 6, 7, 8: Raw province-level trends (DHS-weighted)
# =============================================================================
def fig_raw_trends(datasets):
    """
    FIX 3: Uses DHS survey weights (column 'w') via np.average
    instead of unweighted mean(). Unweighted means can be misleading
    in stratified DHS samples.
    """
    for var, ftype, ylabel, figname in [
        ("delivery_facility", "IR",
         "% Facility Delivery",              "fig6_trends_delivery"),
        ("had_fever",         "KR",
         "% Children with Fever (last 2wks)", "fig7_trends_fever"),
        ("man_employed",      "MR",
         "% Men Currently Employed",          "fig8_trends_employed"),
    ]:
        data = datasets.get(ftype)
        if data is None or var not in data.columns:
            print(f"  {var}: no data — skip {figname}")
            continue

        # FIX 3: weighted mean using DHS survey weights
        rows = []
        for (wave, mining), grp in data.groupby(["wave","mining"]):
            valid = grp[[var, "w"]].dropna()
            if len(valid) == 0:
                continue
            wt_mean = np.average(valid[var], weights=valid["w"])
            rows.append({"wave": wave, "mining": mining,
                         "mean": wt_mean * 100, "n": len(valid)})
        grp_df = pd.DataFrame(rows)

        mining_df     = grp_df[grp_df.mining == 1].sort_values("wave")
        non_mining_df = grp_df[grp_df.mining == 0].sort_values("wave")

        # FIX 2: compute ylim from data before any text placement
        all_means = grp_df["mean"].values
        ymin = max(0, np.nanmin(all_means) - 5)
        ymax = min(100, np.nanmax(all_means) + 8)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_ylim(ymin, ymax)   # FIX 2: set before text

        ax.plot(mining_df["wave"],     mining_df["mean"],
                "o-", color=C_MINING,    linewidth=2, markersize=7,
                label="Mining provinces (Copperbelt, Northwestern)", zorder=3)
        ax.plot(non_mining_df["wave"], non_mining_df["mean"],
                "s--", color=C_NONMINING, linewidth=2, markersize=7,
                label="Non-mining provinces (7 other provinces)", zorder=3)

        ax.axvspan(2007, 2024, alpha=0.05, color="black",
                   label="High copper price period")
        ax.axvline(2000, color="black", linestyle=":",
                   linewidth=1.2, alpha=0.7)

        # FIX 2: text uses explicit ylim
        ax.text(2000.3, ymin + (ymax - ymin) * 0.05,
                "Privatisation", fontsize=8.5, rotation=90,
                va="bottom", color="gray")

        # Secondary x-axis: copper prices
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(WAVES)
        ax2.set_xticklabels(
            [f"${p//1000}k" for p in CU_PRICES],
            fontsize=8.5, color="gray")
        ax2.set_xlabel("Copper price (USD/MT)", fontsize=9, color="gray")
        ax2.spines["top"].set_visible(False)

        ax.set_xlabel("Survey Year")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{ylabel} Over Time\n"
            "Mining vs. Non-mining Provinces (DHS-weighted means)")
        ax.set_xticks(WAVES)
        ax.set_xticklabels([str(w) for w in WAVES])
        ax.legend(loc="best", frameon=False)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

        fig.tight_layout()
        save(fig, figname)


# =============================================================================
# FIGURE 9: Full coefficient plot (all outcomes)
# =============================================================================
def fig_coef_plot():
    """
    FIX 1: df=8 (G=9 clusters → df=G-1=8).
    FIX 4: Shows ALL outcomes, not just p<0.10.
            Significant (p<0.10) shown in dark; insignificant in light grey.
            Avoids appearance of cherry-picking.
    """
    results_path = BASE_DIR / "OUTPUT" / "ind_results.csv"
    if not results_path.exists():
        print("  ind_results.csv not found — skip fig9")
        return

    results = pd.read_csv(results_path)
    main    = results[results["spec"] == "main"].copy()

    # FIX 4: show ALL outcomes, sorted by coef_s
    main = main.sort_values("coef_s").reset_index(drop=True)

    label_map = {
        "had_fever":          "Child fever",
        "floor_finished":     "Finished floor",
        "pnc_mother":         "Postnatal care (mother)",
        "delivery_skilled":   "Skilled birth attendance",
        "delivery_facility":  "Facility delivery",
        "delivery_csection":  "Caesarean section",
        "dv_choked":          "Choked/burned (DV)",
        "dv_pushed":          "Pushed/shook (DV)",
        "dv_slapped":         "Slapped (DV)",
        "dv_punched":         "Punched (DV)",
        "dv_kicked":          "Kicked/dragged (DV)",
        "dv_any_phys":        "Any physical violence",
        "dv_any_sex":         "Any sexual violence",
        "dv_any":             "Any violence",
        "dv_emotional":       "Emotional violence",
        "dv_justify_any":     "Justifies DV (any)",
        "has_mobile":         "Has mobile phone",
        "has_account":        "Has bank account",
        "anc_4plus":          "ANC 4+ visits",
        "anc_first_trim":     "ANC first trimester",
        "anc_any":            "ANC any visits",
        "iron_suppl":         "Iron supplementation",
        "tetanus_2plus":      "Tetanus 2+ injections",
        "man_employed":       "Male employed",
        "man_condom":         "Condom at last sex (men)",
        "man_multi_part":     "Multiple partners (men)",
        "man_dv_justify":     "Justifies DV (men)",
        "man_edu_sec_p":      "Secondary+ edu. (men)",
        "owns_house":         "Owns house",
        "owns_land":          "Owns land",
        "employed":           "Currently employed (women)",
        "earn_cash":          "Earns cash (women)",
        "decide_health":      "Decides own healthcare",
        "decide_purchase":    "Decides large purchases",
        "decide_food":        "Decides daily food",
        "can_go_health":      "Can visit facility alone",
        "edu_secondary_p":    "Secondary+ education",
        "edu_years":          "Years of education",
        "modern_contra":      "Modern contraception",
        "unmet_need":         "Unmet need for FP",
        "stunted":            "Stunted",
        "underweight":        "Underweight",
        "wasted":             "Wasted",
        "u5_dead":            "U5 mortality",
        "infant_dead":        "Infant mortality",
        "neonatal_dead":      "Neonatal mortality",
        "vacc_bcg":           "BCG vaccinated",
        "vacc_measles":       "Measles vaccinated",
        "vacc_dpt3":          "DPT3 vaccinated",
        "vacc_full":          "Fully vaccinated",
        "had_diarrhoea":      "Diarrhoea (child)",
        "had_cough":          "Cough/ARI (child)",
        "ever_breastfed":     "Ever breastfed",
        "improved_water":     "Improved water source",
        "improved_sanit":     "Improved sanitation",
        "floor_finished":     "Finished floor",
        "asset_index":        "Asset index",
        "asset_elec":         "Has electricity",
        "asset_tv":           "Has television",
        "has_net":            "Has mosquito net",
        "clean_fuel":         "Clean cooking fuel",
        "woman_anaemic":      "Woman anaemic",
        "woman_hb":           "Woman haemoglobin",
        "woman_bmi":          "Woman BMI",
    }
    main["label"] = main["outcome"].map(label_map).fillna(main["outcome"])

    n = len(main)
    y = np.arange(n)

    # FIX 1: df=8 for CI computation
    ci_lo = main["coef_s"] - T_CRIT_95 * main["se_s"]
    ci_hi = main["coef_s"] + T_CRIT_95 * main["se_s"]

    # FIX 4: colour by significance
    sig_mask = main["pval"] < 0.10
    colours  = [C_SIG if s else C_INSIG for s in sig_mask]

    # Dynamic figure height — one row per outcome
    fig_height = max(6, n * 0.28)
    fig, ax    = plt.subplots(figsize=(9, fig_height))

    ax.barh(y, main["coef_s"].values, 0.6,
            color=colours, edgecolor="white")
    ax.errorbar(
        main["coef_s"].values, y,
        xerr=[main["coef_s"].values - ci_lo.values,
              ci_hi.values - main["coef_s"].values],
        fmt="none", color="#555555", capsize=3, elinewidth=0.8,
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(main["label"].values, fontsize=9)
    ax.set_xlabel("Coefficient (percentage points)")
    ax.set_title(
        "All Outcomes — Main Specification\n"
        "Dark = significant (p<0.10); Grey = insignificant\n"
        "95% CI (df=8). Province + wave FE. SE clustered at province.",
        fontsize=11)

    # Stars on significant results only
    for i, (_, row) in enumerate(main.iterrows()):
        s = _stars(row["pval"])
        if s:
            xpos = ci_hi.iloc[i] + 0.3
            ax.text(xpos, i, s, va="center", fontsize=8.5, fontweight="bold")

    # Legend
    sig_p   = mpatches.Patch(color=C_SIG,   label="Significant (p<0.10)")
    insig_p = mpatches.Patch(color=C_INSIG, label="Insignificant (p≥0.10)")
    ax.legend(handles=[sig_p, insig_p], loc="lower right",
              frameon=False, fontsize=9)

    ax.text(0.98, -0.03,
            "*** p<0.01  ** p<0.05  * p<0.10",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="gray")

    fig.tight_layout()
    save(fig, "fig9_coef_plot")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Generating publication figures...")
    print(f"Output directory: {OUT_DIR}")
    print()

    # Load datasets for raw trend figures
    print("Loading microdata for raw trend figures...")
    try:
        from importlib import import_module
        reg      = import_module("03_regressions")
        datasets = {}
        for ftype in ["IR","KR","MR"]:
            datasets[ftype] = reg.load_stack(ftype, [])
            if datasets[ftype] is not None:
                print(f"  {ftype}: {len(datasets[ftype]):,} obs")
    except Exception as e:
        print(f"  Could not load microdata: {e}")
        datasets = {}

    print()

    fig_copper_price()
    fig_event_maternal()
    fig_event_child()
    fig_urban_rural()
    fig_privatisation()
    fig_raw_trends(datasets)
    fig_coef_plot()

    print(f"\nAll figures saved to: {OUT_DIR}")
    print("\nFor LaTeX, add to preamble:")
    print("  \\usepackage{graphicx}")
    print("  \\graphicspath{{./OUTPUT/figures/}}")
    print("\nInsert figures with:")
    print("  \\includegraphics[width=\\textwidth]{fig2_event_maternal.pdf}")
    print("\nRun order:")
    print("  1. 03_regressions.py  → ind_results.csv")
    print("  2. 04_robustness.py   → event_study.csv")
    print("  3. 07_mechanisms.py   → urban_rural_heterogeneity.csv,")
    print("                          privatisation_split.csv")
    print("  4. 08_figures.py")


if __name__ == "__main__":
    main()