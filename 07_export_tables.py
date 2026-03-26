"""
07_export_tables.py
-------------------
Reads every results CSV from OUTPUT/ and writes publication-ready LaTeX
table fragments to OUTPUT/tables/.

Column names match the EXACT headers in each CSV file.
Includes a read_csv() wrapper that neutralises the pandas >=2.0
ArrowDtype / StringDtype bug where chained boolean filters silently
return zero rows.

Run from the project root:
    python 07_export_tables.py
"""

import os
import math
import pandas as pd
from scipy import stats as _stats

# t critical value for 95% CI with df = G-1 = 8 (G=9 province clusters)
T_CRIT_95 = _stats.t.ppf(0.975, df=8)   # ≈ 2.306

# ── paths ──────────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(ROOT, "OUTPUT")
TABLES = os.path.join(OUT, "tables")
os.makedirs(TABLES, exist_ok=True)

# ── ArrowDtype-safe CSV reader ─────────────────────────────────────────────

def read_csv(path):
    """Read CSV and convert any ArrowDtype/StringDtype columns to plain
    Python object dtype so that df[mask] boolean indexing works correctly
    in pandas >= 2.0."""
    df = pd.read_csv(path)
    for c in df.columns:
        try:
            dtype_str = str(df[c].dtype).lower()
            if dtype_str in ("str", "string") or dtype_str.startswith("string"):
                df[c] = df[c].astype(object)
        except Exception:
            pass
    return df

# ── formatting helpers ─────────────────────────────────────────────────────

def star(p):
    try:
        p = float(p)
        if math.isnan(p): return ""
    except Exception:
        return ""
    if p < 0.01:  return r"$^{***}$"
    if p < 0.05:  return r"$^{**}$"
    if p < 0.10:  return r"$^{*}$"
    return ""

def fmt_coef(val, p=None):
    try:
        v = float(val)
        if math.isnan(v): return "---"
    except Exception:
        return "---"
    sign = "+" if v >= 0 else ""
    s = star(p) if p is not None else ""
    return f"${sign}{v:.2f}${s}"

def fmt_se(val):
    try:
        v = float(val)
        if math.isnan(v): return "---"
    except Exception:
        return "---"
    return f"{v:.2f}"

def fmt_p(val):
    try:
        v = float(val)
        if math.isnan(v): return "---"
    except Exception:
        return str(val)
    if v < 0.001: return r"$<$0.001"
    return f"{v:.3f}"

def fmt_n(val):
    try:
        return f"{int(float(val)):,}"
    except Exception:
        return str(val)

def fmt_ci(lo, hi):
    """Format a 95% CI as [lo, hi] in percentage points.
    Returns '---' if either bound is missing."""
    try:
        l = float(lo)
        h = float(hi)
        if math.isnan(l) or math.isnan(h): return "---"
    except Exception:
        return "---"
    lsign = "+" if l >= 0 else ""
    hsign = "+" if h >= 0 else ""
    return f"$[{lsign}{l:.2f},\\,{hsign}{h:.2f}]$"

def ci_from_se(coef_s, se_s):
    """Compute 95% CI bounds from an already-scaled coefficient and SE,
    using the t_{df=8} critical value (G=9 province clusters, df=G-1=8).
    Returns (lo, hi) as floats, or (nan, nan) if inputs are invalid."""
    try:
        c = float(coef_s)
        s = float(se_s)
        if math.isnan(c) or math.isnan(s) or s < 0:
            return (math.nan, math.nan)
        return (c - T_CRIT_95 * s, c + T_CRIT_95 * s)
    except Exception:
        return (math.nan, math.nan)

def write_tex(fname, body):
    path = os.path.join(TABLES, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"  wrote OUTPUT/tables/{fname}")

# ── human-readable outcome labels ─────────────────────────────────────────
# Used wherever the CSV stores raw variable names rather than labels.

LABELS = {
    "delivery_facility":  "Facility delivery",
    "delivery_skilled":   "Skilled birth attendance",
    "anc_4plus":          "ANC 4+ visits",
    "pnc_mother":         "Postnatal care (mother)",
    "had_fever":          "Child fever",
    "had_diarrhoea":      "Diarrhoea last 2 weeks",
    "stunted":            "Stunted",
    "underweight":        "Underweight",
    "man_employed":       "Male employed",
    "floor_finished":     "Finished floor",
    "dv_choked":          "Choked or burned (ever)",
    "has_mobile":         "Has mobile phone",
    "dv_justify_any":     "Justifies DV (any reason)",
    "owns_house":         "Owns house",
    "man_edu_sec_p":      r"Secondary+ education",
    "man_edu_level":      "Education level (0--3)",
    "man_condom":         "Condom at last sex",
    "man_multi_part":     "2+ partners last year",
    "man_dv_justify":     "Justifies DV",
}

def lbl(outcome):
    return LABELS.get(str(outcome), str(outcome))


# ══════════════════════════════════════════════════════════════════════════
# TABLE A1 – Urban vs Rural Heterogeneity
# CSV columns: outcome, sample (urban/rural), coef_s, se_s, pval, nobs
# NOTE: ci_lo/ci_hi not in this CSV — SE shown instead.
# ══════════════════════════════════════════════════════════════════════════

def make_urban_rural():
    df = read_csv(os.path.join(OUT, "urban_rural_heterogeneity.csv"))
    outcomes = [
        "delivery_facility", "delivery_skilled", "anc_4plus",
        "pnc_mother", "had_fever", "stunted", "underweight", "man_employed",
    ]
    rows = []
    for o in outcomes:
        u = df.loc[(df["outcome"] == o) & (df["sample"] == "urban")]
        r = df.loc[(df["outcome"] == o) & (df["sample"] == "rural")]
        bu = fmt_coef(u["coef_s"].iloc[0], u["pval"].iloc[0]) if not u.empty else "---"
        pu = fmt_p(u["pval"].iloc[0])                          if not u.empty else "---"
        br = fmt_coef(r["coef_s"].iloc[0], r["pval"].iloc[0]) if not r.empty else "---"
        pr = fmt_p(r["pval"].iloc[0])                          if not r.empty else "---"
        rows.append(f"    {lbl(o)} & {bu} & {pu} & {br} & {pr} \\\\")

    body = r"""\begin{threeparttable}
\begin{tabular}{lcccc}
\toprule
Outcome & Urban $\hat{\beta}$ (pp) & $p$ & Rural $\hat{\beta}$ (pp) & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} Each cell is a separate regression on the urban or
rural subsample. Province and wave fixed effects absorbed by iterative
demeaning. SE clustered at province level ($G = 9$).
$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.
\end{tablenotes}
\end{threeparttable}
"""
    write_tex("tab_urban_rural.tex", body)


# ══════════════════════════════════════════════════════════════════════════
# TABLE A2 – Pre vs Post Privatisation Split
# CSV columns: outcome, sample (pre_priv/post_priv), coef_s, se_s, pval, nobs
# NOTE: ci_lo/ci_hi not in this CSV — SE shown instead.
# ══════════════════════════════════════════════════════════════════════════

def make_privatisation():
    df = read_csv(os.path.join(OUT, "privatisation_split.csv"))
    outcomes = [
        "delivery_facility", "delivery_skilled", "anc_4plus",
        "pnc_mother", "had_fever", "stunted", "man_employed",
    ]
    rows = []
    for o in outcomes:
        pre  = df.loc[(df["outcome"] == o) & (df["sample"] == "pre_priv")]
        post = df.loc[(df["outcome"] == o) & (df["sample"] == "post_priv")]
        bp  = fmt_coef(pre["coef_s"].iloc[0],  pre["pval"].iloc[0])  if not pre.empty  else "---"
        pp  = fmt_p(pre["pval"].iloc[0])                              if not pre.empty  else "---"
        bpo = fmt_coef(post["coef_s"].iloc[0], post["pval"].iloc[0]) if not post.empty else "---"
        ppo = fmt_p(post["pval"].iloc[0])                             if not post.empty else "---"
        rows.append(f"    {lbl(o)} & {bp} & {pp} & {bpo} & {ppo} \\\\")

    body = r"""\begin{threeparttable}
\begin{tabular}{lcccc}
\toprule
Outcome & Pre-priv.\ $\hat{\beta}$ (pp) & $p$ & Post-priv.\ $\hat{\beta}$ (pp) & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} Pre-privatisation sample: waves 1992 and 1996.
Post-privatisation sample: waves 2007, 2014, 2018, and 2024.
Wave 2002 excluded from both samples to avoid boundary overlap.
Postnatal care data unavailable before 2007.
Pre-privatisation point estimates for delivery outcomes are very large
due to near-singular design matrices with only two treated provinces in
two waves; we rely on the sign and the post-privatisation estimates for
substantive interpretation.
SE clustered at province level ($G = 9$).
$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.
\end{tablenotes}
\end{threeparttable}
"""
    write_tex("tab_privatisation.tex", body)


# ══════════════════════════════════════════════════════════════════════════
# TABLE A3 – Wild Cluster Bootstrap vs Analytic p-values
# CSV columns: outcome, coef_s, se_s, tstat, pval_analytic, pval_bootstrap,
#              nobs, n_clusters
# NOTE: Bootstrap table reports both p-values; CIs not applicable here.
# ══════════════════════════════════════════════════════════════════════════

def make_bootstrap():
    df = read_csv(os.path.join(OUT, "bootstrap_results.csv"))
    rows = []
    for _, r in df.iterrows():
        beta = fmt_coef(r["coef_s"], r["pval_analytic"])
        se   = fmt_se(r["se_s"])
        pan  = fmt_p(r["pval_analytic"])
        pbs  = fmt_p(r["pval_bootstrap"])
        n    = fmt_n(r["nobs"])
        rows.append(
            f"    {lbl(r['outcome'])} & {beta} & {se} & {pan} & {pbs} & {n} \\\\"
        )

    body = r"""\begin{threeparttable}
\begin{tabular}{lccccc}
\toprule
Outcome & $\hat{\beta}$ (pp) & SE & $p$ (analytic) & $p$ (bootstrap) & $N$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} Webb (2014) six-point weights, $B = 999$ replications,
$G = 9$ clusters, two treated. H$_0$ imposed via restricted residuals.
Bootstrap $p$-values are larger than analytic $p$-values because only two
of nine clusters are treated, producing a coarse null distribution
\citep{mackinnon2019wild}. This is a known limitation of the bootstrap with
very few treated clusters, not evidence against the main results.
\end{tablenotes}
\end{threeparttable}
"""
    write_tex("tab_bootstrap.tex", body)


# ══════════════════════════════════════════════════════════════════════════
# TABLE A4 – High vs Low Price Wave Robustness
# CSV columns: outcome, sample (high_price/low_price), coef_s, se_s, pval,
#              nobs, n_clusters
# NOTE: ci_lo/ci_hi not in this CSV — SE shown instead.
# ══════════════════════════════════════════════════════════════════════════

def make_continuous():
    df = read_csv(os.path.join(OUT, "continuous_robustness.csv"))
    outcomes = [
        "delivery_facility", "delivery_skilled", "anc_4plus", "pnc_mother",
        "had_fever", "floor_finished", "dv_choked", "has_mobile",
        "man_employed", "owns_house", "dv_justify_any",
    ]
    rows = []
    for o in outcomes:
        hi = df.loc[(df["outcome"] == o) & (df["sample"] == "high_price")]
        lo = df.loc[(df["outcome"] == o) & (df["sample"] == "low_price")]
        bh = fmt_coef(hi["coef_s"].iloc[0], hi["pval"].iloc[0]) if not hi.empty else "---"
        ph = fmt_p(hi["pval"].iloc[0])                           if not hi.empty else "---"
        bl = fmt_coef(lo["coef_s"].iloc[0], lo["pval"].iloc[0]) if not lo.empty else "---"
        pl = fmt_p(lo["pval"].iloc[0])                           if not lo.empty else "---"
        rows.append(f"    {lbl(o)} & {bh} & {ph} & {bl} & {pl} \\\\")

    body = r"""\begin{threeparttable}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{High-price waves} & \multicolumn{2}{c}{Low-price waves} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Outcome & $\hat{\beta}$ (pp) & $p$ & $\hat{\beta}$ (pp) & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} High-price waves: 2007, 2014, 2018, 2024
(price $>$ \$6{,}500/MT). Low-price waves: 1992, 1996, 2002
(price $<$ \$2{,}300/MT). Dashes indicate the outcome was unavailable or
the subsample was too small. Coefficients in percentage points.
SE clustered at province level ($G = 9$), $df = 8$.
$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.
\end{tablenotes}
\end{threeparttable}
"""
    write_tex("tab_continuous.tex", body)


# ══════════════════════════════════════════════════════════════════════════
# TABLE A5 – Event Study: Wave-by-Wave Coefficients
# CSV columns: outcome, wave, beta, se, pval, ci_lo, ci_hi, n
# ══════════════════════════════════════════════════════════════════════════

def make_eventstudy():
    df = read_csv(os.path.join(OUT, "event_study.csv"))
    df["wave"] = df["wave"].astype(int)

    waves    = [1992, 1996, 2002, 2007, 2014, 2018, 2024]
    outcomes = ["delivery_facility", "delivery_skilled", "anc_4plus", "had_fever"]
    headers  = ["Facility delivery", "Skilled birth attend.", "ANC 4+ visits", "Child fever"]

    rows = []
    for w in waves:
        cells = [str(w)]
        for o in outcomes:
            sub = df.loc[(df["outcome"] == o) & (df["wave"] == w)]
            if sub.empty:
                cells.append("---")
            elif w == 2002:
                cells.append("$0$ (base)")
            else:
                b = sub["beta"].iloc[0]
                p = sub["pval"].iloc[0]
                cells.append(fmt_coef(b, p))
        rows.append("    " + " & ".join(cells) + " \\\\")

    header_row = "Wave & " + " & ".join(headers) + r" \\"

    body = r"""\begin{threeparttable}
\begin{tabular}{lcccc}
\toprule
""" + header_row + r"""
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} All coefficients in percentage points.
Base wave is 2002 ($\hat{\beta}_{2002} = 0$ by construction).
Province and wave fixed effects; SE clustered at province level
($G = 9$), $df = 8$.
$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.
\end{tablenotes}
\end{threeparttable}
"""
    write_tex("tab_eventstudy.tex", body)


# ══════════════════════════════════════════════════════════════════════════
# TABLE A6 – Placebo / Composition Stability (placebo_results.csv)
# CSV columns: outcome, label, coef_s, se_s, pval, ci_lo, ci_hi, nobs
# ══════════════════════════════════════════════════════════════════════════

def make_placebo():
    df = read_csv(os.path.join(OUT, "placebo_results.csv"))
    rows = []
    for _, r in df.iterrows():
        beta = fmt_coef(r["coef_s"], r["pval"])
        ci   = fmt_ci(r["ci_lo"], r["ci_hi"])
        pv   = fmt_p(r["pval"])
        n    = fmt_n(r["nobs"])
        rows.append(f"    {r['label']} & {beta} & {ci} & {pv} & {n} \\\\")

    body = r"""\begin{threeparttable}
\begin{tabular}{lcccc}
\toprule
Outcome & $\hat{\beta}$ & 95\% CI & $p$-value & $N$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} Outcomes that copper prices cannot causally affect.
None significant at any conventional level, supporting stability of
provincial population composition across the commodity cycle.
95\% confidence intervals computed from CR1S clustered SE,
$t_{df=8}$ critical value. SE clustered at province level ($G = 9$), $df = 8$.
\end{tablenotes}
\end{threeparttable}
"""
    write_tex("tab_placebo.tex", body)


# ══════════════════════════════════════════════════════════════════════════
# TABLE A6b – Extended Composition Stability (composition_stability.csv)
# CSV columns: outcome, label, coef_s, se_s, pval, ci_lo, ci_hi, nobs
# ══════════════════════════════════════════════════════════════════════════

def make_composition():
    df = read_csv(os.path.join(OUT, "composition_stability.csv"))
    rows = []
    for _, r in df.iterrows():
        beta = fmt_coef(r["coef_s"], r["pval"])
        ci   = fmt_ci(r["ci_lo"], r["ci_hi"])
        pv   = fmt_p(r["pval"])
        n    = fmt_n(r["nobs"])
        rows.append(f"    {r['label']} & {beta} & {ci} & {pv} & {n} \\\\")

    body = r"""\begin{threeparttable}
\begin{tabular}{lcccc}
\toprule
Outcome & $\hat{\beta}$ & 95\% CI & $p$-value & $N$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} Extended composition stability test including education,
marital status, and urbanisation share. None significant at conventional
levels. 95\% confidence intervals computed from CR1S clustered SE,
$t_{df=8}$ critical value. SE clustered at province level ($G = 9$), $df = 8$.
\end{tablenotes}
\end{threeparttable}
"""
    write_tex("tab_composition.tex", body)


# ══════════════════════════════════════════════════════════════════════════
# TABLE A7 – Long-Resident Robustness: Women (longresident_robustness.csv)
# CSV columns: outcome, label, beta_full, se_full, p_full, n_full,
#              beta_lr, se_lr, p_lr, n_lr
# CIs computed here from se_full / se_lr using T_CRIT_95 (df=8).
# ══════════════════════════════════════════════════════════════════════════

def make_longresident_women():
    df = read_csv(os.path.join(OUT, "longresident_robustness.csv"))
    rows = []
    for _, r in df.iterrows():
        lo_f, hi_f = ci_from_se(r["beta_full"], r["se_full"])
        lo_l, hi_l = ci_from_se(r["beta_lr"],   r["se_lr"])
        bf  = fmt_coef(r["beta_full"], r["p_full"])
        cif = fmt_ci(lo_f, hi_f)
        pf  = fmt_p(r["p_full"])
        nf  = fmt_n(r["n_full"])
        bl  = fmt_coef(r["beta_lr"], r["p_lr"])
        cil = fmt_ci(lo_l, hi_l)
        pl  = fmt_p(r["p_lr"])
        nl  = fmt_n(r["n_lr"])
        rows.append(
            f"    {r['label']} & {bf} & {cif} & {pf} & {nf} "
            f"& {bl} & {cil} & {pl} & {nl} \\\\"
        )

    body = r"""\begin{threeparttable}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{4}{c}{Full sample} & \multicolumn{4}{c}{Long-resident} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Outcome & $\hat{\beta}$ & 95\% CI & $p$ & $N$ &
          $\hat{\beta}$ & 95\% CI & $p$ & $N$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} Long-resident $=$ lived in current province
5 or more years (v104 $\geq 5$) or always (v104 $= 95$).
Visitors and those who did not state duration excluded.
Coefficients in percentage points. 95\% confidence intervals from
CR1S clustered SE, $t_{df=8}$ critical value.
SE clustered at province level ($G = 9$), $df = 8$.
\end{tablenotes}
\end{threeparttable}
"""
    write_tex("tab_longresident_women.tex", body)


# ══════════════════════════════════════════════════════════════════════════
# TABLE A8 – Long-Resident Robustness: Men
# CSV columns: outcome, label, scale, beta_full, se_full, p_full, n_full,
#              g_full, r2_full, beta_lr, se_lr, p_lr, n_lr, g_lr, r2_lr
# CIs computed here from se_full / se_lr using T_CRIT_95 (df=8).
# ══════════════════════════════════════════════════════════════════════════

def make_longresident_men():
    df = read_csv(os.path.join(OUT, "mens_longresident_robustness.csv"))
    rows = []
    for _, r in df.iterrows():
        lo_f, hi_f = ci_from_se(r["beta_full"], r["se_full"])
        lo_l, hi_l = ci_from_se(r["beta_lr"],   r["se_lr"])
        bf  = fmt_coef(r["beta_full"], r["p_full"])
        cif = fmt_ci(lo_f, hi_f)
        pf  = fmt_p(r["p_full"])
        nf  = fmt_n(r["n_full"])
        bl  = fmt_coef(r["beta_lr"], r["p_lr"])
        cil = fmt_ci(lo_l, hi_l)
        pl  = fmt_p(r["p_lr"])
        nl  = fmt_n(r["n_lr"])
        rows.append(
            f"    {r['label']} & {bf} & {cif} & {pf} & {nf} "
            f"& {bl} & {cil} & {pl} & {nl} \\\\"
        )

    body = r"""\begin{threeparttable}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{4}{c}{Full sample} & \multicolumn{4}{c}{Long-resident} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Outcome & $\hat{\beta}$ & 95\% CI & $p$ & $N$ &
          $\hat{\beta}$ & 95\% CI & $p$ & $N$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes:} Long-resident $=$ mv104 $\geq 5$ years or mv104 $= 95$
(always). 76.7 percent of men in the pooled MR sample meet this criterion.
Employment gains strengthen in the long-resident subsample; condom use
attenuates to zero, suggesting the full-sample result partly reflects
in-migration of higher-risk men.
95\% confidence intervals from CR1S clustered SE, $t_{df=8}$ critical value.
SE clustered at province level ($G = 9$), $df = 8$.
$^{**}p<0.05$, $^{*}p<0.10$.
\end{tablenotes}
\end{threeparttable}
"""
    write_tex("tab_longresident_men.tex", body)


# ══════════════════════════════════════════════════════════════════════════
# TABLES A9–A14 – Full Results by module (ind_results.csv, spec == 'main')
# CSV columns: table, outcome, label, spec, scale, coef_s, se_s, pval,
#              ci_lo, ci_hi, nobs
# ══════════════════════════════════════════════════════════════════════════

MODULE_MAP = {
    "Table 1: Wealth and Assets": (
        "tab_full_wealth.tex",
        r"Controls: household size, urban residence.",
    ),
    "Table 2: Child Health": (
        "tab_full_child.tex",
        r"Controls: child age, child sex, birth order, urban residence. "
        r"Anthropometric outcomes available from 2007 onwards only. "
        r"Mortality outcomes are omitted (zero within-group variation after demeaning).",
    ),
    "Table 3: Maternal and Reproductive Health": (
        "tab_full_maternal.tex",
        r"Controls: woman's age, urban residence, married/in union, education level.",
    ),
    "Table 4: Women's Empowerment": (
        "tab_full_empowerment.tex",
        r"Controls: woman's age, urban residence, married/in union.",
    ),
    "Table 5: Domestic Violence": (
        "tab_full_dv.tex",
        r"Controls: woman's age, urban residence, secondary education.",
    ),
    "Table 6: Men's Outcomes": (
        "tab_full_men.tex",
        r"Controls: man's age, urban residence. MR files begin in 1996. "
        r"See Table~\ref{tab:longresident_men} for long-resident robustness check.",
    ),
}

SKIP_OUTCOMES = {"u5_dead", "infant_dead", "neonatal_dead", "can_go_health"}

def make_full_tables():
    df   = read_csv(os.path.join(OUT, "ind_results.csv"))
    main = df.loc[df["spec"] == "main"].copy()
    main = main.loc[~main["outcome"].isin(SKIP_OUTCOMES)]

    for tname, (fname, note) in MODULE_MAP.items():
        sub  = main.loc[main["table"] == tname]
        rows = []
        for _, r in sub.iterrows():
            beta = fmt_coef(r["coef_s"], r["pval"])
            ci   = fmt_ci(r["ci_lo"], r["ci_hi"])
            pv   = fmt_p(r["pval"])
            n    = fmt_n(r["nobs"])
            rows.append(f"    {r['label']} & {beta} & {ci} & {pv} & {n} \\\\")

        body = (
            r"""\begin{threeparttable}
\begin{tabular}{lcccc}
\toprule
Outcome & $\hat{\beta}$ (pp) & 95\% CI & $p$-value & $N$ \\
\midrule
"""
            + "\n".join(rows)
            + f"""
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Notes:}} {note}
95\\% confidence intervals from CR1S clustered SE, $t_{{df=8}}$ critical value.
$^{{***}}p<0.01$, $^{{**}}p<0.05$, $^{{*}}p<0.10$.
\\end{{tablenotes}}
\\end{{threeparttable}}
"""
        )
        write_tex(fname, body)


# ── run all ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Exporting appendix table fragments to OUTPUT/tables/\n")
    make_urban_rural()
    make_privatisation()
    make_bootstrap()
    make_continuous()
    make_eventstudy()
    make_placebo()
    make_composition()
    make_longresident_women()
    make_longresident_men()
    make_full_tables()
    n = len(os.listdir(TABLES))
    print(f"\nDone — {n} fragments written to OUTPUT/tables/")
    print("Next step: pdflatex zambia_appendix.tex")