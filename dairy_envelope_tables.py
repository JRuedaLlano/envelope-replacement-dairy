"""
dairy_envelope_tables.py

Utility functions to build and export the LaTeX tables used in Section 7 of
*A renewal–reward envelope approach to optimal replacement* (Rueda-Llano, 2025).

This script:
    - Loads precomputed envelope-rule results from data/*.csv
    - Formats them into the LaTeX table structure required by the manuscript
    - Writes:

        tables/table_section7_percow_baseline.tex
        tables/table_section7_percow_weaker.tex
        tables/table_section7_summary_baseline.tex
        tables/table_section7_summary_weaker.tex
        tables/table_section7_scenarios_compare.tex

It contains **no simulation logic**; it only reshapes and presents results
for direct inclusion via \input{...} in the LaTeX source.

Author: José Rueda-Llano (2025)

"""


# --- Standard library imports ---
from pathlib import Path

# --- Third-party imports ---
import pandas as pd



# ----------- CONFIG ----------- #
# Summary CSVs produced by dairy_envelope_application.py
BASELINE_CSV = Path("data/S0_baseline/section7_summary_baseline.csv")
WEAKER_CSV   = Path("data/S1_weaker/section7_summary_weaker.csv")

# Directory where all LaTeX tables will be written
TABLES_DIR = Path("tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
    

# Columns we expect from your current pipeline
REQUIRED = [
    "cow_id", "policy_branch",
    "t_star_cow_only", "t_hat_switch",
    "lambda_", "Lambda_star",
    "in_sample_R2"
]

# Optional (used if present)
OPTIONAL = [
    "t_env", "A_env",        # envelope age & value
    "repaired",              # whether constrained refit applied
]

# Pretty names for LaTeX headers
PRETTY = {
    "cow_id": "Cow ID",
    "policy_branch": "Decision",
    "t_star_cow_only": "$t_j^*$ (days)",
    "t_hat_switch": "$\\hat t_j$ (days)",
    "lambda_": "$\\lambda_j^*$ (€/day)",
    "Lambda_star": "$\\Lambda_j^*$ (€/day)",
    "in_sample_R2": "$R^2$",
    "t_env": "$t^{\\text{env}}$ (days)",
    "A_env": "$A^{\\text{env}}$ (€/day)",
    "repaired": "Repaired"
}

def _load(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    # soft-check: keep only columns that exist
    cols = [c for c in REQUIRED + OPTIONAL if c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")
    return df[cols].copy()

def _format_percow(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-cow table with consistent rounding and clean decision labels."""
    out = df.copy()
    # Cosmetic: Decision labels
    out["policy_branch"] = out["policy_branch"].replace({
        "COW_ONLY": "Cow-only",
        "SWITCH": "Switch"
    })
    # Round numerics for display (do not mutate original precision elsewhere)
    round2 = ["lambda_", "Lambda_star", "A_env"]
    round0 = ["t_star_cow_only", "t_hat_switch", "t_env"]
    for c in round2:
        if c in out.columns:
            out[c] = out[c].astype(float).round(2)
    for c in round0:
        if c in out.columns:
            out[c] = out[c].astype(float).round(0).astype("Int64")
    if "in_sample_R2" in out.columns:
        out["in_sample_R2"] = out["in_sample_R2"].astype(float).round(3)
    if "repaired" in out.columns:
        out["repaired"] = out["repaired"].map({True:"Yes", False:"No"}).fillna("")
    # Column order
    cols = [c for c in [
        "cow_id", "policy_branch",
        "t_star_cow_only", "t_hat_switch",
        "lambda_", "Lambda_star",
        "t_env", "A_env",
        "in_sample_R2", "repaired"
    ] if c in out.columns]
    return out[cols]

def _summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Scenario-level stats used in Section 7 text."""
    # Basic shares
    n = len(df)
    share_switch = (df["policy_branch"] == "SWITCH").mean()
    share_cow    = 1.0 - share_switch

    # Differences used in the narrative (if both cols exist)
    delta_t = (df["t_hat_switch"] - df["t_star_cow_only"]).abs()
    delta_L = (df["Lambda_star"] - df["lambda_"]).abs()

    def mean_sd(x): return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"

    rows = [
        ("N (cows)", n),
        ("Share Switch", f"{100*share_switch:.1f}\\%"),
        ("Share Cow-only", f"{100*share_cow:.1f}\\%"),
        ("$t_j^*$ (days): mean ± sd", mean_sd(df["t_star_cow_only"])),
        ("$\\hat t_j$ (days): mean ± sd", mean_sd(df["t_hat_switch"])),
        ("$|\\hat t_j - t_j^*|$ (days): mean ± sd", mean_sd(delta_t)),
        ("$\\lambda_j^*$ (€/day): mean ± sd", mean_sd(df["lambda_"])),
        ("$\\Lambda_j^*$ (€/day): mean ± sd", mean_sd(df["Lambda_star"])),
        ("$|\\Lambda_j^* - \\lambda_j^*|$ (€/day): mean ± sd", mean_sd(delta_L)),
        ("In-sample $R^2$: mean ± sd", mean_sd(df["in_sample_R2"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])

def _write_tex(df: pd.DataFrame, path: Path, caption: str, label: str):
    # Pretty headers
    header_map = {c: PRETTY.get(c, c) for c in df.columns}
    df = df.rename(columns=header_map)
    tex = df.to_latex(index=False, escape=False)
    # Wrap with caption+label so \listoftables works nicely
    wrapped = (
        "\\begin{table}[t]\n\\centering\n"
        + tex
        + f"\\caption{{{caption}}}\n"
        + f"\\label{{{label}}}\n"
        + "\\end{table}\n"
    )
    path.write_text(wrapped, encoding="utf-8")

def _write_compare_tex(baseline_stats: pd.DataFrame, weaker_stats: pd.DataFrame, out_path: Path):
    """Two-column comparison table for baseline vs weaker."""
    b = baseline_stats.set_index("Metric")
    w = weaker_stats.set_index("Metric")
    # Keep shared metrics in the same order as baseline
    metrics = b.index.tolist()
    cmp_df = pd.DataFrame({
        "Metric": metrics,
        "Baseline": [b.loc[m, "Value"] if m in b.index else "" for m in metrics],
        "Weaker heifer": [w.loc[m, "Value"] if m in w.index else "" for m in metrics],
    })
    tex = cmp_df.to_latex(index=False, escape=False)
    wrapped = (
        "\\begin{table}[t]\n\\centering\n"
        + tex
        + "\\caption{Baseline vs weaker heifer benchmark: summary comparison for Section~7.}\n"
        + "\\label{tab:sec7-scenarios-compare}\n"
        + "\\end{table}\n"
    )
    out_path.write_text(wrapped, encoding="utf-8")

def main():
    # --- Load
    df_b = _load(BASELINE_CSV)
    df_w = _load(WEAKER_CSV)

    # --- Per-cow tables
    _write_tex(
        _format_percow(df_b),
        TABLES_DIR / "table_section7_percow_baseline.tex",
        "Per-cow results under the baseline heifer benchmark.",
        "tab:sec7-percow-baseline"
    )
    _write_tex(
        _format_percow(df_w),
        TABLES_DIR / "table_section7_percow_weaker.tex",
        "Per-cow results under the weaker heifer benchmark.",
        "tab:sec7-percow-weaker"
    )

    # --- Scenario summaries
    stats_b = _summary_stats(df_b)
    stats_w = _summary_stats(df_w)
    _write_tex(
        stats_b,
        TABLES_DIR / "table_section7_summary_baseline.tex",
        "Summary statistics under the baseline heifer benchmark (Section~7).",
        "tab:sec7-summary-baseline"
    )
    _write_tex(
        stats_w,
        TABLES_DIR / "table_section7_summary_weaker.tex",
        "Summary statistics under the weaker heifer benchmark (Section~7).",
        "tab:sec7-summary-weaker"
    )

    # --- Side-by-side comparison
    _write_compare_tex(stats_b, stats_w, TABLES_DIR / "table_section7_scenarios_compare.tex")


if __name__ == "__main__":
    main()
