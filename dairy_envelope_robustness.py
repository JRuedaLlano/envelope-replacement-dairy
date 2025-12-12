"""
dairy_envelope_robustness.py

Optional robustness experiments for the envelope replacement rule.

This script is not required to reproduce the main tables in Section 7 of
*A renewal–reward envelope approach to optimal replacement* (Rueda-Llano, 2025),
but can be used to explore additional scenarios beyond the weaker heifer case
reported in the manuscript.

Possible uses include:
    - Varying noise levels in cumulative-profit trajectories
    - Perturbing economic parameters (prices, costs) around the baseline
    - Checking how sensitive envelope decisions are to misspecification
      of the surrogate fit or reward components

Outputs:
    - Robustness CSV and LaTeX tables under data/<scenario_name>/
    - Diagnostic figures (PDF/PNG) under data/<scenario_name>/


These outputs may be provided as supplementary material but are not referenced
in the main LaTeX manuscript.

Author: José Rueda-Llano (2025)

"""

from __future__ import annotations

# Standard library imports
import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Third-party imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local application imports
import dairy_envelope_application as s6  # Located in the same directory

# Configure matplotlib for headless environments
matplotlib.use("Agg")




# Short aliases
fit_cubic      = s6.fit_cubic_anchored
eval_cubic     = s6.eval_cubic
envelope_poly  = s6.envelope_from_poly
profit_from_m  = s6.cumulative_profit_from_milk
BL0            = s6.BL   # Baseline economics dataclass
OUTPUT_ROOT    = s6.OUTPUT_DIR


# -------------------------- Perturbations --------------------------

@dataclass
class Perturbation:
    """Base class; implement .apply(...) in subclasses."""
    name: str

    def apply(self, *, t: np.ndarray, milk: np.ndarray,
              bl: s6.Baseline,
              T_h_star: float, P_h_Tstar: float,
              rng: np.random.Generator) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class MilkNoise(Perturbation):
    """Add iid N(0, sigma) to milk, clip at 0."""
    sigma: float = 0.6
    def apply(self, *, t, milk, bl, T_h_star, P_h_Tstar, rng):
        m = np.maximum(0.0, milk + rng.normal(0.0, self.sigma, size=milk.shape))
        return dict(t=t, milk=m, bl=bl, T_h_star=T_h_star, P_h_Tstar=P_h_Tstar)


@dataclass
class TailTruncation(Perturbation):
    """Cut the last K days (if any)."""
    cut_days: int = 60
    def apply(self, *, t, milk, bl, T_h_star, P_h_Tstar, rng):
        if self.cut_days <= 0 or self.cut_days >= len(t):
            return dict(t=t, milk=milk, bl=bl, T_h_star=T_h_star, P_h_Tstar=P_h_Tstar)
        keep = len(t) - self.cut_days
        return dict(t=t[:keep], milk=milk[:keep], bl=bl, T_h_star=T_h_star, P_h_Tstar=P_h_Tstar)


@dataclass
class MarginScale(Perturbation):
    """Scale the milk margin (pm - cf) by (1+eps), holding fixed costs."""
    eps: float = +0.05
    def apply(self, *, t, milk, bl, T_h_star, P_h_Tstar, rng):
        # build a new Baseline with scaled milk price to get the desired margin
        cur_margin = bl.milk_margin
        new_margin = (1.0 + self.eps) * cur_margin
        # keep var_cost fixed; adjust milk price so that price - var_cost = new_margin
        new_price = bl.var_cost + new_margin
        bl2 = s6.Baseline(
            milk_price=new_price, var_cost=bl.var_cost,
            maint_cost=bl.maint_cost, stall_cost=bl.stall_cost,
            rearing_cost=bl.rearing_cost, salvage_value=bl.salvage_value
        )
        return dict(t=t, milk=milk, bl=bl2, T_h_star=T_h_star, P_h_Tstar=P_h_Tstar)


@dataclass
class HeiferShift(Perturbation):
    """Shift the benchmark: T_h* and P_h(T_h*) both by their multipliers."""
    t_mult: float = 1.00
    P_mult: float = 1.00
    def apply(self, *, t, milk, bl, T_h_star, P_h_Tstar, rng):
        return dict(t=t, milk=milk, bl=bl,
                    T_h_star=self.t_mult * T_h_star,
                    P_h_Tstar=self.P_mult * P_h_Tstar)
    


# ------------------Diagnostic figures-----------------------------------

def _ordered_perturbations(df: pd.DataFrame) -> list[str]:
    """
    Preserve the creation order of perturbations as they appeared during
    iteration (drop_duplicates keeps first occurrence).
    """
    return list(df["perturbation"].drop_duplicates())

def save_diagnostic_figures(df: pd.DataFrame, out_dir: str, suffix: str = "") -> None:
    """
    Make two small diagnostics:
      (1) Boxplots of |Δt_env| and |ΔA_env| by perturbation.
      (2) Bar chart of branch-flip rates by perturbation.
    Saved as sec7_diagnostics.pdf and sec7_fliprates.pdf in out_dir.
    """
    if df.empty:
        return

    order = _ordered_perturbations(df)
    labels = [p.replace("_", r"\_") for p in order]  # TeX-safe in case you paste

    # ---------- Figure 1: Boxplots (|Δt_env| and |ΔA_env|) ----------
    data_dt = [np.abs(df.loc[df["perturbation"] == p, "delta_t_env"].to_numpy())
               for p in order]
    data_dA = [np.abs(df.loc[df["perturbation"] == p, "delta_A_env"].to_numpy())
               for p in order]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=200, constrained_layout=True)
    bp1 = axes[0].boxplot(data_dt, showfliers=False, patch_artist=True)
    axes[0].set_title(r"$|\Delta t_{\mathrm{env}}|$ (days)")
    axes[0].set_xticks(range(1, len(order)+1), labels, rotation=25, ha="right")
    axes[0].grid(True, axis="y", ls=":", alpha=0.4)

    bp2 = axes[1].boxplot(data_dA, showfliers=False, patch_artist=True)
    axes[1].set_title(r"$|\Delta A_{\mathrm{env}}|$ (€/day)")
    axes[1].set_xticks(range(1, len(order)+1), labels, rotation=25, ha="right")
    axes[1].grid(True, axis="y", ls=":", alpha=0.4)

    # Light fill to improve readability
    for bp in (bp1, bp2):
        for box in bp["boxes"]:
            box.set_facecolor("#e8eef7")
            box.set_edgecolor("#3b6ea8")
        for med in bp["medians"]:
            med.set_color("#b2182b")
            med.set_linewidth(1.6)

    fig.savefig(os.path.join(out_dir, f"sec7_diagnostics_{suffix}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"sec7_diagnostics_{suffix}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ---------- Figure 2: Flip rates ----------
    flip_rate = (df.groupby("perturbation", as_index=False)["branch_flip"]
                   .mean()
                   .set_index("perturbation")
                   .reindex(order)["branch_flip"]
                   .to_numpy())

    fig2, ax2 = plt.subplots(figsize=(10.0, 3.4), dpi=200)
    ax2.bar(range(len(order)), flip_rate)
    ax2.set_xticks(range(len(order)), labels, rotation=25, ha="right")
    ax2.set_ylim(0.0, max(0.01, float(flip_rate.max())*1.15))
    ax2.set_ylabel("Flip rate")
    ax2.set_title("Branch flips (cow-only ↔ switch)")
    ax2.grid(True, axis="y", ls=":", alpha=0.4)

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"sec7_fliprates_{suffix}.pdf"), bbox_inches="tight")
    fig2.savefig(os.path.join(out_dir, f"sec7_fliprates_{suffix}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig2)


# ----------------------- Core recomputation ------------------------

def _indices_from_series(t: np.ndarray, milk: np.ndarray,
                         bl: s6.Baseline,
                         T_h_star: float, P_h_Tstar: float) -> Dict[str, float]:
    """Cubic fit (anchored) + envelope — exactly like Section 7."""
    P = profit_from_m(milk, bl)
    coef = fit_cubic(t, P)
    env  = envelope_poly(coef, max(1.0, t.min()+1.0), t.max(), T_h_star, P_h_Tstar)
    return dict(
        t_star=env["t_star"], lambda_=env["lambda_"],
        t_hat=env["t_hat"],   Lambda=env["Lambda"],
        t_env=env["t_env"],   A_env=env["A_env"],
        branch=env["branch"],
        used_boundary_cow=bool(env["used_boundary_cow"]),
        used_boundary_switch=bool(env["used_boundary_switch"]),
    )


def _load_baseline_daily(folder: str) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Read the per-cow daily CSVs produced by Section 7 and return
    [(cow_id, t, milk), ...] in creation order.
    """
    files = sorted(glob.glob(os.path.join(folder, "[0-9][0-9]_cow_*_daily.csv")))
    out: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for fp in files:
        df = pd.read_csv(fp)
        df = df.sort_values("day")
        # day 0 row exists; drop it to keep t=1..T, just like the fit
        df = df[df["day"] > 0].copy()
        cid = int(df["cow_id"].iloc[0])
        t   = df["day"].to_numpy(float)
        milk= df["milk_kg"].to_numpy(float)
        out.append((cid, t, milk))
    return out


# ----------------------- Public driver -----------------------------

def run_section7(*,
                 scenario_name: str,
                 T_h_star: float,
                 P_h_Tstar: float,
                 replicates: int = 200,
                 seed: int = 7771) -> pd.DataFrame:
    """
    For every cow in ./data/<scenario_name>, recompute envelope decisions under
    a battery of small perturbations.  Write:
      - sec7_robustness.csv
      - sec7_summary.tex (LaTeX table of medians / IQRs by perturbation)
      - sec7_branch_flips.tex (LaTeX table of branch-flip counts)

    Returns the long-form DataFrame.
    """
    scen_dir = os.path.join(OUTPUT_ROOT, scenario_name)
    assert os.path.isdir(scen_dir), f"Scenario folder not found: {scen_dir}"

    # Baseline envelopes for later deltas (compute from the same files)
    cows = _load_baseline_daily(scen_dir)
    rng_master = np.random.default_rng(seed)

    # Define the perturbation set
    perts: List[Perturbation] = [
        MilkNoise(name="milk_noise_sd0.6", sigma=0.6),
        MilkNoise(name="milk_noise_sd1.0", sigma=1.0),
        TailTruncation(name="truncate_60d", cut_days=60),
        MarginScale(name="margin_plus5", eps=+0.05),
        MarginScale(name="margin_minus5", eps=-0.05),
        HeiferShift(name="heifer_t+10%_P-10%", t_mult=1.10, P_mult=0.90),
        HeiferShift(name="heifer_t-10%_P+10%", t_mult=0.90, P_mult=1.10),
    ]

    rows: List[Dict[str, Any]] = []

    # Pre-compute baseline envelopes
    base_env: Dict[int, Dict[str, float]] = {}
    for cid, t, milk in cows:
        base_env[cid] = _indices_from_series(t, milk, BL0, T_h_star, P_h_Tstar)

    # Run perturbations
    for cid, t, milk in cows:
        for pert in perts:
            for rep in range(replicates):
                rng = np.random.default_rng(rng_master.integers(0, 2**31-1))
                spec = pert.apply(t=t, milk=milk, bl=BL0,
                                  T_h_star=T_h_star, P_h_Tstar=P_h_Tstar, rng=rng)
                # skip degenerate truncations
                if len(spec["t"]) < 5:
                    continue

                env_p = _indices_from_series(spec["t"], spec["milk"], spec["bl"],
                                             spec["T_h_star"], spec["P_h_Tstar"])

                env_0 = base_env[cid]
                rows.append(dict(
                    scenario=scenario_name,
                    cow_id=cid,
                    perturbation=pert.name,
                    replicate=rep+1,
                    # perturbed envelope
                    t_env=env_p["t_env"], A_env=env_p["A_env"], branch=env_p["branch"],
                    t_star=env_p["t_star"], lambda_=env_p["lambda_"],
                    t_hat=env_p["t_hat"],  Lambda=env_p["Lambda"],
                    # deltas vs baseline envelope
                    delta_t_env=float(env_p["t_env"] - env_0["t_env"]),
                    delta_A_env=float(env_p["A_env"] - env_0["A_env"]),
                    branch_flip=(env_p["branch"] != env_0["branch"]),
                    # keep whether any boundary was used (could explain odd ties)
                    used_boundary_cow=env_p["used_boundary_cow"],
                    used_boundary_switch=env_p["used_boundary_switch"],
                ))

    df = pd.DataFrame(rows)

    # --- Add scenario-aware suffix ---
    scen_lower = scenario_name.lower()
    scen_suffix = "weaker" if "weaker" in scen_lower else "baseline"
    
    # Save long data
    out_csv = os.path.join(scen_dir, f"sec7_robustness_{scen_suffix}.csv")
    df.to_csv(out_csv, index=False)

    # Summaries by perturbation
    summ = (df
            .groupby("perturbation", as_index=False)
            .agg(n=("cow_id", "size"),
                 med_abs_dt=("delta_t_env", lambda x: float(np.median(np.abs(x)))),
                 iqr_abs_dt=("delta_t_env", lambda x: float(np.percentile(np.abs(x), 75) - np.percentile(np.abs(x), 25))),
                 med_abs_dA=("delta_A_env", lambda x: float(np.median(np.abs(x)))),
                 iqr_abs_dA=("delta_A_env", lambda x: float(np.percentile(np.abs(x), 75) - np.percentile(np.abs(x), 25))),
                 flip_rate=("branch_flip", "mean")))

    # Write LaTeX tables (tabular only; captions/labels go in the .tex manuscript)
    tex_summary = os.path.join(scen_dir, f"sec7_summary_{scen_suffix}.tex")
    with open(tex_summary, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "Perturbation & $n$ & median $|\\Delta t_{\\mathrm{env}}|$ & "
            "IQR $|\\Delta t_{\\mathrm{env}}|$ & median $|\\Delta A_{\\mathrm{env}}|$ & flip rate \\\\\n"
        )
        f.write("\\midrule\n")

        for _, r in summ.iterrows():
            name = str(r["perturbation"]).replace("_", "\\_").replace("%", "\\%")
            f.write(
                f"{name} & {int(r['n'])} & "
                f"{r['med_abs_dt']:.1f} & {r['iqr_abs_dt']:.1f} & "
                f"{r['med_abs_dA']:.3f} & {r['flip_rate']:.3f}\\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


    # Branch flips per cow (useful diagnostic)
    flips = (df.groupby(["perturbation", "cow_id"], as_index=False)
               .agg(flip_any=("branch_flip", "max")))
    flips_pivot = (flips.pivot_table(index="cow_id", columns="perturbation",
                                     values="flip_any", fill_value=False)
                        .astype(int)
                        .sort_index(axis=1))
    tex_flips = os.path.join(scen_dir, f"sec7_branch_flips_{scen_suffix}.tex")
    flips_pivot.to_latex(tex_flips, index=True, escape=False)

    # Diagnostic figures
    save_diagnostic_figures(df, scen_dir, suffix=scen_suffix)

    print(f"[Section 7 — {scenario_name}] wrote:")
    print(f"  - {out_csv}")
    print(f"  - {tex_summary}")
    print(f"  - {tex_flips}")
    print(f"  - sec7_diagnostics_{scen_suffix}.(pdf|png)")
    print(f"  - sec7_fliprates_{scen_suffix}.(pdf|png)")
    
    return df


# ---------------------- CLI usage ----------------------

if __name__ == "__main__":
    # Match your Section 7 scenarios; adjust if you use different names
    run_section7(scenario_name="S0_baseline", T_h_star=400.0, P_h_Tstar=600.0)
    run_section7(scenario_name="S1_weaker",   T_h_star=450.0, P_h_Tstar=450.0)
