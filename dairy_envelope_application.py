"""
dairy_envelope_application.py

Main simulation and envelope-rule analysis used in Section 7 of the manuscript
*A renewal–reward envelope approach to optimal replacement* (Rueda-Llano, 2025).

This script:
    - Simulates synthetic cumulative-profit trajectories for 10 cows, P_j(t)
    - Simulates a baseline heifer cumulative-profit curve P_h(t)
    - Constructs a weaker heifer benchmark for sensitivity analysis
    - Fits cubic surrogate functions to all cumulative-profit curves
    - Computes:
        * Cow-only optimal age t_j^*
        * Switch age \hat{t}_j relative to the heifer benchmark
        * Average-profit indices \lambda_j^* and \Lambda_j^*
        * Envelope ages and envelope average profits
    - Exports:
        * Per-cow daily CSVs
        * Scenario summary CSVs
        * Figures for the cow trajectories and decision ages

Typical outputs:
    data/S0_baseline/section7_summary_baseline.csv
    data/S1_weaker/section7_summary_weaker.csv
    data/S0_baseline/<idx>_cow_<id>_daily.csv
    data/S1_weaker/<idx>_cow_<id>_daily.csv
    data/S0_baseline/<idx>_fig_S0_baseline_cow_<id>.pdf/.png
    data/S1_weaker/<idx>_fig_S1_weaker_cow_<id>.pdf/.png

These outputs are consumed by:
    - dairy_envelope_tables.py for LaTeX tables (using the summary CSVs)
    - dairy_envelope_robustness.py for robustness experiments.

Author: José Rueda-Llano (2025)
"""


# --- Standard library imports ---
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os

# --- Third-party imports ---
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import offset_copy

# Configure matplotlib for headless environments
matplotlib.use("Agg")


# ------------------ I/O and reproducibility ------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# All scenario data and figures live under ./data/<scenario_name>/
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


MASTER_SEED = 17001
np.random.seed(MASTER_SEED)


# ------------------ Economics (exact values requested) ------------------

MILK_PRICE_KG  = 0.50
FEED_VAR_COST  = 0.23
MAINT_COST     = 3.0
STALL_FIX_COST = 2.0
REARING_COST   = 2700.0
SALVAGE_PRICE  = 1200.0


@dataclass
class Baseline:
    """
    Economic parameters for profit accounting.

    Attributes
    ----------
    milk_price : float
        Price per kg of milk.
    var_cost : float
        Variable feed cost per kg of milk.
    maint_cost : float
        Daily maintenance cost (€/day).
    stall_cost : float
        Daily fixed stall cost (€/day).
    rearing_cost : float
        One-time rearing cost (€, applied at day 0 as a negative offset).
    salvage_value : float
        One-time salvage value (€, applied at day 0 as a positive offset).
    """
    milk_price: float = MILK_PRICE_KG
    var_cost:   float = FEED_VAR_COST
    maint_cost: float = MAINT_COST
    stall_cost: float = STALL_FIX_COST
    rearing_cost: float = REARING_COST
    salvage_value: float = SALVAGE_PRICE

    @property
    def milk_margin(self) -> float:
        """Return milk margin = milk price – variable feed cost (€/kg)."""
        return self.milk_price - self.var_cost


BL = Baseline()


# -------------------Plotting-----------------------

def plot_intercept_marker(ax, y, color, marker, size=28):
    """
    Draw a marker at (0, y) without shifting in x; use a white halo so it
    reads on top of the y-axis spine instead of nudging it right.
    """
    halo = [pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()]
    ax.scatter([0.0], [y],
               s=size, marker=marker, color=color,
               zorder=7, clip_on=False, edgecolors="none",
               path_effects=halo)

def draw_origin_label(ax, text="0", dx_pt=-7, dy_pt=-7):
    """
    Place a small '0' just *below-left* (south-west) of the (0,0) origin.
    Uses a display-space offset so it’s robust to DPI and spine width.
    """
    # offset_copy moves the text by a fixed number of points from data (0,0)
    txt_trans = offset_copy(ax.transData, fig=ax.figure,
                            x=dx_pt, y=dy_pt, units='points')
    ax.text(0.0, 0.0, text,
            transform=txt_trans, ha="right", va="top",
            fontsize=8, color="black", zorder=8)

def _render_cow_plot(ax, *,
                     cow_id: int,
                     t, P, Pfit,
                     t_star, t_hat,
                     P0_data, P0_fit,
                     repaired: bool,
                     panel: bool):
    """Draw one cow's P(t), cubic fit, and decision ages on `ax`."""
    # series
    ax.plot(t, P,    lw=1.5, color="#c98a1b", label="Observed $P_j(t)$", zorder=3)
    ax.plot(t, Pfit, lw=1.8, color="#2b6cb0", label="Cubic fit $\\tilde{P}_j(t)$", zorder=4)

    # axes crossing at (0, 0) — use the allowed keyword 'data'
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # hide automatic "0" tick labels on both axes
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: '' if np.isclose(v, 0) else f'{int(v):d}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: '' if np.isclose(v, 0) else f'{int(v):d}'))

    # Small “0” in the south-west quadrant of the origin
    draw_origin_label(ax, text="0", dx_pt=-7, dy_pt=-7)

    # Intercept markers at x=0 (no x-offset; white halo keeps them legible)
    plot_intercept_marker(ax, P0_data, color="#e7a53f", marker="o", size=28)
    plot_intercept_marker(ax, P0_fit,  color="#2b6cb0", marker="x", size=34)

    # decision lines
    eps_close = 8.0
    col_tstar, col_that = "black", "#b2182b"
    ls_tstar = (0, (4, 3))
    ls_that  = (0, (2, 2, 1, 2))

    def top_marker(ax_, x_, color_, marker_):
        ymin_, ymax_ = ax_.get_ylim()
        y_pos = ymax_ - 0.03 * (ymax_ - ymin_)
        ax_.scatter([x_], [y_pos], marker=marker_, s=24,
                    color=color_, edgecolor="none", zorder=6, clip_on=False)

    if abs(t_star - t_hat) <= eps_close:
        x0, x1 = sorted([t_star, t_hat])
        ax.axvspan(x0, x1 if x1 > x0 else x0 + 0.3, color="#b2182b", alpha=0.12, zorder=1)
        xmid = 0.5 * (x0 + x1)
        ax.axvline(xmid, color=col_tstar, lw=1.1, ls=ls_tstar, alpha=0.9, zorder=5)
        top_marker(ax, xmid, col_tstar, "^")
        legend_lines = [("$(t_j^* = \\hat t_j)$", col_tstar, ls_tstar)]
    else:
        ax.axvline(t_star, color=col_tstar, lw=1.1, ls=ls_tstar, alpha=0.9, zorder=5)
        ax.axvline(t_hat,  color=col_that,  lw=1.1, ls=ls_that,  alpha=0.9, zorder=5)
        top_marker(ax, t_star, col_tstar, "^")
        top_marker(ax, t_hat,  col_that,  "v")
        legend_lines = [("$t_j^*$", col_tstar, ls_tstar),
                        ("$\\hat t_j$", col_that,  ls_that)]

    # baseline grid and zero line
    ax.axhline(0.0, color="k", lw=0.7, alpha=0.45, zorder=2)

    if panel:
        ttl = " (repaired)" if repaired else ""
        ax.set_title(f"Cow {cow_id}{ttl}", fontsize=7.5, pad=1.0)
        ax.grid(True, ls=":", lw=0.4, alpha=0.25)
        ax.tick_params(labelsize=7, pad=1)
        ax.set_xlabel(""); ax.set_ylabel("")
    else:
        title = f"Cow {cow_id} — cumulative profit & cubic fit"
        if repaired:
            title += " (repaired)"
        ax.set_title(title)
        ax.set_xlabel("Day"); ax.set_ylabel("Cumulative profit (€)")
        ax.grid(True, ls=":", alpha=0.35)
        custom = [
            Line2D([0], [0], color="#e7a53f", lw=1.6, label="Observed $P_j(t)$"),
            Line2D([0], [0], color="#2b6cb0", lw=1.8, label="Cubic fit $\\tilde{P}_j(t)$"),
        ]
        for label, col, lstyle in legend_lines:
            custom.append(Line2D([0], [0], color=col, lw=1.1, ls=lstyle, label=label))
        ax.legend(handles=custom, fontsize=7.5, framealpha=0.85,
                  loc="upper left", bbox_to_anchor=(0.065, 0.98),
                  borderaxespad=0.2, handlelength=1.2,
                  handletextpad=0.35, labelspacing=0.28)



# ------------------ Lactation and milk generation ------------------

def wood_curve(t: np.ndarray, peak: float, b: float = 0.20, c: float = 0.008) -> np.ndarray:
    """
    Wood lactation curve evaluated on days t.

    Parameters
    ----------
    t : np.ndarray
        Day indices within a lactation (1..DIM).
    peak : float
        Intended milk peak (kg/day) for this lactation.
    b, c : float
        Wood curve parameters; t_peak ≈ b/c by construction.

    Returns
    -------
    np.ndarray
        Daily milk kg values for this lactation (nonnegative).
    """
    tpk = max(1e-6, b / c)
    a = peak / ((tpk**b) * np.exp(-c * tpk))
    q = a * (np.maximum(t, 1e-6)**b) * np.exp(-c * t)
    q[q < 0] = 0.0
    return q


def unimodal_parity_peaks(n_lact: int,
                          base_peak: float = 29.0,
                          max_peak: float = 47.0,
                          peak_at_parity: int = 3,
                          min_peak: float = 27.0,
                          jitter: float = 0.03,
                          rng: Optional[np.random.Generator] = None) -> List[float]:
    """
    Construct a strictly unimodal sequence of parity peaks (kg/day).

    The sequence increases linearly up to `peak_at_parity` and decreases linearly
    thereafter, with small multiplicative jitter per parity.

    Parameters
    ----------
    n_lact : int
        Number of lactations (parities).
    base_peak, max_peak, min_peak : float
        Peak levels controlling the left slope, the apex, and the right tail.
    peak_at_parity : int
        Parity at which the maximum occurs (clipped to [2, n_lact]).
    jitter : float
        Multiplicative ±jitter applied to each parity’s peak.
    rng : np.random.Generator, optional
        RNG for reproducibility.

    Returns
    -------
    List[float]
        Parity-by-parity intended peaks (length n_lact).
    """
    rng = np.random.default_rng() if rng is None else rng
    peak_at_parity = max(2, min(int(peak_at_parity), n_lact))
    up_steps   = max(1, peak_at_parity - 1)
    down_steps = max(1, n_lact - peak_at_parity)
    up_slope   = (max_peak - base_peak) / up_steps
    down_slope = (max_peak - min_peak) / down_steps

    peaks = []
    for parity in range(1, n_lact + 1):
        if parity <= peak_at_parity:
            val = base_peak + (parity - 1) * up_slope
        else:
            val = max_peak - (parity - peak_at_parity) * down_slope
        val *= rng.uniform(1.0 - jitter, 1.0 + jitter)
        peaks.append(val)
    return peaks


def truncate_zero_tail(df_daily: pd.DataFrame,
                        dry_max_days: int = 75,
                        grace_extra: int = 10) -> pd.DataFrame:
    """
    If the series ends with a long run of zero milk (beyond a plausible dry period),
    truncate the life right after an allowed dry period.

    Parameters
    ----------
    df_daily : DataFrame with columns ['day','milk_kg',...], increasing 'day'
    dry_max_days : int
        Biological maximum dry period in your generator (50–75). Use the same here.
    grace_extra : int
        Small buffer to allow short non-marketed spells, vet days, etc.

    Returns
    -------
    DataFrame
        Possibly shortened copy of df_daily.
    """
    df = df_daily.sort_values("day").reset_index(drop=True)
    if df.empty:
        return df

    # Last day with positive milk
    pos_idx = np.flatnonzero(df["milk_kg"].to_numpy() > 0)
    if pos_idx.size == 0:
        return df  # (degenerate, but keep as is)

    last_pos = int(pos_idx[-1])
    zero_tail_len = len(df) - 1 - last_pos
    allow = int(dry_max_days + grace_extra)

    if zero_tail_len > allow:
        cut_at = last_pos + allow  # keep ≤ one allowed dry period after last milk
        df = df.iloc[:cut_at + 1].copy()

    return df


def generate_one_cow(cow_id,
                     seed=None,
                     min_lac=4, max_lac=6,
                     lact_len_mu=305, lact_len_sd=20, lact_len_min=270, lact_len_max=360,
                     dry_min=50, dry_max=75,
                     peak_day_min=45, peak_day_max=75,
                     base_peak_first=29.0, max_peak=47.0, min_peak=27.0, peak_at_parity=None,
                     ar_sigma=0.6,
                     shocks_per_lact=1,
                     shock_depth_range=(0.04, 0.08),
                     shock_hw_range=(4, 10),
                     enforce_mean_floor=True, target_mean_lact=23.5, max_scale_up=1.25,
                     peaks_override: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Generate daily milk data for a single cow across multiple lactations.

    Behavior:
      * If `peaks_override` is provided, it sets the parity-peak vector exactly
        (its length defines the number of lactations L).
      * Else, L is drawn in {4,5,6}, and a unimodal parity-peak vector is built,
        with randomized max-parity in {2,3,4} clipped to L.

    Daily milk = Wood curve (with randomized peak day) + mild AR(1)-like noise
    + occasional short multiplicative dips (“shocks”). Dry periods with zero milk
    are inserted between lactations.

    Parameters
    ----------
    cow_id : int
        Identifier for the cow (also used in filenames).
    seed : int, optional
        Per-cow seed for reproducibility.
    min_lac, max_lac : int
        Allowed range for number of lactations, unless overridden by peaks_override.
    lact_len_mu, lact_len_sd : float
        Mean and sd of lactation length (truncated to [lact_len_min, lact_len_max]).
    dry_min, dry_max : int
        Uniform range (inclusive) for dry period lengths between lactations.
    peak_day_min, peak_day_max : int
        Uniform range for the lactation day of the Wood-curve peak.
    base_peak_first, max_peak, min_peak : float
        Typical peak levels for unimodal parity peaks (ignored if peaks_override).
    peak_at_parity : int, optional
        Preferred parity for the maximum (ignored if peaks_override is used).
    ar_sigma : float
        Std. dev. of innovations in an AR(1)-like noise added to milk.
    shocks_per_lact : int
        Expected number of short dips per lactation.
    shock_depth_range : tuple(float, float)
        Range for shock depth (fraction of parity peak).
    shock_hw_range : tuple(int, int)
        Range for half-width of the local shock window (in days).
    enforce_mean_floor : bool
        If True, rescales lactation up to a minimal mean if too low.
    target_mean_lact : float
        Target minimal mean (kg/day) when enforcing the floor.
    max_scale_up : float
        Cap on rescaling factor to avoid unrealistic inflation.
    peaks_override : list[float], optional
        Parity-peak vector to use verbatim (length = number of lactations).

    Returns
    -------
    pd.DataFrame
        Columns: cow_id, day, lactation, milk_kg
    """
    rng = np.random.default_rng(seed)

    if peaks_override is not None:
        peaks_override = list(peaks_override)
        L = len(peaks_override)
        # tiny jitter for realism, but keep the pattern
        jit = rng.uniform(0.985, 1.015, size=L)
        peaks = (np.array(peaks_override, dtype=float) * jit).tolist()
    else:
        # original path (random L and unimodal peaks)
        L = int(rng.integers(min_lac, max_lac + 1))
        if peak_at_parity is None:
            upper = min(4, L)
            peak_parity_this_cow = int(rng.integers(2, upper + 1))
        else:
            peak_parity_this_cow = int(np.clip(peak_at_parity, 2, L))
        peaks = unimodal_parity_peaks(
            n_lact=L, base_peak=base_peak_first, max_peak=max_peak,
            peak_at_parity=peak_parity_this_cow, min_peak=min_peak,
            jitter=0.025, rng=rng
        )

    # Draw durations and peak days
    rloc = np.random.default_rng(seed)
    lact_lengths = np.clip(rloc.normal(lact_len_mu, lact_len_sd, size=L),
                           lact_len_min, lact_len_max).astype(int)
    dry_lengths  = (rloc.integers(dry_min, dry_max + 1, size=L - 1) if L > 1 else np.array([], dtype=int))
    peak_days    = rloc.integers(peak_day_min, peak_day_max + 1, size=L)

    # Construct daily milk series
    records = []
    day_counter = 0
    local_rng = np.random.default_rng(seed)

    for ell in range(L):
        dim = int(lact_lengths[ell])
        t = np.arange(1, dim + 1)
        pd_ell = float(peak_days[ell])
        b = 0.20
        c = b / pd_ell
        milk = wood_curve(t, float(peaks[ell]), b=b, c=c)

        # Mild AR(1)-like jaggedness
        eps = local_rng.normal(0.0, ar_sigma, size=dim)
        ar = np.zeros(dim)
        for i in range(1, dim):
            ar[i] = 0.55 * ar[i-1] + eps[i]
        milk = np.maximum(milk + ar, 0.0)

        # Occasional transient dips ("shocks")
        for _ in range(shocks_per_lact):
            c0 = local_rng.uniform(0.15 * dim, 0.85 * dim)
            hw = local_rng.integers(shock_hw_range[0], shock_hw_range[1] + 1)
            depth = local_rng.uniform(*shock_depth_range) * max(1.0, peaks[ell])
            lo = max(0, int(c0 - hw)); hi = min(dim, int(c0 + hw) + 1)
            milk[lo:hi] *= (1.0 - depth / peaks[ell])

        # Enforce optional mean-floor
        if enforce_mean_floor:
            cur_mean = float(np.mean(milk))
            if cur_mean < target_mean_lact:
                scale = min(max_scale_up, target_mean_lact / max(cur_mean, 1e-6))
                milk *= scale

        # Write lactation days
        for d in range(dim):
            day_counter += 1
            records.append(dict(cow_id=cow_id, day=day_counter, lactation=ell + 1, milk_kg=float(milk[d])))

        # Dry period days (zero milk)
        if ell < L - 1:
            D = int(dry_lengths[ell])
            for _ in range(D):
                day_counter += 1
                records.append(dict(cow_id=cow_id, day=day_counter, lactation=ell + 1, milk_kg=0.0))

    df_out = pd.DataFrame.from_records(records, columns=["cow_id","day","lactation","milk_kg"])
    df_out = truncate_zero_tail(df_out, dry_max_days=75, grace_extra=10)
    return df_out


# ------------------ Profit accumulation (Excel logic) ------------------

def cumulative_profit_from_milk(milk: np.ndarray, bl: Baseline) -> np.ndarray:
    """
    Construct cumulative profit P(t) from daily milk series using Excel logic.

    Daily contribution:
        (milk_price - feed_var_cost) * milk_kg - (maint_cost + stall_cost)

    Cumulative P(t) is the running sum of contributions with a day-0 offset:
        P(0) = -rearing_cost + salvage_value

    Parameters
    ----------
    milk : np.ndarray
        Daily milk production series (kg).
    bl : Baseline
        Economic parameters.

    Returns
    -------
    np.ndarray
        Cumulative profit series aligned with days 1..T; P(0) is not included here
        but is used when plotting/intercepting at x=0.
    """
    pi = bl.milk_margin
    c_day = bl.maint_cost + bl.stall_cost
    contrib = pi * milk - c_day
    P = np.cumsum(contrib)
    return P - bl.rearing_cost + bl.salvage_value


# ------------------ Cubic fit and differentiable indices ------------------

def fit_cubic_anchored(t: np.ndarray, P: np.ndarray, P0: Optional[float] = None) -> np.ndarray:
    """
    Fit a cubic polynomial P̃(t) = a t³ + b t² + c t + d in descending order,
    with intercept constrained so that P̃(0) = P0.

    Parameters
    ----------
    t : np.ndarray
        Time (days).
    P : np.ndarray
        Observed cumulative profit.
    P0 : float, optional
        Baseline value at t=0 (defaults to -REARING_COST + SALVAGE_PRICE).

    Returns
    -------
    np.ndarray
        Coefficients [a, b, c, d] (descending order).
    """
    if P0 is None:
        P0 = -REARING_COST + SALVAGE_PRICE

    # shift response so the intercept is effectively fixed
    y = P - P0
    X = np.column_stack([t**3, t**2, t])  # descending order, no constant term
    coef3, coef2, coef1 = np.linalg.lstsq(X, y, rcond=None)[0]
    coef0 = P0  # fixed intercept
    return np.array([coef3, coef2, coef1, coef0], dtype=float)




def eval_cubic(coef: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Evaluate P̃(t) = a t^3 + b t^2 + c t + d on a numpy array t.

    Parameters
    ----------
    coef : np.ndarray
        Coefficients [a, b, c, d].
    t : np.ndarray
        Points at which to evaluate.

    Returns
    -------
    np.ndarray
        P̃(t) values.
    """
    a, b, c, d = coef
    return ((a*t + b)*t + c)*t + d


def switch_opt_from_cubic(coef: np.ndarray, t_min: float, t_max: float,
                          t_h_star: float, C_h: float) -> float:
    """
    Compute the switch-with-heifer optimal age t̂ via FOC for
        max_t (P̃(t) + C_h) / (t + t_h_star).

    The derivative condition yields a cubic in t:
        2 a t^3 + (b + 3 a t_h*) t^2 + (2 b t_h*) t + (c t_h* - d - C_h) = 0.

    Parameters
    ----------
    coef : np.ndarray
        Cubic coefficients [a, b, c, d].
    t_min, t_max : float
        Search interval (strictly positive).
    t_h_star : float
        Heifer cycle length (days).
    C_h : float
        Heifer cumulative profit at its optimal horizon: P_h(t_h*).

    Returns
    -------
    float
        Optimal switching age t̂.
    """
    a, b, c, d = coef
    roots = np.roots([2*a, b + 3*a*t_h_star, 2*b*t_h_star, c*t_h_star - d - C_h])
    cand = [float(r.real) for r in roots if abs(r.imag) < 1e-8 and t_min < r.real < t_max]
    cand += [t_min + 1e-6, t_max - 1e-6]

    def S(tt):
        P = eval_cubic(coef, np.array([tt]))[0]
        return (P + C_h)/(tt + t_h_star)

    return max(cand, key=S)


def average_profit_from_cubic(coef: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute Ã(t) = P̃(t) / t on a vector of t (t>0).

    Parameters
    ----------
    coef : np.ndarray
        Cubic coefficients [a, b, c, d].
    t : np.ndarray
        Evaluation grid (strictly positive entries).

    Returns
    -------
    np.ndarray
        Ã(t) values; NaN at nonpositive t (if any).
    """
    P = eval_cubic(coef, t)
    A = np.full_like(t, np.nan, dtype=float)
    m = t > 0
    A[m] = P[m]/t[m]
    return A


def needs_repair(coef: np.ndarray, t_min: float, t_max: float) -> bool:
    """
    Decide if the fitted cubic requires repair:
    (i) Ã(t) has multiple local maxima, OR
    (ii) tail concavity violation (P'' > 0) near the dominant peak.
    """
    # grid for tests
    t_grid = np.linspace(t_min, t_max, 1500)
    Pfit_grid = np.polyval(coef, t_grid)
    Agrid = np.full_like(t_grid, np.nan, dtype=float)
    m = t_grid > 0
    Agrid[m] = Pfit_grid[m] / t_grid[m]

    # (i) multimodality
    peak_count = sum(
        1 for i in range(1, len(Agrid) - 1)
        if np.isfinite(Agrid[i-1]) and np.isfinite(Agrid[i+1])
        and Agrid[i] >= Agrid[i-1] and Agrid[i] >= Agrid[i+1]
    )
    if peak_count > 1:
        return True

    # (ii) tail concavity test around dominant peak
    i_dom = int(np.nanargmax(Agrid))
    # finite-difference second derivative on a local window [i_dom, end)
    # use a conservative slice of the right tail
    i0 = max(i_dom, int(0.6 * len(t_grid)))
    if i0 < len(t_grid) - 2:
        dt = t_grid[1] - t_grid[0]
        d2 = (Pfit_grid[i0+1:-1] - 2*Pfit_grid[i0:-2] + Pfit_grid[i0-1:-3]) / (dt**2)
        # if any clear positive curvature is found, flag for repair
        if np.nanmax(d2) > 0:
            return True

    return False



def constrained_refit_monodec_second_derivative(t: np.ndarray, Pfit: np.ndarray,
                                                window: Tuple[float, float]) -> np.ndarray:
    """
    Local constrained refit to enforce P''(t) ≤ 0 on a window [τ1, τ2].

    We keep a cubic but grid-search on negative 'a' (curvature) and choose 'b'
    so that P''(τ1) ≤ 0 and P''(τ2) ≤ 0 hold. The linear terms (c, d) are
    re-estimated by least squares. Returns the best (a,b,c,d) minimizing MSE
    against the original Pfit on the full observed grid.

    Parameters
    ----------
    t : np.ndarray
        Observed day grid (1..T).
    Pfit : np.ndarray
        Original cubic fit evaluated on t.
    window : (float, float)
        Window [τ1, τ2] around the dominant Ã peak where concavity is enforced.

    Returns
    -------
    np.ndarray
        New coefficients [a, b, c, d] if found, else None.
    """
    τ1, τ2 = window
    a_uc = fit_cubic_anchored(t, Pfit)[0]
    a_min = -abs(a_uc) if a_uc != 0 else -1e-6
    a_grid = np.linspace(1.5 * a_min, -1e-9, 60)  # strictly negative curvature
    best_coef, best_err = None, np.inf

    T = t
    for a in a_grid:
        # choose b to keep P''(t)=6 a t + 2 b ≤ 0 at t=τ1, τ2
        b = min(-3.0 * a * τ1, -3.0 * a * τ2)
        X = np.column_stack([T, np.ones_like(T)])
        y = Pfit - (a*T**3 + b*T**2)
        cd, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        c, d = float(cd[0]), float(cd[1])

        P_hat = a*T**3 + b*T**2 + c*T + d
        err = np.mean((P_hat - Pfit)**2)
        if err < best_err:
            best_err = err
            # return in *descending* order [a,b,c,d]
            best_coef = np.array([a, b, c, d], dtype=float)

    return best_coef

# --------------------Polynomial fit robustness------------------------

def envelope_from_poly(
    coef: np.ndarray,
    t_min: float,
    t_max: float,
    t_h_star: float,
    C_h: float,
    tol: float = 1e-6,
) -> dict:
    """
    Envelope decision from an arbitrary polynomial P(t) with coefficients in
    NumPy's descending order (as returned by np.polyfit).

    Returns a dict with:
      - t_star, lambda_          : cow-only argmax and level of A(t) = P(t)/t
      - t_hat,  Lambda           : switch argmax and level of S(t) = (P(t)+C_h)/(t+t_h*)
      - branch                   : "COW_ONLY" or "SWITCH" (whoever has higher value)
      - t_env,  A_env            : envelope argmax and value
      - used_boundary_cow        : True if cow-only maximizer is a boundary
      - used_boundary_switch     : True if switch maximizer is a boundary
    """
    if not (np.isfinite(t_min) and np.isfinite(t_max) and t_min < t_max):
        raise ValueError("t_min and t_max must be finite with t_min < t_max.")
    if t_min <= 0:
        # We divide by t; enforce strictly positive domain.
        t_min = max(t_min, 1e-6)

    p  = np.poly1d(coef)     # P(t)
    p1 = p.deriv()           # P'(t)
    tpoly = np.poly1d([1.0, 0.0])  # the polynomial 't'

    # Average-profit and switch indices
    def A_of(t: float) -> float:
        return float(p(t) / t)

    def S_of(t: float) -> float:
        return float((p(t) + C_h) / (t + t_h_star))

    # Helper: real roots strictly inside (lo, hi)
    def real_roots_in(poly1d_obj, lo: float, hi: float) -> list[float]:
        r = np.roots(poly1d_obj)
        return [float(z.real) for z in r if (z.imag == 0.0) and (lo < z.real < hi)]

    # Use a tiny absolute and relative margin to stay away from exact endpoints
    eps = max(1e-9, 1e-6 * (t_max - t_min))
    lo, hi = t_min + eps, t_max - eps

    # ---------- 1) Cow-only: maximize A(t) on [t_min, t_max] ----------
    # FOC: t*P'(t) - P(t) = 0
    F1 = tpoly * p1 - p
    interior_cand_star = real_roots_in(F1, lo, hi)

    # Candidate set = interior roots + boundaries
    cand_star = interior_cand_star + [t_min, t_max]
    # Evaluate A on all candidates and pick the best
    vals_star = [A_of(t) for t in cand_star]
    idx_star = int(np.argmax(vals_star))
    t_star = float(cand_star[idx_star])
    lambda_ = float(vals_star[idx_star])
    used_boundary_cow = (idx_star >= len(interior_cand_star))

    # ---------- 2) Switch-with-heifer: maximize S(t) on [t_min, t_max] ----------
    # FOC: (t+t_h*) P'(t) - (P(t)+C_h) = 0
    F2 = (tpoly + t_h_star) * p1 - (p + C_h)
    interior_cand_hat = real_roots_in(F2, lo, hi)

    cand_hat = interior_cand_hat + [t_min, t_max]
    vals_hat = [S_of(t) for t in cand_hat]

    # Pick the true maximizer of S; if ties within tol, prefer interior,
    # then minimize |P'(t) - S(t)| among the tied candidates.
    max_S = max(vals_hat)
    near_best = [k for k, v in enumerate(vals_hat) if abs(v - max_S) <= tol]

    def foc_gap(t: float) -> float:
        return abs(float(p1(t)) - S_of(t))

    near_best_interior = [k for k in near_best if k < len(interior_cand_hat)]
    if near_best_interior:
        k_hat = min(near_best_interior, key=lambda k: foc_gap(cand_hat[k]))
    else:
        k_hat = min(near_best, key=lambda k: foc_gap(cand_hat[k]))


    t_hat = float(cand_hat[k_hat])
    Lambda = float(vals_hat[k_hat])
    used_boundary_switch = (k_hat >= len(interior_cand_hat))

    # ---------- 3) Envelope ----------
    if lambda_ >= Lambda:
        branch, t_env, A_env = "COW_ONLY", t_star, lambda_
    else:
        branch, t_env, A_env = "SWITCH", t_hat, Lambda

    return dict(
        t_star=t_star, lambda_=lambda_,
        t_hat=t_hat,  Lambda=Lambda,
        branch=branch,
        t_env=t_env,  A_env=A_env,
        used_boundary_cow=used_boundary_cow,
        used_boundary_switch=used_boundary_switch,
    )


def is_envelope_consistent(t_star: float, t_hat: float,
                           lam: float, Lambda: float,
                           used_boundary_switch: bool,
                           t_tol: float = 5.0,          # days tolerance for “same time”
                           a_tol_abs: float = 5e-3,     # €/day absolute tolerance
                           a_tol_rel: float = 5e-3      # relative tolerance (0.5%)
                           ) -> bool:
    """
    Return True unless there is a *material* violation of the interior-geometry order:
      - If t_hat > t_star, we expect Lambda <= lam (switch later → lower long-run rate).
      - If t_hat < t_star, we expect Lambda >= lam (switch earlier → higher long-run rate).

    Near-ties in time (|t_hat - t_star| <= t_tol) or in indices (<= tolerances) are treated as consistent.
    Endpoints for the switch optimum (used_boundary_switch=True) are exempted because S(t) may
    be monotone at the boundary.

    Parameters mirror the envelope outputs you already compute.
    """
    # near-equal decision times → treat as consistent
    if abs(t_hat - t_star) <= t_tol:
        return True
    # boundary switch → do not enforce interior-order check
    if used_boundary_switch:
        return True

    # tolerant comparisons for indices
    def ge(a, b):  # a >= b within tolerance
        return (a + a_tol_abs >= b) or (abs(a - b) <= a_tol_rel * max(1.0, abs(b)))
    def le(a, b):  # a <= b within tolerance
        return (a <= b + a_tol_abs) or (abs(a - b) <= a_tol_rel * max(1.0, abs(b)))

    if t_hat > t_star:
        return le(Lambda, lam)   # expect Λ ≤ λ
    else:  # t_hat < t_star
        return ge(Lambda, lam)   # expect Λ ≥ λ


# ------------------ Daily table (transparency / Excel columns) ------------------

def make_daily_table(cow_id: int, df_daily: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    """
    Build a transparent per-cow daily table that backs each figure.

    Columns
    -------
    cow_id, day, lactation, milk_kg,
    acum_milk_kg, acum_profit, time_avg_profit,
    marketed_flag, marketed_milk_kg

    The first row (day=0) is added to match the Excel convention, with
    P(0) = -REARING_COST + SALVAGE_PRICE and NaN time-average.

    Parameters
    ----------
    cow_id : int
        Identifier.
    df_daily : pd.DataFrame
        Output of generate_one_cow (days 1..T).
    P : np.ndarray
        Cumulative profit aligned with days 1..T.

    Returns
    -------
    pd.DataFrame
        Daily table including the day-0 row.
    """
    out = df_daily.copy().sort_values("day").reset_index(drop=True)
    out["acum_milk_kg"] = out["milk_kg"].cumsum()
    out["acum_profit"]  = P
    out["time_avg_profit"] = out["acum_profit"] / out["day"].replace(0, np.nan)
    out["marketed_flag"]    = 1
    out["marketed_milk_kg"] = out["milk_kg"]

    row0 = pd.DataFrame([{
        "cow_id": cow_id, "day": 0, "lactation": 0, "milk_kg": 0.0,
        "acum_milk_kg": 0.0, "acum_profit": -REARING_COST + SALVAGE_PRICE,
        "time_avg_profit": np.nan, "marketed_flag": 1, "marketed_milk_kg": 0.0,
    }])

    return pd.concat([row0, out], ignore_index=True)


# ------------------ Main pipeline ------------------

def run(n_cows: int = 10,
        horizon: int = 2400,
        scenario_name: str = "S0_baseline",
        T_H_STAR: float = 400.0,
        P_H_AT_TSTAR: float = 600.0) -> pd.DataFrame:
    """
    End-to-end pipeline for a cohort of cows under a chosen heifer benchmark.

    Steps per cow
    -------------
    1) Generate daily milk (strictly unimodal parity peaks), except the first cow,
       which is a “stress test” with oscillating parity peaks (Hi/Lo/Hi/Lo...) to
       induce multiple local maxima in basic A(t).
    2) Construct P(t) via Excel accounting logic.
    3) Fit cubic P̃(t) and compute indices via FOCs (t*, t̂).
    4) If Ã(t) shows multiple local maxima, try local constrained refit; on the
       stress-test cow we apply it unconditionally when feasible (to ensure a
       repaired example is present).
    5) Produce the per-cow daily CSV and figure; collect summary metrics.

    Parameters
    ----------
    n_cows : int
        Number of cows to simulate.
    horizon : int
        Maximum number of days to keep in the series (pad or trim to this).
    scenario_name : str
        Name of the heifer-benchmark scenario; used as subfolder name.
    T_H_STAR : float
        Heifer cycle length (days) used in the switch index.
    P_H_AT_TSTAR : float
        Heifer cumulative profit at T_H_STAR used in the switch index.

    Returns
    -------
    pd.DataFrame
        Summary table written to ./data/<scenario_name>/section7_summary_<baseline|weaker>.csv
    """
    rng = np.random.default_rng(MASTER_SEED)
    out_dir = os.path.join(OUTPUT_DIR, scenario_name)
    os.makedirs(out_dir, exist_ok=True)

    # Unique cow IDs in [100, 998]
    cow_ids = []
    while len(cow_ids) < n_cows:
        cid = int(rng.integers(100, 999))
        if cid not in cow_ids:
            cow_ids.append(cid)

    rows: list[dict] = []

    fig_names = []
    for i, cid in enumerate(cow_ids):
        per_seed = int(rng.integers(0, 2**31 - 1))

        # ---- Stress-test cow (first one): oscillating parity peaks (Hi/Lo/Hi/Lo...) ----
        is_stress = (i == 0)
        peaks_override = None
        if is_stress:
            base = np.array([30.0, 45.0, 35.0, 46.0, 34.0, 29.0], dtype=float)  # 6 lactations
            local_rng = np.random.default_rng(per_seed)
            jit = local_rng.uniform(0.98, 1.02, size=base.shape)               # tiny jitter
            peaks_override = (base * jit).tolist()                              # controls L too

        # Generate daily milk
        df = generate_one_cow(
            cow_id=cid, seed=per_seed,
            min_lac=4, max_lac=6,
            base_peak_first=29.0, max_peak=47.0, min_peak=27.0,
            ar_sigma=0.6, shocks_per_lact=1, shock_depth_range=(0.04, 0.08),
            peaks_override=peaks_override
        )

        # Only trim if exceeding the horizon; otherwise keep actual life length
        if df["day"].iloc[-1] > horizon:
            df = df[df["day"] <= horizon].copy()
        

        # Construct P(t)
        t = df["day"].to_numpy(float)
        milk = df["milk_kg"].to_numpy(float)
        P = cumulative_profit_from_milk(milk, BL)
        t_min = max(1.0, t.min() + 1.0)
        t_max = t.max()

        # Cubic fit
        coef = fit_cubic_anchored(t, P)
        Pfit = eval_cubic(coef, t)

        # ---- Unimodality check for Ã(t) via stationary-condition zeros: t P'(t) - P(t) ----
        # A'(t) = (t P'(t) - P(t)) / t^2, so zeros of N(t) := t P'(t) - P(t) are the extrema of Ã.
        repaired = False
        t_grid = np.linspace(max(1.0, t.min() + 1.0), t.max(), 1500)

        # Build N(t) on a fine grid
        a, b, c, _ = coef
        P_grid = eval_cubic(coef, t_grid)
        Pprime_grid = 3.0*a*t_grid**2 + 2.0*b*t_grid + c
        N_grid = t_grid * Pprime_grid - P_grid

        # Count sign changes of N(t) (ignoring tiny values near zero)
        sgn = np.sign(N_grid)
        sgn[np.abs(N_grid) < 1e-10] = 0.0
        # compress zeros so they don’t create spurious flips
        sgn_nz = sgn[sgn != 0]
        num_flips = np.sum(sgn_nz[1:] * sgn_nz[:-1] < 0) if sgn_nz.size > 1 else 0
        A_unimodal = (num_flips <= 1)

        if not A_unimodal:
            # Focus a repair window around the dominant extremum of Ã
            Agrid = np.full_like(t_grid, np.nan, dtype=float)
            mask = t_grid > 0
            Agrid[mask] = P_grid[mask] / t_grid[mask]
            i_dom = int(np.nanargmax(Agrid))
            t_dom = float(t_grid[i_dom])
            τ1 = max(t_grid.min(), t_dom - 120.0)
            τ2 = min(t_grid.max(), t_dom + 120.0)

            coef_rep = constrained_refit_monodec_second_derivative(t, Pfit, (τ1, τ2))
            if coef_rep is not None:
                # Compare envelopes using the SAME function & scenario
                env_uc = envelope_from_poly(coef,     t_min, t_max, T_H_STAR, P_H_AT_TSTAR)
                env_rp = envelope_from_poly(coef_rep, t_min, t_max, T_H_STAR, P_H_AT_TSTAR)

                lam_uc = env_uc["lambda_"]
                lam_rp = env_rp["lambda_"]

                # Stress cow: keep the repaired example for the paper
                if is_stress:
                    coef = coef_rep
                    Pfit = eval_cubic(coef, t)
                    repaired = True
                else:
                    # Otherwise adopt only if it materially changes λ (≥ 2%)
                    if abs(lam_rp - lam_uc) / max(1.0, abs(lam_uc)) > 0.02:
                        coef = coef_rep
                        Pfit = eval_cubic(coef, t)
                        repaired = True


        # Envelope quantities (now centrally computed for cubic as well)
        env_cubic = envelope_from_poly(coef, t_min, t_max, T_H_STAR, P_H_AT_TSTAR)
        t_star  = env_cubic["t_star"]
        lam     = env_cubic["lambda_"]
        t_hat   = env_cubic["t_hat"]
        Lambda  = env_cubic["Lambda"]
        branch  = env_cubic["branch"]
        t_env   = env_cubic["t_env"]
        A_env   = env_cubic["A_env"]

        # Boundary flags and diagnostics
        used_boundary_cow    = bool(env_cubic["used_boundary_cow"])
        used_boundary_switch = bool(env_cubic["used_boundary_switch"])

        # Exact time gap and “interior-equal” flag
        t_diff = float(abs(t_star - t_hat))
        times_equal_interior = (
            (t_diff <= 1e-6) and
            (not used_boundary_cow) and
            (not used_boundary_switch)
        )


        # 1) Check concavity of tail on [min(t*,t^), t_max]
        def cubic_tail_concave_on(coef, L, U):
            a,b,_,_ = coef
            val_L = 6.0*a*L + 2.0*b
            val_U = 6.0*a*U + 2.0*b
            return max(val_L, val_U) <= 0.0

        tail_concave = cubic_tail_concave_on(coef, min(t_star, t_hat), t_max)

        # 2) Boundary flags (tolerance = 1e-3 days)
        eps = 1e-3
        star_at_boundary = (abs(t_star - t_min) < eps) or (abs(t_star - t_max) < eps)
        hat_at_boundary  = (abs(t_hat  - t_min) < eps) or (abs(t_hat  - t_max) < eps)

        # 3) Second-order test at stationary points (negative => local max)
        #    For a cubic: P''(t) = 6 a t + 2 b
        a,b,_,_ = coef
        P2_star = 6.0*a*t_star + 2.0*b
        P2_hat  = 6.0*a*t_hat  + 2.0*b

        # 4) If geometry looks impossible, write a tiny debug file with the raw numbers
        if (t_hat > t_star) and (Lambda > lam) and tail_concave and (not star_at_boundary) and (not hat_at_boundary):
            dbg_path = os.path.join(out_dir, f"debug_cubic_geometry_cow_{cid}.txt")
            with open(dbg_path, "w", encoding="utf-8") as g:
                g.write(f"cow_id={cid}\n")
                g.write(f"t_min={t_min:.6f}, t_max={t_max:.6f}\n")
                g.write(f"t_star={t_star:.6f}, lambda={lam:.6f}, P''(t_star)={P2_star:.6e}\n")
                g.write(f"t_hat ={t_hat :.6f}, Lambda={Lambda:.6f}, P''(t_hat) ={P2_hat :.6e}\n")
                g.write(f"tail_concave={tail_concave}, star_at_boundary={star_at_boundary}, hat_at_boundary={hat_at_boundary}\n")
                # also log derivative monotonicity sample
                ts = np.linspace(min(t_star, t_hat), t_max, 6)
                P1 = 3*a*ts**2 + 2*b*ts + (coef[2])
                g.write("sample_t," + ",".join(f"{x:.1f}" for x in ts) + "\n")
                g.write("sample_Pprime," + ",".join(f"{x:.6f}" for x in P1) + "\n")


        # Fit diagnostics
        resid = P - Pfit
        mse = float(np.mean(resid**2))
        mae = float(np.mean(np.abs(resid)))
        sst = float(np.sum((P - P.mean())**2))
        r2  = 1.0 - float(np.sum(resid**2))/sst if sst > 0 else np.nan

        # Save daily table (transparency) — prefix files with generation index
        idx = f"{i+1:02d}"
        daily = make_daily_table(cid, df, P)
        daily.to_csv(os.path.join(out_dir, f"{idx}_cow_{cid}_daily.csv"),
                     index=False, encoding="utf-8")

        # --------- Figures: full + panel versions ---------
        P0_data = -REARING_COST + SALVAGE_PRICE
        P0_fit  = float(eval_cubic(coef, np.array([0.0]))[0])

        # ----- (1) Full stand-alone figure -----
        fig, ax = plt.subplots(figsize=(7.8, 4.0), dpi=300)
        _render_cow_plot(ax,
            cow_id=cid,
            t=t, P=P, Pfit=Pfit,
            t_star=t_star, t_hat=t_hat,
            P0_data=P0_data, P0_fit=P0_fit,
            repaired=repaired, panel=False)
        fig.tight_layout()

        base_name = f"{idx}_fig_{scenario_name}_cow_{cid}"
        png_path  = os.path.join(out_dir, base_name + ".png")
        pdf_path  = os.path.join(out_dir, base_name + ".pdf")
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        # ----- (2) Compact panel for mosaic (no legends/labels) -----
        fig_p, ax_p = plt.subplots(figsize=(3.2, 2.1), dpi=300)
        _render_cow_plot(ax_p,
            cow_id=cid,
            t=t, P=P, Pfit=Pfit,
            t_star=t_star, t_hat=t_hat,
            P0_data=P0_data, P0_fit=P0_fit,
            repaired=repaired, panel=True)
        fig_p.tight_layout(pad=0.4)
        panel_name = f"{idx}_fig_{scenario_name}_cow_{cid}_panel"
        fig_p.savefig(os.path.join(out_dir, panel_name + ".png"), bbox_inches="tight", dpi=300)
        fig_p.savefig(os.path.join(out_dir, panel_name + ".pdf"), bbox_inches="tight")
        plt.close(fig_p)

        # track names if you’re making a LaTeX grid later
        fig_names.append(base_name)          # full
        fig_names.append(panel_name)         # panel (use these in the gallery)

        rows.append(dict(
            scenario=scenario_name,
            generation_index=i+1,     # keep creation order
            cow_id=cid,
            seed=per_seed,
            stress_test=is_stress,
            repaired=repaired,
            in_sample_MSE=mse,
            in_sample_MAE=mae,
            in_sample_R2=r2,
            t_star_cow_only=t_star,
            lambda_=lam,
            t_hat_switch=t_hat,
            Lambda_star=Lambda,
            policy_branch=branch,
            P0_data=(-REARING_COST + SALVAGE_PRICE),
            P0_fit=P0_fit,
            # --- new envelope outputs (add them here) ---
            t_env=t_env,
            A_env=A_env,
            # NEW: diagnostics that explain “same t, different λ”
            t_diff_days=t_diff,
            used_boundary_cow=used_boundary_cow,
            used_boundary_switch=used_boundary_switch,
            times_equal_interior=times_equal_interior,
        ))

    # Keep generation order (no sorting) and write scenario-specific summary
    summary_df = pd.DataFrame(rows)

    # Decide suffix for the filename (baseline vs weaker)
    scen_l = scenario_name.lower()
    scen_suffix = "weaker" if "weaker" in scen_l else "baseline"
    summary_filename = f"section7_summary_{scen_suffix}.csv"

    summary_path = os.path.join(out_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8", float_format="%.8f")

    with open(os.path.join(out_dir, "section7_readme.txt"), "w", encoding="utf-8") as f:
        f.write(f"Section 7 pipeline — Scenario: {scenario_name}\n")
        f.write(" - Master seed: 17001\n")
        f.write(" - Economics: milk_price=0.50, var_cost=0.23, maint=3, stall=2, rearing=2700, salvage=1200 (P(0)=-1500)\n")
        f.write(f" - Heifer benchmark: t_h*={T_H_STAR}, P_h(t_h*)={P_H_AT_TSTAR}, A_h={P_H_AT_TSTAR/T_H_STAR:.2f} €/day\n")
        f.write(" - Cow #1 is a stress test with oscillating parity peaks; constrained refit applied if feasible.\n")
        f.write(" - Other cows: strictly unimodal parity peaks with randomized max-parity in {2,3,4}.\n")
        f.write(" - Files are prefixed by generation index (01_, 02_, …) to match creation order.\n")
        f.write(" - Note: When the cow-only and switch optima occur at a boundary (e.g., t_max), "
                "they can coincide in time while their indices differ, since FOCs need not hold at endpoints.\n")

    # write a simple manifest with one filename per line (creation order)
    with open(os.path.join(out_dir, f"fig_manifest_{scenario_name}.txt"), "w", encoding="utf-8") as mf:
        for nm in fig_names:
            mf.write(nm + "\n")

    print(f"\n[{scenario_name}] Results saved in:")
    print("   " + os.path.abspath(out_dir))
    print(f"   - {summary_filename}")
    # If your figure filenames include the scenario, mirror that here.
    # Example if you save as f\"{idx}_fig_{scen_suffix}_cow_{cid}.png\"
    print(f"   - {{idx}}_fig_{scenario_name}_cow_<id>.png and {{idx}}_cow_<id>_daily.csv (one per cow)")
    print("   - section7_readme.txt")
    print(f"   - fig_manifest_{scenario_name}.txt\n")

    # --- Create shared legend figure (for LaTeX panel use) ---
    fig_leg, ax_leg = plt.subplots(figsize=(5.5, 0.35))
    ax_leg.axis("off")

    handles = [
        Line2D([0], [0], color="#e7a53f", lw=1.6, label="Observed $P_j(t)$"),
        Line2D([0], [0], color="#2b6cb0", lw=1.8, label="Cubic fit $\\tilde{P}_j(t)$"),
        Line2D([0], [0], color="black", lw=1.0, ls=(0, (4, 3)), label="$t_j^*$ (cow-only)"),
        Line2D([0], [0], color="#b2182b", lw=1.0, ls=(0, (2, 2, 1, 2)), label="$\\hat t_j$ (switch-with-heifer)")
    ]

    ax_leg.legend(handles=handles, loc="center", ncol=4, frameon=False, fontsize=8)
    leg_name = f"legend_panel_{scenario_name}.pdf"
    fig_leg.savefig(os.path.join(out_dir, leg_name), bbox_inches="tight")
    plt.close(fig_leg)

    return summary_df




# ------------------ Example runs for two benchmark scenarios ------------------

if __name__ == "__main__":
    # S0: baseline heifer (A_h = 600/400 = 1.50 €/day) → likely more SWITCH
    run(n_cows=10, horizon=2400,
        scenario_name="S0_baseline",
        T_H_STAR=400.0, P_H_AT_TSTAR=600.0)

    # S1: weaker heifer (A_h = 450/450 = 1.00 €/day) → some COW_ONLY cases appear
    run(n_cows=10, horizon=2400,
        scenario_name="S1_weaker",
        T_H_STAR=450.0, P_H_AT_TSTAR=450.0)
