# fig9_plot.py
# Pure plotting/styling for Appendix Fig. 9.
#
# Inputs:
#   - day1/day2 precomputed DataFrames from fig9_data.compute_fig9_day_products
#
# Output:
#   - PDF (and optional PNG) figure.

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


# -----------------------------------------------------------------------------
# Paper style (mirrors fig5_plot so the whole paper is consistent)
# -----------------------------------------------------------------------------
PAPER_STYLE_RCPARAMS: Dict[str, Any] = {
    "figure.dpi": 300,
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "lines.linewidth": 1.2,
    "lines.markersize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def get_paper_style_rcparams() -> Dict[str, Any]:
    return dict(PAPER_STYLE_RCPARAMS)


def get_paper_style_manifest_rcparams() -> Dict[str, Any]:
    """A JSON-friendly rcparams dict (same convention as Fig.5 manifest)."""
    return {k.replace(".", "_"): v for k, v in PAPER_STYLE_RCPARAMS.items()}


def apply_paper_style() -> None:
    plt.rcParams.update(get_paper_style_rcparams())


# -----------------------------------------------------------------------------
# Tick formatter: wrap all tick labels in mathtext
# -----------------------------------------------------------------------------
def mathtext_formatter(x, pos):
    """Format tick labels as mathtext for font consistency."""
    if x == int(x):
        return rf"${int(x)}$"
    else:
        return rf"${x:g}$"


# -----------------------------------------------------------------------------
# Fig.9 styling knobs (EDIT HERE when you want to tweak the look)
# -----------------------------------------------------------------------------

# Day-to-day colors for Panel B (signal/control breakdown)
FIG9_DAY_COLORS = {
    "Day1": "#548c8d",  
    "Day2": "#c2a855", 
}

# Day-to-day colors for Panel A (delta mode) - distinct from Fig3 signal/control
FIG9_DELTA_COLORS = {
    "Day1": "#548c8d",  
    "Day2": "#c2a855", 
}

# Marker for Panel A delta mode (distinct from signal/control markers)
FIG9_DELTA_MARKER = "^"  # thin diamond (smaller footprint than "D")
FIG9_DELTA_MARKERSIZE = 6  # slightly smaller than default 6

# Pair styles (consistent with Fig.5: signal is solid, control is dashed)
FIG9_PAIR_STYLES = {
    "signal": dict(marker="o", linestyle="-", markerfacecolor=None),
    "control": dict(marker="s", linestyle="--", markerfacecolor="white"),
}


def _etas_intersection(day1_df: pd.DataFrame, day2_df: pd.DataFrame, col: str = "eta3") -> list[float]:
    e1 = {float(x) for x in day1_df[col].unique()}
    e2 = {float(x) for x in day2_df[col].unique()}
    return sorted(e1.intersection(e2))


def plot_fig9(
    *,
    day1_summary_df: pd.DataFrame,
    day2_summary_df: pd.DataFrame,
    day1_delta_df: pd.DataFrame,
    day2_delta_df: pd.DataFrame,
    out_pdf: str,
    out_png: str | None = None,
    n_key: int = 7,
    metric_mode: str = "delta",  # {"delta", "V"}
    signal_pair: Tuple[int, int] = (0, 1),
    control_pair: Tuple[int, int] = (1, 4),
    include_control_amp: bool = True,
    fig_size_inches: Tuple[float, float] = (10.0, 3.72),
) -> None:
    """Render Appendix Fig. 9.

    Panel A: Day1 vs Day2 overlay for either
      - ΔV_circ(eta3) at fixed n (default), or
      - V_circ(eta3) at fixed n for both signal+control.

    Panel B: amp_min whiskers (q10 -> mean) at fixed n.
    """

    metric_mode = str(metric_mode).lower().strip()
    if metric_mode not in {"delta", "v"}:
        raise ValueError("metric_mode must be one of {'delta','V'}.")

    apply_paper_style()

    # Determine which eta values are comparable across days.
    if metric_mode == "delta":
        etas = _etas_intersection(day1_delta_df[day1_delta_df["n"] == int(n_key)], day2_delta_df[day2_delta_df["n"] == int(n_key)])
    else:
        etas = _etas_intersection(day1_summary_df[day1_summary_df["n"] == int(n_key)], day2_summary_df[day2_summary_df["n"] == int(n_key)])

    if len(etas) == 0:
        raise ValueError(f"No overlapping eta3 values found between Day1 and Day2 at n={n_key}.")

    # Use categorical x positions for clean offsets.
    x0 = np.arange(len(etas), dtype=float)
    xticklabels = [rf"${e:g}$" for e in etas]

    # Offsets: day and pair are separated so points don't collide.
    off_day = {"Day1": -0.15, "Day2": +0.15}
    off_pair = {"signal": -0.05, "control": +0.05}

    # Layout: 1x2 (matches fig3_plot style)
    fig = plt.figure(figsize=fig_size_inches, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # Panel labels - mathtext
    axA.text(-0.05, 1.05, r"$\mathbf{(A)}$", transform=axA.transAxes, fontsize=16, va="bottom", ha="right")
    axB.text(-0.05, 1.05, r"$\mathbf{(B)}$", transform=axB.transAxes, fontsize=16, va="bottom", ha="right")

    # ------------------------------------------------------------------
    # Panel A
    # ------------------------------------------------------------------
    if metric_mode == "delta":
        for day_label, ddf in [("Day1", day1_delta_df), ("Day2", day2_delta_df)]:
            df = ddf[ddf["n"] == int(n_key)].set_index("eta3")
            df = df.loc[etas]
            y = df["delta_V_circ"].to_numpy(float)
            lo = df["delta_shotbs_ci_lo"].to_numpy(float)
            hi = df["delta_shotbs_ci_hi"].to_numpy(float)

            x = x0 + off_day[day_label]
            yerr = np.vstack([y - lo, hi - y])
            c = FIG9_DELTA_COLORS[day_label]
            axA.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=FIG9_DELTA_MARKER,
                markersize=FIG9_DELTA_MARKERSIZE,
                color=c,
                markerfacecolor=c,
                markeredgewidth=1.2,
                capsize=3,
                linewidth=1.2,
                label=rf"$\mathrm{{{day_label}}}$",
                zorder=10,
            )
            axA.plot(x, y, color=c, linewidth=1.2, zorder=9)

        axA.set_ylabel(r"$\Delta V_{\mathrm{circ}} = V_{\mathrm{circ}}(\mathrm{sig}) - V_{\mathrm{circ}}(\mathrm{ctl})$")

    else:
        pair_map = {
            "signal": f"({signal_pair[0]},{signal_pair[1]})",
            "control": f"({control_pair[0]},{control_pair[1]})",
        }
        for day_label, sdf in [("Day1", day1_summary_df), ("Day2", day2_summary_df)]:
            for role, pair_str in pair_map.items():
                df = sdf[(sdf["n"] == int(n_key)) & (sdf["pair"] == pair_str)].set_index("eta3")
                df = df.loc[etas]
                y = df["kappa_ctx_circ_var_point"].to_numpy(float)
                lo = df["kappa_ctx_circ_var_shotbs_ci95_lo"].to_numpy(float)
                hi = df["kappa_ctx_circ_var_shotbs_ci95_hi"].to_numpy(float)

                x = x0 + off_day[day_label] + off_pair[role]
                yerr = np.vstack([y - lo, hi - y])
                sty = FIG9_PAIR_STYLES[role]

                axA.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt=sty["marker"],
                    markersize=6,
                    color=FIG9_DAY_COLORS[day_label],
                    markerfacecolor=(FIG9_DAY_COLORS[day_label] if sty["markerfacecolor"] is None else sty["markerfacecolor"]),
                    markeredgewidth=1.2,
                    capsize=3,
                    linewidth=1.0,
                    zorder=10 if role == "signal" else 9,
                )
                axA.plot(
                    x,
                    y,
                    color=FIG9_DAY_COLORS[day_label],
                    linestyle=sty["linestyle"],
                    linewidth=1.2,
                    zorder=8,
                )

        axA.set_ylabel(r"$\kappa\mathrm{-profile\ dispersion}\ V_{\mathrm{circ}}$")

    axA.set_xlabel(r"$\eta_3$")
    axA.set_xticks(x0)
    axA.set_xticklabels(xticklabels)
    axA.yaxis.set_major_formatter(FuncFormatter(mathtext_formatter))

    # ------------------------------------------------------------------
    # Panel B (amp_min quality)
    # ------------------------------------------------------------------
    pair_map_amp = {"signal": f"({signal_pair[0]},{signal_pair[1]})"}
    if include_control_amp:
        pair_map_amp["control"] = f"({control_pair[0]},{control_pair[1]})"

    for day_label, sdf in [("Day1", day1_summary_df), ("Day2", day2_summary_df)]:
        for role, pair_str in pair_map_amp.items():
            df = sdf[(sdf["n"] == int(n_key)) & (sdf["pair"] == pair_str)].set_index("eta3")
            df = df.loc[etas]
            mean = df["amp_min_mean_ctx_trial"].to_numpy(float)
            q10 = df["amp_min_q10_ctx_trial"].to_numpy(float)

            x = x0 + off_day[day_label] + off_pair[role]
            c = FIG9_DAY_COLORS[day_label]
            sty = FIG9_PAIR_STYLES[role]

            # Whisker q10 -> mean (works even if q10 > mean in weird edge cases)
            cap = 0.06
            m = np.isfinite(x) & np.isfinite(mean) & np.isfinite(q10)
            axB.vlines(x[m], q10[m], mean[m], color=c, linewidth=1.0, alpha=0.95, zorder=8)
            axB.hlines(q10[m], x[m] - cap, x[m] + cap, color=c, linewidth=1.0, alpha=0.95, zorder=8)
            axB.plot(
                x,
                mean,
                color=c,
                linestyle=sty["linestyle"],
                marker=sty["marker"],
                markerfacecolor=(c if sty["markerfacecolor"] is None else sty["markerfacecolor"]),
                markeredgewidth=1.2,
                markersize=6,
                linewidth=1.2,
                zorder=10 if role == "signal" else 9,
            )

    axB.set_xlabel(r"$\eta_3$")
    axB.set_ylabel(r"$\mathrm{amp}_{\min}$")
    axB.set_xticks(x0)
    axB.set_xticklabels(xticklabels)
    axB.set_ylim(0.0, 1.1)
    axB.yaxis.set_major_formatter(FuncFormatter(mathtext_formatter))

    # ------------------------------------------------------------------
    # Legends (compact; avoid overloading Supplementary) - all mathtext
    # ------------------------------------------------------------------
    # Panel A: Day legend with delta colors/marker
    delta_day_handles = [
        Line2D([0], [0], color=FIG9_DELTA_COLORS["Day1"], marker=FIG9_DELTA_MARKER, linestyle="-", markersize=FIG9_DELTA_MARKERSIZE, label=r"$\mathrm{Day1}$"),
        Line2D([0], [0], color=FIG9_DELTA_COLORS["Day2"], marker=FIG9_DELTA_MARKER, linestyle="-", markersize=FIG9_DELTA_MARKERSIZE, label=r"$\mathrm{Day2}$"),
    ]

    # Panel B: Day legend with standard colors
    day_handles = [
        Line2D([0], [0], color=FIG9_DAY_COLORS["Day1"], marker="o", linestyle="-", markersize=6, label=r"$\mathrm{Day1}$"),
        Line2D([0], [0], color=FIG9_DAY_COLORS["Day2"], marker="o", linestyle="-", markersize=6, label=r"$\mathrm{Day2}$"),
    ]

    pair_handles = [
        Line2D([0], [0], color="#333333", marker=FIG9_PAIR_STYLES["signal"]["marker"], linestyle=FIG9_PAIR_STYLES["signal"]["linestyle"], markersize=6, label=rf"$\mathrm{{Signal}}\ ({signal_pair[0]},{signal_pair[1]})$"),
        Line2D([0], [0], color="#333333", marker=FIG9_PAIR_STYLES["control"]["marker"], linestyle=FIG9_PAIR_STYLES["control"]["linestyle"], markerfacecolor="white", markersize=6, label=rf"$\mathrm{{Control}}\ ({control_pair[0]},{control_pair[1]})$"),
    ]

    # Panel A: Day legend only (delta colors)
    axA.legend(handles=delta_day_handles, loc="upper left", frameon=False, framealpha=0.9, edgecolor="lightgray", fontsize=9, handlelength=2.5)

    # Panel B: Day + Pair legends side by side (fig5 style)
    leg_day = axB.legend(
        handles=day_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=False,
        fontsize=9,
        handlelength=2.5,
    )
    axB.add_artist(leg_day)

    axB.legend(
        handles=pair_handles,
        loc="lower left",
        bbox_to_anchor=(0.28, 0.0129),
        frameon=False,
        fontsize=9,
        handlelength=2.5,
    )

    # Save
    fig.savefig(out_pdf, bbox_inches="tight")
    if out_png:
        fig.savefig(out_png,  bbox_inches="tight", dpi=300)
    plt.close(fig)
