"""
plot_score_decomposition.py — Score predictability from structural factors.

Quantifies how much of the score variance is explained by:
  - Evaluator model
  - Writer model (cl / cv_cl conditions only; absent in cv_only)
  - Fit tier  (high-fit: CV_Idx 1–25 vs moderate-fit: CV_Idx 26–50)
  - Job ID

Outputs (saved to output_plots/cl_features_no_gemini2/):
  score_variance_decomposition.png  — OLS R² bar charts per eval_type
  tier_sensitivity_heatmap.png      — score gap (high-fit − moderate-fit) per evaluator × job
  writer_effect_analysis.png        — within-CV writer effects + partial R² comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aggregate_plots import MODEL_DISPLAY, DISPLAY_COLORS, UNIQUE_EVALUATORS, RAW_WRITERS

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

MASTER_PATH = "output_eval/master_df_no_gemini2.parquet"
OUT_DIR     = "output_plots/cl_features_no_gemini2"
fs          = plt.rcParams["font.size"]

EVAL_TYPES = {
    "cv_only":           "CV Only",
    "cv_cl_evaluations": "CV + Cover Letter",
    "cl_evaluations":    "Cover Letter Only",
}

JOB_LABELS = {
    "job_615":  "Construction",
    "job_1079": "HR Manager",
    "job_1169": "Teacher",
    "job_2561": "Accountant",
    "job_3373": "Web Dev",
    "job_3697": "UX Des. A",
    "job_3701": "UX Des. B",
    "job_3946": "CFO",
    "job_4142": "Sales Dir.",
    "job_4165": "Sales Mgr",
}

FACTOR_COLORS = {
    "Evaluator": "#1565C0",
    "Writer":    "#2E7D32",
    "Tier":      "#E65100",
    "Job":       "#6A1B9A",
}
COMBO_COLOR = "#546E7A"


# ── helpers ───────────────────────────────────────────────────────────────────

def r2_ols(df_sub, predictors):
    """OLS R² with dummy-encoded categorical predictors (numpy lstsq for stability)."""
    X = pd.get_dummies(df_sub[predictors], drop_first=True).astype(float).values
    y = df_sub["Score"].values
    if X.shape[1] == 0 or len(y) < X.shape[1] + 2:
        return np.nan
    X_int = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
    with np.errstate(all="ignore"):
        ss_r = ((y - X_int @ coef) ** 2).sum()
    ss_t = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_r / ss_t) if ss_t > 0 else np.nan


def predictor_sets(has_writer):
    """Ordered list of (label, [predictors], bar_color) from simple → full."""
    if has_writer:
        return [
            ("Evaluator",               ["Evaluator"],                               FACTOR_COLORS["Evaluator"]),
            ("Writer",                  ["Writer"],                                  FACTOR_COLORS["Writer"]),
            ("Tier",                    ["Tier"],                                    FACTOR_COLORS["Tier"]),
            ("Job",                     ["Job_ID"],                                  FACTOR_COLORS["Job"]),
            ("Eval + Writer",           ["Evaluator", "Writer"],                     COMBO_COLOR),
            ("Eval + Writer + Job",     ["Evaluator", "Writer", "Job_ID"],           COMBO_COLOR),
            ("Eval + Writer + Tier + Job (full)", ["Evaluator", "Writer", "Tier", "Job_ID"], COMBO_COLOR),
        ]
    else:
        return [
            ("Evaluator",               ["Evaluator"],                               FACTOR_COLORS["Evaluator"]),
            ("Tier",                    ["Tier"],                                    FACTOR_COLORS["Tier"]),
            ("Job",                     ["Job_ID"],                                  FACTOR_COLORS["Job"]),
            ("Eval + Tier",             ["Evaluator", "Tier"],                       COMBO_COLOR),
            ("Eval + Tier + Job (full)",["Evaluator", "Tier", "Job_ID"],             COMBO_COLOR),
        ]


# ── Plot 1: Variance decomposition ────────────────────────────────────────────

def plot_variance_decomposition(scores):
    n    = len(EVAL_TYPES)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5))

    for ax, (et, title) in zip(axes, EVAL_TYPES.items()):
        sub        = scores[scores["Eval_Type"] == et]
        has_writer = sub["Writer"].nunique() > 1
        sets       = predictor_sets(has_writer)

        labels = [s[0] for s in sets]
        values = [r2_ols(sub, s[1]) for s in sets]
        colors = [s[2] for s in sets]

        y_pos = list(range(len(labels)))
        ax.barh(y_pos,
                [v if not np.isnan(v) else 0 for v in values],
                color=colors, edgecolor="white", height=0.6)

        for i, val in enumerate(values):
            if not np.isnan(val):
                ax.text(val + 0.003, i, f"{val:.3f}",
                        va="center", ha="left",
                        fontsize=fs + 2, fontweight="bold")

        max_r2 = max((v for v in values if not np.isnan(v)), default=0.5)
        ax.set_xlim(0, max_r2 * 1.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=fs + 2)
        ax.set_xlabel("R²", fontsize=fs + 2)
        ax.set_title(title, fontsize=fs + 5, pad=10)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)

        # draw a separator line between single-predictor and combined bars
        n_single = 3 if not has_writer else 4
        ax.axhline(n_single - 0.5, color="black", linewidth=1.2,
                   linestyle="--", alpha=0.5)

    # shared colour legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=lbl)
               for lbl, c in {**FACTOR_COLORS, "Combined": COMBO_COLOR}.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=fs + 1, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Score Predictability from Structural Factors\n"
        "OLS R²  |  Unit = (evaluator, writer, CV, job) mean score across runs\n"
        "Dashed line separates individual factors (below) from combined models (above)",
        fontsize=fs + 5)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    out = os.path.join(OUT_DIR, "score_variance_decomposition.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out)}")


# ── Plot 2: Tier sensitivity heatmap ─────────────────────────────────────────

def plot_tier_sensitivity(scores):
    n    = len(EVAL_TYPES)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))

    for ax, (et, title) in zip(axes, EVAL_TYPES.items()):
        sub = scores[scores["Eval_Type"] == et]

        # mean score per (evaluator, job, tier) — averaged across writers and CV_Idx within tier
        tier_means = (sub.groupby(["Evaluator", "Job_ID", "Tier"])["Score"]
                      .mean().unstack("Tier"))

        if "high_fit" not in tier_means.columns or "mod_fit" not in tier_means.columns:
            ax.text(0.5, 0.5, "No tier data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=fs + 2)
            continue

        tier_means["gap"] = tier_means["high_fit"] - tier_means["mod_fit"]
        pivot = tier_means["gap"].unstack("Job_ID")

        ordered_evals = [e for e in UNIQUE_EVALUATORS if e in pivot.index]
        pivot = pivot.loc[ordered_evals]

        pivot.index   = [MODEL_DISPLAY.get(e, e) for e in pivot.index]
        pivot.columns = [JOB_LABELS.get(j, j) for j in pivot.columns]

        vabs  = max(np.nanmax(np.abs(pivot.values)), 0.01)
        annot = pivot.copy().astype(object)
        for r in pivot.index:
            for c in pivot.columns:
                v = pivot.loc[r, c]
                annot.loc[r, c] = f"{v:+.2f}" if not np.isnan(v) else ""

        sns.heatmap(pivot.astype(float), annot=annot, fmt="", ax=ax,
                    cmap="RdBu", center=0, vmin=-vabs, vmax=vabs,
                    linewidths=0.4, linecolor="lightgrey",
                    annot_kws={"size": fs + 1}, cbar=True)

        ax.set_title(f"{title}\nHigh-fit − Moderate-fit (score gap)", fontsize=fs + 4, pad=10)
        ax.set_xlabel("Job", fontsize=fs + 2)
        ax.set_ylabel("Evaluator" if ax is axes[0] else "", fontsize=fs + 2)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs + 1)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fs + 1, fontweight="bold")
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))

    fig.suptitle(
        "Tier Sensitivity: Do Evaluators Score High-Fit Candidates Higher?\n"
        "Blue = high-fit scored higher (correct discrimination)  |  "
        "Red = moderate-fit scored higher (inverted)\n"
        "For CL and CV+CL conditions, gap is averaged across writer models",
        fontsize=fs + 5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(OUT_DIR, "tier_sensitivity_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out)}")


# ── Plot 3: Writer effect — marginal means + partial R² ──────────────────────

ET_WRITER = ["cl_evaluations", "cv_cl_evaluations"]


def plot_writer_effect(scores):
    """
    Left/middle: within-CV demeaned writer effects (bar chart ± 95% CI).
    Right: standalone R²(Writer) vs partial R²(Writer | CV, Evaluator, Job).
    """
    sub = scores[scores["Eval_Type"].isin(ET_WRITER)].copy()

    # within-CV demeaning: remove (Evaluator × CV × Job) group mean
    grp_mean = sub.groupby(["Eval_Type", "Evaluator", "CV_Idx", "Job_ID"])["Score"].transform("mean")
    sub["writer_effect"] = sub["Score"] - grp_mean

    writer_agg = (sub.groupby(["Eval_Type", "Writer"])["writer_effect"]
                  .agg(mean="mean", sem=lambda x: x.sem())
                  .reset_index())

    # partial R²: residualise on (Evaluator + Job_ID + CV_Idx), then fit Writer
    standalone, partial = {}, {}
    for et in ET_WRITER:
        s = scores[scores["Eval_Type"] == et]
        y = s["Score"].values

        standalone[et] = r2_ols(s, ["Writer"])

        X_ctrl = pd.get_dummies(s[["Evaluator", "Job_ID", "CV_Idx"]],
                                drop_first=True).astype(float).values
        X_ctrl = np.column_stack([np.ones(len(X_ctrl)), X_ctrl])
        coef, _, _, _ = np.linalg.lstsq(X_ctrl, y, rcond=None)
        with np.errstate(all="ignore"):
            resid = y - X_ctrl @ coef

        X_wr = pd.get_dummies(s[["Writer"]], drop_first=True).astype(float).values
        X_wr = np.column_stack([np.ones(len(X_wr)), X_wr])
        coef_wr, _, _, _ = np.linalg.lstsq(X_wr, resid, rcond=None)
        with np.errstate(all="ignore"):
            pred_wr = X_wr @ coef_wr
        ss_r = ((resid - pred_wr) ** 2).sum()
        ss_t = ((resid - resid.mean()) ** 2).sum()
        partial[et] = float(1 - ss_r / ss_t) if ss_t > 0 else np.nan

    # layout: two wide bar panels + one narrow R² panel
    fig = plt.figure(figsize=(26, 6))
    gs  = fig.add_gridspec(1, 3, width_ratios=[2, 2, 1.2], wspace=0.38)
    ax_cl   = fig.add_subplot(gs[0])
    ax_cvcl = fig.add_subplot(gs[1])   # no sharey — each panel has its own order
    ax_r2   = fig.add_subplot(gs[2])

    for ax, et in [(ax_cl, "cl_evaluations"), (ax_cvcl, "cv_cl_evaluations")]:
        sub_et = (writer_agg[writer_agg["Eval_Type"] == et]
                  .sort_values("mean", ascending=True))   # lowest at bottom, highest at top
        disp   = [MODEL_DISPLAY.get(w, w) for w in sub_et["Writer"]]
        colors = [DISPLAY_COLORS.get(MODEL_DISPLAY.get(w, w), "grey") for w in sub_et["Writer"]]

        ax.barh(disp, sub_et["mean"].values,
                xerr=sub_et["sem"].values * 1.96,
                color=colors, edgecolor="white", height=0.65,
                error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "black"})
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

        for i, (_, row) in enumerate(sub_et.iterrows()):
            v = row["mean"]
            ax.text(v + (0.02 if v >= 0 else -0.02), i,
                    f"{v:+.2f}", va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=fs + 1, fontweight="bold")

        ax.set_xlabel("Score deviation from within-CV mean", fontsize=fs + 2)
        ax.set_title(EVAL_TYPES[et], fontsize=fs + 5, pad=10)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))

    # R² comparison
    x   = np.arange(len(ET_WRITER))
    w   = 0.35
    labels_r2 = [EVAL_TYPES[et] for et in ET_WRITER]
    ax_r2.bar(x - w / 2, [standalone[et] for et in ET_WRITER],
              width=w, color="#90A4AE", label="Standalone R²(Writer)")
    ax_r2.bar(x + w / 2, [partial[et] for et in ET_WRITER],
              width=w, color="#1565C0", label="Partial R²  (after CV, Eval, Job)")

    for i, et in enumerate(ET_WRITER):
        ax_r2.text(i - w / 2, standalone[et] + 0.001, f"{standalone[et]:.3f}",
                   ha="center", fontsize=fs + 2, fontweight="bold")
        ax_r2.text(i + w / 2, partial[et] + 0.001,   f"{partial[et]:.3f}",
                   ha="center", fontsize=fs + 2, fontweight="bold")

    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(labels_r2, fontsize=fs + 1, rotation=15, ha="right")
    ax_r2.set_ylabel("R²", fontsize=fs + 2)
    ax_r2.set_ylim(0, 0.06)
    ax_r2.set_title("Standalone vs Partial R²\n(partial = after CV, Eval, Job removed)",
                    fontsize=fs + 2, pad=8)
    ax_r2.legend(fontsize=fs - 1, frameon=False, loc="upper left")
    ax_r2.grid(axis="y", alpha=0.4)
    ax_r2.set_axisbelow(True)

    fig.suptitle(
        "Writer Model Effect on Score — Within-CV Analysis\n"
        "Per-writer bonus/penalty after removing CV quality, evaluator, and job effects  (±95% CI)",
        fontsize=fs + 5)
    fig.subplots_adjust(top=0.85)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(OUT_DIR, "writer_effect_analysis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    master_df = pd.read_parquet(MASTER_PATH)

    # average across runs → one score per (eval_type, evaluator, writer, CV_Idx, job)
    scores = (master_df
              .groupby(["Eval_Type", "Evaluator", "Writer", "CV_Idx", "Job_ID"])["Score"]
              .mean().reset_index())
    scores["Tier"] = np.where(scores["CV_Idx"] <= 25, "high_fit", "mod_fit")

    print("Plotting variance decomposition...")
    plot_variance_decomposition(scores)

    print("Plotting tier sensitivity...")
    plot_tier_sensitivity(scores)

    print("Plotting writer effect analysis...")
    plot_writer_effect(scores)

    print(f"\nDone — outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
