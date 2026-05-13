"""
agreement_corr_plots.py

Generates three evaluator-agreement correlation heatmaps → output_plots/paper_plots_v2/

  agreement_corr_cv_only.png
      Pairwise Pearson r between evaluators on raw CV-only scores.
      Measures baseline agreement on candidate quality.

  agreement_corr_STYLE_cl_evaluations.png
  agreement_corr_STYLE_cv_cl_evaluations.png
      Pairwise Pearson r of within-candidate score residuals
      (score minus that evaluator's mean across all writers for the same candidate).
      Removes candidate-quality variance; measures agreement on cover-letter style.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aggregate_plots import (
    BASE_DIR, OUT_DIR_V2, TIER_RANGE, CACHE_PATH,
    UNIQUE_EVALUATORS, TITLE_MAP,
    MODEL_DISPLAY, DISPLAY_COLORS,
    build_master_df,
)


def plot_agreement_corr_v2(tier_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 3

    def _build_corr(vecs):
        df = pd.DataFrame(vecs).dropna()
        return df.corr(method="pearson")

    def _draw(corr, title, fname):
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        sns.heatmap(corr, annot=corr.round(2).astype(str), fmt="", ax=ax,
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="gray",
                    annot_kws={"size": fs - 1})
        ax.set_title(title, fontsize=fs + 5, pad=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                           fontsize=fs, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=fs, fontweight="bold")
        for tick in ax.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ax.set_xlabel("Evaluator Model", fontsize=fs + 4)
        ax.set_ylabel("Evaluator Model", fontsize=fs + 4)
        cbar = ax.collections[0].colorbar
        cbar.set_label("Pearson r", fontsize=fs + 4)
        cbar.ax.tick_params(labelsize=fs)
        plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")

    # CV Only: raw score agreement
    cv_sub = tier_df[tier_df["Eval_Type"] == "cv_only"]
    if not cv_sub.empty:
        vecs = {
            MODEL_DISPLAY.get(ev, ev): (
                cv_sub[cv_sub["Evaluator"] == ev]
                .groupby(["Job_ID", "CV_Idx"])["Score"].mean()
            )
            for ev in UNIQUE_EVALUATORS
        }
        _draw(_build_corr(vecs),
              "Evaluator Agreement — CV Only\n(Pearson r of raw scores across candidates)",
              "agreement_corr_cv_only.png")

    # CL and CV+CL: style residual agreement
    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        sub = tier_df[tier_df["Eval_Type"] == etype].copy()
        if sub.empty:
            continue
        sub["Residual"] = sub["Score"] - sub.groupby(
            ["Evaluator", "Job_ID", "CV_Idx"])["Score"].transform("mean")
        vecs = {
            MODEL_DISPLAY.get(ev, ev): (
                sub[sub["Evaluator"] == ev]
                .groupby(["Job_ID", "CV_Idx", "Writer"])["Residual"].mean()
            )
            for ev in UNIQUE_EVALUATORS
        }
        _draw(_build_corr(vecs),
              f"Evaluator Style Agreement — {TITLE_MAP[etype]}\n"
              "(Pearson r of within-candidate score residuals)",
              f"agreement_corr_STYLE_{etype}.png")


def main():
    if not os.path.exists(BASE_DIR):
        print(f"ERROR: {BASE_DIR} does not exist.")
        return
    print("Loading master dataframe...")
    master_df = build_master_df(BASE_DIR)
    if master_df.empty:
        print("ERROR: no data found.")
        return
    start_cv, end_cv = TIER_RANGE
    tier_df = master_df[
        (master_df["CV_Idx"] >= start_cv) & (master_df["CV_Idx"] <= end_cv)
    ]
    print("Generating agreement correlation plots...")
    plot_agreement_corr_v2(tier_df, OUT_DIR_V2)
    print("Done.")


if __name__ == "__main__":
    main()
