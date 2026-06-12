"""
plot_feature_score_corr.py — Spearman correlations between cover-letter features and scores.

For each evaluator model, computes ρ between every text/lexical/emotion/semantic feature
and the score assigned in cv_cl_evaluations and cl_evaluations.

Input  : output_eval/cl_features.parquet
         output_eval/master_df.parquet
Output : output_plots/cl_features/feature_score_corr.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from aggregate_plots import WRITER_COLORS, MODEL_DISPLAY, DISPLAY_COLORS, UNIQUE_EVALUATORS

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

FEATURES_PATH = "output_eval/cl_features_no_gemini2.parquet"
MASTER_PATH   = "output_eval/master_df_no_gemini2.parquet"
OUT_PATH      = "output_plots/cl_features_no_gemini2/feature_score_corr.png"
fs            = plt.rcParams["font.size"]

EVAL_TYPES = {
    "cv_cl_evaluations": "CV + Cover Letter",
    "cl_evaluations":    "Cover Letter Only",
}

FEATURE_GROUPS = [
    # (column,               display label,              group)
    ("word_count",           "Word Count",               "Length & Structure"),
    ("sentence_count",       "Sentence Count",           "Length & Structure"),
    ("paragraph_count",      "Paragraph Count",          "Length & Structure"),
    ("comma_count",          "Comma Count",              "Length & Structure"),
    ("avg_word_length",      "Avg Word Length",          "Language Complexity"),
    ("ttr",                  "Type-Token Ratio",         "Language Complexity"),
    ("flesch_reading_ease",  "Flesch Reading Ease",      "Language Complexity"),
    ("vader_compound",       "VADER Sentiment",          "Sentiment & Affect"),
    ("vad_valence",          "VAD Valence",              "Sentiment & Affect"),
    ("vad_arousal",          "VAD Arousal",              "Sentiment & Affect"),
    ("vad_dominance",        "VAD Dominance",            "Sentiment & Affect"),
    ("emo_joy",              "Joy",                      "Emotions"),
    ("emo_neutral",          "Neutral",                  "Emotions"),
    ("emo_surprise",         "Surprise",                 "Emotions"),
    ("emo_fear",             "Fear",                     "Emotions"),
    ("emo_anger",            "Anger",                    "Emotions"),
    ("emo_disgust",          "Disgust",                  "Emotions"),
    ("emo_sadness",          "Sadness",                  "Emotions"),
    ("job_cosine_sim",       "Cosine Sim. to Job Ad",    "Semantic Fit"),
    ("cv_cosine_sim",        "Cosine Sim. to CV",        "Semantic Fit"),
]

FEATURE_COLS   = [col   for col, _, _     in FEATURE_GROUPS]
FEATURE_LABELS = {col: lbl for col, lbl, _ in FEATURE_GROUPS}
FEATURE_GROUPS_MAP = {col: grp for col, _, grp in FEATURE_GROUPS}

# group boundary rows (for drawing separator lines)
GROUPS_ORDER = list(dict.fromkeys(grp for _, _, grp in FEATURE_GROUPS))


def compute_correlations(merged: pd.DataFrame) -> dict:
    """
    Returns {(eval_type, evaluator, feature_col): (rho, p_value)}.
    merged has columns: Eval_Type, Evaluator, Score, + all feature cols.
    """
    results = {}
    for (eval_type, evaluator), grp in merged.groupby(["Eval_Type", "Evaluator"]):
        for col in FEATURE_COLS:
            sub = grp[[col, "Score"]].dropna()
            if len(sub) < 10:
                results[(eval_type, evaluator, col)] = (np.nan, np.nan)
                continue
            rho, p = stats.spearmanr(sub[col], sub["Score"])
            results[(eval_type, evaluator, col)] = (rho, p)
    return results


def build_matrix(results, eval_type, evaluators):
    """Build rho and significance matrices for one eval_type."""
    rho_mat = pd.DataFrame(index=FEATURE_COLS,
                           columns=[MODEL_DISPLAY[e] for e in evaluators],
                           dtype=float)
    sig_mat = pd.DataFrame(index=FEATURE_COLS,
                           columns=[MODEL_DISPLAY[e] for e in evaluators],
                           dtype=bool)
    for evaluator in evaluators:
        for col in FEATURE_COLS:
            rho, p = results.get((eval_type, evaluator, col), (np.nan, np.nan))
            rho_mat.loc[col, MODEL_DISPLAY[evaluator]] = rho
            sig_mat.loc[col, MODEL_DISPLAY[evaluator]] = (p < 0.05) if not np.isnan(p) else False
    rho_mat.index = [FEATURE_LABELS[c] for c in FEATURE_COLS]
    sig_mat.index = [FEATURE_LABELS[c] for c in FEATURE_COLS]
    return rho_mat, sig_mat


def draw_heatmap(ax, rho_mat, sig_mat, title, show_yticklabels=True):
    annot = rho_mat.copy().astype(object)
    for row in rho_mat.index:
        for col in rho_mat.columns:
            v = rho_mat.loc[row, col]
            star = "*" if sig_mat.loc[row, col] else ""
            annot.loc[row, col] = f"{v:.2f}{star}" if not np.isnan(v) else ""

    sns.heatmap(rho_mat.astype(float), annot=annot, fmt="", ax=ax,
                cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                linewidths=0.4, linecolor="lightgrey",
                annot_kws={"size": fs + 1},
                cbar=False,
                yticklabels=show_yticklabels)

    ax.set_title(title, fontsize=fs + 6, pad=14)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                       fontsize=fs + 2, fontweight="bold")
    for tick in ax.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))

    if show_yticklabels:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs + 2)
        ax.set_ylabel("")
    else:
        ax.set_yticks([])

    # group separator lines + labels in data coordinates
    n = len(FEATURE_COLS)
    current_group, group_start = None, 0
    for i in range(n + 1):
        grp = FEATURE_GROUPS_MAP.get(FEATURE_COLS[i]) if i < n else None
        if grp != current_group:
            if i > 0:
                ax.axhline(i, color="black", linewidth=1.8)
            if show_yticklabels and current_group is not None:
                mid_y = (group_start + i) / 2
                ncols  = rho_mat.shape[1]
                ax.text(-0.6, mid_y, current_group,
                        ha="right", va="center",
                        fontsize=fs + 1, color="dimgrey", fontstyle="italic",
                        transform=ax.transData, clip_on=False)
            current_group = grp
            group_start   = i


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # ── load & prepare ──
    feat_df   = pd.read_parquet(FEATURES_PATH).drop(columns=["embedding"], errors="ignore")
    master_df = pd.read_parquet(MASTER_PATH)

    scores = (
        master_df[master_df["Eval_Type"].isin(EVAL_TYPES)]
        .groupby(["Job_ID", "Writer", "CV_Idx", "Evaluator", "Eval_Type"])["Score"]
        .mean()
        .reset_index()
    )
    merged = scores.merge(feat_df, on=["Job_ID", "Writer", "CV_Idx"])

    # ── correlations ──
    print("Computing Spearman correlations...")
    results = compute_correlations(merged)

    evaluators = [e for e in UNIQUE_EVALUATORS]

    # ── plot ──
    fig, axes = plt.subplots(1, 2, figsize=(30, 18))

    for ax, (eval_type, title) in zip(axes, EVAL_TYPES.items()):
        show_y = (ax is axes[0])
        rho_mat, sig_mat = build_matrix(results, eval_type, evaluators)
        draw_heatmap(ax, rho_mat, sig_mat, title, show_yticklabels=show_y)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                norm=plt.Normalize(vmin=-0.5, vmax=0.5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.015, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=fs + 2)
    cbar.ax.tick_params(labelsize=fs)

    fig.suptitle("Feature–Score Correlations by Evaluator Model  (* p < 0.05, uncorrected)\n"
                 r"$\it{Note:}$ Gemini 2.0 Flash has no evaluations for Claude Sonnet 4.6, "
                 r"Gemini 3.5 Flash, GPT-5 (model deprecated before those writers existed)",
                 fontsize=fs + 6)
    fig.tight_layout(rect=[0.05, 0, 0.97, 0.96])
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
