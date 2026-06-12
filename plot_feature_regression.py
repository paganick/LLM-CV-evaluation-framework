"""
plot_feature_regression.py — Feature-preference regression analysis.

For each evaluator, fits OLS(Score ~ std_features + job_FEs) to extract feature
weights. Then computes a predicted preference matrix (evaluator × writer) as the
dot product of those weights with each writer's mean feature profile, and compares
it to the observed score patterns.

Outputs:
  regression_coef_heatmap.png      — what each evaluator weights, per feature
  predicted_preference_matrix.png  — predicted vs actual preference (eval × writer)
  predicted_vs_actual_scatter.png  — scatter of predicted vs actual, self-pref highlighted
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sst
from sklearn.linear_model import RidgeCV

from aggregate_plots import (
    WRITER_COLORS, MODEL_DISPLAY, DISPLAY_COLORS,
    UNIQUE_EVALUATORS, RAW_WRITERS,
)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

FEATURES_PATH = "output_eval/cl_features_no_gemini2.parquet"
MASTER_PATH   = "output_eval/master_df_no_gemini2.parquet"
OUT_DIR       = "output_plots/cl_features_no_gemini2"
fs            = plt.rcParams["font.size"]

EVAL_TYPES = {
    "cv_cl_evaluations": "CV + Cover Letter",
    "cl_evaluations":    "Cover Letter Only",
}

FEATURE_GROUPS = [
    ("word_count",          "Word Count",            "Length & Structure"),
    ("sentence_count",      "Sentence Count",        "Length & Structure"),
    ("paragraph_count",     "Paragraph Count",       "Length & Structure"),
    ("comma_count",         "Comma Count",           "Length & Structure"),
    ("avg_word_length",     "Avg Word Length",       "Language Complexity"),
    ("ttr",                 "Type-Token Ratio",      "Language Complexity"),
    ("flesch_reading_ease", "Flesch Reading Ease",   "Language Complexity"),
    ("vader_compound",      "VADER Sentiment",       "Sentiment & Affect"),
    ("vad_valence",         "VAD Valence",           "Sentiment & Affect"),
    ("vad_arousal",         "VAD Arousal",           "Sentiment & Affect"),
    ("vad_dominance",       "VAD Dominance",         "Sentiment & Affect"),
    ("emo_joy",             "Joy",                   "Emotions"),
    ("emo_neutral",         "Neutral",               "Emotions"),
    ("emo_surprise",        "Surprise",              "Emotions"),
    ("emo_fear",            "Fear",                  "Emotions"),
    ("emo_anger",           "Anger",                 "Emotions"),
    ("emo_disgust",         "Disgust",               "Emotions"),
    ("emo_sadness",         "Sadness",               "Emotions"),
    ("job_cosine_sim",      "Cosine Sim. to Job Ad", "Semantic Fit"),
    ("cv_cosine_sim",       "Cosine Sim. to CV",     "Semantic Fit"),
]

FEATURE_COLS       = [col for col, _, _   in FEATURE_GROUPS]
FEATURE_LABELS     = {col: lbl for col, lbl, _ in FEATURE_GROUPS}
FEATURE_GROUPS_MAP = {col: grp for col, _, grp in FEATURE_GROUPS}

COSINE_COLS  = {"job_cosine_sim", "cv_cosine_sim"}
STYLE_COLS   = [c for c in FEATURE_COLS if c not in COSINE_COLS]


# ── regression ────────────────────────────────────────────────────────────────

def fit_regressions(merged, eval_type, feat_cols):
    """
    For each evaluator: Ridge(Score_res ~ std_features_res), where _res means
    within-job demeaned (partials out job fixed effects without including job
    dummies in the Ridge penalty).  RidgeCV picks alpha via leave-one-out CV.

    Returns:
      coef_df    : DataFrame (raw evaluator names × feat_cols), Ridge β
      feat_means : Series used to standardise writer profiles consistently
      feat_stds  : Series used to standardise writer profiles consistently
    """
    sub = merged[merged["Eval_Type"] == eval_type].copy()

    feat_means = sub[feat_cols].mean()
    feat_stds  = sub[feat_cols].std().replace(0, 1)
    sub[feat_cols] = (sub[feat_cols] - feat_means) / feat_stds

    alphas = np.logspace(-2, 3, 40)
    coef_rows = {}

    for evaluator in UNIQUE_EVALUATORS:
        ev = sub[sub["Evaluator"] == evaluator].dropna(
            subset=feat_cols + ["Score"]).copy()
        if len(ev) < 30:
            coef_rows[evaluator] = {c: np.nan for c in feat_cols}
            continue

        # partial out job fixed effects via within-job demeaning
        for col in feat_cols + ["Score"]:
            ev[col] = ev[col] - ev.groupby("Job_ID")[col].transform("mean")

        X = ev[feat_cols].values
        y = ev["Score"].values

        ridge = RidgeCV(alphas=alphas, fit_intercept=False)
        ridge.fit(X, y)
        coef_rows[evaluator] = dict(zip(feat_cols, ridge.coef_))

    coef_df = pd.DataFrame(coef_rows).T   # evaluators × features
    return coef_df, feat_means, feat_stds


def writer_profiles(feat_df, feat_means, feat_stds, feat_cols):
    """Mean standardised feature value per writer."""
    profiles = feat_df.groupby("Writer")[feat_cols].mean()
    return (profiles - feat_means) / feat_stds


def predicted_pref(coef_df, profiles, feat_cols):
    """
    Dot product: (n_eval × n_feat) @ (n_feat × n_writer) → (n_eval × n_writer).
    Returns DataFrame with MODEL_DISPLAY names.
    """
    writers    = [w for w in RAW_WRITERS    if w in profiles.index]
    evaluators = [e for e in UNIQUE_EVALUATORS if e in coef_df.index]
    C = coef_df.loc[evaluators, feat_cols].values
    P = profiles.loc[writers,   feat_cols].values
    pred = C @ P.T
    return pd.DataFrame(pred,
                        index=[MODEL_DISPLAY[e] for e in evaluators],
                        columns=[MODEL_DISPLAY[w] for w in writers])


def actual_pref(merged, eval_type):
    """
    Mean score per (evaluator, writer), mean-centred within evaluator
    (removes strictness/leniency differences).
    """
    sub = merged[merged["Eval_Type"] == eval_type]
    mat = sub.groupby(["Evaluator", "Writer"])["Score"].mean().unstack("Writer")
    mat = mat.sub(mat.mean(axis=1), axis=0)
    mat.index   = [MODEL_DISPLAY.get(e, e) for e in mat.index]
    mat.columns = [MODEL_DISPLAY.get(w, w) for w in mat.columns]
    return mat


# ── helpers ───────────────────────────────────────────────────────────────────

def _add_group_separators(ax, show_labels, feat_cols):
    """Horizontal separator lines + italic group labels on a feature-row heatmap."""
    n = len(feat_cols)
    current_group, group_start = None, 0
    for i in range(n + 1):
        grp = FEATURE_GROUPS_MAP.get(feat_cols[i]) if i < n else None
        if grp != current_group:
            if i > 0:
                ax.axhline(i, color="black", linewidth=1.8)
            if show_labels and current_group is not None:
                ax.text(-0.6, (group_start + i) / 2, current_group,
                        ha="right", va="center", fontsize=fs + 1,
                        color="dimgrey", fontstyle="italic",
                        transform=ax.transData, clip_on=False)
            current_group = grp
            group_start   = i


def _color_xticklabels(ax):
    for tick in ax.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))


def _color_yticklabels(ax):
    for tick in ax.get_yticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))


def _highlight_diagonal(ax, row_labels, col_labels):
    """Black border on cells where row label == col label."""
    for i, r in enumerate(row_labels):
        if r in col_labels:
            j = list(col_labels).index(r)
            ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                       fill=False, edgecolor="black", linewidth=2.5))


# ── Plot 1: regression coefficient heatmap ───────────────────────────────────

def _draw_coef_panel(ax, coef_disp, title, show_ylabels, feat_cols):
    """
    coef_disp : DataFrame (display evaluator names × feat_cols).
    Transposed to (features × evaluators) for the heatmap.
    """
    mat = coef_disp[feat_cols].T.copy()
    mat.index = [FEATURE_LABELS[c] for c in feat_cols]

    vals = mat.values[~np.isnan(mat.values)]
    vabs = min(np.abs(vals).max() if len(vals) else 1.0, 1.5)

    annot = mat.copy().astype(object)
    for r in mat.index:
        for c in mat.columns:
            v = mat.loc[r, c]
            annot.loc[r, c] = f"{v:.2f}" if not np.isnan(v) else ""

    sns.heatmap(mat.astype(float), annot=annot, fmt="", ax=ax,
                cmap="RdBu_r", center=0, vmin=-vabs, vmax=vabs,
                linewidths=0.4, linecolor="lightgrey",
                annot_kws={"size": fs + 1}, cbar=False,
                yticklabels=show_ylabels)

    ax.set_title(title, fontsize=fs + 6, pad=14)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                       fontsize=fs + 2, fontweight="bold")
    _color_xticklabels(ax)
    if show_ylabels:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs + 2)
        ax.set_ylabel("")
    else:
        ax.set_yticks([])
    _add_group_separators(ax, show_ylabels, feat_cols)


def plot_coef_heatmap(coef_by_type, feat_cols, suffix, title_note):
    fig, axes = plt.subplots(1, 2, figsize=(30, 18))
    for ax, (eval_type, title) in zip(axes, EVAL_TYPES.items()):
        _draw_coef_panel(ax, coef_by_type[eval_type], title, ax is axes[0], feat_cols)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-2.5, vmax=2.5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_label("Ridge β  (standardised features)", fontsize=fs + 2)
    cbar.ax.tick_params(labelsize=fs)

    fig.suptitle(
        f"Feature Weights per Evaluator  —  Ridge Regression with Job Fixed Effects{title_note}\n"
        "β = change in score per 1 SD increase in feature  |  "
        "Job effects partialled out via within-job demeaning  |  α chosen by LOO-CV",
        fontsize=fs + 6)
    fig.tight_layout(rect=[0.05, 0, 0.97, 0.96])
    out = os.path.join(OUT_DIR, f"regression_coef_heatmap{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out)}")


# ── Plot 2: predicted vs actual preference matrices ──────────────────────────

def _draw_pref_panel(ax, mat, title):
    vals = mat.values[~np.isnan(mat.values)]
    vabs = np.abs(vals).max() if len(vals) else 1.0

    annot = mat.copy().astype(object)
    for r in mat.index:
        for c in mat.columns:
            v = mat.loc[r, c]
            annot.loc[r, c] = f"{v:.2f}" if not np.isnan(v) else ""

    sns.heatmap(mat.astype(float), annot=annot, fmt="", ax=ax,
                cmap="RdBu_r", center=0, vmin=-vabs, vmax=vabs,
                linewidths=0.4, linecolor="lightgrey",
                annot_kws={"size": fs}, cbar=False)

    ax.set_title(title, fontsize=fs + 4, pad=10)
    ax.set_xlabel("Writer", fontsize=fs + 2)
    ax.set_ylabel("Evaluator", fontsize=fs + 2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                       fontsize=fs + 1, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontsize=fs + 1, fontweight="bold")
    _color_xticklabels(ax)
    _color_yticklabels(ax)
    _highlight_diagonal(ax, list(mat.index), list(mat.columns))


def plot_preference_matrices(pred_by_type, actual_by_type, suffix, title_note):
    fig, axes = plt.subplots(2, 2, figsize=(28, 20))

    writer_disp  = [MODEL_DISPLAY[w] for w in RAW_WRITERS]
    eval_disp    = [MODEL_DISPLAY[e] for e in UNIQUE_EVALUATORS]

    for col, (eval_type, title) in enumerate(EVAL_TYPES.items()):
        pred   = pred_by_type[eval_type]
        actual = actual_by_type[eval_type]

        writers = [d for d in writer_disp  if d in pred.columns and d in actual.columns]
        evals   = [d for d in eval_disp    if d in pred.index   and d in actual.index]

        _draw_pref_panel(axes[0, col], pred.loc[evals, writers],
                         f"Predicted — {title}")
        _draw_pref_panel(axes[1, col], actual.loc[evals, writers],
                         f"Actual score gap — {title}")

    fig.suptitle(
        f"Predicted vs Actual Preference Matrix{title_note}\n"
        "Predicted = evaluator feature weights · writer feature profile  "
        "| Actual = mean score, centred per evaluator\n"
        "Black border = self-evaluation  (evaluator model == writer model)",
        fontsize=fs + 5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, f"predicted_preference_matrix{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out)}")


# ── Plot 3: predicted vs actual scatter ──────────────────────────────────────

def plot_scatter(pred_by_type, actual_by_type, suffix, title_note):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, (eval_type, title) in zip(axes, EVAL_TYPES.items()):
        pred   = pred_by_type[eval_type]
        actual = actual_by_type[eval_type]

        rows = []
        for ev in pred.index:
            for wr in pred.columns:
                if wr not in actual.columns or ev not in actual.index:
                    continue
                pv = pred.loc[ev, wr]
                av = actual.loc[ev, wr]
                if np.isnan(pv) or np.isnan(av):
                    continue
                rows.append({"pred": pv, "actual": av,
                             "is_self": ev == wr, "eval": ev, "writer": wr})
        df = pd.DataFrame(rows)

        non_self = df[~df["is_self"]]
        self_df  = df[df["is_self"]]

        ax.scatter(non_self["pred"], non_self["actual"],
                   alpha=0.35, s=60, color="steelblue", label="Other-evaluation")
        ax.scatter(self_df["pred"], self_df["actual"],
                   alpha=0.9, s=140, color="firebrick", zorder=5,
                   marker="D", label="Self-evaluation")

        for _, row in self_df.iterrows():
            ax.annotate(row["eval"], (row["pred"], row["actual"]),
                        textcoords="offset points", xytext=(7, 4),
                        fontsize=fs - 1, color="firebrick", fontweight="bold")

        rho, p = sst.spearmanr(df["pred"], df["actual"])
        ax.set_title(f"{title}\nSpearman ρ = {rho:.2f}  (p = {p:.3f})",
                     fontsize=fs + 4)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Predicted preference  (feature alignment score)", fontsize=fs + 2)
        ax.set_ylabel("Actual score gap  (centred per evaluator)", fontsize=fs + 2)
        ax.legend(fontsize=fs)

    fig.suptitle(
        f"Does Feature Alignment Predict Score Gaps?{title_note}\n"
        "Each point = one (evaluator, writer) pair  |  "
        "Red diamonds = self-evaluations",
        fontsize=fs + 6)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"predicted_vs_actual_scatter{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    feat_df   = pd.read_parquet(FEATURES_PATH).drop(columns=["embedding"], errors="ignore")
    master_df = pd.read_parquet(MASTER_PATH)

    scores = (
        master_df[master_df["Eval_Type"].isin(EVAL_TYPES)]
        .groupby(["Job_ID", "Writer", "CV_Idx", "Evaluator", "Eval_Type"])["Score"]
        .mean()
        .reset_index()
    )
    merged = scores.merge(feat_df, on=["Job_ID", "Writer", "CV_Idx"])

    variants = [
        (FEATURE_COLS, "",       ""),
        (STYLE_COLS,   "_style", "  [style features only — cosine similarities excluded]"),
    ]

    for feat_cols, suffix, title_note in variants:
        print(f"\n── Running {'full' if not suffix else 'style-only'} feature set ──")

        coef_by_type   = {}
        pred_by_type   = {}
        actual_by_type = {}

        for eval_type in EVAL_TYPES:
            print(f"  Fitting regressions — {eval_type}...")
            coef_df, feat_means, feat_stds = fit_regressions(merged, eval_type, feat_cols)
            profiles = writer_profiles(feat_df, feat_means, feat_stds, feat_cols)

            coef_disp = coef_df.copy()
            coef_disp.index = [MODEL_DISPLAY.get(e, e) for e in coef_df.index]

            coef_by_type[eval_type]   = coef_disp
            pred_by_type[eval_type]   = predicted_pref(coef_df, profiles, feat_cols)
            actual_by_type[eval_type] = actual_pref(merged, eval_type)

        print("  Plotting...")
        plot_coef_heatmap(coef_by_type, feat_cols, suffix, title_note)
        plot_preference_matrices(pred_by_type, actual_by_type, suffix, title_note)
        plot_scatter(pred_by_type, actual_by_type, suffix, title_note)

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
