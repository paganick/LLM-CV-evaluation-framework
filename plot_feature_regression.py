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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

# In the cover-letter-only condition, cv_cosine_sim (how closely the letter
# echoes the CV) cannot be something an evaluator is reacting to as an
# explanatory feature: the CV is never in that evaluator's prompt. Any Ridge
# coefficient the regression assigns it there would be a spurious correlate
# of something else the letter's CV-echo happens to track, not a genuine
# preference. Same reasoning already applied to the CV Consistency block in
# plot_model_explainability.py (_drop_cv_consistency) — mirrored here for
# the feature-level (not block-level) regressions in this module.
CL_ONLY_FEATURE_COLS = [c for c in FEATURE_COLS if c != "cv_cosine_sim"]

# Same category colour scheme already used for feature-category coding
# elsewhere (plot_model_explainability.BLOCK_COLORS) — duplicated here
# (rather than imported) because that module imports FROM this one, so
# importing it back would be circular. Job-Ad Fit / CV Consistency are the
# two Semantic Fit features individually, matching that module's granularity.
FEATURE_CATEGORY_COLORS = {
    "Job-Ad Fit":          "#424242",
    "CV Consistency":      "#AD1457",
    "Length & Structure":  "#1565C0",
    "Language Complexity": "#2E7D32",
    "Sentiment & Affect":  "#E65100",
    "Emotions":            "#6A1B9A",
}


def _feature_category_name(col):
    if col == "job_cosine_sim":
        return "Job-Ad Fit"
    if col == "cv_cosine_sim":
        return "CV Consistency"
    return FEATURE_GROUPS_MAP.get(col)


def _feature_name_color(col):
    return FEATURE_CATEGORY_COLORS.get(_feature_category_name(col), "black")


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
    """Gold fill + black border on cells where row label == col label."""
    for i, r in enumerate(row_labels):
        if r in col_labels:
            j = list(col_labels).index(r)
            ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                       facecolor="gold", alpha=0.25, zorder=1))
            ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                       fill=False, edgecolor="black", linewidth=2.5, zorder=2))


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
    ax.set_xlabel("Writer  (sorted by Avg, descending)", fontsize=fs + 2)
    ax.set_ylabel("Evaluator", fontsize=fs + 2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                       fontsize=fs + 1, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontsize=fs + 1, fontweight="bold")
    _color_xticklabels(ax)
    _color_yticklabels(ax)
    _highlight_diagonal(ax, list(mat.index), list(mat.columns))

    if mat.index[-1] == "Avg":
        ax.axhline(len(mat.index) - 1, color="black", linewidth=2.5)
        for tick in ax.get_yticklabels():
            if tick.get_text() == "Avg":
                tick.set_fontstyle("italic")
                tick.set_color("black")


def _with_avg_row_sorted(mat, order):
    """Reindex mat's columns by `order`, then append an 'Avg' row = column
    mean (mean across evaluators, i.e. common/universal preference for that
    writer under this metric)."""
    mat = mat[order]
    avg = mat.mean(axis=0)
    avg.name = "Avg"
    return pd.concat([mat, avg.to_frame().T])


def plot_preference_matrices(pred_by_type, actual_by_type, suffix, title_note):
    fig, axes = plt.subplots(3, 2, figsize=(28, 30))

    writer_disp  = [MODEL_DISPLAY[w] for w in RAW_WRITERS]
    eval_disp    = [MODEL_DISPLAY[e] for e in UNIQUE_EVALUATORS]

    for col, (eval_type, title) in enumerate(EVAL_TYPES.items()):
        pred   = pred_by_type[eval_type]
        actual = actual_by_type[eval_type]

        writers = [d for d in writer_disp  if d in pred.columns and d in actual.columns]
        evals   = [d for d in eval_disp    if d in pred.index   and d in actual.index]

        pred_sub   = pred.loc[evals, writers]
        actual_sub = actual.loc[evals, writers]
        residual   = actual_sub - pred_sub

        # order writers by average PREDICTED preference (descending), so the
        # same column order lines up across all three rows for this eval_type
        order = pred_sub.mean(axis=0).sort_values(ascending=False).index
        # evaluators (rows) follow that same rank order, restricted to the
        # subset that are also evaluators — keeps self-evaluation cells as
        # close to the diagonal as the row/column mismatch (9 vs 11) allows
        row_order = [w for w in order if w in evals]

        pred_sub   = pred_sub.loc[row_order]
        actual_sub = actual_sub.loc[row_order]
        residual   = residual.loc[row_order]

        pred_sub   = _with_avg_row_sorted(pred_sub, order)
        actual_sub = _with_avg_row_sorted(actual_sub, order)
        residual   = _with_avg_row_sorted(residual, order)

        _draw_pref_panel(axes[0, col], pred_sub,   f"Predicted — {title}")
        _draw_pref_panel(axes[1, col], actual_sub, f"Actual score gap — {title}")
        _draw_pref_panel(axes[2, col], residual,   f"Residual (actual − predicted) — {title}")

    fig.suptitle(
        f"Predicted vs Actual Preference Matrix{title_note}\n"
        "Predicted = evaluator feature weights · writer feature profile  "
        "| Actual = mean score, centred per evaluator  "
        "| Residual = actual − predicted\n"
        "Gold fill + black border = self-evaluation  |  "
        "Red residual = actual preference exceeds feature prediction  |  "
        "Blue residual = features over-predict preference",
        fontsize=fs + 5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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

        rho, p_sp  = sst.spearmanr(df["pred"], df["actual"])
        r, _       = sst.pearsonr(df["pred"],  df["actual"])
        r2         = r ** 2

        # OLS fit line
        m, b = np.polyfit(df["pred"], df["actual"], 1)
        x_range = np.linspace(df["pred"].min(), df["pred"].max(), 200)
        ax.plot(x_range, m * x_range + b, color="navy", linewidth=1.5,
                linestyle="-", alpha=0.6, zorder=3, label="OLS fit")

        ax.set_title(
            f"{title}\n"
            f"Pearson r = {r:.2f}  |  R² = {r2:.2f}  |  Spearman ρ = {rho:.2f}  (p = {p_sp:.3f})",
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


def _plot_contribution_heatmap(contrib, feat_cols, title, subtitle, out_name, vabs=None):
    """
    contrib: DataFrame, models (index, display names) x feat_cols (columns, raw codes).
    Plots features (rows, grouped) x models (columns), with a bottom "Total" row
    (= column sums) and a right "Avg |contrib|" column (= mean absolute row value).
    """
    order = contrib.sum(axis=1).sort_values(ascending=False).index
    contrib = contrib.loc[order]

    avg_abs = contrib[feat_cols].abs().mean(axis=0)
    mat = contrib[feat_cols].T.copy()
    mat.index = [FEATURE_LABELS[c] for c in feat_cols]
    mat["Avg |contrib|"] = avg_abs.values
    total = mat.iloc[:, :-1].sum(axis=0)
    total.name = "Total"
    mat = pd.concat([mat, total.to_frame().T])

    if vabs is None:
        vabs = mat.iloc[:-1, :-1].abs().to_numpy().max()

    fig, ax = plt.subplots(figsize=(1.1 * mat.shape[1] + 4, 0.55 * mat.shape[0] + 3))
    annot = mat.round(2).astype(str)
    sns.heatmap(mat, annot=annot, fmt="", ax=ax,
                cmap="RdBu_r", center=0, vmin=-vabs, vmax=vabs,
                linewidths=0.4, linecolor="lightgrey",
                annot_kws={"size": fs - 1},
                cbar_kws={"label": "Contribution to preference score", "shrink": 0.7})
    ax.set_title(f"{title}\n{subtitle}", fontsize=fs + 4, pad=12)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right", fontsize=fs)
    for tick in ax.get_xticklabels():
        label = tick.get_text()
        tick.set_color(DISPLAY_COLORS.get(label, "black"))
        tick.set_fontweight("bold" if label not in ("Avg |contrib|",) else "normal")
        if label == "Avg |contrib|":
            tick.set_fontstyle("italic")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs, rotation=0)
    for tick in ax.get_yticklabels():
        if tick.get_text() == "Total":
            tick.set_fontweight("bold")
            tick.set_fontstyle("italic")

    n_feat_rows = len(feat_cols)
    _add_group_separators(ax, show_labels=True, feat_cols=feat_cols)
    ax.axvline(mat.shape[1] - 1, color="black", linewidth=2.5)
    ax.axhline(n_feat_rows, color="black", linewidth=2.5)

    fig.tight_layout(rect=[0.12, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


def plot_common_preference_drivers(coef_df, profiles, feat_cols,
                                    out_name="common_preference_drivers.png", title_note=""):
    """
    Decomposes each writer's common-preference score into per-feature contributions:
    (average evaluator taste) x (writer's own standardized style), so it's clear
    WHICH features drive a writer being broadly liked or disliked by evaluators
    in general — independent of who is judging. The "Total" row is exactly the
    answer to "do the best writers dominate in the features evaluators reward
    the most universally?": it's each writer's style profile dotted with the
    average evaluator's taste, sorted descending left to right.
    """
    common_models = [m for m in UNIQUE_EVALUATORS if m in coef_df.index and m in profiles.index]
    mean_taste = coef_df.loc[common_models, feat_cols].mean(axis=0)
    writers = [w for w in RAW_WRITERS if w in profiles.index]

    contrib = profiles.loc[writers, feat_cols].mul(mean_taste, axis=1)
    contrib.index = [MODEL_DISPLAY[w] for w in writers]

    _plot_contribution_heatmap(
        contrib, feat_cols,
        f"What Drives Common (Universal) Preference for Each Writer?{title_note}",
        "Contribution = avg-evaluator taste x writer's own style  |  "
        "Total (bottom) = each writer's common-preference score  |  "
        "Avg |contrib| (right) = how much that feature matters on average",
        out_name)


def plot_self_preference_drivers(coef_df, profiles, feat_cols):
    """
    Decomposes each model's idiosyncratic self-preference into per-feature
    contributions: (own taste - avg taste) x (own standardized style).
    """
    common_models = [m for m in UNIQUE_EVALUATORS if m in coef_df.index and m in profiles.index]
    mean_taste = coef_df.loc[common_models, feat_cols].mean(axis=0)

    contrib = pd.DataFrame({
        MODEL_DISPLAY[m]: (coef_df.loc[m, feat_cols] - mean_taste) * profiles.loc[m, feat_cols]
        for m in common_models
    }).T

    _plot_contribution_heatmap(
        contrib, feat_cols,
        "What Drives Idiosyncratic Self-Preference for Each Model?",
        "Contribution = (own taste − avg taste) x own style  |  "
        "Total (bottom) = each model's idiosyncratic self-preference score",
        "self_preference_drivers.png")


def compute_pairwise_contributions(coef_df, profiles, feat_cols):
    """contribution[writer][evaluator] = Series over feat_cols of
    profile[writer, f] * coef[evaluator, f] — the full (writer x evaluator x
    feature) breakdown that predicted_pref/plot_common_preference_drivers
    only ever summarize (by averaging over evaluators or over features)."""
    common_models = [m for m in UNIQUE_EVALUATORS if m in coef_df.index]
    writers = [w for w in RAW_WRITERS if w in profiles.index]
    return {
        w: {m: profiles.loc[w, feat_cols] * coef_df.loc[m, feat_cols] for m in common_models}
        for w in writers
    }


def plot_divergence_dotplot(pairwise, out_name, top_n=25, title_note=""):
    """
    Forest/dot-plot of the most divergent (writer, feature) pairs: one row
    each, with one dot per evaluator showing its actual SIGNED contribution
    (writer's style x that evaluator's taste). Unlike a std-based heatmap,
    this keeps direction visible — you can see which specific evaluators are
    for/against a feature, not just that they disagree by some amount.
    """
    rows = []
    for w, by_eval in pairwise.items():
        per_eval = pd.DataFrame(by_eval).T  # evaluators x features
        for f in per_eval.columns:
            col = per_eval[f]
            rows.append({
                "writer_display": MODEL_DISPLAY[w],
                "label": f"{MODEL_DISPLAY[w]} — {FEATURE_LABELS[f]}",
                "spread": col.max() - col.min(),
                "values": col,
            })
    rows = sorted(rows, key=lambda r: r["spread"], reverse=True)[:top_n][::-1]

    evaluators = list(rows[0]["values"].index)
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(rows) + 2))
    for i, r in enumerate(rows):
        for m, v in r["values"].items():
            is_self = MODEL_DISPLAY[m] == r["writer_display"]
            ax.scatter(v, i, color=DISPLAY_COLORS.get(MODEL_DISPLAY[m], "grey"),
                       s=100, edgecolor="white", linewidth=0.6, zorder=(4 if is_self else 3),
                       alpha=(1.0 if is_self else 0.3))
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=fs)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(DISPLAY_COLORS.get(r["writer_display"], "black"))
        tick.set_fontweight("bold")
    ax.set_xlabel("Contribution to predicted preference  (writer's style x evaluator's taste)",
                  fontsize=fs + 1)
    ax.set_title(f"Most Divergent (Writer, Feature) Pairs — Each Dot Is One Evaluator's Signed Preference "
                 f"(opaque = self-evaluation){title_note}",
                 fontsize=fs + 3, pad=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=DISPLAY_COLORS.get(MODEL_DISPLAY[m], "grey"), label=MODEL_DISPLAY[m])
               for m in evaluators]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=fs - 1,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


def plot_category_dotplot(pairwise, feat_cols, category, out_path, xlim, feat_df):
    """
    Same dot-plot idea as plot_divergence_dotplot, but exhaustive rather than
    top-N: every writer x every feature in one category, grouped by feature
    (separator lines between blocks) and sorted within each feature block by
    mean contribution across evaluators (most-helped writer at top, most-hurt
    at bottom). xlim is passed in and shared across all category plots, so
    the size of the dots' spread is directly comparable category to category.

    Self-evaluation dots (evaluator == the row's writer) are drawn opaque;
    every other evaluator's dot is faded, so self-preference stands out.

    A second panel shows a box plot of that feature's actual raw values per
    writer (500 letters) — one small subplot per feature, stacked to line up
    with that feature's block on the left, each with its OWN x-axis (features
    in one category can have very different natural units, e.g. word count
    vs. comma count, so a single shared scale would flatten most of them).
    """
    writers = [w for w in RAW_WRITERS if w in pairwise]
    blocks_ordered = {}  # feature -> [(writer, per_eval_series), ...] sorted descending by mean
    rows = []
    for f in feat_cols:
        feat_vals = []
        for w in writers:
            per_eval = pd.DataFrame(pairwise[w]).T[f]  # Series indexed by evaluator
            feat_vals.append((w, per_eval))
        feat_vals.sort(key=lambda t: t[1].mean(), reverse=True)
        blocks_ordered[f] = feat_vals
        for w, per_eval in feat_vals:
            rows.append({
                "writer": w,
                "feature": f,
                "writer_display": MODEL_DISPLAY[w],
                "label": f"{MODEL_DISPLAY[w]} — {FEATURE_LABELS[f]}",
                "values": per_eval,
            })
    block_starts = [i * len(writers) for i in range(len(feat_cols))]
    rows = rows[::-1]
    n = len(rows)
    block_starts = [n - 1 - b for b in block_starts]  # convert to reversed-plot coordinates

    evaluators = list(rows[0]["values"].index)
    body_h, top_h, legend_h = 0.4 * n, 1.2, 1.0
    total_h = body_h + top_h + legend_h
    fig = plt.figure(figsize=(18, total_h))
    top_frac, bottom_frac = 1 - top_h / total_h, legend_h / total_h
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.12,
                          top=top_frac, bottom=bottom_frac, left=0.32, right=0.98)
    ax = fig.add_subplot(gs[0, 0])
    box_gs = gs[0, 1].subgridspec(len(feat_cols), 1, hspace=0.35)
    box_axes = [fig.add_subplot(box_gs[i, 0]) for i in range(len(feat_cols))]

    for i, r in enumerate(rows):
        is_self = {m: MODEL_DISPLAY[m] == r["writer_display"] for m in r["values"].index}
        for m, v in r["values"].items():
            ax.scatter(v, i, color=DISPLAY_COLORS.get(MODEL_DISPLAY[m], "grey"),
                       s=100, edgecolor="white", linewidth=0.6, zorder=(4 if is_self[m] else 3),
                       alpha=(1.0 if is_self[m] else 0.3))
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    for b in block_starts[1:]:
        ax.axhline(b + 0.5, color="grey", linewidth=1.2, alpha=0.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("Contribution to predicted preference  (writer's style x evaluator's taste)",
                  fontsize=fs + 1)
    ax.set_title("Predicted preference per evaluator\n(opaque = self-evaluation)", fontsize=fs + 2, pad=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=fs)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(DISPLAY_COLORS.get(r["writer_display"], "black"))
        tick.set_fontweight("bold")

    for bi, f in enumerate(feat_cols):
        ax_sub = box_axes[bi]
        feat_vals = blocks_ordered[f]  # already top-to-bottom order
        raw_by_writer = [feat_df.loc[feat_df["Writer"] == w, f].dropna().values for w, _ in feat_vals]
        local_y = list(range(len(feat_vals) - 1, -1, -1))  # idx0 (top) -> highest local y
        for (w, _), y_local, vals in zip(feat_vals, local_y, raw_by_writer):
            box_color = DISPLAY_COLORS.get(MODEL_DISPLAY[w], "grey")
            ax_sub.boxplot(vals, positions=[y_local], vert=False, widths=0.6, patch_artist=True,
                          showfliers=False,
                          boxprops=dict(facecolor=box_color, alpha=0.5, edgecolor=box_color),
                          medianprops=dict(color="black", linewidth=1.5),
                          whiskerprops=dict(color=box_color), capprops=dict(color=box_color))
        ax_sub.set_ylim(-0.5, len(feat_vals) - 0.5)
        ax_sub.set_yticks([])
        all_vals = np.concatenate(raw_by_writer)
        pad = (all_vals.max() - all_vals.min()) * 0.1 or 1
        ax_sub.set_xlim(all_vals.min() - pad, all_vals.max() + pad)
        ax_sub.set_xlabel(FEATURE_LABELS[f], fontsize=fs)
        ax_sub.tick_params(axis="x", labelsize=fs - 1)
        ax_sub.grid(axis="x", alpha=0.3)
        ax_sub.set_axisbelow(True)
    box_axes[0].set_title("Actual distribution of raw values\n(each feature has its own scale)",
                          fontsize=fs + 2, pad=10)

    fig.suptitle(f"{category}: Every Writer x Every Feature", fontsize=fs + 4, y=1.0)

    handles = [Patch(facecolor=DISPLAY_COLORS.get(MODEL_DISPLAY[m], "grey"), label=MODEL_DISPLAY[m])
               for m in evaluators]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=fs - 1,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_all_category_dotplots(pairwise, feat_df, out_dir):
    """One exhaustive dot-plot per feature category, all on the same x-axis
    scale so relative importance is comparable across categories at a glance."""
    os.makedirs(out_dir, exist_ok=True)

    categories = list(dict.fromkeys(grp for _, _, grp in FEATURE_GROUPS))
    cat_features = {cat: [c for c, _, g in FEATURE_GROUPS if g == cat] for cat in categories}

    all_vals = [v for w in pairwise for e in pairwise[w] for v in pairwise[w][e].values]
    vmax = max(abs(min(all_vals)), abs(max(all_vals))) * 1.05
    xlim = (-vmax, vmax)

    slug = {"Semantic Fit": "semantic_fit", "Length & Structure": "length_structure",
            "Language Complexity": "language_complexity", "Sentiment & Affect": "sentiment_affect",
            "Emotions": "emotions"}
    for cat in categories:
        out_path = os.path.join(out_dir, f"{slug.get(cat, cat.lower().replace(' & ', '_').replace(' ', '_'))}.png")
        plot_category_dotplot(pairwise, cat_features[cat], cat, out_path, xlim, feat_df)


def plot_self_points_summary(pairwise, feat_cols, feat_df, out_path):
    """
    One row per feature (all categories, in the same order as the per-category
    dot-plots, with group separators) — but unlike those plots, only the SELF
    point is drawn: for feature f, one dot per model that is both a writer and
    an evaluator, at that model's own contribution to its own predicted
    preference (profile[w, f] x coef[w, f]). There is no faded/opaque
    distinction here since every dot already is a self-evaluation.

    Two independent visual channels, each answering a different question:
      - x-position   : self contribution to predicted preference (own style x
                        own taste) — same quantity as before.
      - dot size      : how EXTREME that model's own writing is on this
                        feature, as an effect size: |own mean - feature's
                        overall mean| divided by that feature's own
                        population std (letter-to-letter variability) — the
                        same standardisation logic as the Cohen's d
                        tier-effect heatmap elsewhere in this project. Sizes
                        are scaled with ONE global min-max across every
                        (feature, model) pair, not per row, so a feature
                        where all models cluster tightly (e.g. VADER
                        Sentiment) comes out uniformly small, and a feature
                        with real spread (e.g. Cosine Sim. to Job Ad) shows
                        a real range — size is comparable feature to
                        feature, not just within one row.
      - marker shape  : ^ if this model's own mean is ABOVE the feature's
                        overall mean (across all writers), v if below — the
                        *direction* of that extremity, which size alone
                        collapses to a magnitude.

    (A third channel, marker border colour for the sign of the raw feature
    value, was tried and dropped: nearly every feature here is strictly
    positive by construction — only VAD Arousal ever goes negative, and
    uniformly so across every model — so the signal was real but too sparse
    to be worth a whole legend entry.)
    """
    dual_role = [w for w in RAW_WRITERS if w in UNIQUE_EVALUATORS]

    # Pass 1: for every (feature, model), an effect-size-style extremity —
    # |own mean - feature's overall mean| standardised by that feature's own
    # population std (letter-to-letter variability), the same logic used for
    # Cohen's d elsewhere in this project. Centring on the MEAN rather than
    # (min+max)/2 matters: several features (e.g. VADER Sentiment) have a
    # long low tail from a handful of outlier letters, which drags the
    # min-max midpoint far from where almost every letter actually sits and
    # would make every model look artificially "extreme."
    per_feat = {}
    for f in feat_cols:
        all_vals = feat_df[f].dropna().values
        feat_mean = float(all_vals.mean())
        pop_std = float(all_vals.std()) or 1.0

        own_mean, effect = {}, {}
        for w in dual_role:
            vals = feat_df.loc[feat_df["Writer"] == w, f].dropna().values
            m = float(vals.mean()) if len(vals) else np.nan
            own_mean[w] = m
            effect[w] = abs(m - feat_mean) / pop_std
        per_feat[f] = {"feat_mean": feat_mean, "own_mean": own_mean, "effect": effect}

    # Pass 2: ONE global min-max over every (feature, model) effect size, so
    # a feature where all models cluster tightly comes out with uniformly
    # small dots, and a feature with real spread shows a real size range —
    # sizes are comparable feature to feature, not just within a row.
    all_effects = [e for d in per_feat.values() for e in d["effect"].values()]
    gmin, gmax = min(all_effects), max(all_effects)

    def to_size(e):
        return 30.0 if gmax == gmin else 30.0 + 470.0 * (e - gmin) / (gmax - gmin)

    rows = []
    for f in feat_cols:
        d = per_feat[f]
        rows.append({
            "feature": f,
            "label": FEATURE_LABELS[f],
            "points": {w: (float(pairwise[w][w][f]), to_size(d["effect"][w]),
                           "^" if d["own_mean"][w] >= d["feat_mean"] else "v")
                       for w in dual_role},
        })

    n = len(rows)
    BIG = 5  # extra points added to every font size in this figure
    fig, ax = plt.subplots(figsize=(16, 0.5 * n + 1.4))
    for i, r in enumerate(rows[::-1]):
        for w, (v, s, marker) in r["points"].items():
            ax.scatter(v, i, s=s, marker=marker,
                       color=DISPLAY_COLORS.get(MODEL_DISPLAY[w], "grey"),
                       edgecolor="white", linewidth=0.7, alpha=0.9, zorder=3)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)

    _add_group_separators_by_feat(ax, [r["feature"] for r in rows[::-1]])

    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows[::-1]], fontsize=fs + BIG)
    ax.tick_params(axis="x", labelsize=fs + BIG - 1)
    ax.set_xlabel("Self contribution to predicted preference  "
                  "(own style x own taste)", fontsize=fs + BIG)
    ax.set_title("How Each Model's Own Writing Fares Under Its Own Evaluation Taste\n"
                 "(size $\\propto$ effect size of own mean vs. the feature's overall mean — "
                 "see legend for shape/size)",
                 fontsize=fs + BIG + 1, pad=14)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # Shape and size legends live OUTSIDE the axes, to its right — nearly
    # every row has real dots across almost the full x-range (not just a
    # couple of outlier rows), so there's no empty region left to safely
    # inset a legend into without covering data.
    shape_handles = [
        Line2D([0], [0], marker="^", linestyle="none", markersize=13,
               markerfacecolor="grey", markeredgecolor="grey",
               label="Above feature's mean"),
        Line2D([0], [0], marker="v", linestyle="none", markersize=13,
               markerfacecolor="grey", markeredgecolor="grey",
               label="Below feature's mean"),
    ]
    # fig-level (not ax-level) legends with figure-fraction anchors are what
    # reliably survive bbox_inches="tight" in this codebase (see the model
    # legend below) — ax-level legends anchored past the axes' own bounds
    # via bbox_transform=ax.transAxes were getting clipped at save time.
    shape_legend = fig.legend(handles=shape_handles, loc="upper left", fontsize=fs + BIG - 2,
                              frameon=True, framealpha=0.9, borderpad=0.9,
                              bbox_to_anchor=(0.835, 0.91),
                              title="Own mean, vs.", title_fontsize=fs + BIG - 1)
    fig.add_artist(shape_legend)

    # Size legend: three reference effect sizes spanning the data's own
    # range, so the labels are the real min/mid/max rather than round
    # numbers that may not occur in the data.
    ref_effects = [gmin, (gmin + gmax) / 2, gmax]
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markersize=(to_size(e) ** 0.5) * 0.85,
               markerfacecolor="grey", markeredgecolor="white",
               label=f"{e:.2f}")
        for e in ref_effects
    ]
    size_legend = fig.legend(handles=size_handles, loc="upper left", fontsize=fs + BIG - 2,
                             frameon=True, framealpha=0.9, borderpad=0.9, labelspacing=1.4,
                             bbox_to_anchor=(0.835, 0.72),
                             title="Effect size", title_fontsize=fs + BIG - 1)
    fig.add_artist(size_legend)

    model_handles = [Patch(facecolor=DISPLAY_COLORS.get(MODEL_DISPLAY[w], "grey"), label=MODEL_DISPLAY[w])
                      for w in dual_role]
    fig.legend(handles=model_handles, loc="lower center", ncol=3, fontsize=fs + BIG - 2,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.09, 0.82, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out_path)}")


def plot_self_points_summary_ranked(pairwise, feat_cols, feat_df, out_path, title_note="",
                                     legend_side="left"):
    """
    Same figure as plot_self_points_summary (a separate output, not a
    replacement), with two differences, mirroring what
    plot_common_preference_summary_ranked does to plot_common_preference_summary:

      - Rows are sorted by the MAXIMUM |self contribution| across all nine
        dual-role models (descending, most-extreme-at-top), not grouped by
        feature category. A large positive value on this axis is genuine
        self-preference; a large negative value is the opposite — a model
        actively penalising itself for a trait it leans into — which is
        just as interesting a signal, hence ranking by |value| rather than
        by the signed value or by feature category.
      - Feature-name row labels are coloured by category (same scheme as
        plot_common_preference_summary_ranked / plot_model_explainability's
        BLOCK_COLORS), since category grouping no longer holds the rows
        together visually once they're re-sorted.
    """
    dual_role = [w for w in RAW_WRITERS if w in UNIQUE_EVALUATORS]

    per_feat = {}
    for f in feat_cols:
        all_vals = feat_df[f].dropna().values
        feat_mean = float(all_vals.mean())
        pop_std = float(all_vals.std()) or 1.0

        own_mean, effect = {}, {}
        for w in dual_role:
            vals = feat_df.loc[feat_df["Writer"] == w, f].dropna().values
            m = float(vals.mean()) if len(vals) else np.nan
            own_mean[w] = m
            effect[w] = abs(m - feat_mean) / pop_std
        per_feat[f] = {"feat_mean": feat_mean, "own_mean": own_mean, "effect": effect}

    all_effects = [e for d in per_feat.values() for e in d["effect"].values()]
    gmin, gmax = min(all_effects), max(all_effects)

    def to_size(e):
        return 30.0 if gmax == gmin else 30.0 + 470.0 * (e - gmin) / (gmax - gmin)

    rows = []
    for f in feat_cols:
        d = per_feat[f]
        points = {w: (float(pairwise[w][w][f]), to_size(d["effect"][w]),
                      "^" if d["own_mean"][w] >= d["feat_mean"] else "v")
                  for w in dual_role}
        max_abs = max(abs(v) for v, _, _ in points.values())
        rows.append({"feature": f, "label": FEATURE_LABELS[f], "points": points, "max_abs": max_abs})

    # Ascending, plotted directly (no reversal) — largest |value| ends up at
    # the top, same leaderboard convention as the common-preference ranking.
    rows.sort(key=lambda r: r["max_abs"])

    n = len(rows)
    BIG = 10
    fig, ax = plt.subplots(figsize=(16, 0.5 * n + 1.4))
    for i, r in enumerate(rows):
        for w, (v, s, marker) in r["points"].items():
            ax.scatter(v, i, s=s, marker=marker,
                       color=DISPLAY_COLORS.get(MODEL_DISPLAY[w], "grey"),
                       edgecolor="white", linewidth=0.7, alpha=0.9, zorder=3)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)

    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=fs + BIG)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(_feature_name_color(r["feature"]))
        tick.set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=fs + BIG - 1)
    ax.set_xlabel("Self contribution to predicted preference  "
                  "(own style x own taste)", fontsize=fs + BIG - 3)
    ax.set_title("How Each Model's Own Writing Fares Under Its Own Evaluation Taste"
                 f"{title_note}\n(sorted by max |self contribution| across models)",
                 fontsize=fs + BIG + 1, pad=14)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # All three marker legends now live INSIDE the axes: the low-ranked
    # rows at the bottom only ever have small, near-zero dots, leaving both
    # bottom corners mostly empty (which corner is actually clearer can
    # flip depending on the data, hence legend_side). Category and shape
    # (just above it) stack on one side; effect size sits in the other
    # bottom corner, clear of both.
    SMALL = fs + BIG - 7  # these three legends stay compact so they fit in
                          # the corners without covering the smaller dots
                          # that still populate the low-ranked rows
    side_loc  = "lower left" if legend_side == "left" else "lower right"
    other_loc = "lower right" if legend_side == "left" else "lower left"
    anchor_x  = 0.0 if legend_side == "left" else 1.0

    # Category legend goes in first, anchored to the true bottom corner on
    # the chosen side; shape legend is then placed using category's ACTUAL
    # rendered height (via get_window_extent, post-draw) rather than a
    # guessed offset, so the two never overlap regardless of font metrics.
    # Only categories actually present among feat_cols are listed — e.g.
    # CV Consistency has no row at all when cv_cosine_sim is excluded.
    present_cats = {_feature_category_name(f) for f in feat_cols}
    category_handles = [Patch(facecolor=c, label=cat) for cat, c in FEATURE_CATEGORY_COLORS.items()
                        if cat in present_cats]
    category_legend = ax.legend(handles=category_handles, loc=side_loc, fontsize=SMALL,
                                frameon=True, framealpha=0.9, borderpad=0.6, ncol=2,
                                title="Feature category (row label colour)",
                                title_fontsize=SMALL + 1)
    ax.add_artist(category_legend)

    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    cat_top_axes_y = inv.transform(category_legend.get_window_extent())[1, 1]

    shape_handles = [
        Line2D([0], [0], marker="^", linestyle="none", markersize=10,
               markerfacecolor="grey", markeredgecolor="grey",
               label="Above feature's mean"),
        Line2D([0], [0], marker="v", linestyle="none", markersize=10,
               markerfacecolor="grey", markeredgecolor="grey",
               label="Below feature's mean"),
    ]
    shape_legend = ax.legend(handles=shape_handles, loc=side_loc, fontsize=SMALL,
                             frameon=True, framealpha=0.9, borderpad=0.6,
                             bbox_to_anchor=(anchor_x, cat_top_axes_y + 0.02),
                             title="Own mean, vs.", title_fontsize=SMALL + 1)
    ax.add_artist(shape_legend)

    # Marker matches the shape actually used in the plot (a plain circle,
    # used previously, never appears in this chart — every point is a
    # triangle, so the legend swatch should be one too).
    ref_effects = [gmin, (gmin + gmax) / 2, gmax]
    size_handles = [
        Line2D([0], [0], marker="^", linestyle="none",
               markersize=(to_size(e) ** 0.5) * 0.75,
               markerfacecolor="grey", markeredgecolor="white",
               label=f"{e:.2f}")
        for e in ref_effects
    ]
    ax.legend(handles=size_handles, loc=other_loc, fontsize=SMALL,
             frameon=True, framealpha=0.9, borderpad=0.6, labelspacing=1.1,
             title="Effect size", title_fontsize=SMALL + 1)

    model_handles = [Patch(facecolor=DISPLAY_COLORS.get(MODEL_DISPLAY[w], "grey"), label=MODEL_DISPLAY[w])
                      for w in dual_role]
    fig.legend(handles=model_handles, loc="lower center", ncol=3, fontsize=fs + BIG - 2,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.09, 0.98, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out_path)}")


def plot_common_preference_summary(coef_df, feat_cols, feat_df, out_path):
    """
    Companion to plot_self_points_summary, but about CONSENSUS rather than
    self-preference. Two panels, sharing the same feature rows:

    LEFT — how much evaluators reward each feature: one small dot per
    EVALUATOR (all nine, not just the dual-role ones — this isn't about
    self-preference, so every evaluator's taste counts) at that evaluator's
    own Ridge coefficient for the feature, plus one bold diamond for the
    cross-evaluator MEAN coefficient. Coefficients are already standardised
    (features were z-scored before fitting), so they're directly comparable
    feature to feature without further rescaling.

    RIGHT — how much each WRITER (all eleven, including the two that are
    never evaluators) actually uses that feature: one dot per writer at its
    own mean, standardised the same way ((mean - population mean) /
    population std, using feat_df directly, mirroring what the regression's
    own standardisation does) with a horizontal error bar for that writer's
    own std — so it's a real distribution, not just a point estimate. Both
    panels share one x-scale in standard-deviation units, so a row is
    directly readable end to end: "writers spread out this much on this
    feature (right), and evaluators reward it this much and this
    consistently (left)."

    Rows where all nine evaluators agree in sign (unanimous reward or
    unanimous penalty) get a light shaded background band spanning BOTH
    panels, since those are the features "most appreciated (or most
    disliked) by all models" — the specific question this plot exists to
    answer.
    """
    evaluators = [e for e in UNIQUE_EVALUATORS if e in coef_df.index]

    rows = []
    for f in feat_cols:
        vals = {e: float(coef_df.loc[e, f]) for e in evaluators}
        unanimous = len({v >= 0 for v in vals.values()}) == 1

        all_vals = feat_df[f].dropna().values
        feat_mean, feat_std = float(all_vals.mean()), float(all_vals.std()) or 1.0
        writer_z = {}
        for w in RAW_WRITERS:
            wv = feat_df.loc[feat_df["Writer"] == w, f].dropna().values
            if len(wv) == 0:
                continue
            writer_z[w] = ((float(wv.mean()) - feat_mean) / feat_std,
                           float(wv.std()) / feat_std)

        rows.append({"feature": f, "label": FEATURE_LABELS[f],
                     "vals": vals, "mean": float(np.mean(list(vals.values()))),
                     "unanimous": unanimous, "writer_z": writer_z})

    n = len(rows)
    BIG = 5
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 0.5 * n + 1.4),
                                  sharey=True, gridspec_kw={"wspace": 0.05})
    rows_r = rows[::-1]
    for i, r in enumerate(rows_r):
        if r["unanimous"]:
            band_color = "#C8E6C9" if r["mean"] >= 0 else "#FFCDD2"
            ax.axhspan(i - 0.5, i + 0.5, color=band_color, alpha=0.5, zorder=0)
            ax2.axhspan(i - 0.5, i + 0.5, color=band_color, alpha=0.5, zorder=0)
        for e, v in r["vals"].items():
            ax.scatter(v, i, s=110, color=DISPLAY_COLORS.get(MODEL_DISPLAY[e], "grey"),
                       edgecolor="white", linewidth=0.6, alpha=0.75, zorder=3)
        ax.scatter(r["mean"], i, s=260, marker="D", color="black",
                   edgecolor="white", linewidth=1.0, zorder=4)
        for w, (z, zerr) in r["writer_z"].items():
            ax2.errorbar(z, i, xerr=zerr, fmt="o", markersize=9.5, capsize=3,
                        color=WRITER_COLORS.get(w, "grey"), ecolor=WRITER_COLORS.get(w, "grey"),
                        markeredgecolor="white", markeredgewidth=0.6, alpha=0.85, zorder=3)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax2.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)

    _add_group_separators_by_feat(ax, [r["feature"] for r in rows_r])
    _add_group_separators_by_feat(ax2, [r["feature"] for r in rows_r])

    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows_r], fontsize=fs + BIG)
    ax.tick_params(axis="x", labelsize=fs + BIG - 1)
    ax2.tick_params(axis="x", labelsize=fs + BIG - 1, labelleft=False)
    ax.set_xlabel("Ridge coefficient (standardised feature -> Score, within-job demeaned)",
                  fontsize=fs + BIG - 1)
    ax2.set_xlabel("Writer's own mean $\\pm$ 1 std  (standard-deviation units)",
                   fontsize=fs + BIG - 1)
    ax.set_title("How Much Evaluators Reward It", fontsize=fs + BIG + 1, pad=14)
    ax2.set_title("How Much Writers Actually Use It", fontsize=fs + BIG + 1, pad=14)
    ax.grid(axis="x", alpha=0.3)
    ax2.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax2.set_axisbelow(True)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=11,
               markerfacecolor="grey", markeredgecolor="white", alpha=0.75,
               label="Individual evaluator (left)"),
        Line2D([0], [0], marker="D", linestyle="none", markersize=13,
               markerfacecolor="black", markeredgecolor="white",
               label="Cross-evaluator mean (left)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=11,
               markerfacecolor="grey", markeredgecolor="white", alpha=0.85,
               label="Individual writer, mean $\\pm$ std (right)"),
        Patch(facecolor="#C8E6C9", alpha=0.5, label="Unanimous reward"),
        Patch(facecolor="#FFCDD2", alpha=0.5, label="Unanimous penalty"),
    ]
    fig.legend(handles=legend_handles, loc="upper left", fontsize=fs + BIG - 2,
              frameon=True, framealpha=0.9, borderpad=0.9,
              bbox_to_anchor=(1.005, 0.91))

    model_handles = [Patch(facecolor=DISPLAY_COLORS.get(MODEL_DISPLAY[w], "grey"), label=MODEL_DISPLAY[w])
                      for w in RAW_WRITERS]
    fig.legend(handles=model_handles, loc="lower center", ncol=6, fontsize=fs + BIG - 2,
               frameon=False, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Which Features Do Evaluators Reward, and Do Writers Actually Use Them?",
                fontsize=fs + BIG + 2, y=1.0)
    fig.tight_layout(rect=[0, 0.14, 0.85, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out_path)}")


def plot_common_preference_summary_ranked(coef_df, feat_cols, feat_df, out_path, title_note=""):
    """
    Same two-panel figure as plot_common_preference_summary (a separate
    output, not a replacement), with two differences:

      - Rows are sorted by cross-evaluator mean coefficient (descending),
        not grouped by feature category. Ranking is the point of this
        version, so category grouping — which would scatter rows all over
        once sorted by reward — is dropped along with its separator lines.
      - Feature-name row labels are coloured by category instead of plain
        black, using the same scheme as plot_model_explainability's
        BLOCK_COLORS (Job-Ad Fit / CV Consistency / Length & Structure /
        Language Complexity / Sentiment & Affect / Emotions), so category
        membership is still visible despite the interleaved order.
    """
    evaluators = [e for e in UNIQUE_EVALUATORS if e in coef_df.index]

    rows = []
    for f in feat_cols:
        vals = {e: float(coef_df.loc[e, f]) for e in evaluators}
        unanimous = len({v >= 0 for v in vals.values()}) == 1

        all_vals = feat_df[f].dropna().values
        feat_mean, feat_std = float(all_vals.mean()), float(all_vals.std()) or 1.0
        writer_z = {}
        for w in RAW_WRITERS:
            wv = feat_df.loc[feat_df["Writer"] == w, f].dropna().values
            if len(wv) == 0:
                continue
            writer_z[w] = ((float(wv.mean()) - feat_mean) / feat_std,
                           float(wv.std()) / feat_std)

        rows.append({"feature": f, "label": FEATURE_LABELS[f],
                     "vals": vals, "mean": float(np.mean(list(vals.values()))),
                     "unanimous": unanimous, "writer_z": writer_z})

    # Ascending, plotted directly (no reversal): row i=0 sits at the bottom
    # of the axes and the largest mean at the top, so the plot reads like a
    # leaderboard — most-rewarded feature at the top.
    rows.sort(key=lambda r: r["mean"])

    n = len(rows)
    BIG = 10
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 0.5 * n + 1.4),
                                  sharey=True, gridspec_kw={"wspace": 0.1})
    rows_r = rows
    for i, r in enumerate(rows_r):
        if r["unanimous"]:
            band_color = "#C8E6C9" if r["mean"] >= 0 else "#FFCDD2"
            ax.axhspan(i - 0.5, i + 0.5, color=band_color, alpha=0.5, zorder=0)
            ax2.axhspan(i - 0.5, i + 0.5, color=band_color, alpha=0.5, zorder=0)
        for e, v in r["vals"].items():
            ax.scatter(v, i, s=110, color=DISPLAY_COLORS.get(MODEL_DISPLAY[e], "grey"),
                       edgecolor="white", linewidth=0.6, alpha=0.75, zorder=3)
        # Cross-evaluator mean: a short vertical bar confined to this row,
        # rather than a big diamond that visually dominates the row.
        ax.plot([r["mean"], r["mean"]], [i - 0.32, i + 0.32],
               color="black", linewidth=3.2, solid_capstyle="butt", zorder=4)
        for w, (z, zerr) in r["writer_z"].items():
            ax2.errorbar(z, i, xerr=zerr, fmt="o", markersize=9.5, capsize=3,
                        color=WRITER_COLORS.get(w, "grey"), ecolor=WRITER_COLORS.get(w, "grey"),
                        markeredgecolor="white", markeredgewidth=0.6, alpha=0.85, zorder=3)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax2.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)

    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows_r], fontsize=fs + BIG)
    for tick, r in zip(ax.get_yticklabels(), rows_r):
        tick.set_color(_feature_name_color(r["feature"]))
        tick.set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=fs + BIG - 1)
    ax2.tick_params(axis="x", labelsize=fs + BIG - 1, labelleft=False)
    ax.set_xlabel("Ridge coefficient (standardised feature -> Score, within-job demeaned)",
                  fontsize=fs + BIG - 4)
    ax2.set_xlabel("Writer's own mean $\\pm$ 1 std  (standard-deviation units)",
                   fontsize=fs + BIG - 4)
    ax.set_title("How Much Evaluators Reward It\n(sorted by cross-evaluator mean)",
                fontsize=fs + BIG + 1, pad=14)
    ax2.set_title("How Much Writers Actually Use It", fontsize=fs + BIG + 1, pad=14)
    ax.grid(axis="x", alpha=0.3)
    ax2.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax2.set_axisbelow(True)

    # Marker/shading legend now lives INSIDE the left panel itself (bottom
    # right corner is empty here: the lowest-ranked rows only ever have
    # negative-side dots, so nothing sits there to cover).
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=11,
               markerfacecolor="grey", markeredgecolor="white", alpha=0.75,
               label="Individual evaluator (left)"),
        Line2D([0], [0], marker="|", linestyle="none", markersize=18, markeredgewidth=3.2,
               markerfacecolor="black", markeredgecolor="black",
               label="Cross-evaluator mean (left)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=11,
               markerfacecolor="grey", markeredgecolor="white", alpha=0.85,
               label="Individual writer, mean $\\pm$ std (right)"),
        Patch(facecolor="#C8E6C9", alpha=0.5, label="Unanimous reward"),
        Patch(facecolor="#FFCDD2", alpha=0.5, label="Unanimous penalty"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=fs + BIG - 2,
             frameon=True, framealpha=0.9, borderpad=0.9)

    # Category legend and model legend sit together at the bottom, category
    # to the left of the (now 3-column) model legend. Only categories
    # actually present among feat_cols are listed — e.g. CV Consistency has
    # no row at all when cv_cosine_sim is excluded.
    present_cats = {_feature_category_name(f) for f in feat_cols}
    category_handles = [Patch(facecolor=c, label=cat) for cat, c in FEATURE_CATEGORY_COLORS.items()
                        if cat in present_cats]
    fig.legend(handles=category_handles, loc="upper center", fontsize=fs + BIG - 2,
              frameon=True, framealpha=0.9, borderpad=0.9, title="Feature category (row label colour)",
              title_fontsize=fs + BIG - 1, bbox_to_anchor=(0.24, -0.02))

    model_handles = [Patch(facecolor=DISPLAY_COLORS.get(MODEL_DISPLAY[w], "grey"), label=MODEL_DISPLAY[w])
                      for w in RAW_WRITERS]
    fig.legend(handles=model_handles, loc="upper center", ncol=3, fontsize=fs + BIG - 2,
               frameon=False, bbox_to_anchor=(0.68, -0.02))

    fig.suptitle("Which Features Do Evaluators Reward, Ranked, and Do Writers Actually Use Them?"
                f"{title_note}",
                fontsize=fs + BIG + 2, y=1.0)
    fig.tight_layout(rect=[0, 0.16, 0.99, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out_path)}")


def _add_group_separators_by_feat(ax, feats_by_y):
    """Like _add_group_separators, but for a plot with one row per feature
    (not per writer x feature) — feats_by_y[i] is the feature plotted at
    y = i, so a boundary is drawn wherever consecutive entries change group."""
    n = len(feats_by_y)
    current_group = None
    for i in range(n + 1):
        grp = FEATURE_GROUPS_MAP.get(feats_by_y[i]) if i < n else None
        if grp != current_group and i > 0:
            ax.axhline(i - 0.5, color="grey", linewidth=1.2, alpha=0.6)
        current_group = grp


def print_alignment_summary(coef_df, profiles, feat_cols, pairwise):
    """Per-writer text summary: top universal driver (common preference) and
    top controversial (highest-divergence) feature, naming which evaluator is
    most for/against it — directly answers 'which features match, which are
    opposite, across models'."""
    common_models = [m for m in UNIQUE_EVALUATORS if m in coef_df.index]
    mean_taste = coef_df.loc[common_models, feat_cols].mean(axis=0)

    print("\n=== Per-writer alignment summary (full features incl. semantic fit, Cover Letter Only) ===")
    for w, by_eval in pairwise.items():
        per_eval = pd.DataFrame(by_eval).T  # evaluators x features
        universal = (profiles.loc[w, feat_cols] * mean_taste).sort_values(key=abs, ascending=False)
        top_universal = universal.index[0]

        divergence = per_eval.std(axis=0).sort_values(ascending=False)
        top_div_feat = divergence.index[0]
        col = per_eval[top_div_feat].sort_values()
        low_eval, high_eval = col.index[0], col.index[-1]

        print(f"\n{MODEL_DISPLAY[w]}:")
        print(f"  Universal driver:   {FEATURE_LABELS[top_universal]:<22s} "
              f"contributes {universal[top_universal]:+.3f} with every evaluator's average taste")
        print(f"  Most divergent:     {FEATURE_LABELS[top_div_feat]:<22s} "
              f"std={divergence[top_div_feat]:.3f}  "
              f"({MODEL_DISPLAY[low_eval]}: {col[low_eval]:+.3f}  vs.  {MODEL_DISPLAY[high_eval]}: {col[high_eval]:+.3f})")


def plot_common_vs_self_preference(coef_by_type, profiles_by_type, feat_cols):
    """
    Splits apparent style-based self-preference into two components:

    - Common (universal) preference: (average evaluator's taste) . (writer's style).
      How much a writer's style is liked by a "typical" evaluator, independent of
      who is actually judging it.
    - Idiosyncratic self-preference: (own taste − average taste) . (own style).
      Whether a model specifically overweights its OWN stylistic quirks beyond
      what an average evaluator would already like about that style. This is
      NOT the same as raw self-preference: a model can be strongly liked by
      everyone (high common preference) while still rating itself BELOW what
      an average evaluator already would (negative idiosyncratic term) — the
      two panels side by side are what make that distinction visible, rather
      than the diagonal of a raw preference matrix, which conflates "liked by
      everyone" with "self-preferring."

    Only defined for models that are both an evaluator and a writer.
    One row per Eval_Type (CV + Cover Letter, Cover Letter Only), so the two
    scenarios can be compared directly — same layout as
    plot_self_preference_ranking.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    for row, (eval_type, row_title) in enumerate(EVAL_TYPES.items()):
        coef_df, profiles = coef_by_type[eval_type], profiles_by_type[eval_type]
        common_models = [m for m in UNIQUE_EVALUATORS if m in coef_df.index and m in profiles.index]
        mean_taste = coef_df.loc[common_models, feat_cols].mean(axis=0)

        common_pref = (profiles.loc[[w for w in RAW_WRITERS if w in profiles.index], feat_cols]
                       @ mean_taste)
        common_pref.index = [MODEL_DISPLAY[w] for w in common_pref.index]
        common_pref = common_pref.sort_values(ascending=False)

        idio_rows = {}
        for m in common_models:
            idio_coef = coef_df.loc[m, feat_cols] - mean_taste
            idio_rows[MODEL_DISPLAY[m]] = float(idio_coef @ profiles.loc[m, feat_cols])
        idio_pref = pd.Series(idio_rows).sort_values(ascending=False)

        ax = axes[row, 0]
        colors = [DISPLAY_COLORS.get(w, "grey") for w in common_pref.index]
        ax.barh(common_pref.index[::-1], common_pref.values[::-1],
                color=colors[::-1], edgecolor="white", height=0.65)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        xmax = common_pref.abs().max() * 1.3
        for i, v in enumerate(common_pref.values[::-1]):
            ax.text(v + (0.008 * xmax if v >= 0 else -0.008 * xmax), i, f"{v:+.3f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=fs, fontweight="bold")
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
            tick.set_fontweight("bold")
        ax.set_xlabel("Avg-evaluator taste · writer's own style", fontsize=fs + 1)
        ax.set_title(f"{row_title}: Common (Universal) Preference\n"
                     f"How much a 'typical' evaluator likes this writer's style",
                     fontsize=fs + 3, pad=10)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(-xmax, xmax)

        ax = axes[row, 1]
        colors = [DISPLAY_COLORS.get(w, "grey") for w in idio_pref.index]
        ax.barh(idio_pref.index[::-1], idio_pref.values[::-1],
                color=colors[::-1], edgecolor="white", height=0.65)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        xmax = idio_pref.abs().max() * 1.3
        for i, v in enumerate(idio_pref.values[::-1]):
            ax.text(v + (0.008 * xmax if v >= 0 else -0.008 * xmax), i, f"{v:+.3f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=fs, fontweight="bold")
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
            tick.set_fontweight("bold")
        ax.set_xlabel("(Own taste − avg taste) · own style", fontsize=fs + 1)
        ax.set_title(f"{row_title}: Idiosyncratic Self-Preference\nDoes a model overweight its OWN quirks?  "
                     f"(mean = {idio_pref.mean():+.3f})",
                     fontsize=fs + 3, pad=10)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(-xmax, xmax)

    fig.suptitle("Separating Common Preference from Genuine Self-Preference (style-only)",
                 fontsize=fs + 6)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "common_vs_self_preference.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out)}")


def _self_preference_contrib(coef_df, profiles, feat_cols):
    """Per-(model, feature) idiosyncratic self-preference contribution:
    (own taste − avg taste) x (own standardized style). Models x features."""
    common_models = [m for m in UNIQUE_EVALUATORS if m in coef_df.index and m in profiles.index]
    mean_taste = coef_df.loc[common_models, feat_cols].mean(axis=0)
    return pd.DataFrame({
        MODEL_DISPLAY[m]: (coef_df.loc[m, feat_cols] - mean_taste) * profiles.loc[m, feat_cols]
        for m in common_models
    }).T


def plot_self_preference_ranking(coef_by_type, profiles_by_type, feat_cols):
    """
    Ranks models by how much idiosyncratic self-preference shows up, using the
    SAME per-(model, feature) contributions as the divergence dot-plots
    (full feature set, incl. semantic fit) — this is the "sum the dots" version
    of that plot.

    Contributions are already on a common, comparable scale across features:
    every feature is standardized (z-scored) before regression, so a
    contribution = (own taste − avg taste) x (own standardized style) is
    unit-free and safe to sum or compare feature-to-feature.

    Two aggregates, because summing signed values can hide real effects via
    cancellation (see self_preference_drivers.png's GPT-4o mini / Claude
    Haiku 4.5 examples):
      - Net self-preference    = sum of signed per-feature contributions.
        Can be near zero even when individual features show a real, consistent
        self-preference, if they happen to point in opposite directions.
      - Self-preference magnitude = sum of |per-feature contributions|.
        Never cancels — answers "how much does self-preference show up,
        feature by feature" regardless of sign.

    One row per Eval_Type (CV + Cover Letter, Cover Letter Only), so the two
    scenarios can be compared directly.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    for row, (eval_type, row_title) in enumerate(EVAL_TYPES.items()):
        contrib = _self_preference_contrib(coef_by_type[eval_type], profiles_by_type[eval_type], feat_cols)
        net = contrib.sum(axis=1).sort_values(ascending=False)
        mag = contrib.abs().sum(axis=1).sort_values(ascending=False)

        ax = axes[row, 0]
        colors = [DISPLAY_COLORS.get(m, "grey") for m in net.index]
        ax.barh(net.index[::-1], net.values[::-1], color=colors[::-1], edgecolor="white", height=0.65)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        xmax = net.abs().max() * 1.35 or 1.0
        for i, v in enumerate(net.values[::-1]):
            ax.text(v + (0.008 * xmax if v >= 0 else -0.008 * xmax), i, f"{v:+.3f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=fs, fontweight="bold")
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
            tick.set_fontweight("bold")
        ax.set_xlabel("Sum of signed per-feature contributions", fontsize=fs + 1)
        ax.set_title(f"{row_title}: Net Self-Preference\nCan cancel across features  (mean = {net.mean():+.3f})",
                     fontsize=fs + 3, pad=10)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(-xmax, xmax)

        ax = axes[row, 1]
        colors = [DISPLAY_COLORS.get(m, "grey") for m in mag.index]
        ax.barh(mag.index[::-1], mag.values[::-1], color=colors[::-1], edgecolor="white", height=0.65)
        for i, v in enumerate(mag.values[::-1]):
            ax.text(v + 0.008 * mag.max(), i, f"{v:.3f}", va="center", ha="left", fontsize=fs, fontweight="bold")
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
            tick.set_fontweight("bold")
        ax.set_xlabel("Sum of |per-feature contributions|", fontsize=fs + 1)
        ax.set_title(f"{row_title}: Self-Preference Magnitude\nNever cancels — how much shows up, feature by feature",
                     fontsize=fs + 3, pad=10)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(0, mag.max() * 1.2)

    fig.suptitle("Ranking Models by Idiosyncratic Self-Preference (full features incl. semantic fit)",
                 fontsize=fs + 6)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "self_preference_ranking.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out)}")


def plot_self_preference_and_received(coef_df, profiles, feat_cols, out_path, title_note=""):
    """
    Two ranked bar charts side by side — a separate figure from
    self_preference_ranking.png (not a replacement), meant to close out the
    generative-preference-hierarchy analysis in the main text:

      LEFT  — Net Self-Preference: identical definition to
        self_preference_ranking.png's left column (dual-role models only):
        sum over features of (own taste - avg taste) x own standardised
        style. Idiosyncratic self-bias; can cancel across features.

      RIGHT — Total Preference Received: for every writer (all eleven,
        including the two that are never evaluators — this one isn't
        self-referential), sum over ALL NINE evaluators of (that
        evaluator's taste . the writer's own standardised style) — how
        much the writer's actual style is liked by the evaluator panel as
        a WHOLE, not just by itself. Same ranking as the "Total" row of
        common_preference_drivers_full.png; expressed here as a summed
        (not averaged) quantity across evaluators, so magnitudes are
        exactly 9x that row's, per the user's specification.
    """
    BIG = 8

    contrib = _self_preference_contrib(coef_df, profiles, feat_cols)
    net = contrib.sum(axis=1).sort_values(ascending=False)

    evaluators = [e for e in UNIQUE_EVALUATORS if e in coef_df.index]
    writers = [w for w in RAW_WRITERS if w in profiles.index]
    evaluator_sum_taste = coef_df.loc[evaluators, feat_cols].sum(axis=0)
    received = profiles.loc[writers, feat_cols].dot(evaluator_sum_taste)
    received.index = [MODEL_DISPLAY[w] for w in writers]
    received = received.sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9.5))

    ax = axes[0]
    colors = [DISPLAY_COLORS.get(m, "grey") for m in net.index]
    ax.barh(net.index[::-1], net.values[::-1], color=colors[::-1], edgecolor="white", height=0.65)
    ax.axvline(0, color="black", linewidth=1.4, linestyle="--", alpha=0.7)
    xmax = net.abs().max() * 1.35 or 1.0
    for i, v in enumerate(net.values[::-1]):
        ax.text(v + (0.012 * xmax if v >= 0 else -0.012 * xmax), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=fs + BIG, fontweight="bold")
    ax.set_yticks(range(len(net)))
    ax.set_yticklabels(net.index[::-1], fontsize=fs + BIG)
    for tick in ax.get_yticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        tick.set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=fs + BIG - 1)
    ax.set_xlabel("Sum of signed per-feature contributions", fontsize=fs + BIG)
    ax.set_title(f"Net Self-Preference{title_note}\n(idiosyncratic own-taste bias; can cancel across features)",
                fontsize=fs + BIG + 2, pad=14)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(-xmax, xmax)

    ax = axes[1]
    colors = [DISPLAY_COLORS.get(m, "grey") for m in received.index]
    ax.barh(received.index[::-1], received.values[::-1], color=colors[::-1], edgecolor="white", height=0.65)
    ax.axvline(0, color="black", linewidth=1.4, linestyle="--", alpha=0.7)
    xmax2 = max(abs(received.min()), abs(received.max())) * 1.35 or 1.0
    for i, v in enumerate(received.values[::-1]):
        ax.text(v + (0.012 * xmax2 if v >= 0 else -0.012 * xmax2), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=fs + BIG, fontweight="bold")
    ax.set_yticks(range(len(received)))
    ax.set_yticklabels(received.index[::-1], fontsize=fs + BIG)
    for tick in ax.get_yticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        tick.set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=fs + BIG - 1)
    ax.set_xlabel("Writer's style $\\cdot$ sum of all nine evaluators' taste", fontsize=fs + BIG)
    ax.set_title(f"Total Preference Received{title_note}\n(from the whole evaluator panel, not just self)",
                fontsize=fs + BIG + 2, pad=14)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(-xmax2, xmax2)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out_path)}")


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

    full_coef_by_type, full_profiles_by_type = {}, {}
    style_coef_by_type, style_profiles_by_type = {}, {}

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

            if suffix == "_style" and eval_type == "cl_evaluations":
                style_coef_df, style_profiles = coef_df, profiles
            if suffix == "_style":
                style_coef_by_type[eval_type]     = coef_df
                style_profiles_by_type[eval_type] = profiles
            if suffix == "":
                full_coef_by_type[eval_type]     = coef_df
                full_profiles_by_type[eval_type] = profiles

        print("  Plotting...")
        plot_coef_heatmap(coef_by_type, feat_cols, suffix, title_note)
        plot_preference_matrices(pred_by_type, actual_by_type, suffix, title_note)
        plot_scatter(pred_by_type, actual_by_type, suffix, title_note)

    plot_common_vs_self_preference(style_coef_by_type, style_profiles_by_type, STYLE_COLS)
    plot_common_preference_drivers(style_coef_df, style_profiles, STYLE_COLS)
    plot_self_preference_drivers(style_coef_df, style_profiles, STYLE_COLS)

    full_coef_df, full_profiles = full_coef_by_type["cl_evaluations"], full_profiles_by_type["cl_evaluations"]
    pairwise_full = compute_pairwise_contributions(full_coef_df, full_profiles, FEATURE_COLS)
    plot_divergence_dotplot(pairwise_full, "preference_divergence_dotplot.png",
                             title_note="  [full features incl. semantic fit]")
    print_alignment_summary(full_coef_df, full_profiles, FEATURE_COLS, pairwise_full)

    # Same "common preference drivers" decomposition as above, but with the
    # FULL feature set (semantic fit included) — needed because Cosine Sim.
    # to Job Ad turned out to be the single most universally-rewarded
    # feature of all twenty, so a style-only version would miss the biggest
    # driver of which writers benefit most from universal evaluator taste.
    plot_common_preference_drivers(full_coef_df, full_profiles, FEATURE_COLS,
                                    out_name="common_preference_drivers_full.png",
                                    title_note="  [full features incl. semantic fit]")

    plot_self_preference_ranking(full_coef_by_type, full_profiles_by_type, FEATURE_COLS)

    plot_all_category_dotplots(pairwise_full, feat_df, os.path.join(OUT_DIR, "preference_by_category"))

    # cv_cosine_sim excluded here: in the cover-letter-only condition the
    # evaluator's prompt never contains the CV, so a Ridge coefficient on
    # how much the letter echoes it cannot reflect a genuine preference —
    # see CL_ONLY_FEATURE_COLS. Refit specifically for these four figures
    # rather than reusing full_coef_df/pairwise_full (which still include
    # cv_cosine_sim and remain correct for the CV+CL-condition plots above,
    # where the evaluator does see the CV).
    print("  Refitting cover-letter-only regression without cv_cosine_sim...")
    clonly_coef_df, clonly_means, clonly_stds = fit_regressions(merged, "cl_evaluations", CL_ONLY_FEATURE_COLS)
    clonly_profiles = writer_profiles(feat_df, clonly_means, clonly_stds, CL_ONLY_FEATURE_COLS)
    pairwise_clonly = compute_pairwise_contributions(clonly_coef_df, clonly_profiles, CL_ONLY_FEATURE_COLS)

    plot_self_points_summary(pairwise_clonly, CL_ONLY_FEATURE_COLS, feat_df,
                              os.path.join(OUT_DIR, "self_points_summary.png"))

    plot_self_points_summary_ranked(pairwise_clonly, CL_ONLY_FEATURE_COLS, feat_df,
                                     os.path.join(OUT_DIR, "self_points_summary_ranked.png"))

    plot_common_preference_summary(clonly_coef_df, CL_ONLY_FEATURE_COLS, feat_df,
                                    os.path.join(OUT_DIR, "common_preference_summary.png"))

    plot_common_preference_summary_ranked(clonly_coef_df, CL_ONLY_FEATURE_COLS, feat_df,
                                           os.path.join(OUT_DIR, "common_preference_summary_ranked.png"))

    # CV + Cover Letter counterparts of both ranked figures, for the
    # Appendix — same computation, just the other Eval_Type's coefficients.
    cvcl_coef_df, cvcl_profiles = full_coef_by_type["cv_cl_evaluations"], full_profiles_by_type["cv_cl_evaluations"]
    pairwise_cvcl = compute_pairwise_contributions(cvcl_coef_df, cvcl_profiles, FEATURE_COLS)

    plot_self_points_summary_ranked(pairwise_cvcl, FEATURE_COLS, feat_df,
                                     os.path.join(OUT_DIR, "self_points_summary_ranked_cvcl.png"),
                                     title_note="  [CV + Cover Letter]", legend_side="right")

    plot_common_preference_summary_ranked(cvcl_coef_df, FEATURE_COLS, feat_df,
                                           os.path.join(OUT_DIR, "common_preference_summary_ranked_cvcl.png"),
                                           title_note="  [CV + Cover Letter]")

    # Semantic Fit category dot-plot, for the paper appendix, in both
    # scopes: CL-only necessarily drops cv_cosine_sim entirely (same
    # exclusion as CL_ONLY_FEATURE_COLS — it's not a valid regressor when
    # the evaluator never sees the CV), so that version has only
    # job_cosine_sim rows; the CV+CL version keeps both.
    cat_dir = os.path.join(OUT_DIR, "preference_by_category")
    clonly_semfit_vals = [v for w in pairwise_clonly for e in pairwise_clonly[w] for v in pairwise_clonly[w][e].values]
    clonly_semfit_vmax = max(abs(min(clonly_semfit_vals)), abs(max(clonly_semfit_vals))) * 1.05
    plot_category_dotplot(pairwise_clonly, ["job_cosine_sim"], "Semantic Fit (Cover-Letter-Only)",
                          os.path.join(cat_dir, "semantic_fit_clonly.png"),
                          (-clonly_semfit_vmax, clonly_semfit_vmax), feat_df)

    cvcl_semfit_vals = [v for w in pairwise_cvcl for e in pairwise_cvcl[w] for v in pairwise_cvcl[w][e].values]
    cvcl_semfit_vmax = max(abs(min(cvcl_semfit_vals)), abs(max(cvcl_semfit_vals))) * 1.05
    plot_category_dotplot(pairwise_cvcl, ["job_cosine_sim", "cv_cosine_sim"], "Semantic Fit (CV + Cover Letter)",
                          os.path.join(cat_dir, "semantic_fit_cvcl.png"),
                          (-cvcl_semfit_vmax, cvcl_semfit_vmax), feat_df)

    # Closing figure for the main text: Net Self-Preference + Total
    # Preference Received, CL-only, CV-consistency-corrected (uses the same
    # clonly_coef_df/clonly_profiles as the other CL-only figures above).
    plot_self_preference_and_received(clonly_coef_df, clonly_profiles, CL_ONLY_FEATURE_COLS,
                                       os.path.join(OUT_DIR, "self_preference_and_received.png"))

    # CV + Cover Letter companion, for the Appendix.
    plot_self_preference_and_received(cvcl_coef_df, cvcl_profiles, FEATURE_COLS,
                                       os.path.join(OUT_DIR, "self_preference_and_received_cvcl.png"),
                                       title_note="  [CV + Cover Letter]")

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
