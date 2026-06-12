"""
plot_cl_variance.py — two analyses of cover-letter feature variance by writer model.

1. Intra-model homogeneity (within-group std)
   For each (model, job): std of each feature across the 50 CVs.
   Averaged across jobs, normalised by global std → "relative within-group variability".
   Low value = model produces similar letters regardless of candidate.

2. Tier sensitivity (Cohen's d: High-Fit vs Moderate-Fit)
   For each model: effect size of the tier split on each feature.
   Positive d = High-Fit letters score higher on that feature.

3. Embedding homogeneity
   Mean pairwise cosine similarity within each (model, job) group,
   averaged across jobs.  High = the model writes near-identical letters.

Outputs:
  output_plots/cl_features/cl_variance_heatmaps.png
  output_plots/cl_features/cl_embedding_homogeneity.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

from aggregate_plots import WRITER_COLORS, MODEL_DISPLAY, DISPLAY_COLORS

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

IN_PATH = "output_eval/cl_features_no_gemini2.parquet"
OUT_DIR = "output_plots/cl_features_no_gemini2"
fs      = plt.rcParams["font.size"]

WRITERS = list(WRITER_COLORS)

FEATURE_GROUPS = [
    # 1 — Length & Structure
    ("word_count",           "Word Count",           "Length & Structure"),
    ("sentence_count",       "Sentence Count",       "Length & Structure"),
    ("paragraph_count",      "Paragraph Count",      "Length & Structure"),
    ("comma_count",          "Comma Count",          "Length & Structure"),
    # 2 — Language Complexity
    ("avg_word_length",      "Avg Word Len",         "Language Complexity"),
    ("ttr",                  "TTR",                  "Language Complexity"),
    ("flesch_reading_ease",  "Flesch Ease",          "Language Complexity"),
    # 3 — Sentiment & Affect
    ("vader_compound",       "VADER",                "Sentiment & Affect"),
    ("vad_valence",          "VAD Valence",          "Sentiment & Affect"),
    ("vad_arousal",          "VAD Arousal",          "Sentiment & Affect"),
    ("vad_dominance",        "VAD Dominance",        "Sentiment & Affect"),
    # 4 — Emotions
    ("emo_joy",              "Joy",                  "Emotions"),
    ("emo_neutral",          "Neutral",              "Emotions"),
    ("emo_surprise",         "Surprise",             "Emotions"),
    ("emo_fear",             "Fear",                 "Emotions"),
    ("emo_anger",            "Anger",                "Emotions"),
    ("emo_disgust",          "Disgust",              "Emotions"),
    ("emo_sadness",          "Sadness",              "Emotions"),
    # 5 — Semantic Fit
    ("job_cosine_sim",       "Job Sim.",             "Semantic Fit"),
    ("cv_cosine_sim",        "CV Sim.",              "Semantic Fit"),
]

FEATURE_COLS   = [c for c, _, _ in FEATURE_GROUPS]
FEATURE_LABELS = {c: l for c, l, _ in FEATURE_GROUPS}
FEATURE_GROUP  = {c: g for c, _, g in FEATURE_GROUPS}
GROUPS_ORDER   = list(dict.fromkeys(g for _, _, g in FEATURE_GROUPS))


# ── helpers ───────────────────────────────────────────────────────────────────

def cohens_d(a: pd.Series, b: pd.Series) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return float((m1 - m2) / pooled) if pooled > 0 else 0.0


def mean_pairwise_cosine(vecs: np.ndarray) -> float:
    """Mean cosine similarity over all pairs in vecs (n_samples × n_dims)."""
    if len(vecs) < 2:
        return np.nan
    sim = cosine_similarity(vecs)
    n = len(vecs)
    # upper triangle only, excluding diagonal
    idx = np.triu_indices(n, k=1)
    return float(sim[idx].mean())


# ── analysis 1: within-model variability ─────────────────────────────────────

def compute_within_variability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each feature is first scaled to [0, 1] using the 5th–95th percentile range
    (robust to outliers).  Then, for each (Job_ID, Writer), compute std of the
    scaled values across the 50 CVs.  Average across jobs.

    Result is directly comparable across features:
      0   → all 50 letters identical on this feature
      0.5 → letters span the full observed range (maximum possible spread)
    Returns DataFrame: writers (index) × features (columns).
    """
    scaled = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        p5, p95 = df[col].quantile(0.05), df[col].quantile(0.95)
        rng = p95 - p5
        if rng > 0:
            scaled[col] = ((df[col] - p5) / rng).clip(0, 1)
        else:
            scaled[col] = 0.0

    scaled["Job_ID"] = df["Job_ID"].values
    scaled["Writer"] = df["Writer"].values

    within_std = (
        scaled.groupby(["Job_ID", "Writer"])[FEATURE_COLS]
        .std(ddof=1)
        .reset_index()
        .groupby("Writer")[FEATURE_COLS]
        .mean()
        .reindex(WRITERS)
    )
    within_std.index   = [MODEL_DISPLAY[w] for w in WRITERS]
    within_std.columns = [FEATURE_LABELS[c] for c in FEATURE_COLS]
    return within_std


# ── analysis 2: tier sensitivity ──────────────────────────────────────────────

def compute_between_model_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, feature): z-score of the model's mean relative to the
    global mean and std.  Positive = model writes higher than average on this
    feature; negative = lower.  Comparable across features by construction.
    Returns DataFrame: writers (index) × features (columns).
    """
    global_mean = df[FEATURE_COLS].mean()
    global_std  = df[FEATURE_COLS].std().replace(0, np.nan)

    deviation = (
        df.groupby("Writer")[FEATURE_COLS]
        .mean()
        .reindex(WRITERS)
        .sub(global_mean)
        .div(global_std)
    )
    deviation.index   = [MODEL_DISPLAY[w] for w in WRITERS]
    deviation.columns = [FEATURE_LABELS[c] for c in FEATURE_COLS]
    return deviation


def compute_eta_squared(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Job_ID, Writer): η² = SS_between_tiers / SS_total across 50 CVs.
    Average across jobs.  Range 0–1:
      0 = tier explains none of the within-job variance on this feature
      1 = all within-job variance is explained by the tier split
    """
    df = df.copy()
    df["Tier"] = df["CV_Idx"].apply(lambda x: "High-Fit" if x <= 25 else "Moderate-Fit")
    job_ids = df["Job_ID"].unique()

    rows = {}
    for writer in WRITERS:
        per_job = []
        for job_id in job_ids:
            sub   = df[(df["Writer"] == writer) & (df["Job_ID"] == job_id)]
            high  = sub[sub["Tier"] == "High-Fit"][FEATURE_COLS]
            mod   = sub[sub["Tier"] == "Moderate-Fit"][FEATURE_COLS]
            grand = sub[FEATURE_COLS].mean()
            n_h, n_m = len(high), len(mod)
            ss_between = n_h * (high.mean() - grand) ** 2 + n_m * (mod.mean() - grand) ** 2
            ss_total   = ((sub[FEATURE_COLS] - grand) ** 2).sum()
            eta2 = (ss_between / ss_total.replace(0, np.nan)).clip(0, 1)
            per_job.append(eta2)
        rows[MODEL_DISPLAY[writer]] = {
            FEATURE_LABELS[c]: np.nanmean([d[c] for d in per_job])
            for c in FEATURE_COLS
        }
    return pd.DataFrame(rows).T.reindex([MODEL_DISPLAY[w] for w in WRITERS])


def compute_within_variability_globalstd(df: pd.DataFrame) -> pd.DataFrame:
    """Same as compute_within_variability but normalised by global std."""
    global_std = df[FEATURE_COLS].std().replace(0, np.nan)
    within_std = (
        df.groupby(["Job_ID", "Writer"])[FEATURE_COLS]
        .std(ddof=1)
        .reset_index()
        .groupby("Writer")[FEATURE_COLS]
        .mean()
        .div(global_std)
        .reindex(WRITERS)
    )
    within_std.index   = [MODEL_DISPLAY[w] for w in WRITERS]
    within_std.columns = [FEATURE_LABELS[c] for c in FEATURE_COLS]
    return within_std


def compute_tier_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Job_ID, Writer): Cohen's d (High-Fit − Moderate-Fit) per feature,
    using 25 candidates per tier within each job.
    Then average d across jobs.
    Returns DataFrame: writers (index) × features (columns).
    """
    df = df.copy()
    df["Tier"] = df["CV_Idx"].apply(lambda x: "High-Fit" if x <= 25 else "Moderate-Fit")
    job_ids = df["Job_ID"].unique()

    rows = {}
    for writer in WRITERS:
        per_job = []
        for job_id in job_ids:
            sub  = df[(df["Writer"] == writer) & (df["Job_ID"] == job_id)]
            high = sub[sub["Tier"] == "High-Fit"]
            mod  = sub[sub["Tier"] == "Moderate-Fit"]
            per_job.append({c: cohens_d(high[c], mod[c]) for c in FEATURE_COLS})
        rows[MODEL_DISPLAY[writer]] = {
            FEATURE_LABELS[c]: np.nanmean([d[c] for d in per_job])
            for c in FEATURE_COLS
        }
    return pd.DataFrame(rows).T.reindex([MODEL_DISPLAY[w] for w in WRITERS])


# ── analysis 3: embedding homogeneity (within-job vs cross-job) ──────────────

def compute_embedding_homogeneity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each writer:
      within_job  : mean pairwise cosine sim over pairs from the SAME job
      cross_job   : mean pairwise cosine sim over pairs from DIFFERENT jobs
    Computed efficiently via the full gram matrix.
    Returns DataFrame indexed by display name, columns [within_job, cross_job].
    """
    rows = []
    for writer in WRITERS:
        sub     = df[df["Writer"] == writer].sort_values(["Job_ID", "CV_Idx"])
        job_ids = sub["Job_ID"].to_numpy(dtype=str)
        embs    = np.array(sub["embedding"].tolist(), dtype=np.float32)

        # normalise
        norms     = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_norm = embs / np.where(norms == 0, 1, norms)

        sim = embs_norm @ embs_norm.T          # (500, 500)
        same_job = (job_ids[:, None] == job_ids[None, :])

        n     = len(embs)
        upper = np.triu_indices(n, k=1)
        s     = sim[upper]
        mask  = same_job[upper]

        rows.append({
            "Writer":    MODEL_DISPLAY[writer],
            "within_job": float(s[mask].mean()),
            "cross_job":  float(s[~mask].mean()),
        })
    return pd.DataFrame(rows).set_index("Writer").reindex(
        [MODEL_DISPLAY[w] for w in WRITERS]
    )


# ── plotting ──────────────────────────────────────────────────────────────────

def _draw_heatmap(ax, mat: pd.DataFrame, title: str, cmap: str,
                  vmin: float, vmax: float, center: float,
                  cbar_label: str, show_yticklabels: bool = True,
                  add_avg_col: bool = True, add_avg_row: bool = False,
                  add_spacer_row: bool = False):
    mat = mat.copy().astype(float)
    n_feat_rows = len(FEATURE_COLS)   # number of feature rows before any avg row

    if add_avg_col:
        mat["Avg"] = mat.iloc[:n_feat_rows].mean(axis=1).reindex(mat.index)
    if add_avg_row:
        avg = mat.mean(axis=0)
        avg.name = "Avg"
        mat = pd.concat([mat, avg.to_frame().T])
    elif add_spacer_row:
        spacer = pd.Series(np.nan, index=mat.columns, name=" ")
        mat = pd.concat([mat, spacer.to_frame().T])

    annot = mat.round(2).astype(str)
    sns.heatmap(mat, annot=annot, fmt="", ax=ax,
                cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                linewidths=0.4, linecolor="lightgrey",
                annot_kws={"size": fs},
                cbar_kws={"label": cbar_label, "shrink": 0.6},
                yticklabels=show_yticklabels)
    ax.set_title(title, fontsize=fs + 5, pad=10)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right", fontsize=fs + 1)
    for tick in ax.get_xticklabels():
        label = tick.get_text()
        tick.set_color(DISPLAY_COLORS.get(label, "black"))
        tick.set_fontweight("bold" if label != "Avg" else "normal")
        if label == "Avg":
            tick.set_fontstyle("italic")
    if show_yticklabels:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs + 1)
        for tick in ax.get_yticklabels():
            if tick.get_text() == "Avg":
                tick.set_fontweight("bold")
                tick.set_fontstyle("italic")
    else:
        ax.set_yticks([])

    # group separators + labels (only over feature rows)
    current_group, group_start = None, 0
    for i in range(n_feat_rows + 1):
        grp = FEATURE_GROUP.get(FEATURE_COLS[i]) if i < n_feat_rows else None
        if grp != current_group:
            if i > 0:
                ax.axhline(i, color="black", linewidth=1.5)
            if show_yticklabels and current_group is not None:
                ax.text(-0.6, (group_start + i) / 2, current_group,
                        ha="right", va="center", fontsize=fs,
                        color="dimgrey", fontstyle="italic",
                        transform=ax.transData, clip_on=False)
            current_group = grp
            group_start   = i

    # thick vertical separator before Avg column
    if add_avg_col:
        ax.axvline(mat.shape[1] - 1, color="black", linewidth=2.5)

    # thick horizontal separator before avg / spacer row
    if add_avg_row or add_spacer_row:
        ax.axhline(n_feat_rows, color="black", linewidth=2.5)


def plot_between_deviation_standalone(between: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(16, 14))
    _draw_heatmap(ax, between.T,
                  "Between-Model Deviation\n(z-score of model mean vs. global mean)",
                  cmap="RdBu_r", vmin=-1.5, vmax=1.5, center=0,
                  cbar_label="z-score", show_yticklabels=True,
                  add_avg_col=False, add_avg_row=False, add_spacer_row=False)
    fig.tight_layout(rect=[0.06, 0, 1, 0.97])
    path = os.path.join(OUT_DIR, "cl_between_model_deviation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cl_between_model_deviation.png")


def plot_heatmaps(eta2: pd.DataFrame, within_gstd: pd.DataFrame, tier: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(42, 14))

    _draw_heatmap(axes[0], within_gstd.T,
                  "Intra-Model Variability\n(within-job std / global std)",
                  cmap="YlOrRd", vmin=0, vmax=1, center=0.5,
                  cbar_label="Within-job std / global std", show_yticklabels=True,
                  add_avg_col=True, add_avg_row=True)

    _draw_heatmap(axes[1], tier.T,
                  "Tier Sensitivity\n(Cohen's d: High-Fit − Moderate-Fit)",
                  cmap="RdBu_r", vmin=-0.5, vmax=0.5, center=0,
                  cbar_label="Cohen's d", show_yticklabels=False,
                  add_avg_col=True, add_avg_row=True)

    _draw_heatmap(axes[2], eta2.T,
                  "Tier-Explained Variance\n(η²: fraction of within-job var. explained by tier)",
                  cmap="YlOrRd", vmin=0, vmax=0.5, center=0.25,
                  cbar_label="η²", show_yticklabels=False,
                  add_avg_col=True, add_avg_row=False, add_spacer_row=True)

    fig.suptitle("Cover-Letter Feature Variance by Writer Model",
                 fontsize=fs + 8, y=1.01)
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])
    path = os.path.join(OUT_DIR, "cl_variance_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cl_variance_heatmaps.png")


def plot_embedding_homogeneity(homogeneity: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 6))

    n       = len(WRITERS)
    bw      = 0.35
    gap     = 0.1
    step    = 2 * bw + gap + 0.2

    for i, writer in enumerate(WRITERS):
        label   = MODEL_DISPLAY[writer]
        color   = WRITER_COLORS[writer]
        x_c     = i * step
        v_in    = homogeneity.loc[label, "within_job"]
        v_cross = homogeneity.loc[label, "cross_job"]

        b1 = ax.bar(x_c - (bw + gap) / 2, v_in,    width=bw, color=color, alpha=0.9)
        b2 = ax.bar(x_c + (bw + gap) / 2, v_cross, width=bw, color=color, alpha=0.4,
                    hatch="///", edgecolor=color)
        for bar, val in [(b1, v_in), (b2, v_cross)]:
            ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                    bar[0].get_height() + 0.001,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=fs - 1)

    ax.set_xticks([i * step for i in range(n)])
    ax.set_xticklabels([MODEL_DISPLAY[w] for w in WRITERS],
                       rotation=35, ha="right", fontsize=fs + 1, fontweight="bold")
    for tick, w in zip(ax.get_xticklabels(), WRITERS):
        tick.set_color(WRITER_COLORS[w])

    # legend
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="grey", alpha=0.9,           label="Same job"),
            mpatches.Patch(facecolor="grey", alpha=0.4, hatch="///", edgecolor="grey",
                           label="Different jobs"),
        ],
        fontsize=fs + 1, frameon=True,
    )
    ax.set_ylabel("Mean pairwise cosine similarity", fontsize=fs + 2)
    ax.set_title("Embedding Homogeneity: same job vs. different jobs\n"
                 "(same model, pairs drawn within vs. across jobs)",
                 fontsize=fs + 4, pad=10)
    ax.set_ylim(0, homogeneity.values.max() * 1.12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cl_embedding_homogeneity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cl_embedding_homogeneity.png")


# ── analysis 4: cross-model similarity ───────────────────────────────────────

def compute_cross_model_similarity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For every (Job_ID, CV_Idx) pair, compute cosine similarity between the
    embeddings produced by each pair of writer models.
    Returns (mean_matrix, std_matrix), both 8×8 DataFrames indexed by display name.
    """
    labels = [MODEL_DISPLAY[w] for w in WRITERS]
    sims   = {(a, b): [] for a in WRITERS for b in WRITERS}

    for (_, __), group in df.groupby(["Job_ID", "CV_Idx"]):
        emb = {row["Writer"]: np.array(row["embedding"], dtype=np.float32)
               for _, row in group.iterrows()
               if row["Writer"] in WRITERS}
        for a in WRITERS:
            for b in WRITERS:
                if a in emb and b in emb:
                    sim = float(cosine_similarity(
                        emb[a].reshape(1, -1), emb[b].reshape(1, -1)
                    )[0, 0])
                    sims[(a, b)].append(sim)

    mean_mat = pd.DataFrame(index=labels, columns=labels, dtype=float)
    std_mat  = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a in WRITERS:
        for b in WRITERS:
            vals = sims[(a, b)]
            mean_mat.loc[MODEL_DISPLAY[a], MODEL_DISPLAY[b]] = np.mean(vals) if vals else np.nan
            std_mat.loc[MODEL_DISPLAY[a], MODEL_DISPLAY[b]]  = np.std(vals)  if vals else np.nan
    return mean_mat, std_mat


def plot_cross_model_similarity(mean_mat: pd.DataFrame, std_mat: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 9))

    annot = pd.DataFrame(
        [[f"{mean_mat.iloc[i, j]:.2f}\n±{std_mat.iloc[i, j]:.2f}"
          for j in range(len(mean_mat.columns))]
         for i in range(len(mean_mat.index))],
        index=mean_mat.index, columns=mean_mat.columns,
    )

    sns.heatmap(mean_mat.astype(float), annot=annot, fmt="", ax=ax,
                cmap="YlOrRd", vmin=0.5, vmax=1.0,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": fs + 1},
                cbar_kws={"label": "Mean cosine similarity", "shrink": 0.7})

    ax.set_title("Cross-Model Embedding Similarity\n"
                 "(same candidate & job, mean ± std across all 500 pairs)",
                 fontsize=fs + 5, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                       fontsize=fs + 1, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs + 1, fontweight="bold")
    for tick in ax.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    for tick in ax.get_yticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "cl_cross_model_similarity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cl_cross_model_similarity.png")




# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading features...")
    df = pd.read_parquet(IN_PATH)

    print("Computing between-model deviation...")
    between     = compute_between_model_deviation(df)

    print("Computing eta-squared...")
    eta2        = compute_eta_squared(df)

    print("Computing within-model variability...")
    within_gstd = compute_within_variability_globalstd(df)

    print("Computing tier sensitivity...")
    tier        = compute_tier_sensitivity(df)

    print("Computing embedding homogeneity...")
    homogeneity = compute_embedding_homogeneity(df)

    print("Computing cross-model similarity...")
    mean_mat, std_mat = compute_cross_model_similarity(df)

    df["Tier"] = df["CV_Idx"].apply(lambda x: "High-Fit" if x <= 25 else "Moderate-Fit")

    plot_heatmaps(eta2, within_gstd, tier)
    plot_between_deviation_standalone(between)
    plot_embedding_homogeneity(homogeneity)
    plot_cross_model_similarity(mean_mat, std_mat)
    print(f"\nAll plots saved to {OUT_DIR}/")
    print("\nEmbedding homogeneity:")
    print(homogeneity.sort_values("within_job", ascending=False).to_string())


if __name__ == "__main__":
    main()
