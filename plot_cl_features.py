"""
plot_cl_features.py — visualise cover-letter text features by writer model and candidate tier.

Tiers  :  CV_Idx 1-25 → High-Fit  |  CV_Idx 26-50 → Moderate-Fit
Input  :  output_eval/cl_features.parquet
Output :  output_plots/cl_features/cl_features_overview.png
          output_plots/cl_features/cl_features_<group>.png  (one per group)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from aggregate_plots import WRITER_COLORS, MODEL_DISPLAY, DISPLAY_COLORS, UNIQUE_EVALUATORS

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

IN_PATH = "output_eval/cl_features_no_gemini2.parquet"
OUT_DIR = "output_plots/cl_features_no_gemini2"
fs      = plt.rcParams["font.size"]

FEATURES = [
    # (column,               display label,                group)
    # 1 — length & surface structure
    ("word_count",           "Word Count",                 "Length & Structure"),
    ("sentence_count",       "Sentence Count",             "Length & Structure"),
    ("paragraph_count",      "Paragraph Count",            "Length & Structure"),
    ("comma_count",          "Comma Count",                "Length & Structure"),
    # 2 — vocabulary richness & readability (language complexity)
    ("avg_word_length",      "Avg Word Length",            "Language Complexity"),
    ("ttr",                  "Type-Token Ratio",           "Language Complexity"),
    ("flesch_reading_ease",  "Flesch Reading Ease",        "Language Complexity"),
    # 3 — sentiment & affect
    ("vader_compound",       "VADER Sentiment",            "Sentiment & Affect"),
    ("vad_valence",          "VAD Valence",                "Sentiment & Affect"),
    ("vad_arousal",          "VAD Arousal",                "Sentiment & Affect"),
    ("vad_dominance",        "VAD Dominance",              "Sentiment & Affect"),
    # 4 — semantic fit
    ("job_cosine_sim",       "Cosine Sim. to Job Ad",      "Semantic Fit"),
    ("cv_cosine_sim",        "Cosine Sim. to CV",          "Semantic Fit"),
    ("job_minus_cv_sim",     "Job − CV Similarity",        "Semantic Fit"),
]

# Emotion columns handled separately via plot_emotion_heatmap
EMO_COLS   = ["emo_joy", "emo_neutral", "emo_surprise",
              "emo_fear", "emo_anger", "emo_disgust", "emo_sadness"]
EMO_LABELS = ["Joy", "Neutral", "Surprise", "Fear", "Anger", "Disgust", "Sadness"]

TIERS      = ["High-Fit", "Moderate-Fit"]
TIER_ALPHA = {"High-Fit": 0.9, "Moderate-Fit": 0.45}
TIER_HATCH = {"High-Fit": "",  "Moderate-Fit": "///"}
WRITERS    = list(WRITER_COLORS)          # canonical order
BAR_W      = 0.3                          # width of each tier bar
TIER_GAP   = 0.05                         # gap between the two tier bars
MODEL_GAP  = 0.35                         # gap between model groups
# x-centre of model i:  i * (2*BAR_W + TIER_GAP + MODEL_GAP)


def _add_tier_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Tier"] = df["CV_Idx"].apply(
        lambda x: "High-Fit" if x <= 25 else "Moderate-Fit"
    )
    return df


def _draw_feature(ax, stats: pd.DataFrame, label: str, ylim=None, fs_override=None,
                   title_fs_override=None, show_xticks=True):
    """One group per writer model; two adjacent bars per group (High-Fit | Moderate-Fit)."""
    f = fs_override if fs_override is not None else fs
    tf = title_fs_override if title_fs_override is not None else f + 3
    step = 2 * BAR_W + TIER_GAP + MODEL_GAP
    for model_idx, writer in enumerate(WRITERS):
        x_center = model_idx * step
        color = WRITER_COLORS[writer]
        for tier_idx, tier in enumerate(TIERS):
            row = stats[(stats["Writer"] == writer) & (stats["Tier"] == tier)]
            if row.empty:
                continue
            m, s = float(row["mean"].iloc[0]), float(row["std"].iloc[0])
            x = x_center + (tier_idx - 0.5) * (BAR_W + TIER_GAP / 2)
            ax.bar(x, m, width=BAR_W,
                   color=color,
                   alpha=TIER_ALPHA[tier],
                   hatch=TIER_HATCH[tier],
                   edgecolor="white" if TIER_HATCH[tier] == "" else color,
                   yerr=s,
                   error_kw={"capsize": 3, "linewidth": 1.4, "ecolor": "black"})

    if show_xticks:
        ax.set_xticks([i * step for i in range(len(WRITERS))])
        ax.set_xticklabels([MODEL_DISPLAY[w] for w in WRITERS],
                           rotation=40, ha="right", fontsize=f)
    else:
        # Model identity is conveyed by bar colour + the shared legend only;
        # no per-panel labels, so the x-axis carries no tick marks either.
        ax.set_xticks([])
        ax.tick_params(axis="x", bottom=False)
        ax.set_xlim(-step / 2, (len(WRITERS) - 1) * step + step / 2)
    ax.set_title(label, fontsize=tf, pad=10)
    ax.tick_params(axis="y", labelsize=f)
    ax.grid(True, alpha=0.3, axis="y")
    if ylim is not None:
        ax.set_ylim(ylim)


def _legend_handles():
    """Model colour legend + tier hatch legend."""
    model_handles = [
        mpatches.Patch(color=WRITER_COLORS[w], label=MODEL_DISPLAY[w])
        for w in WRITERS
    ]
    tier_handles = [
        mpatches.Patch(facecolor="grey", alpha=TIER_ALPHA[t],
                       hatch=TIER_HATCH[t], edgecolor="grey", label=t)
        for t in TIERS
    ]
    return model_handles + [mpatches.Patch(visible=False)] + tier_handles


def _finalise(fig, title, path):
    fig.legend(handles=_legend_handles(),
               loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=5, fontsize=fs, frameon=True)
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    fig.suptitle(title, fontsize=fs + 8, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_group(stats: pd.DataFrame, group: str, group_features: list, save_dir: str):
    n = len(group_features)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows),
                             )
    axes = np.array(axes).flatten()

    for i, (col, label, _) in enumerate(group_features):
        _draw_feature(axes[i], stats[stats["col"] == col], label)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fname = f"cl_features_{group.lower().replace(' & ', '_').replace(' ', '_')}.png"
    _finalise(fig, f"Cover-Letter Features — {group}",
              os.path.join(save_dir, fname))
    print(f"  Saved {fname}")


COMPREHENSIVE_FEATURES = [
    ("word_count",           "Word Count",             "Length & Structure"),
    ("sentence_count",       "Sentence Count",         "Length & Structure"),
    ("paragraph_count",      "Paragraph Count",        "Length & Structure"),
    ("comma_count",          "Comma Count",             "Length & Structure"),
    ("avg_word_length",      "Avg Word Length",        "Language Complexity"),
    ("ttr",                  "Type-Token Ratio",       "Language Complexity"),
    ("flesch_reading_ease",  "Flesch Reading Ease",    "Language Complexity"),
    ("vader_compound",       "VADER Sentiment",        "Sentiment & Affect"),
    ("vad_valence",          "VAD Valence",            "Sentiment & Affect"),
    ("vad_arousal",          "VAD Arousal",            "Sentiment & Affect"),
    ("vad_dominance",        "VAD Dominance",          "Sentiment & Affect"),
    ("emo_joy",              "Joy",                    "Emotion"),
    ("emo_neutral",          "Neutral",                "Emotion"),
    ("job_cosine_sim",       "Cosine Sim. to Job Ad",  "Semantic Fit"),
    ("cv_cosine_sim",        "Cosine Sim. to CV",      "Semantic Fit"),
    ("job_minus_cv_sim",     "Job − CV Similarity",    "Semantic Fit"),
]
COMPREHENSIVE_GROUPS = ["Length & Structure", "Sentiment & Affect",
                         "Language Complexity", "Semantic Fit", "Emotion"]


def plot_comprehensive(stats: pd.DataFrame, save_dir: str):
    """One figure, one row per feature class, panels = individual features
    within that class (same bar style as the per-class figures). Emotion
    collapses to just Joy/Neutral (the other five are low and
    indistinguishable across models — see the heatmap for the full set).
    Vocabulary/keyness/topical fingerprints are intentionally excluded.

    Model names are dropped from every panel's x-axis (bar colour + one
    shared legend carries that instead), which frees up the horizontal
    space rotated tick labels used to need and lets titles/tick numbers
    run bigger. The grid is ragged (4/4/3/3/2 panels per row, most-filled
    row on top down to least-filled at the bottom); rather than leave
    that as dead whitespace, the two empty cells in the Emotion row
    are merged into one panel that holds the model-colour legend, and the
    one empty cell in the Semantic Fit row holds the tier legend.

    Font sizes here are deliberately much larger than the module default:
    this figure is meant to be printed at close to full text width in a
    single-column journal layout, which shrinks a 4-panel-wide, 6in-per-
    panel source figure by a factor of ~0.28. Sizes below are chosen so
    that, after that shrink, printed text lands in a legible 8-13pt range
    rather than the 3-5pt it would be at the module's default font size.
    """
    rows_features = {g: [f for f in COMPREHENSIVE_FEATURES if f[2] == g]
                     for g in COMPREHENSIVE_GROUPS}
    ncols = max(len(v) for v in rows_features.values())
    nrows = len(COMPREHENSIVE_GROUPS)

    Y_TICK_FS = 22
    ROW_FS    = 28
    LEGEND_FS = 18
    TITLE_FS  = 24
    SUPTITLE_FS = 32

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows),
                              gridspec_kw={"hspace": 0.8})

    emotion_row = COMPREHENSIVE_GROUPS.index("Emotion")
    semfit_row  = COMPREHENSIVE_GROUPS.index("Semantic Fit")
    n_semfit    = len(rows_features["Semantic Fit"])

    for row_idx, group in enumerate(COMPREHENSIVE_GROUPS):
        group_features = rows_features[group]
        n = len(group_features)
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            if col_idx < n:
                col, label, _ = group_features[col_idx]
                _draw_feature(ax, stats[stats["col"] == col], label,
                              fs_override=Y_TICK_FS, title_fs_override=TITLE_FS,
                              show_xticks=False)
            elif row_idx == emotion_row and col_idx >= n:
                pass  # filled in below, after layout is finalised
            elif row_idx == semfit_row and col_idx == n_semfit:
                pass  # filled in below, after layout is finalised
            else:
                ax.set_visible(False)

    # Reserve margins FIRST, then read back final axis positions — anything
    # placed before tight_layout would use stale (pre-layout) coordinates.
    fig.tight_layout(rect=[0.04, 0.015, 1, 0.95])

    # Merge the Emotion row's two empty cells into one legend panel.
    box_a = axes[emotion_row, 2].get_position()
    box_b = axes[emotion_row, 3].get_position()
    axes[emotion_row, 2].set_visible(False)
    axes[emotion_row, 3].set_visible(False)
    legend_ax = fig.add_axes([box_a.x0, box_a.y0,
                               box_b.x1 - box_a.x0, box_a.y1 - box_a.y0])
    legend_ax.axis("off")
    model_handles = [mpatches.Patch(color=WRITER_COLORS[w], label=MODEL_DISPLAY[w])
                      for w in WRITERS]
    legend_ax.legend(handles=model_handles, loc="center right", ncol=2,
                      fontsize=LEGEND_FS, frameon=True, title="Writer Model",
                      title_fontsize=LEGEND_FS + 2)

    # The Semantic Fit row's one empty cell holds the tier (hatch) legend.
    tier_ax = axes[semfit_row, n_semfit]
    tier_ax.set_visible(True)
    tier_ax.axis("off")
    tier_handles = [mpatches.Patch(facecolor="grey", alpha=TIER_ALPHA[t],
                                    hatch=TIER_HATCH[t], edgecolor="grey", label=t)
                     for t in TIERS]
    tier_ax.legend(handles=tier_handles, loc="center right", fontsize=LEGEND_FS,
                    frameon=True, title="Candidate Tier", title_fontsize=LEGEND_FS + 2)

    # Horizontal section header above each row, left-aligned over its first
    # panel — a rotated sidebar label would have to fit e.g. "Language
    # Complexity" (20 characters) within one row's height, which overflows
    # into neighbouring rows once rows are this short. Anchored to each
    # row's TIGHT bbox (title text included, not just the axes box) so the
    # header clears the panel titles regardless of font size.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for row_idx, group in enumerate(COMPREHENSIVE_GROUPS):
        row_axes = [a for a in axes[row_idx] if a.get_visible()]
        tops = [inv.transform(a.get_tightbbox(renderer))[1, 1] for a in row_axes]
        row_top = max(tops)
        row_left = axes[row_idx, 0].get_position().x0
        fig.text(row_left, row_top + 0.006, group,
                  va="bottom", ha="left", fontsize=ROW_FS, fontweight="bold")

    fig.suptitle("Cover-Letter Text Features by Writer Model and Candidate Tier",
                 fontsize=SUPTITLE_FS, y=0.985)
    path = os.path.join(save_dir, "cl_features_comprehensive.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cl_features_comprehensive.png")


def plot_overview(stats: pd.DataFrame, save_dir: str):
    n     = len(FEATURES)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows),
                             )
    axes = np.array(axes).flatten()

    for i, (col, label, _) in enumerate(FEATURES):
        _draw_feature(axes[i], stats[stats["col"] == col], label)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    _finalise(fig, "Cover-Letter Text Features by Writer Model and Candidate Tier",
              os.path.join(save_dir, "cl_features_overview.png"))
    print("  Saved cl_features_overview.png")



def plot_radar_group(df: pd.DataFrame, group: str, group_features: list, save_dir: str):
    """Radar plot for one feature group: one polygon per model, min-max scaled to [0,1]."""
    cols   = [col for col, _, _ in group_features]
    labels = [lbl for _, lbl, _ in group_features]
    N      = len(cols)
    if N < 3:
        return  # radar needs at least 3 axes

    # min-max scale per feature across all observations
    scaled = df[["Writer"] + cols].copy()
    for col in cols:
        mn, mx = df[col].min(), df[col].max()
        scaled[col] = (df[col] - mn) / (mx - mn) if mx > mn else 0.0

    means = scaled.groupby("Writer")[cols].mean().reindex(WRITERS)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for writer in WRITERS:
        vals = means.loc[writer, cols].tolist()
        vals_closed = vals + vals[:1]
        color = WRITER_COLORS[writer]
        ax.plot(angles_closed, vals_closed, color=color, linewidth=2,
                label=MODEL_DISPLAY[writer])
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.07)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=fs + 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=fs - 1, color="grey")
    ax.grid(True, alpha=0.4)
    ax.set_title(f"Feature Profile — {group}\n(min-max scaled to [0,1])",
                 fontsize=fs + 4, pad=18)

    handles = [mpatches.Patch(color=WRITER_COLORS[w], label=MODEL_DISPLAY[w])
               for w in WRITERS]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=fs, frameon=True)

    fname = f"cl_radar_{group.lower().replace(' & ', '_').replace(' ', '_')}.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_emotion_heatmap(df: pd.DataFrame, save_dir: str):
    """Model × emotion heatmap, two panels (High-Fit | Moderate-Fit)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, tier in zip(axes, TIERS):
        sub = df[df["Tier"] == tier]
        mat = (
            sub.groupby("Writer")[EMO_COLS]
            .mean()
            .reindex(WRITERS)
            .rename(index=MODEL_DISPLAY, columns=dict(zip(EMO_COLS, EMO_LABELS)))
        )
        annot = mat.round(2).astype(str)
        sns.heatmap(mat, annot=annot, fmt="", ax=ax,
                    cmap="YlOrRd", vmin=0, vmax=mat.values.max() * 1.05,
                    linewidths=0.5, linecolor="white",
                    annot_kws={"size": fs + 1},
                    cbar_kws={"label": "Mean probability", "shrink": 0.7})
        ax.set_title(f"Emotions — {tier}", fontsize=fs + 5, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs + 1, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=fs + 1)
        for tick in ax.get_yticklabels():
            writer_key = next((w for w in WRITERS if MODEL_DISPLAY[w] == tick.get_text()), None)
            if writer_key:
                tick.set_color(WRITER_COLORS[writer_key])

    fig.suptitle("Emotion Profile by Writer Model and Candidate Tier",
                 fontsize=fs + 7, y=1.01)
    fig.tight_layout()
    path = os.path.join(save_dir, "cl_features_emotions_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cl_features_emotions_heatmap.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(IN_PATH)
    df["job_minus_cv_sim"] = df["job_cosine_sim"] - df["cv_cosine_sim"]
    df = _add_tier_column(df)

    # Aggregate mean ± std per (Writer, Tier) for every feature (union of
    # FEATURES and COMPREHENSIVE_FEATURES, since the latter also needs
    # emo_joy/emo_neutral)
    feature_cols = sorted(set(col for col, _, _ in FEATURES) |
                           set(col for col, _, _ in COMPREHENSIVE_FEATURES))
    rows = []
    for col in feature_cols:
        sub = (
            df.groupby(["Writer", "Tier"])[col]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        sub["col"] = col
        rows.append(sub)
    stats = pd.concat(rows, ignore_index=True)

    # Per-group bar plots (all groups except Emotions)
    groups = {}
    for col, label, group in FEATURES:
        groups.setdefault(group, []).append((col, label, group))
    for group, group_features in groups.items():
        plot_group(stats, group, group_features, OUT_DIR)

    # Emotion heatmap (separate)
    plot_emotion_heatmap(df, OUT_DIR)

    # Comprehensive single-figure overview for the paper (grouped by class)
    plot_comprehensive(stats, OUT_DIR)

    # Radar plots (one per feature group + emotions)
    for group, group_features in groups.items():
        plot_radar_group(df, group, group_features, OUT_DIR)
    emo_features = [(c, l, "Emotions") for c, l in zip(EMO_COLS, ["Joy","Neutral","Surprise","Fear","Anger","Disgust","Sadness"])]
    plot_radar_group(df, "Emotions", emo_features, OUT_DIR)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
