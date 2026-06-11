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

IN_PATH = "output_eval/cl_features.parquet"
OUT_DIR = "output_plots/cl_features"
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
    ("flesch_kincaid_grade", "Flesch-Kincaid Grade",       "Language Complexity"),
    # 3 — sentiment & affect
    ("vader_compound",       "VADER Sentiment",            "Sentiment & Affect"),
    ("vad_valence",          "VAD Valence",                "Sentiment & Affect"),
    ("vad_arousal",          "VAD Arousal",                "Sentiment & Affect"),
    ("vad_dominance",        "VAD Dominance",              "Sentiment & Affect"),
    # 4 — semantic fit
    ("job_cosine_sim",       "Cosine Sim. to Job Ad",      "Semantic Fit"),
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


def _draw_feature(ax, stats: pd.DataFrame, label: str, ylim=None):
    """One group per writer model; two adjacent bars per group (High-Fit | Moderate-Fit)."""
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
                   error_kw={"capsize": 2, "linewidth": 1, "ecolor": "black"})

    ax.set_xticks([i * step for i in range(len(WRITERS))])
    ax.set_xticklabels([MODEL_DISPLAY[w] for w in WRITERS],
                       rotation=40, ha="right", fontsize=fs)
    ax.set_title(label, fontsize=fs + 3, pad=6)
    ax.tick_params(axis="y", labelsize=fs)
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
    df = _add_tier_column(df)

    # Aggregate mean ± std per (Writer, Tier) for every feature
    feature_cols = [col for col, _, _ in FEATURES]
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

    # Overview plot
    plot_overview(stats, OUT_DIR)

    # Per-group bar plots (all groups except Emotions)
    groups = {}
    for col, label, group in FEATURES:
        groups.setdefault(group, []).append((col, label, group))
    for group, group_features in groups.items():
        plot_group(stats, group, group_features, OUT_DIR)

    # Emotion heatmap (separate)
    plot_emotion_heatmap(df, OUT_DIR)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
