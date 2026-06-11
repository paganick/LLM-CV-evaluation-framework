"""
analyze_empath.py — Empath topical analysis of cover letters by writer model.

For each model, compares its mean Empath category scores against all other models
combined, using Cohen's d as the effect size (positive d = category overrepresented
in the target model relative to the rest).

Produces:
  - output_plots/empath/empath_global.png   : aggregated across all jobs
  - output_plots/empath/empath_job_<id>.png : one per job

Empath has 194 predefined topical categories (achievement, work, positive_emotion, …).
Categories with negligible global mean are filtered out before ranking.
"""

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from empath import Empath

from aggregate_plots import WRITER_COLORS, MODEL_DISPLAY

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

CL_DIR  = "output_cl"
OUT_DIR = "output_plots/empath"
fs      = plt.rcParams["font.size"]

TOP_K       = 15     # categories to show per model
MIN_MEAN    = 5e-4   # global mean must exceed this to be eligible
WRITERS     = list(WRITER_COLORS)

# Curated subset of Empath categories relevant to professional/cover-letter text.
# Excludes categories that produce false positives via semantic extension
# (e.g. "execute" → crime, "strong" → masculine).
RELEVANT_CATS = {
    # work & career
    "work", "occupation", "office", "business", "white_collar_job",
    "achievement", "competing", "leader", "power", "strength",
    "independence", "pride", "order", "meeting", "negotiate",
    "dominant_personality", "heroic",
    # communication & knowledge
    "communication", "speaking", "writing", "reading", "journalism",
    "school", "college", "science", "technology", "computer",
    "internet", "programming",
    # positive affect
    "positive_emotion", "cheerfulness", "optimism", "contentment",
    "joy", "zest", "anticipation", "trust", "warmth", "affection",
    "celebration", "fun",
    # social & relational
    "help", "giving", "sympathy", "healing", "politeness", "friends",
    # economic
    "money", "economics", "payment", "valuable", "wealthy", "gain", "banking",
    # governance & institutions
    "government", "law", "politics",
    # health (relevant for medical/HR roles)
    "health",
    # creative
    "art", "music",
    # negative affect (worth tracking even if rare)
    "negative_emotion", "disappointment", "nervousness",
}


# ── Empath scoring ────────────────────────────────────────────────────────────

def score_texts(texts: list[str], lexicon: Empath) -> np.ndarray:
    """Returns (n_texts, n_cats) array; preserves category order from lexicon.cats."""
    cats = list(lexicon.cats)
    rows = []
    for text in texts:
        scores = lexicon.analyze(text, normalize=True) or {}
        rows.append([scores.get(c, 0.0) for c in cats])
    return np.array(rows, dtype=np.float32), cats


# ── effect size ───────────────────────────────────────────────────────────────

def cohens_d(target: np.ndarray, rest: np.ndarray) -> float:
    n1, n2 = len(target), len(rest)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = target.mean(), rest.mean()
    v1 = target.var(ddof=1)
    v2 = rest.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return float((m1 - m2) / pooled) if pooled > 0 else 0.0


def compute_effects(scores_by_writer: dict[str, np.ndarray],
                    cats: list[str]) -> dict[str, list[tuple]]:
    """
    Returns {writer: [(category, cohen_d), ...]} sorted by d descending.
    Only positive d (overrepresented) retained; categories with negligible
    global mean filtered out.
    """
    all_scores  = np.vstack(list(scores_by_writer.values()))
    global_mean = all_scores.mean(axis=0)
    eligible    = np.where(
        (global_mean > MIN_MEAN) &
        np.array([c in RELEVANT_CATS for c in cats])
    )[0]

    results = {}
    for writer in WRITERS:
        if writer not in scores_by_writer:
            results[writer] = []
            continue
        target = scores_by_writer[writer]
        rest   = np.vstack([v for w, v in scores_by_writer.items() if w != writer])
        effects = []
        for ci in eligible:
            d = cohens_d(target[:, ci], rest[:, ci])
            if d > 0:
                effects.append((cats[ci], d))
        effects.sort(key=lambda x: x[1], reverse=True)
        results[writer] = effects
    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def _fmt_cat(name: str) -> str:
    return name.replace("_", " ").title()


def plot_empath(effects: dict, title: str, out_path: str):
    ncols, nrows = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 6))
    axes = axes.flatten()

    for ax, writer in zip(axes, WRITERS):
        top = effects.get(writer, [])[:TOP_K]
        if not top:
            ax.set_visible(False)
            continue
        cats, ds = zip(*top)
        cats = [_fmt_cat(c) for c in reversed(cats)]
        ds   = list(reversed(ds))
        color = WRITER_COLORS[writer]
        ax.barh(range(len(cats)), ds, color=color, alpha=0.85)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=fs + 1)
        ax.set_xlabel("Cohen's d", fontsize=fs)
        ax.set_title(MODEL_DISPLAY[writer], fontsize=fs + 3,
                     fontweight="bold", color=color, pad=6)
        ax.tick_params(axis="x", labelsize=fs)
        ax.grid(True, alpha=0.3, axis="x")
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(len(WRITERS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=fs + 8, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out_path)}")


# ── data loading ──────────────────────────────────────────────────────────────

def _job_id(folder: str) -> str:
    m = re.match(r"(job_\d+)", folder)
    return m.group(1) if m else folder


def load_texts() -> dict:
    """Returns {job_id: {writer: [text, ...]}}"""
    data = defaultdict(lambda: defaultdict(list))
    for job_folder in sorted(os.listdir(CL_DIR)):
        job_dir = os.path.join(CL_DIR, job_folder)
        if not os.path.isdir(job_dir):
            continue
        jid = _job_id(job_folder)
        for fname in sorted(os.listdir(job_dir)):
            if not fname.endswith(".txt"):
                continue
            m = re.match(r"(.+)_cover_letter_cv\d+\.txt", fname)
            if not m:
                continue
            with open(os.path.join(job_dir, fname)) as f:
                data[jid][m.group(1)].append(f.read().strip())
    return data


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading Empath lexicon...")
    lexicon = Empath()
    cats    = list(lexicon.cats)
    print(f"  {len(cats)} categories")

    print("Loading cover letters...")
    data    = load_texts()
    job_ids = sorted(data.keys())

    # single scoring pass over all texts
    flat_texts, flat_index = [], []
    for jid in job_ids:
        for writer, texts in data[jid].items():
            for text in texts:
                flat_texts.append(text)
                flat_index.append((jid, writer))

    print(f"Scoring {len(flat_texts)} cover letters with Empath...")
    all_scores, cats = score_texts(flat_texts, lexicon)

    # reconstruct per-(job, writer) score matrices
    global_scores: dict[str, list] = defaultdict(list)
    per_job_scores: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for (jid, writer), row in zip(flat_index, all_scores):
        global_scores[writer].append(row)
        per_job_scores[jid][writer].append(row)

    global_arrays  = {w: np.array(v) for w, v in global_scores.items()}
    per_job_arrays = {jid: {w: np.array(v) for w, v in ws.items()}
                      for jid, ws in per_job_scores.items()}

    # ── global ──
    print("\nComputing global Empath effects...")
    plot_empath(compute_effects(global_arrays, cats),
                "Topical Distinctiveness by Writer Model — Empath (Global, model vs. rest)",
                os.path.join(OUT_DIR, "empath_global.png"))

    # ── per job ──
    print("Computing per-job Empath effects...")
    for jid in job_ids:
        plot_empath(compute_effects(per_job_arrays[jid], cats),
                    f"Topical Distinctiveness — Empath  |  {jid}",
                    os.path.join(OUT_DIR, f"empath_{jid}.png"))

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
