"""
analyze_keyness.py — log-likelihood keyness analysis of cover letters by writer model.

For each model, compares its lemma distribution against all other models combined (model vs. rest).
Produces:
  - output_plots/keyness/keyness_global.png   : aggregated across all jobs
  - output_plots/keyness/keyness_job_<id>.png : one per job

Method: G² (log-likelihood ratio) test, standard in corpus linguistics.
  Positive G² = lemma overrepresented in target model.
  Only top-K positive key terms are shown (distinctive vocabulary).

Filters: stopwords, punctuation, numbers, tokens < 3 chars removed via spaCy.
Minimum frequency: lemma must appear ≥ MIN_FREQ times in the target corpus.
"""

import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import spacy
import seaborn as sns

from aggregate_plots import WRITER_COLORS, MODEL_DISPLAY, UNIQUE_EVALUATORS

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

CL_DIR  = "output_cl"
OUT_DIR = "output_plots/keyness"
fs      = plt.rcParams["font.size"]

TOP_K    = 15   # key terms to show per model
MIN_FREQ = 3    # min occurrences in target corpus to be eligible
WRITERS  = list(WRITER_COLORS)


# ── lemmatisation ─────────────────────────────────────────────────────────────

def _keep(token) -> bool:
    return (
        not token.is_stop
        and not token.is_punct
        and not token.like_num
        and token.is_alpha
        and len(token.lemma_) >= 3
    )


def lemmatise_texts(texts: list[str], nlp) -> list[list[str]]:
    result = []
    for doc in nlp.pipe(texts, batch_size=64, disable=["ner", "parser"]):
        result.append([t.lemma_.lower() for t in doc if _keep(t)])
    return result


# ── keyness ───────────────────────────────────────────────────────────────────

def log_likelihood(a: int, b: int, N1: int, N2: int) -> float:
    """G² score; positive = overrepresented in target (a side)."""
    if a + b == 0:
        return 0.0
    E1 = N1 * (a + b) / (N1 + N2)
    E2 = N2 * (a + b) / (N1 + N2)
    score = 0.0
    if a > 0 and E1 > 0:
        score += a * np.log(a / E1)
    if b > 0 and E2 > 0:
        score += b * np.log(b / E2)
    score *= 2
    if N1 > 0 and N2 > 0 and a / N1 < b / N2:
        score = -score
    return score


def compute_keyness(freq_by_writer: dict[str, Counter]) -> dict[str, list[tuple]]:
    """
    For each writer: G² vs. pooled rest.
    Returns {writer: [(lemma, g2_score), ...]} sorted descending by g2.
    """
    all_lemmas = set(w for c in freq_by_writer.values() for w in c)
    total_by_writer = {w: sum(c.values()) for w, c in freq_by_writer.items()}
    results = {}

    for target in WRITERS:
        if target not in freq_by_writer:
            results[target] = []
            continue
        target_freq  = freq_by_writer[target]
        ref_freq     = Counter()
        for w, c in freq_by_writer.items():
            if w != target:
                ref_freq += c
        N1 = total_by_writer[target]
        N2 = sum(v for w, v in total_by_writer.items() if w != target)

        scores = []
        for lemma in all_lemmas:
            a = target_freq.get(lemma, 0)
            if a < MIN_FREQ:
                continue
            b  = ref_freq.get(lemma, 0)
            g2 = log_likelihood(a, b, N1, N2)
            scores.append((lemma, g2))
        scores.sort(key=lambda x: x[1], reverse=True)
        results[target] = scores
    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_keyness(keyness: dict, title: str, out_path: str):
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 6))
    axes = axes.flatten()

    for ax, writer in zip(axes, WRITERS):
        top = [(lm, g2) for lm, g2 in keyness.get(writer, []) if g2 > 0][:TOP_K]
        if not top:
            ax.set_visible(False)
            continue
        lemmas, scores = zip(*top)
        # reverse so highest is at top
        lemmas  = list(reversed(lemmas))
        scores  = list(reversed(scores))
        color   = WRITER_COLORS[writer]
        bars    = ax.barh(range(len(lemmas)), scores, color=color, alpha=0.85)
        ax.set_yticks(range(len(lemmas)))
        ax.set_yticklabels(lemmas, fontsize=fs + 1)
        ax.set_xlabel("G²", fontsize=fs)
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
            writer = m.group(1)
            with open(os.path.join(job_dir, fname)) as f:
                data[jid][writer].append(f.read().strip())
    return data


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading cover letters...")
    data = load_texts()
    job_ids = sorted(data.keys())
    print(f"  {len(job_ids)} jobs, {sum(len(ws) for ws in data.values())} writer groups")

    # flatten all texts for a single lemmatisation pass
    # index: (job_id, writer, text_idx) → position in flat list
    flat_texts, flat_index = [], []
    for jid in job_ids:
        for writer, texts in data[jid].items():
            for i, t in enumerate(texts):
                flat_texts.append(t)
                flat_index.append((jid, writer))

    print(f"Lemmatising {len(flat_texts)} texts...")
    flat_lemmas = lemmatise_texts(flat_texts, nlp)

    # reconstruct per (job_id, writer) lemma counters
    global_freq: dict[str, Counter] = defaultdict(Counter)
    per_job_freq: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for (jid, writer), lemmas in zip(flat_index, flat_lemmas):
        global_freq[writer].update(lemmas)
        per_job_freq[jid][writer].update(lemmas)

    # ── global plot ──
    print("\nComputing global keyness...")
    global_keyness = compute_keyness(global_freq)
    plot_keyness(global_keyness,
                 "Keyword Distinctiveness by Writer Model (Global, model vs. rest)",
                 os.path.join(OUT_DIR, "keyness_global.png"))

    # ── per-job plots ──
    print("Computing per-job keyness...")
    for jid in job_ids:
        keyness = compute_keyness(per_job_freq[jid])
        plot_keyness(keyness,
                     f"Keyword Distinctiveness — {jid}",
                     os.path.join(OUT_DIR, f"keyness_{jid}.png"))

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
