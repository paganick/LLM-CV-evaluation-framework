"""
plot_model_explainability.py — Per-model explainability of scoring/writing practices.

Extends the nested-model idea (plot_nested_model_decomposition.py) to the level
of individual models. Two separate kinds of plot, kept apart on purpose: fine
per-feature detail needs its own scale, or the (much larger) semantic
contribution flattens it into unreadable slivers.

Job + candidate fit Tier (High-Fit/Moderate-Fit) are baseline controls in every
analysis below. Tier matters specifically because cosine similarity to the
job/CV is itself correlated with candidate quality (a stronger CV is, by
construction, more semantically aligned with the job) — without controlling
for Tier, "Semantic" partly stands in for "this is just a better candidate"
rather than something attributable to the writer's letter.

1. Style breakdown, per model (evaluator_explainability.png / writer_explainability.png)

   Baseline = Job + Tier + Semantic + Writer-or-Evaluator (semantic is removed
   first, like the nested-model script). ΔR² from adding STYLE features only,
   per model, stacked by individual feature (own fine scale). Category totals
   are exact Shapley (LMG); the within-category feature-level split is an
   approximate share (Ridge coefficient x correlation with residual score,
   renormalized to the category's exact total).

2. Semantic vs. style balance, per model (evaluator_semantic_vs_style.png / writer_semantic_vs_style.png)

   Baseline = Job + Tier + Writer-or-Evaluator. Exact Shapley split of R²
   across 5 blocks: Semantic Fit + the 4 style categories (no individual-
   feature detail — this plot is about balance, not drivers). Shows which
   models' explainable score variance is dominated by semantic fit vs. spread
   more evenly into style.

3. Genuine candidate fit vs. Writer identity, per evaluator
   (evaluator_tier_vs_writer.png / evaluator_cvjob_vs_writer.png)

   Baseline = Job only. Exact Shapley split of R² between genuine candidate
   fit — either Tier (categorical) or CV-Job cosine similarity (continuous,
   computed on the raw CV/job text, before any cover letter is written) — and
   Writer identity (categorical — captures anything writer-specific: style,
   self-preference, unmeasured quirks). Semantic Fit is deliberately excluded
   from this comparison: it's measured on the produced letter, so it isn't a
   clean candidate-only signal the way Tier / CV-Job Fit are.

4. Does style matter more for high- or moderate-fit candidates?
   (tier_semantic_vs_style.png)

   Pooled (across all evaluators), baseline = Job + Writer + Evaluator.
   Semantic-vs-style Shapley split computed separately within each Tier.

5. Genuine fit vs. everything else the letter reflects
   (tier_vs_rest.png / cvjob_vs_rest.png / tier_and_cvjob_vs_rest.png)

   Pooled, baseline = Job + Evaluator. Exact Shapley split of R² between
   genuine candidate fit (Tier, continuous CV-Job Fit, or both together) and
   Semantic Fit + the 4 style categories.

Outputs:
  evaluator_explainability.png        writer_explainability.png
  evaluator_semantic_vs_style.png     writer_semantic_vs_style.png
  evaluator_tier_vs_writer.png        evaluator_cvjob_vs_writer.png
  tier_semantic_vs_style.png
  tier_vs_rest.png  cvjob_vs_rest.png  tier_and_cvjob_vs_rest.png
"""

import os
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from sklearn.linear_model import RidgeCV
from adjustText import adjust_text

from aggregate_plots import MODEL_DISPLAY, DISPLAY_COLORS, UNIQUE_EVALUATORS, RAW_WRITERS
from plot_feature_regression import (
    FEATURES_PATH, STYLE_COLS, COSINE_COLS, FEATURE_LABELS, FEATURE_GROUPS_MAP, fit_regressions,
)
from plot_nested_model_decomposition import MASTER_PATH, OUT_DIR, EVAL_TYPES, r2_ols

fs = plt.rcParams["font.size"]

MIN_N  = 30
ALPHAS = np.logspace(-2, 3, 40)

ALL_COLS = STYLE_COLS + list(COSINE_COLS)

STYLE_CATEGORY_ORDER = ["Length & Structure", "Language Complexity", "Sentiment & Affect", "Emotions"]
STYLE_CATEGORY_FEATURES = {cat: [c for c in STYLE_COLS if FEATURE_GROUPS_MAP[c] == cat] for cat in STYLE_CATEGORY_ORDER}
STYLE_CATEGORY_CMAPS = {
    "Length & Structure":  "Blues",
    "Language Complexity": "Greens",
    "Sentiment & Affect":  "Oranges",
    "Emotions":            "Purples",
}

BLOCK_ORDER = ["Job-Ad Fit", "CV Consistency"] + STYLE_CATEGORY_ORDER
BLOCK_FEATURES = {"Job-Ad Fit": ["job_cosine_sim"], "CV Consistency": ["cv_cosine_sim"], **STYLE_CATEGORY_FEATURES}
BLOCK_COLORS = {
    "Job-Ad Fit":          "#424242",
    "CV Consistency":      "#AD1457",
    "Length & Structure":  "#1565C0",
    "Language Complexity": "#2E7D32",
    "Sentiment & Affect":  "#E65100",
    "Emotions":            "#6A1B9A",
}

TIER_BLOCK_ORDER = ["Tier"] + BLOCK_ORDER
TIER_BLOCK_COLORS = {"Tier": "#FBC02D", **BLOCK_COLORS}

CVJOB_BLOCK_ORDER = ["CV-Job Fit"] + BLOCK_ORDER
CVJOB_BLOCK_COLORS = {"CV-Job Fit": "#00897B", **BLOCK_COLORS}

BOTH_BLOCK_ORDER = ["Tier", "CV-Job Fit"] + BLOCK_ORDER
BOTH_BLOCK_COLORS = {"Tier": "#FBC02D", "CV-Job Fit": "#00897B", **BLOCK_COLORS}


def _drop_cv_consistency(eval_type, order, features=None):
    """"CV Consistency" (letter-to-CV cosine similarity) isn't a meaningful
    block in Cover-Letter-Only: the evaluator's prompt never contains the CV,
    so it cannot be reacting to how well the letter echoes it — any observed
    effect would be a spurious correlate of something else, not a genuine
    consistency check. Dropped for that Eval_Type everywhere blocks are
    scored/interpreted (same reasoning as compute_evaluator_inflation_reward's
    CV-Echo Inflation)."""
    if eval_type != "cl_evaluations":
        return (order, features) if features is not None else order
    order2 = [b for b in order if b != "CV Consistency"]
    if features is None:
        return order2
    features2 = {k: v for k, v in features.items() if k != "CV Consistency"}
    return order2, features2

# "Substance" = this evaluator's OWN CV-only score for this candidate (mean over
# its 4 runs) — a holistic, self-referential quality judgment, not just a lexical
# proxy (cv_job_cosine_sim) or a coarse label (Tier). Used per-evaluator, so
# "own" always means the SAME model being analyzed in that row of the plot.
SUBSTANCE_BLOCK_ORDER = ["Substance (Own CV View)"] + BLOCK_ORDER
SUBSTANCE_BLOCK_FEATURES = {"Substance (Own CV View)": ["cv_only_score_self"], **BLOCK_FEATURES}
SUBSTANCE_BLOCK_COLORS = {"Substance (Own CV View)": "#5D4037", **BLOCK_COLORS}


def _feature_colors():
    """One shade per style feature, drawn from its category's colormap."""
    colors = {}
    for cat in STYLE_CATEGORY_ORDER:
        cols = STYLE_CATEGORY_FEATURES[cat]
        cmap = cm.get_cmap(STYLE_CATEGORY_CMAPS[cat])
        shades = cmap(np.linspace(0.45, 0.9, len(cols)))
        for c, shade in zip(cols, shades):
            colors[c] = shade
    return colors


FEATURE_COLORS = _feature_colors()


def _residualize(sub, baseline_cats, continuous_baseline_cols=None):
    """Score residual after removing baseline categorical FE + optional standardized continuous cols."""
    y = sub["Score"].values.astype(float)
    parts = []
    if baseline_cats:
        parts.append(pd.get_dummies(sub[list(baseline_cats)], drop_first=True).astype(float).values)
    if continuous_baseline_cols:
        x = sub[list(continuous_baseline_cols)].astype(float)
        x = (x - x.mean()) / x.std().replace(0, 1)
        parts.append(x.values)
    if parts:
        X = np.column_stack(parts)
        X_int = np.column_stack([np.ones(len(X)), X])
    else:
        X_int = np.ones((len(sub), 1))
    coef, *_ = np.linalg.lstsq(X_int, y, rcond=None)
    return y - X_int @ coef


def _shapley_over_blocks(y_resid, blocks):
    """Exact Shapley (LMG) value of each block's R² contribution to y_resid.
    blocks: dict[name -> design matrix] (categorical dummies or standardized continuous)."""
    names = list(blocks)
    n = len(names)

    def r2_subset(subset):
        if not subset:
            return 0.0
        X = np.column_stack([blocks[c] for c in subset])
        return r2_ols(y_resid, X)

    all_subsets = [frozenset(c) for r in range(n + 1) for c in combinations(names, r)]
    r2_cache = {s: r2_subset(s) for s in all_subsets}

    shapley = {}
    for name in names:
        others = [c for c in names if c != name]
        total = 0.0
        for r in range(len(others) + 1):
            for combo in combinations(others, r):
                S = frozenset(combo)
                weight = factorial(r) * factorial(n - r - 1) / factorial(n)
                total += weight * (r2_cache[S | {name}] - r2_cache[S])
        shapley[name] = total
    return shapley, r2_cache[frozenset(names)]


# ── 1. Style breakdown (semantic removed via baseline, individual feature detail) ──

def compute_style_contributions(sub, baseline_cats):
    """Returns a Series indexed by STYLE_COLS (per-feature ΔR² contribution,
    semantic already removed via baseline) and the total style ΔR²."""
    y_resid = _residualize(sub, baseline_cats, COSINE_COLS)

    Xstd_all = sub[STYLE_COLS].astype(float)
    Xstd_all = (Xstd_all - Xstd_all.mean()) / Xstd_all.std().replace(0, 1)
    Xstd_by_cat = {cat: Xstd_all[cols].values for cat, cols in STYLE_CATEGORY_FEATURES.items()}

    cat_shapley, total_r2 = _shapley_over_blocks(y_resid, Xstd_by_cat)

    ridge = RidgeCV(alphas=ALPHAS, fit_intercept=False)
    ridge.fit(Xstd_all.values, y_resid)
    coefs = pd.Series(ridge.coef_, index=STYLE_COLS)
    corrs = pd.Series({c: np.corrcoef(Xstd_all[c], y_resid)[0, 1] for c in STYLE_COLS})
    pratt = (coefs * corrs).clip(lower=0)

    contrib = pd.Series(0.0, index=STYLE_COLS)
    for cat, cols in STYLE_CATEGORY_FEATURES.items():
        w = pratt[cols]
        share = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(cols), index=cols)
        contrib[cols] = share * cat_shapley[cat]

    return contrib, total_r2


def compute_style_explainability(merged, eval_type, group_col, group_values, baseline_cats):
    sub_all = merged[merged["Eval_Type"] == eval_type]
    rows, contribs = [], {}
    for g in group_values:
        sub = sub_all[sub_all[group_col] == g].dropna(subset=ALL_COLS + ["Score"]).copy()
        if len(sub) < MIN_N:
            continue
        contrib, total_r2 = compute_style_contributions(sub, baseline_cats)
        contribs[MODEL_DISPLAY[g]] = contrib
        rows.append({"Model": MODEL_DISPLAY[g], "delta_R2": total_r2})

    order_df = pd.DataFrame(rows).sort_values("delta_R2", ascending=False)
    contrib_df = pd.DataFrame(contribs).T.loc[order_df["Model"]]
    return order_df.set_index("Model")["delta_R2"], contrib_df


def plot_style_breakdown(all_results, title, subtitle, out_name):
    """all_results: {eval_type: (delta_r2 Series, contrib_df DataFrame)}"""
    # subtitle is kept short at the call site for figures used in the paper
    # (full methodological detail goes in the LaTeX caption instead); wrap
    # here is a safety net so a long string can't force the canvas wide.
    BIG = 10
    fig, axes = plt.subplots(1, len(all_results), figsize=(10 * len(all_results), 9.5))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (eval_type, (delta_r2, contrib_df)) in zip(axes, all_results.items()):
        models = delta_r2.index[::-1]
        y_pos = list(range(len(models)))

        for j, m in enumerate(models):
            left = 0.0
            for cat in STYLE_CATEGORY_ORDER:
                for feat in STYLE_CATEGORY_FEATURES[cat]:
                    v = contrib_df.loc[m, feat]
                    ax.barh(j, v, left=left, color=FEATURE_COLORS[feat],
                            edgecolor="white", linewidth=0.3, height=0.6)
                    left += v
            ax.text(left + delta_r2.max() * 0.015, j, f"{left:.3f}",
                    va="center", fontsize=fs + BIG - 1, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=fs + BIG + 1, fontweight="bold")
        for tick, m in zip(ax.get_yticklabels(), models):
            tick.set_color(DISPLAY_COLORS.get(m, "black"))
        ax.set_xlim(0, delta_r2.max() * 1.2)
        ax.set_xlabel("ΔR²  (style features)", fontsize=fs + BIG + 2)
        ax.set_title(EVAL_TYPES[eval_type], fontsize=fs + BIG + 5, pad=12)
        ax.tick_params(axis="x", labelsize=fs + BIG - 1)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)

    cat_x = np.linspace(0.16, 0.84, len(STYLE_CATEGORY_ORDER))
    for x, cat in zip(cat_x, STYLE_CATEGORY_ORDER):
        handles = [Patch(facecolor=FEATURE_COLORS[f], label=FEATURE_LABELS[f]) for f in STYLE_CATEGORY_FEATURES[cat]]
        leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(x, 0.21),
                          ncol=1, fontsize=fs + BIG - 5, frameon=False, title=cat, title_fontsize=fs + BIG - 3)
        fig.add_artist(leg)

    fig.suptitle(title, fontsize=fs + BIG + 5, wrap=True)
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=fs + BIG - 4,
                  color="dimgrey", wrap=True)
    fig.tight_layout(rect=[0, 0.22, 1, 0.89])
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


# ── 2 & 3. Block-balance plots (Semantic vs. style categories / Semantic vs. Writer) ──

CATEGORICAL_DUMMY_SENTINELS = {"WRITER_DUMMY": "Writer", "TIER_DUMMY": "Tier"}


def compute_block_balance(merged, eval_type, group_col, group_values, baseline_cats, blocks_spec,
                           sort_col="delta_R2"):
    """blocks_spec: dict[block_name -> list of raw feature columns] to standardize,
    OR one of the CATEGORICAL_DUMMY_SENTINELS keys (e.g. "WRITER_DUMMY", "TIER_DUMMY")
    meaning categorical dummies of that column instead.

    sort_col: which column to rank rows by (descending, largest at top when
    plotted) — defaults to total ΔR², but can be set to a single block name
    (e.g. "Writer") to rank by that block's own contribution instead."""
    sub_all = merged[merged["Eval_Type"] == eval_type]
    rows = {}
    for g in group_values:
        needed_cols = [c for cols in blocks_spec.values()
                       if not (isinstance(cols, str) and cols in CATEGORICAL_DUMMY_SENTINELS) for c in cols]
        sub = sub_all[sub_all[group_col] == g].dropna(subset=needed_cols + ["Score"]).copy()
        if len(sub) < MIN_N:
            continue
        y_resid = _residualize(sub, baseline_cats)
        blocks = {}
        for name, cols in blocks_spec.items():
            if isinstance(cols, str) and cols in CATEGORICAL_DUMMY_SENTINELS:
                col = CATEGORICAL_DUMMY_SENTINELS[cols]
                blocks[name] = pd.get_dummies(sub[col], drop_first=True).astype(float).values
            else:
                x = sub[cols].astype(float)
                x = (x - x.mean()) / x.std().replace(0, 1)
                blocks[name] = x.values
        shapley, total_r2 = _shapley_over_blocks(y_resid, blocks)
        rows[MODEL_DISPLAY[g]] = {**shapley, "delta_R2": total_r2}
    return pd.DataFrame(rows).T.sort_values(sort_col, ascending=False)


def compute_substance_style_drilldown(merged, eval_type, group_col, group_values, blocks_spec):
    """Drills into the STYLE portion of compute_block_balance's substance-vs-
    style split: the SAME baseline (Job_ID only) and the SAME blocks_spec
    (e.g. SUBSTANCE_BLOCK_FEATURES) — so each style category's total here is
    IDENTICAL to evaluator_substance_vs_style.png's bars. This decomposes
    that plot's style portion feature by feature; it is not a separately-
    scoped analysis with its own baseline (unlike the older
    compute_style_explainability, which uses Job+Tier+Writer+Semantic).

    Individual-feature credit within each style category is an approximate
    share (Ridge coefficient x correlation with the residual after removing
    every OTHER block — Substance, Job-Ad Fit, CV Consistency — clipped,
    renormalized to match that category's exact Shapley total from the
    category-level split computed with the identical baseline/blocks)."""
    sub_all = merged[merged["Eval_Type"] == eval_type]
    non_style_blocks = {k: v for k, v in blocks_spec.items() if k not in STYLE_CATEGORY_ORDER}
    non_style_cols = [c for cols in non_style_blocks.values() for c in cols]
    needed_cols = [c for cols in blocks_spec.values() for c in cols]

    rows, contribs = [], {}
    for g in group_values:
        sub = sub_all[sub_all[group_col] == g].dropna(subset=needed_cols + ["Score"]).copy()
        if len(sub) < MIN_N:
            continue

        # exact category-level Shapley — identical setup to compute_block_balance
        y_resid = _residualize(sub, ["Job_ID"])
        blocks = {}
        for name, cols in blocks_spec.items():
            x = sub[cols].astype(float)
            x = (x - x.mean()) / x.std().replace(0, 1)
            blocks[name] = x.values
        shapley, total_r2 = _shapley_over_blocks(y_resid, blocks)
        style_total = sum(shapley[cat] for cat in STYLE_CATEGORY_ORDER)

        # split each category's exact credit across its individual features
        y_resid_style = _residualize(sub, ["Job_ID"], non_style_cols)
        Xstd_all = sub[STYLE_COLS].astype(float)
        Xstd_all = (Xstd_all - Xstd_all.mean()) / Xstd_all.std().replace(0, 1)
        ridge = RidgeCV(alphas=ALPHAS, fit_intercept=False)
        ridge.fit(Xstd_all.values, y_resid_style)
        coefs = pd.Series(ridge.coef_, index=STYLE_COLS)
        corrs = pd.Series({c: np.corrcoef(Xstd_all[c], y_resid_style)[0, 1] for c in STYLE_COLS})
        pratt = (coefs * corrs).clip(lower=0)

        contrib = pd.Series(0.0, index=STYLE_COLS)
        for cat in STYLE_CATEGORY_ORDER:
            cols = STYLE_CATEGORY_FEATURES[cat]
            w = pratt[cols]
            share = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(cols), index=cols)
            contrib[cols] = share * shapley[cat]

        contribs[MODEL_DISPLAY[g]] = contrib
        rows.append({"Model": MODEL_DISPLAY[g], "delta_R2": style_total})

    order_df = pd.DataFrame(rows).sort_values("delta_R2", ascending=False)
    contrib_df = pd.DataFrame(contribs).T.loc[order_df["Model"]]
    return order_df.set_index("Model")["delta_R2"], contrib_df


def compute_block_balance_by_tier(merged, eval_type, group_col, group_values, blocks_spec):
    """Same as compute_block_balance, but computed separately within each
    Tier (High-Fit / Moderate-Fit) for every group value, baseline = Job_ID
    only (Tier itself is the split, so it's excluded from the baseline, same
    as compute_tier_semantic_vs_style / compute_tier_substance_vs_style).
    Returns a DataFrame with a (Model, Tier) row MultiIndex."""
    sub_all = merged[merged["Eval_Type"] == eval_type]
    needed_cols = [c for cols in blocks_spec.values()
                   if not (isinstance(cols, str) and cols in CATEGORICAL_DUMMY_SENTINELS) for c in cols]
    rows = {}
    for g in group_values:
        for tier, label in [("high_fit", "High-Fit"), ("mod_fit", "Moderate-Fit")]:
            sub = sub_all[(sub_all[group_col] == g) & (sub_all["Tier"] == tier)].dropna(
                subset=needed_cols + ["Score"]).copy()
            if len(sub) < MIN_N:
                continue
            y_resid = _residualize(sub, ["Job_ID"])
            blocks = {}
            for name, cols in blocks_spec.items():
                if isinstance(cols, str) and cols in CATEGORICAL_DUMMY_SENTINELS:
                    col = CATEGORICAL_DUMMY_SENTINELS[cols]
                    blocks[name] = pd.get_dummies(sub[col], drop_first=True).astype(float).values
                else:
                    x = sub[cols].astype(float)
                    x = (x - x.mean()) / x.std().replace(0, 1)
                    blocks[name] = x.values
            shapley, total_r2 = _shapley_over_blocks(y_resid, blocks)
            rows[(MODEL_DISPLAY[g], label)] = {**shapley, "delta_R2": total_r2}
    df = pd.DataFrame(rows).T
    df.index.names = ["Model", "Tier"]
    return df


TIER_HATCH = {"High-Fit": "", "Moderate-Fit": "///"}


def plot_block_balance_by_tier(all_results, block_order, block_colors, title, subtitle, out_name):
    """Like plot_block_balance, but each model gets TWO adjacent bars
    (High-Fit plain, Moderate-Fit hatched — same TIER_HATCH motif as
    plot_cl_features.py, no transparency) so the style/substance split can be
    compared directly within a model across tiers. Models are ordered once,
    by mean ΔR² across both tiers in the CV + Cover Letter panel, and that
    same order is reused for every panel."""
    ref_type = next(et for et in EVAL_TYPES if et in all_results)
    ref_df = all_results[ref_type]
    avg_r2 = ref_df["delta_R2"].groupby(level="Model").mean()
    models = avg_r2.sort_values(ascending=True).index

    fig, axes = plt.subplots(1, len(all_results), figsize=(12 * len(all_results), 10))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (eval_type, df) in zip(axes, all_results.items()):
        blocks_here = [b for b in block_order if b in df.columns]
        y_labels, y_ticks = [], []
        row = 0
        for m in models:
            for tier in ["Moderate-Fit", "High-Fit"]:
                if (m, tier) not in df.index:
                    row += 1
                    continue
                hatch = TIER_HATCH[tier]
                edge = "black" if tier == "Moderate-Fit" else "white"
                left = 0.0
                for block in blocks_here:
                    v = df.loc[(m, tier), block]
                    ax.barh(row, v, left=left, color=block_colors[block], edgecolor=edge,
                            linewidth=0.8, height=0.85, hatch=hatch)
                    pct = v / df.loc[(m, tier), "delta_R2"] * 100 if df.loc[(m, tier), "delta_R2"] > 0 else 0
                    if v > df["delta_R2"].max() * 0.045:
                        ax.text(left + v / 2, row, f"{pct:.0f}%", va="center", ha="center",
                                fontsize=fs - 2, color="white", fontweight="bold")
                    left += v
                ax.text(left + df["delta_R2"].max() * 0.015, row, f"{left:.3f}",
                        va="center", fontsize=fs - 1, fontweight="bold")
                y_ticks.append(row)
                y_labels.append(f"{m} — {tier}")
                row += 1
            row += 0.6  # gap between models

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=fs, fontweight="bold")
        for tick, lab in zip(ax.get_yticklabels(), y_labels):
            m = lab.split(" — ")[0]
            tick.set_color(DISPLAY_COLORS.get(m, "black"))
        ax.set_xlim(0, df["delta_R2"].max() * 1.2)
        ax.set_xlabel("ΔR²  (on top of a Job-only baseline)", fontsize=fs + 2)
        ax.set_title(EVAL_TYPES[eval_type], fontsize=fs + 5, pad=10)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)

    handles = [Patch(facecolor=block_colors[b], label=b) for b in block_order]
    handles += [Patch(facecolor="white", edgecolor="black", hatch=TIER_HATCH["High-Fit"], label="High-Fit"),
                Patch(facecolor="white", edgecolor="black", hatch=TIER_HATCH["Moderate-Fit"], label="Moderate-Fit")]
    fig.legend(handles=handles, loc="lower center", ncol=len(block_order) + 2, fontsize=fs,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{title}\n{subtitle}", fontsize=fs + 5)
    fig.tight_layout(rect=[0, 0.08, 1, 0.86])
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


def compute_tier_semantic_vs_style(merged, eval_type):
    """Pooled (across all evaluators/writers/jobs) Semantic-vs-style Shapley
    split, computed separately within High-Fit and Moderate-Fit candidates,
    baseline = Job + Writer + Evaluator. Answers: does style matter more for
    weaker candidates, or for stronger ones?"""
    sub_all = merged[merged["Eval_Type"] == eval_type]
    _, block_features = _drop_cv_consistency(eval_type, BLOCK_ORDER, BLOCK_FEATURES)
    cols = [c for v in block_features.values() for c in v]
    rows = {}
    for tier, label in [("high_fit", "High-Fit"), ("mod_fit", "Moderate-Fit")]:
        sub = sub_all[sub_all["Tier"] == tier].dropna(subset=cols + ["Score"]).copy()
        y_resid = _residualize(sub, ["Job_ID", "Writer", "Evaluator"])
        Xstd_all = sub[cols].astype(float)
        Xstd_all = (Xstd_all - Xstd_all.mean()) / Xstd_all.std().replace(0, 1)
        blocks = {cat: Xstd_all[v].values for cat, v in block_features.items()}
        shapley, total_r2 = _shapley_over_blocks(y_resid, blocks)
        rows[label] = {**shapley, "delta_R2": total_r2}
    return pd.DataFrame(rows).T


def compute_tier_substance_vs_style(merged, eval_type):
    """Same question as compute_tier_semantic_vs_style, but with Substance
    (each evaluator's own CV-only score) added as its own block instead of
    being folded into the baseline — computed separately within High-Fit and
    Moderate-Fit candidates, baseline = Job + Writer + Evaluator (Tier itself
    is the split variable here, so it's excluded from the baseline)."""
    sub_all = merged[merged["Eval_Type"] == eval_type]
    _, block_features = _drop_cv_consistency(eval_type, SUBSTANCE_BLOCK_ORDER, SUBSTANCE_BLOCK_FEATURES)
    needed = [c for cols in block_features.values() for c in cols]
    rows = {}
    for tier, label in [("high_fit", "High-Fit"), ("mod_fit", "Moderate-Fit")]:
        sub = sub_all[sub_all["Tier"] == tier].dropna(subset=needed + ["Score"]).copy()
        y_resid = _residualize(sub, ["Job_ID", "Writer", "Evaluator"])
        Xstd_all = sub[needed].astype(float)
        Xstd_all = (Xstd_all - Xstd_all.mean()) / Xstd_all.std().replace(0, 1)
        blocks = {cat: Xstd_all[cols].values for cat, cols in block_features.items()}
        shapley, total_r2 = _shapley_over_blocks(y_resid, blocks)
        rows[label] = {**shapley, "delta_R2": total_r2}
    return pd.DataFrame(rows).T


def compute_cvjob_quartile_semantic_vs_style(merged, eval_type):
    """Same question as compute_tier_semantic_vs_style, but split into 4
    quartiles of continuous CV-Job Fit (ranked within job) instead of the
    coarse Tier binary — Tier is an exact top/bottom-25 split of this same
    score, so this reveals finer-grained structure a 2-way split can't."""
    sub_all = merged[merged["Eval_Type"] == eval_type].copy()
    job_rank = sub_all.groupby("Job_ID")["cv_job_cosine_sim"].rank(pct=True)
    quartile = pd.cut(job_rank, [0, 0.25, 0.5, 0.75, 1.0],
                       labels=["Q1 (lowest fit)", "Q2", "Q3", "Q4 (highest fit)"], include_lowest=True)

    _, block_features = _drop_cv_consistency(eval_type, BLOCK_ORDER, BLOCK_FEATURES)
    cols = [c for v in block_features.values() for c in v]
    rows = {}
    for label in ["Q1 (lowest fit)", "Q2", "Q3", "Q4 (highest fit)"]:
        sub = sub_all[quartile == label].dropna(subset=cols + ["Score"]).copy()
        y_resid = _residualize(sub, ["Job_ID", "Writer", "Evaluator"])
        Xstd_all = sub[cols].astype(float)
        Xstd_all = (Xstd_all - Xstd_all.mean()) / Xstd_all.std().replace(0, 1)
        blocks = {cat: Xstd_all[v].values for cat, v in block_features.items()}
        shapley, total_r2 = _shapley_over_blocks(y_resid, blocks)
        rows[label] = {**shapley, "delta_R2": total_r2}
    return pd.DataFrame(rows).T


def compute_fit_vs_rest(merged, eval_type, fit_blocks):
    """Pooled (across all evaluators/writers/jobs/CVs) exact Shapley split of
    R² between one or more "genuine candidate fit" blocks (fit_blocks) and
    everything else the letter reflects: Job-Ad Fit / CV Consistency
    (measured on the produced letter, so they can vary by writer even for the
    same candidate) and the 4 style categories. Baseline = Job + Evaluator
    only — the fit block(s) and Writer-driven variation are both left in, on
    purpose.

    fit_blocks: dict[name -> raw column name(s) or "TIER_DUMMY"], e.g.
      {"Tier": "TIER_DUMMY"}
      {"CV-Job Fit": ["cv_job_cosine_sim"]}
      {"Tier": "TIER_DUMMY", "CV-Job Fit": ["cv_job_cosine_sim"]}
    """
    _, block_features = _drop_cv_consistency(eval_type, BLOCK_ORDER, BLOCK_FEATURES)
    cols = [c for v in block_features.values() for c in v]
    needed = [c for fcols in fit_blocks.values() if fcols != "TIER_DUMMY" for c in fcols]
    sub = merged[merged["Eval_Type"] == eval_type].dropna(subset=cols + needed + ["Score"]).copy()
    y_resid = _residualize(sub, ["Job_ID", "Evaluator"])

    fit_arrays = {}
    for name, fcols in fit_blocks.items():
        if fcols == "TIER_DUMMY":
            fit_arrays[name] = pd.get_dummies(sub["Tier"], drop_first=True).astype(float).values
        else:
            x = sub[fcols].astype(float)
            x = (x - x.mean()) / x.std().replace(0, 1)
            fit_arrays[name] = x.values

    Xstd_all = sub[cols].astype(float)
    Xstd_all = (Xstd_all - Xstd_all.mean()) / Xstd_all.std().replace(0, 1)
    blocks = {**fit_arrays, **{cat: Xstd_all[v].values for cat, v in block_features.items()}}
    shapley, total_r2 = _shapley_over_blocks(y_resid, blocks)
    return pd.DataFrame({"All Models": {**shapley, "delta_R2": total_r2}}).T


def plot_block_balance(all_results, block_order, block_colors, title, subtitle, out_name, legend_labels=None):
    # subtitle is kept short at the call site (full methodological detail
    # goes in the LaTeX caption for figures used in the paper) and wrapped
    # here as a safety net, so a long string can't force the canvas wide.
    BIG = 10
    legend_labels = legend_labels or {b: b for b in block_order}
    fig, axes = plt.subplots(1, len(all_results), figsize=(9.5 * len(all_results), 9.5))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (eval_type, df) in zip(axes, all_results.items()):
        models = df.index[::-1]
        y_pos = list(range(len(models)))
        blocks_here = [b for b in block_order if b in df.columns]

        for j, m in enumerate(models):
            left = 0.0
            for block in blocks_here:
                v = df.loc[m, block]
                ax.barh(j, v, left=left, color=block_colors[block], edgecolor="white", height=0.6)
                pct = v / df.loc[m, "delta_R2"] * 100 if df.loc[m, "delta_R2"] > 0 else 0
                if v > df["delta_R2"].max() * 0.035:
                    ax.text(left + v / 2, j, f"{pct:.0f}%", va="center", ha="center",
                            fontsize=fs + BIG - 3, color="white", fontweight="bold")
                left += v
            ax.text(left + df["delta_R2"].max() * 0.015, j, f"{left:.3f}",
                    va="center", fontsize=fs + BIG - 1, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=fs + BIG, fontweight="bold")
        for tick, m in zip(ax.get_yticklabels(), models):
            tick.set_color(DISPLAY_COLORS.get(m, "black"))
        ax.set_xlim(0, df["delta_R2"].max() * 1.2)
        ax.set_xlabel("ΔR²  (on top of baseline)", fontsize=fs + BIG)
        ax.set_title(EVAL_TYPES[eval_type], fontsize=fs + BIG + 4, pad=12)
        ax.tick_params(axis="x", labelsize=fs + BIG - 2)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)

    ncol = min(4, len(block_order))
    handles = [Patch(facecolor=block_colors[b], label=legend_labels[b]) for b in block_order]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=fs + BIG - 1,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=fs + BIG + 4, wrap=True)
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=fs + BIG - 4,
                  color="dimgrey", wrap=True)
    legend_rows = -(-len(block_order) // ncol)  # ceil
    fig.tight_layout(rect=[0, 0.05 * legend_rows + 0.02, 1, 0.90])
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


def compute_echo_residuals(feat_df):
    """For each writer, its average job_cosine_sim / cv_cosine_sim minus the
    grand mean — does this writer echo the job ad / CV more or less than
    other writers, in raw cosine-similarity units.

    This is equivalent here to regressing out Job identity and true candidate
    fit (cv_job_cosine_sim) first: every writer writes for the exact same 500
    (Job, CV) pairs (a perfectly balanced, fully-crossed design), so both of
    those confounds have an identical distribution for every writer and
    cancel out of any writer-to-writer comparison. The simple raw average
    already IS the fit-adjusted version in this dataset — the regression
    would only change the answer if writers had been assigned different or
    unequal candidates."""
    rows = {}
    for target, key in [("job_cosine_sim", "echo_job"), ("cv_cosine_sim", "echo_cv")]:
        grand_mean = feat_df[target].mean()
        per_writer = feat_df.groupby("Writer")[target].mean() - grand_mean
        for w in RAW_WRITERS:
            rows.setdefault(MODEL_DISPLAY[w], {})[key] = per_writer[w]
    return pd.DataFrame(rows).T


def compute_genuine_vs_inflated(feat_df, cv_job_sim):
    """
    Pooled ("genuine") slope of the letter's semantic similarity to the job
    ad / CV against cv_job_cosine_sim (writer-independent, letter-free fit
    score), within-job demeaned. Used as the reference slope that
    compute_evaluator_inflation_reward orthogonalizes job_cosine_sim /
    cv_cosine_sim against, to isolate writer- and evaluator-level inflation
    beyond what genuine fit would predict.

    Returns dict[target -> OLS slope].
    """
    df = feat_df.merge(cv_job_sim, on=["Job_ID", "CV_Idx"], how="left").dropna(
        subset=["cv_job_cosine_sim", "job_cosine_sim", "cv_cosine_sim"])

    pooled_slope = {}
    for target in ["job_cosine_sim", "cv_cosine_sim"]:
        y = df[target] - df.groupby("Job_ID")[target].transform("mean")
        x = df["cv_job_cosine_sim"] - df.groupby("Job_ID")["cv_job_cosine_sim"].transform("mean")
        slope = float(np.polyfit(x, y, 1)[0])
        pooled_slope[target] = slope
    return pooled_slope


def compute_evaluator_inflation_reward(merged, eval_type, pooled_slope):
    """For each evaluator, does its Score reward genuine fit (cv_job_cosine_sim),
    job-ad-echo inflation, or CV-echo inflation? Within-job demeaned,
    standardized, one joint OLS per evaluator. The predictors included depend
    on what's actually visible to the evaluator in this Eval_Type: the CV
    itself is never shown to the evaluator (only cv_job_cosine_sim, a proxy,
    is used as a control) — but CV-Echo Inflation specifically requires the
    letter to be comparable to CV *content*, which is only meaningful when a
    CV is part of the evaluator's prompt (CV + Cover Letter). In Cover-Letter-
    Only, the evaluator never sees the CV, so "CV-Echo Inflation" can't be
    interpreted as the evaluator reacting to echoed CV content and is
    dropped; Job-Echo Inflation stays, since the job ad IS always shown."""
    include_cv_echo = eval_type != "cl_evaluations"

    sub_all = merged[merged["Eval_Type"] == eval_type].dropna(
        subset=["cv_job_cosine_sim", "job_cosine_sim", "cv_cosine_sim", "Score"]).copy()

    for col in ["cv_job_cosine_sim", "job_cosine_sim", "cv_cosine_sim", "Score"]:
        sub_all[f"{col}_dm"] = sub_all[col] - sub_all.groupby("Job_ID")[col].transform("mean")

    sub_all["job_inflation"] = sub_all["job_cosine_sim_dm"] - pooled_slope["job_cosine_sim"] * sub_all["cv_job_cosine_sim_dm"]

    predictors = ["cv_job_cosine_sim_dm", "job_inflation"]
    names = ["CV-Job Fit", "Job-Echo Inflation"]
    if include_cv_echo:
        sub_all["cv_inflation"] = sub_all["cv_cosine_sim_dm"] - pooled_slope["cv_cosine_sim"] * sub_all["cv_job_cosine_sim_dm"]
        predictors.append("cv_inflation")
        names.append("CV-Echo Inflation")

    rows = {}
    for e in UNIQUE_EVALUATORS:
        ev = sub_all[sub_all["Evaluator"] == e]
        if len(ev) < MIN_N:
            continue
        X = ev[predictors].astype(float)
        X = (X - X.mean()) / X.std().replace(0, 1)
        X_int = np.column_stack([np.ones(len(X)), X.values])
        y = ev["Score_dm"].values
        coef, *_ = np.linalg.lstsq(X_int, y, rcond=None)
        rows[MODEL_DISPLAY[e]] = dict(zip(names, coef[1:]))
    return pd.DataFrame(rows).T


def plot_evaluator_inflation_reward(results_by_type, out_name="evaluator_inflation_reward.png"):
    """Grouped bar chart: does each evaluator reward genuine candidate fit,
    or a writer's job-ad/CV echo beyond what genuine fit would justify?"""
    colors = {"CV-Job Fit": "#5D4037", "Job-Echo Inflation": "#C62828", "CV-Echo Inflation": "#00897B"}
    fig, axes = plt.subplots(1, len(results_by_type), figsize=(11 * len(results_by_type), 7))
    if len(results_by_type) == 1:
        axes = [axes]

    for ax, (eval_type, df) in zip(axes, results_by_type.items()):
        cols_here = [c for c in ["CV-Job Fit", "Job-Echo Inflation", "CV-Echo Inflation"] if c in df.columns]
        models = df.sort_values("CV-Job Fit", ascending=True).index
        y_base = np.arange(len(models))
        bar_h = 0.6 / len(cols_here)
        offset0 = -(len(cols_here) - 1) / 2
        for i, col in enumerate(cols_here):
            y_pos = y_base + (offset0 + i) * bar_h
            vals = df.loc[models, col].values
            ax.barh(y_pos, vals, height=bar_h, color=colors[col], edgecolor="white", label=col)
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_yticks(y_base)
        ax.set_yticklabels(models, fontsize=fs + 1, fontweight="bold")
        for tick, m in zip(ax.get_yticklabels(), models):
            tick.set_color(DISPLAY_COLORS.get(m, "black"))
        ax.set_xlabel("Standardized OLS coefficient on Score", fontsize=fs + 1)
        ax.set_title(EVAL_TYPES[eval_type], fontsize=fs + 4, pad=10)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)

    handles = [Patch(facecolor=colors[c], label=c) for c in colors]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=fs + 1,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Does an Evaluator Reward CV-Job Fit (Genuine), or a Writer's Inflated Echo?\n"
        "Within-job demeaned, standardized, one joint OLS per evaluator — the three predictors are "
        "orthogonal by construction (inflation = residual beyond the pooled CV-Job Fit slope)",
        fontsize=fs + 5)
    fig.tight_layout(rect=[0, 0.08, 1, 0.86])
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


def compute_evaluator_cosine_prefs(merged, eval_type):
    """For each evaluator, its Ridge coefficient (within-job demeaned) on
    job_cosine_sim and cv_cosine_sim — how much it rewards a letter that
    mirrors the job ad vs. one that mirrors the CV, holding the other fixed."""
    cos_cols = ["job_cosine_sim", "cv_cosine_sim"]
    coef_df, _, _ = fit_regressions(merged, eval_type, cos_cols)
    df = coef_df.rename(index=MODEL_DISPLAY)[cos_cols]
    df.columns = ["pref_job", "pref_cv"]
    return df


def compute_writer_feature_deviation(feat_df, col_x, col_y):
    """For each writer, its average value of col_x / col_y minus the grand
    mean — same logic as compute_echo_residuals, generalized to any pair of
    raw feature columns. Valid without further adjustment because every
    writer writes for the exact same 500 (Job, CV) pairs (see
    compute_echo_residuals's docstring for why that makes the raw average
    already fit-adjusted)."""
    rows = {}
    for col, key in [(col_x, "x"), (col_y, "y")]:
        grand_mean = feat_df[col].mean()
        per_writer = feat_df.groupby("Writer")[col].mean() - grand_mean
        for w in RAW_WRITERS:
            rows.setdefault(MODEL_DISPLAY[w], {})[key] = per_writer[w]
    return pd.DataFrame(rows).T


def compute_evaluator_feature_prefs(merged, eval_type, col_x, col_y):
    """For each evaluator, its Ridge coefficients on col_x / col_y from ONE
    joint fit over ALL_COLS (mutually adjusted for every other measured
    feature, not just the pair in isolation) — consistent with how
    coefficients are used everywhere else in this script."""
    coef_df, _, _ = fit_regressions(merged, eval_type, ALL_COLS)
    df = coef_df.rename(index=MODEL_DISPLAY)
    return pd.DataFrame({"x": df[col_x], "y": df[col_y]})


def _scatter_panel(ax, df, xcol, ycol, xlabel, ylabel, title,
                    diagonal=False, regression=False, shared_lims=False, start_at_zero=False):
    colors = [DISPLAY_COLORS.get(m, "grey") for m in df.index]
    ax.scatter(df[xcol], df[ycol], c=colors, s=160, edgecolor="white", linewidth=1, zorder=3)

    if shared_lims:
        lo = 0.0 if start_at_zero else min(df[xcol].min(), df[ycol].min())
        hi = max(df[xcol].max(), df[ycol].max())
        pad = (hi - lo) * 0.15
        xlim = ylim = (lo if start_at_zero else lo - pad, hi + pad)
    else:
        pad_x = df[xcol].abs().max() * 0.3
        pad_y = df[ycol].abs().max() * 0.3
        xlim = (df[xcol].min() - pad_x, df[xcol].max() + pad_x)
        ylim = (df[ycol].min() - pad_y, df[ycol].max() + pad_y)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    texts = [ax.text(row[xcol], row[ycol], model, fontsize=fs, fontweight="bold",
                      color=DISPLAY_COLORS.get(model, "black"), zorder=4,
                      bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1))
             for model, row in df.iterrows()]
    adjust_text(texts, x=df[xcol].values, y=df[ycol].values, ax=ax,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.6, alpha=0.6))

    if diagonal:
        lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
        ax.plot([lo, hi], [lo, hi], color="grey", linewidth=1.2, linestyle=":", zorder=1, label="y = x")
    else:
        ax.axhline(0, color="grey", linewidth=1, linestyle="--", alpha=0.7)
        ax.axvline(0, color="grey", linewidth=1, linestyle="--", alpha=0.7)

    if regression:
        m, b = np.polyfit(df[xcol], df[ycol], 1)
        x_range = np.array(xlim)
        ax.plot(x_range, m * x_range + b, color="firebrick", linewidth=1.5,
                alpha=0.7, zorder=2, label=f"fit (slope={m:.2f})")

    if diagonal or regression:
        ax.legend(fontsize=fs - 1, frameon=False, loc="upper left")

    ax.set_xlabel(xlabel, fontsize=fs + 1)
    ax.set_ylabel(ylabel, fontsize=fs + 1)
    ax.set_title(title, fontsize=fs + 3, pad=12)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


def plot_writer_vs_evaluator_pair(writer_df, eval_df, writer_labels, eval_labels, suptitle, out_name,
                                   same_scale=False):
    """writer_df / eval_df: DataFrames with columns "x", "y" (see
    compute_writer_feature_deviation / compute_evaluator_feature_prefs).
    same_scale: if the two features share a natural common scale (e.g. both
    VAD dimensions, both cosine similarities), draw a y=x reference line and
    force shared, zero-based axis limits on the evaluator panel; otherwise
    each axis is scaled independently (different units aren't comparable)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    _scatter_panel(axes[0], writer_df, "x", "y", writer_labels[0], writer_labels[1],
                   "As a WRITER", regression=True)
    _scatter_panel(axes[1], eval_df, "x", "y", eval_labels[0], eval_labels[1],
                   "As an EVALUATOR", regression=True,
                   diagonal=same_scale, shared_lims=same_scale, start_at_zero=same_scale)

    fig.suptitle(suptitle, fontsize=fs + 5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


def plot_writer_vs_evaluator_cosine(echo_df, eval_pref_df, out_name, title_note=""):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    _scatter_panel(
        axes[0], echo_df, "echo_job", "echo_cv",
        "Writer's avg. letter-to-job-ad similarity − average across writers",
        "Writer's avg. letter-to-CV similarity − average across writers",
        "As a WRITER: does it echo the job ad or the CV more?")
    _scatter_panel(
        axes[1], eval_pref_df, "pref_job", "pref_cv",
        "Evaluator's reward for letter-to-job-ad similarity (Ridge coef.)",
        "Evaluator's reward for letter-to-CV similarity (Ridge coef.)",
        "As an EVALUATOR: does it reward job-ad fit or CV fit more?",
        diagonal=True, regression=True, shared_lims=True, start_at_zero=True)

    fig.suptitle(f"Writing Behavior vs. Evaluation Preference: Job Ad vs. CV Alignment{title_note}",
                fontsize=fs + 5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


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
    merged["Tier"] = np.where(merged["CV_Idx"] <= 25, "high_fit", "mod_fit")

    cv_job_sim = pd.read_parquet("output_eval/cv_job_similarity.parquet")
    merged = merged.merge(cv_job_sim, on=["Job_ID", "CV_Idx"], how="left")

    # self-referential "substance" baseline: each evaluator's OWN CV-only score
    # (mean over its 4 runs) for this exact candidate — matched on Evaluator too,
    # so evaluator e's rows get e's own CV judgment, not another model's.
    cv_only_self = (
        master_df[master_df["Eval_Type"] == "cv_only"]
        .groupby(["Evaluator", "Job_ID", "CV_Idx"])["Score"].mean()
        .rename("cv_only_score_self").reset_index()
    )
    merged = merged.merge(cv_only_self, on=["Evaluator", "Job_ID", "CV_Idx"], how="left")

    eval_style, writer_style = {}, {}
    eval_sem_style, writer_sem_style, tier_sem_style = {}, {}, {}
    fit_vs_rest_tier, fit_vs_rest_cvjob, fit_vs_rest_both = {}, {}, {}
    eval_tier_vs_writer, eval_cvjob_vs_writer = {}, {}
    cvjob_quartile_sem_style = {}
    eval_substance_style = {}
    eval_substance_style_by_tier = {}
    eval_style_drilldown = {}
    eval_substance_vs_writer = {}
    tier_substance_style = {}

    for eval_type in EVAL_TYPES:
        print(f"Computing evaluator style breakdown — {eval_type}...")
        eval_style[eval_type] = compute_style_explainability(
            merged, eval_type, "Evaluator", UNIQUE_EVALUATORS, ["Job_ID", "Tier", "Writer"])

        print(f"Computing writer style breakdown — {eval_type}...")
        writer_style[eval_type] = compute_style_explainability(
            merged, eval_type, "Writer", RAW_WRITERS, ["Job_ID", "Tier", "Evaluator"])

        block_features_here = _drop_cv_consistency(eval_type, BLOCK_ORDER, BLOCK_FEATURES)[1]
        substance_features_here = _drop_cv_consistency(eval_type, SUBSTANCE_BLOCK_ORDER, SUBSTANCE_BLOCK_FEATURES)[1]

        print(f"Computing evaluator semantic-vs-style balance — {eval_type}...")
        eval_sem_style[eval_type] = compute_block_balance(
            merged, eval_type, "Evaluator", UNIQUE_EVALUATORS, ["Job_ID", "Tier", "Writer"], block_features_here)

        print(f"Computing evaluator substance-vs-style balance — {eval_type}...")
        eval_substance_style[eval_type] = compute_block_balance(
            merged, eval_type, "Evaluator", UNIQUE_EVALUATORS, ["Job_ID"], substance_features_here,
            sort_col="Substance (Own CV View)")

        print(f"Computing evaluator substance-vs-writer split — {eval_type}...")
        eval_substance_vs_writer[eval_type] = compute_block_balance(
            merged, eval_type, "Evaluator", UNIQUE_EVALUATORS, ["Job_ID"],
            {"Substance (Own CV View)": ["cv_only_score_self"], "Writer": "WRITER_DUMMY"},
            sort_col="Writer")

        print(f"Computing evaluator substance-vs-style balance by tier — {eval_type}...")
        eval_substance_style_by_tier[eval_type] = compute_block_balance_by_tier(
            merged, eval_type, "Evaluator", UNIQUE_EVALUATORS, substance_features_here)

        print(f"Computing evaluator style drill-down — {eval_type}...")
        eval_style_drilldown[eval_type] = compute_substance_style_drilldown(
            merged, eval_type, "Evaluator", UNIQUE_EVALUATORS, substance_features_here)

        print(f"Computing writer semantic-vs-style balance — {eval_type}...")
        writer_sem_style[eval_type] = compute_block_balance(
            merged, eval_type, "Writer", RAW_WRITERS, ["Job_ID", "Tier", "Evaluator"], block_features_here)

        print(f"Computing evaluator Tier-vs-Writer split — {eval_type}...")
        eval_tier_vs_writer[eval_type] = compute_block_balance(
            merged, eval_type, "Evaluator", UNIQUE_EVALUATORS, ["Job_ID"],
            {"Tier": "TIER_DUMMY", "Writer": "WRITER_DUMMY"})

        print(f"Computing evaluator CV-Job Fit-vs-Writer split — {eval_type}...")
        eval_cvjob_vs_writer[eval_type] = compute_block_balance(
            merged, eval_type, "Evaluator", UNIQUE_EVALUATORS, ["Job_ID"],
            {"CV-Job Fit": ["cv_job_cosine_sim"], "Writer": "WRITER_DUMMY"})

        print(f"Computing tier semantic-vs-style balance — {eval_type}...")
        tier_sem_style[eval_type] = compute_tier_semantic_vs_style(merged, eval_type)

        print(f"Computing tier substance-vs-style balance — {eval_type}...")
        tier_substance_style[eval_type] = compute_tier_substance_vs_style(merged, eval_type)

        print(f"Computing CV-Job Fit quartile semantic-vs-style balance — {eval_type}...")
        cvjob_quartile_sem_style[eval_type] = compute_cvjob_quartile_semantic_vs_style(merged, eval_type)

        print(f"Computing fit-vs-rest split (Tier) — {eval_type}...")
        fit_vs_rest_tier[eval_type] = compute_fit_vs_rest(
            merged, eval_type, {"Tier": "TIER_DUMMY"})

        print(f"Computing fit-vs-rest split (CV-Job cosine) — {eval_type}...")
        fit_vs_rest_cvjob[eval_type] = compute_fit_vs_rest(
            merged, eval_type, {"CV-Job Fit": ["cv_job_cosine_sim"]})

        print(f"Computing fit-vs-rest split (Tier + CV-Job cosine) — {eval_type}...")
        fit_vs_rest_both[eval_type] = compute_fit_vs_rest(
            merged, eval_type, {"Tier": "TIER_DUMMY", "CV-Job Fit": ["cv_job_cosine_sim"]})

    plot_style_breakdown(
        eval_style,
        "Which Evaluators' Scoring Practices Are Best Explained by Writing Style?",
        "ΔR² of style features on top of a Job + Tier + Semantic + Writer baseline, per evaluator, split by feature",
        "evaluator_explainability.png")
    plot_style_breakdown(
        writer_style,
        "Which Writers' Received Scores Are Best Explained by Their Own Style?",
        "ΔR² of style features on top of a Job + Tier + Semantic + Evaluator baseline, per writer, split by feature",
        "writer_explainability.png")

    plot_block_balance(
        eval_sem_style, BLOCK_ORDER, BLOCK_COLORS,
        "Is an Evaluator's Scoring More Semantic- or Style-Driven?",
        "Exact Shapley split of R² on top of a Job + Tier + Writer baseline, per evaluator",
        "evaluator_semantic_vs_style.png")
    plot_block_balance(
        eval_substance_style, SUBSTANCE_BLOCK_ORDER, SUBSTANCE_BLOCK_COLORS,
        "Substance vs. Style, Per Evaluator — Ranked by Substance",
        "Job-only baseline; Writer identity not controlled, see caption",
        "evaluator_substance_vs_style.png")
    plot_block_balance(
        eval_substance_vs_writer,
        ["Substance (Own CV View)", "Writer"],
        {"Substance (Own CV View)": "#5D4037", "Writer": "#C62828"},
        "Substance vs. Writer Identity, Per Evaluator — Ranked by Writer-Dependence",
        "Job-only baseline, see caption",
        "evaluator_substance_vs_writer.png",
        legend_labels={"Substance (Own CV View)": "Substance (own CV-only score)",
                        "Writer": "Writer identity (categorical)"})
    plot_style_breakdown(
        eval_style_drilldown,
        "Digging Into the Style Portion: Which Specific Feature, Per Evaluator?",
        "Same baseline as evaluator_substance_vs_style.png, decomposed feature by feature; see caption",
        "evaluator_style_drilldown.png")
    plot_block_balance_by_tier(
        eval_substance_style_by_tier, SUBSTANCE_BLOCK_ORDER, SUBSTANCE_BLOCK_COLORS,
        "Does an Evaluator's Substance-vs-Style Balance Depend on Candidate Fit?",
        "Exact Shapley split of R² on top of a Job-only baseline, per evaluator, computed separately within "
        "High-Fit (solid) and Moderate-Fit (faded) candidates",
        "evaluator_substance_vs_style_by_tier.png")
    plot_block_balance(
        writer_sem_style, BLOCK_ORDER, BLOCK_COLORS,
        "Are a Writer's Received Scores More Semantic- or Style-Driven?",
        "Exact Shapley split of R² on top of a Job + Tier + Evaluator baseline, per writer",
        "writer_semantic_vs_style.png")
    plot_block_balance(
        eval_tier_vs_writer, ["Tier", "Writer"],
        {"Tier": "#FBC02D", "Writer": "#C62828"},
        "Does an Evaluator's Score Reflect Genuine Candidate Fit (Tier) or Which Model Wrote It?",
        "Exact Shapley split of R² on top of a Job-only baseline, per evaluator",
        "evaluator_tier_vs_writer.png",
        legend_labels={"Tier": "Tier (High-Fit/Moderate-Fit)", "Writer": "Writer identity (categorical)"})
    plot_block_balance(
        eval_cvjob_vs_writer, ["CV-Job Fit", "Writer"],
        {"CV-Job Fit": "#00897B", "Writer": "#C62828"},
        "Does an Evaluator's Score Reflect Genuine Candidate Fit (CV-Job Cosine Sim.) or Which Model Wrote It?",
        "Exact Shapley split of R² on top of a Job-only baseline, per evaluator",
        "evaluator_cvjob_vs_writer.png",
        legend_labels={"CV-Job Fit": "CV-Job Fit (continuous)", "Writer": "Writer identity (categorical)"})
    plot_block_balance(
        tier_sem_style, BLOCK_ORDER, BLOCK_COLORS,
        "Does Style Matter More for High-Fit or Moderate-Fit Candidates?",
        "Exact Shapley split of R² on top of a Job + Writer + Evaluator baseline, pooled across all evaluators",
        "tier_semantic_vs_style.png")
    plot_block_balance(
        tier_substance_style, SUBSTANCE_BLOCK_ORDER, SUBSTANCE_BLOCK_COLORS,
        "Does Non-Substance Content Matter More for High-Fit or Moderate-Fit Candidates?",
        "Exact Shapley split of R² on top of a Job + Writer + Evaluator baseline, pooled across all evaluators — "
        "Substance = that evaluator's own CV-only score for this candidate",
        "tier_substance_vs_style.png")
    plot_block_balance(
        cvjob_quartile_sem_style, BLOCK_ORDER, BLOCK_COLORS,
        "Does Style Matter More at Different Levels of Candidate Fit? (Finer-Grained)",
        "Exact Shapley split of R² on top of a Job + Writer + Evaluator baseline, split into quartiles of "
        "continuous CV-Job Fit (Tier is an exact top/bottom-half split of this same score, so this reveals "
        "structure a 2-way split can't)",
        "cvjob_quartile_semantic_vs_style.png")
    plot_block_balance(
        fit_vs_rest_tier, TIER_BLOCK_ORDER, TIER_BLOCK_COLORS,
        "Genuine Candidate Fit (Tier) vs. Everything Else the Letter Reflects",
        "Exact Shapley split of R² on top of a Job + Evaluator baseline, pooled across all models — "
        "Tier (High-Fit/Moderate-Fit) is candidate-intrinsic and writer-independent by construction",
        "tier_vs_rest.png")
    plot_block_balance(
        fit_vs_rest_cvjob, CVJOB_BLOCK_ORDER, CVJOB_BLOCK_COLORS,
        "Genuine Candidate Fit (CV-Job Cosine Sim.) vs. Everything Else the Letter Reflects",
        "Exact Shapley split of R² on top of a Job + Evaluator baseline, pooled across all models — "
        "CV-Job Fit is a continuous, writer-independent 'objective skills match' proxy (CV embedding vs. "
        "job-ad embedding, computed before any cover letter is written)",
        "cvjob_vs_rest.png")
    plot_block_balance(
        fit_vs_rest_both, BOTH_BLOCK_ORDER, BOTH_BLOCK_COLORS,
        "Tier vs. Continuous CV-Job Fit vs. Everything Else",
        "Exact Shapley split of R² on top of a Job + Evaluator baseline, pooled across all models — "
        "both candidate-fit signals included together, to see whether the continuous score adds anything "
        "beyond the coarse High-Fit/Moderate-Fit label",
        "tier_and_cvjob_vs_rest.png")

    print("Computing per-writer echo residuals (job ad vs. CV)...")
    echo_df = compute_echo_residuals(feat_df)
    print(echo_df.sort_values("echo_job", ascending=False).round(4))

    print("Computing genuine-vs-inflated pooled slope...")
    pooled_slope = compute_genuine_vs_inflated(feat_df, cv_job_sim)
    print(f"  Pooled genuine slope — job_cosine_sim: {pooled_slope['job_cosine_sim']:.3f}  "
          f"cv_cosine_sim: {pooled_slope['cv_cosine_sim']:.3f}")

    print("Computing evaluator reward for genuine fit vs. inflation...")
    inflation_reward = {}
    for eval_type in EVAL_TYPES:
        inflation_reward[eval_type] = compute_evaluator_inflation_reward(merged, eval_type, pooled_slope)
    plot_evaluator_inflation_reward(inflation_reward)

    print("Computing per-evaluator cosine-similarity preferences...")
    eval_pref_df = compute_evaluator_cosine_prefs(merged, "cl_evaluations")
    print(eval_pref_df.sort_values("pref_job", ascending=False).round(4))

    plot_writer_vs_evaluator_cosine(echo_df, eval_pref_df, "writer_vs_evaluator_cosine.png")

    # CV + Cover Letter counterpart, for the paper appendix: this is the
    # ONLY condition where "reward for CV-echo" is a meaningful regressor
    # at all — the cover-letter-only evaluator panel above technically
    # fits a coefficient on cv_cosine_sim too, but the evaluator's prompt
    # never contains the CV in that condition, so that coefficient can't
    # reflect a genuine preference (same reasoning as CL_ONLY_FEATURE_COLS
    # in plot_feature_regression.py). Kept as a separate file rather than
    # replacing the CL-only one above, which is left as-is for now.
    eval_pref_df_cvcl = compute_evaluator_cosine_prefs(merged, "cv_cl_evaluations")
    plot_writer_vs_evaluator_cosine(echo_df, eval_pref_df_cvcl, "writer_vs_evaluator_cosine_cvcl.png",
                                     title_note="  [CV + Cover Letter]")

    feature_pairs = [
        ("flesch_reading_ease", "avg_word_length",
         "Flesch Reading Ease", "Avg Word Length",
         "Writing Behavior vs. Evaluation Preference: Readability vs. Word Length",
         "writer_vs_evaluator_flesch_wordlen.png", False),
        ("word_count", "flesch_reading_ease",
         "Word Count", "Flesch Reading Ease",
         "Writing Behavior vs. Evaluation Preference: Length vs. Readability",
         "writer_vs_evaluator_wordcount_flesch.png", False),
        ("vad_dominance", "vad_arousal",
         "VAD Dominance", "VAD Arousal",
         "Writing Behavior vs. Evaluation Preference: Dominance vs. Arousal",
         "writer_vs_evaluator_dominance_arousal.png", False),
        ("emo_joy", "emo_neutral",
         "Joy", "Neutral",
         "Writing Behavior vs. Evaluation Preference: Joy vs. Neutral Tone",
         "writer_vs_evaluator_joy_neutral.png", False),
    ]
    for col_x, col_y, label_x, label_y, suptitle, out_name, same_scale in feature_pairs:
        print(f"Computing writer/evaluator comparison — {label_x} vs. {label_y}...")
        writer_df = compute_writer_feature_deviation(feat_df, col_x, col_y)
        eval_df = compute_evaluator_feature_prefs(merged, "cl_evaluations", col_x, col_y)
        plot_writer_vs_evaluator_pair(
            writer_df, eval_df,
            (f"Writer's avg. {label_x} − average across writers", f"Writer's avg. {label_y} − average across writers"),
            (f"Evaluator's reward for {label_x} (Ridge coef.)", f"Evaluator's reward for {label_y} (Ridge coef.)"),
            suptitle, out_name, same_scale=same_scale)

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
