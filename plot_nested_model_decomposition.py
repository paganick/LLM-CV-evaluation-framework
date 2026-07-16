"""
plot_nested_model_decomposition.py — Nested-model variance decomposition.

ONE figure, three panels (CV Only / CV + Cover Letter / Cover Letter Only),
answering both "how does predictability change across conditions" and "how
much does each block add" at once.

Builds up R² step by step through a shared trunk, then — for the two
letter-inclusive conditions only — branches into two ALTERNATIVE endpoints
(not a continuation of each other):

  Job -> + Substance -> + Evaluator -> + Writer            (branch A)
                                     -> + Text Features     (branch B,
                                        incl. semantic fit)

CV Only has no letter, so it stops at '+ Evaluator' — no Writer, no Text
Features possible for that condition.

"Substance" here = the POOLED AVERAGE CV-only score across all 9 evaluators
for that (Job, CV) pair — a single consensus quality measure, not any one
evaluator's own judgment. This is the same underlying measurement used for
"Substance" in the style-vs-substance section of this notebook, just
aggregated differently: there it's each evaluator's OWN CV-only score
(self-referential, analyzed per evaluator); here it has to be ONE shared
value usable across every evaluator in a single pooled trunk, so it's
averaged across all of them instead. Using each evaluator's own score here
would also be tautological specifically for the CV Only panel, where the
outcome being predicted (Score) IS that exact self-referential measurement —
averaging across evaluators avoids that (a given evaluator's own score is
only 1/9 of the average, so there's real, non-trivial signal left to
explain).

This replaces the earlier version of this figure, which used CV-Job Fit
(cv_job_cosine_sim, an embedding-similarity proxy) — dropped so the notebook
sticks to one single "how good is this candidate" measure throughout,
instead of switching between an embedding proxy in one section and an
elicited CV-only judgment in another.

Each block answers one question: which job posting (baseline), how good is
the candidate really (Substance), who's judging (Evaluator), and finally
either who wrote it (Writer, a catch-all categorical) or what the letter
actually contains (Text Features, incl. semantic fit — a decomposed
alternative to the same variance Writer would otherwise absorb). Branch A and
branch B are two different ways of explaining the SAME remaining variance
after '+ Evaluator', not sequential steps — both are computed from
'+ Evaluator' directly.

"Text Features" includes job_cosine_sim (letter-to-job-ad) plus, for the
CV + Cover Letter condition only, cv_cosine_sim (letter-to-CV) — Cover-Letter-
Only excludes it, since the evaluator's prompt never contains the CV there,
so letter-to-CV similarity isn't something it could plausibly be reacting to.

All three panels share one x-axis scale, so bar length is directly
comparable across conditions.

Output: nested_model_decomposition.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aggregate_plots import MODEL_DISPLAY, DISPLAY_COLORS
from plot_feature_regression import FEATURES_PATH, STYLE_COLS, COSINE_COLS

sns_fs = plt.rcParams["font.size"]

MASTER_PATH = "output_eval/master_df_no_gemini2.parquet"
OUT_DIR     = "output_plots/cl_features_no_gemini2"

CONDITIONS = {
    "cv_only":           "CV Only",
    "cv_cl_evaluations": "CV + Cover Letter",
    "cl_evaluations":    "Cover Letter Only",
}
LETTER_CONDITIONS = {k: v for k, v in CONDITIONS.items() if k != "cv_only"}
EVAL_TYPES = LETTER_CONDITIONS  # backward-compat alias: plot_model_explainability.py imports this name

STEP_COLORS = {
    "Job":                          "#90A4AE",
    "+ Substance":                  "#5D4037",
    "+ Evaluator":                  "#1565C0",
    "+ Writer  [branch A]":         "#2E7D32",
    "+ Text Features  [branch B]":  "#E65100",
}
FULL_ORDER = ["Job", "+ Substance", "+ Evaluator",
              "+ Writer  [branch A]", "+ Text Features  [branch B]"]


def _design(df, categorical_cols, continuous_cols):
    parts = []
    if categorical_cols:
        parts.append(pd.get_dummies(df[list(categorical_cols)], drop_first=True).astype(float))
    if continuous_cols:
        x = df[list(continuous_cols)].astype(float)
        x = (x - x.mean()) / x.std().replace(0, 1)
        parts.append(x)
    if not parts:
        return np.empty((len(df), 0))
    return pd.concat(parts, axis=1).values


def r2_ols(y, X):
    X_int = np.column_stack([np.ones(len(X)), X]) if X.shape[1] > 0 else np.ones((len(X), 1))
    coef, *_ = np.linalg.lstsq(X_int, y, rcond=None)
    ss_r = ((y - X_int @ coef) ** 2).sum()
    ss_t = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_r / ss_t) if ss_t > 0 else np.nan


def compute_steps(sub, condition):
    """Returns an ordered dict of step label -> R^2. CV Only stops at
    '+ Evaluator' (no letter, so no Writer / Text Features branch)."""
    y = sub["Score"].values
    steps = {}
    steps["Job"] = r2_ols(y, _design(sub, ["Job_ID"], []))
    steps["+ Substance"] = r2_ols(y, _design(sub, ["Job_ID"], ["cv_only_score_avg"]))
    steps["+ Evaluator"] = r2_ols(y, _design(sub, ["Job_ID", "Evaluator"], ["cv_only_score_avg"]))

    if condition == "cv_only":
        return steps

    semantic_cols = ["job_cosine_sim"] + (["cv_cosine_sim"] if condition != "cl_evaluations" else [])
    steps["+ Writer  [branch A]"] = r2_ols(
        y, _design(sub, ["Job_ID", "Evaluator", "Writer"], ["cv_only_score_avg"]))
    steps["+ Text Features  [branch B]"] = r2_ols(
        y, _design(sub, ["Job_ID", "Evaluator"], ["cv_only_score_avg"] + semantic_cols + STYLE_COLS))
    return steps


def plot_nested_decomposition(all_steps):
    # Full methodological detail (which blocks, why CL-only drops CV
    # similarity, the "A/B are alternatives" point) lives in the LaTeX
    # caption — the in-image title stays to one short line so the figure
    # doesn't have to be rendered extremely wide just to fit unwrapped text.
    BIG = 10
    conditions = list(all_steps.keys())
    fig, axes = plt.subplots(1, len(conditions), figsize=(7.5 * len(conditions), 8.5))
    if len(conditions) == 1:
        axes = [axes]

    global_max = max(v for steps in all_steps.values() for v in steps.values())

    for ax, condition in zip(axes, conditions):
        steps = all_steps[condition]
        labels = [l for l in FULL_ORDER if l in steps]
        values = [steps[l] for l in labels]
        colors = [STEP_COLORS[l] for l in labels]
        y_pos = list(range(len(labels)))

        ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.6)
        for i, v in enumerate(values):
            ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=sns_fs + BIG - 1, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=sns_fs + BIG)
        ax.set_xlim(0, global_max * 1.22)
        ax.set_xlabel("R²", fontsize=sns_fs + BIG)
        ax.set_title(CONDITIONS[condition], fontsize=sns_fs + BIG + 3, pad=12)
        ax.tick_params(axis="x", labelsize=sns_fs + BIG - 2)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)

        if "+ Writer  [branch A]" in steps:
            # trunk = Job, Substance, Evaluator (indices 0-2); the separator
            # sits right after '+ Evaluator', BEFORE both branches — branch A
            # and branch B are alternatives to each other, not a sequence,
            # so both live above the same line
            eval_idx = labels.index("+ Evaluator")
            ax.axhline(eval_idx + 0.5, color="black", linewidth=1.4, linestyle="--", alpha=0.7)
            ax.annotate("A & B are alternatives",
                        xy=(0, eval_idx + 0.5), xycoords=("axes fraction", "data"),
                        xytext=(0, 6), textcoords="offset points",
                        ha="left", va="bottom", fontsize=sns_fs + BIG - 4, color="dimgrey", fontstyle="italic")

    fig.suptitle("Nested Model Decomposition: How Much Does Each Block Add?",
                 fontsize=sns_fs + BIG + 4)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, "nested_model_decomposition.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(out)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    feat_df   = pd.read_parquet(FEATURES_PATH).drop(columns=["embedding"], errors="ignore")
    master_df = pd.read_parquet(MASTER_PATH)

    # pooled average CV-only score across ALL 9 evaluators, per (Job, CV) —
    # one shared "how good is this candidate" value, not any single
    # evaluator's own judgment
    cv_only_avg = (
        master_df[master_df["Eval_Type"] == "cv_only"]
        .groupby(["Job_ID", "CV_Idx"])["Score"].mean()
        .rename("cv_only_score_avg").reset_index()
    )

    all_steps = {}

    print("Computing nested R^2 steps — cv_only...")
    cvonly_scores = (
        master_df[master_df["Eval_Type"] == "cv_only"]
        .groupby(["Job_ID", "CV_Idx", "Evaluator"])["Score"].mean().reset_index()
    )
    cvonly_merged = cvonly_scores.merge(cv_only_avg, on=["Job_ID", "CV_Idx"], how="left")
    cvonly_merged = cvonly_merged.dropna(subset=["cv_only_score_avg", "Score"])
    all_steps["cv_only"] = compute_steps(cvonly_merged, "cv_only")

    scores = (
        master_df[master_df["Eval_Type"].isin(LETTER_CONDITIONS)]
        .groupby(["Job_ID", "Writer", "CV_Idx", "Evaluator", "Eval_Type"])["Score"]
        .mean()
        .reset_index()
    )
    merged = scores.merge(feat_df, on=["Job_ID", "Writer", "CV_Idx"])
    merged = merged.merge(cv_only_avg, on=["Job_ID", "CV_Idx"], how="left")

    for eval_type in LETTER_CONDITIONS:
        print(f"Computing nested R^2 steps — {eval_type}...")
        needed = list(COSINE_COLS) + STYLE_COLS + ["cv_only_score_avg", "Score"]
        sub = merged[merged["Eval_Type"] == eval_type].dropna(subset=needed)
        all_steps[eval_type] = compute_steps(sub, eval_type)

    print()
    for condition, steps in all_steps.items():
        print(f"--- {CONDITIONS[condition]} ---")
        for label, r2 in steps.items():
            print(f"  {label:<30s} R² = {r2:.4f}")
        print()

    plot_nested_decomposition(all_steps)
    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
