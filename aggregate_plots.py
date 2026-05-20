"""
aggregate_plots.py

Generates 4 fully-aggregated (All_CVs tier, all jobs combined) plots
and their companion CSV files in OUT_DIR:

  1. heatmap_gap_cl_evaluations.png / .csv
  2. heatmap_gap_cv_cl_evaluations.png / .csv
  3. win_matrix_UNBIASED_cv_cl_evaluations.png / .csv
  4. net_advantage_matrix_ALL_COMBINED.png / .csv
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "./output_eval"
OUT_DIR  = "./output_plots/paper_plots"

TIER_NAME  = "All_CVs"
TIER_RANGE = (1, 50)

CACHE_PATH = "./output_eval/master_df.parquet"

UNIQUE_EVALUATORS = [
    "gpt-4o-mini", "gpt-5-mini", "gemini-2.0-flash",
    "gemini-3-flash-preview", "claude-haiku-4-5", "deepseek-chat",
]

RAW_WRITERS = [
    "gpt-4o-mini", "gpt-5-mini", "gemini-2.0-flash",
    "gemini-3-flash-preview", "claude-haiku-4-5",
    "deepseek-chat", "deepseek-r1-8b", "llama3.1-8b",
]

MODEL_PAIRS = [f"{e}_{w}" for e in UNIQUE_EVALUATORS for w in RAW_WRITERS]

SORTED_WRITERS = [e for e in UNIQUE_EVALUATORS if e in RAW_WRITERS] + \
                 [w for w in RAW_WRITERS if w not in UNIQUE_EVALUATORS]

TITLE_MAP = {
    "cl_evaluations":    "Cover Letter Only",
    "cv_cl_evaluations": "CV + Cover Letter",
    "cv_only":           "CV Only",
}

WRITER_COLORS = {
    "gpt-4o-mini":            "#1565C0",  # OpenAI – dark blue
    "gpt-5-mini":             "#42A5F5",  # OpenAI – light blue
    "gemini-2.0-flash":       "#1B5E20",  # Google – dark green
    "gemini-3-flash-preview": "#66BB6A",  # Google – medium green
    "claude-haiku-4-5":       "#E65100",  # Anthropic – burnt orange
    "deepseek-chat":          "#4A148C",  # DeepSeek – dark purple
    "deepseek-r1-8b":         "#AB47BC",  # DeepSeek – medium purple
    "llama3.1-8b":            "#5D4037",  # Meta – brown
}

MODEL_DISPLAY = {
    "gpt-4o-mini":            "GPT-4o mini",
    "gpt-5-mini":             "GPT-5 mini",
    "gemini-2.0-flash":       "Gemini 2.0 Flash",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "claude-haiku-4-5":       "Claude Haiku 4.5",
    "deepseek-chat":          "DeepSeek Chat",
    "deepseek-r1-8b":         "DeepSeek R1 8B",
    "llama3.1-8b":            "Llama 3.1 8B",
}

DISPLAY_COLORS = {MODEL_DISPLAY[k]: v for k, v in WRITER_COLORS.items()}

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

# ==========================
# SHARED HELPERS
# ==========================

def extract_first_number(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            m = re.search(r"(\d+(?:\.\d+)?)", f.read())
            return float(m.group(1)) if m else None
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return None


def detect_cv_count(path):
    max_cv = 0
    if not os.path.exists(path):
        return 0
    for root, _, files in os.walk(path):
        for fname in files:
            m = re.search(r"cv(\d+)", fname, re.IGNORECASE)
            if m:
                max_cv = max(max_cv, int(m.group(1)))
    return max_cv


def format_job_title(job_id, folder_name):
    clean = folder_name.replace(job_id, "").strip("_")
    clean = re.sub(r"_\d+$", "", clean).replace("_", " ")
    return f"{clean} ({job_id.replace('job_', '')})"


def parse_filename(filename, etype):
    """Extracts (evaluator, writer, cv_idx) from a score filename."""
    evaluator = next(
        (e for e in sorted(UNIQUE_EVALUATORS, key=len, reverse=True)
         if filename.startswith(e + "_")),
        None,
    )
    if not evaluator:
        return None, None, None

    if etype == "cv_only":
        writer = "CV_ONLY"
    else:
        remainder = filename[len(evaluator) + 1:]
        writer = next(
            (w for w in sorted(RAW_WRITERS, key=len, reverse=True)
             if remainder.startswith(w + "_") or remainder.startswith(w + ".")
             or remainder == w),
            None,
        )
        if not writer:
            return None, None, None

    m = re.search(r"cv(\d+)", filename, re.IGNORECASE)
    if not m:
        return None, None, None
    return evaluator, writer, int(m.group(1))


# ==========================
# DATA LOADING — Plots 1 & 2
# ==========================

def _scores_from_folder(folder):
    scores = {}
    if not os.path.exists(folder):
        return scores
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            s = extract_first_number(os.path.join(folder, fname))
            if s is not None:
                scores[fname] = s
    return scores


def collect_job_data(version_folders, cv_count):
    etypes = ["cl_evaluations", "cv_cl_evaluations"]
    data = {
        et: {pair: {i: [] for i in range(1, cv_count + 2)} for pair in MODEL_PAIRS}
        for et in etypes
    }
    for _, _, v_path in version_folders:
        if not os.path.exists(v_path):
            continue
        for run_folder in sorted(os.listdir(v_path)):
            run_path = os.path.join(v_path, run_folder)
            if not os.path.isdir(run_path):
                continue
            for etype in etypes:
                for fname, score in _scores_from_folder(os.path.join(run_path, etype)).items():
                    for pair in MODEL_PAIRS:
                        if pair in fname:
                            m = re.search(r"cv(\d+)", fname, re.IGNORECASE)
                            if m:
                                idx = int(m.group(1))
                                if idx <= cv_count:
                                    data[etype][pair][idx].append(score)
    return data


def merge_jobs_data(all_jobs_data):
    etypes = ["cl_evaluations", "cv_cl_evaluations"]
    global_data = {et: {pair: {} for pair in MODEL_PAIRS} for et in etypes}
    for job_data in all_jobs_data.values():
        for et in etypes:
            for pair in MODEL_PAIRS:
                for cv_idx, scores in job_data[et][pair].items():
                    global_data[et][pair].setdefault(cv_idx, []).extend(scores)
    return global_data


# ==========================
# DATA LOADING
# ==========================

def build_master_df(base_dir, force=False):
    if not force and os.path.exists(CACHE_PATH):
        print(f"  Loading master dataframe from cache ({CACHE_PATH})")
        return pd.read_parquet(CACHE_PATH)

    print("  Building master dataframe from raw files (this takes a while)...")
    rows = []
    for job_folder in sorted(os.listdir(base_dir)):
        job_path = os.path.join(base_dir, job_folder)
        if not os.path.isdir(job_path):
            continue
        m = re.match(r"(job_\d+)", job_folder)
        if not m:
            continue
        job_id = m.group(1)
        job_title = format_job_title(job_id, job_folder)
        for run_folder in os.listdir(job_path):
            if not run_folder.startswith("run_"):
                continue
            m_run = re.search(r"run_(\d+)", run_folder)
            run_id = int(m_run.group(1)) if m_run else 0
            run_path = os.path.join(job_path, run_folder)
            for etype in ["cv_only", "cl_evaluations", "cv_cl_evaluations"]:
                eval_path = os.path.join(run_path, etype)
                if not os.path.exists(eval_path):
                    continue
                for fname in os.listdir(eval_path):
                    if not fname.endswith(".txt"):
                        continue
                    evaluator, writer, cv_idx = parse_filename(fname, etype)
                    if evaluator and writer and cv_idx:
                        score = extract_first_number(os.path.join(eval_path, fname))
                        if score is not None:
                            rows.append({
                                "Job_ID": job_id, "Job_Title": job_title,
                                "Run": run_id,
                                "Eval_Type": etype, "Evaluator": evaluator,
                                "Writer": writer, "CV_Idx": cv_idx, "Score": score,
                            })
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE_PATH, index=False)
    print(f"  Cached master dataframe → {CACHE_PATH}")
    return df




def load_competitive_data(base_dir):
    if not os.path.exists(CACHE_PATH):
        return pd.DataFrame()
    df = pd.read_parquet(CACHE_PATH)
    if "Run" not in df.columns:
        print("  WARNING: parquet missing Run column — regenerate cache with raw data present.")
        return pd.DataFrame()
    comp = df[df["Eval_Type"].isin(["cv_only", "cv_cl_evaluations"])].copy()
    comp = comp.rename(columns={"Job_ID": "Job", "Eval_Type": "Type"})
    return comp[["Job", "Run", "Evaluator", "Writer", "CV_Idx", "Score", "Type"]]


def calculate_leapfrog(df, evaluator, baseline_writer, target_writer):
    eval_df = df[df["Evaluator"] == evaluator]
    results = []
    for job_id in eval_df["Job"].unique():
        for run_id in eval_df["Run"].unique():
            env = eval_df[(eval_df["Job"] == job_id) & (eval_df["Run"] == run_id)]
            cv_only = env[env["Type"] == "cv_only"].copy()
            if len(cv_only) != 50:
                continue
            cv_only = cv_only.sort_values(by=["Score", "CV_Idx"], ascending=[False, True])
            cv_only["Baseline_Rank"] = range(1, 51)
            incumbent_cvs  = cv_only[cv_only["Baseline_Rank"] <= 25]["CV_Idx"].tolist()
            challenger_cvs = cv_only[cv_only["Baseline_Rank"] > 25]["CV_Idx"].tolist()
            cv_cl = env[env["Type"] == "cv_cl_evaluations"]
            incumbents  = cv_cl[(cv_cl["Writer"] == baseline_writer) & cv_cl["CV_Idx"].isin(incumbent_cvs)].copy()
            challengers = cv_cl[(cv_cl["Writer"] == target_writer)  & cv_cl["CV_Idx"].isin(challenger_cvs)].copy()
            challengers["Type_Tag"] = "Challenger"
            if len(incumbents) == 0 or len(challengers) == 0:
                continue
            pool = pd.concat([incumbents, challengers])
            pool = pool.sort_values(by=["Score", "CV_Idx"], ascending=[False, True])
            pool["New_Rank"] = range(1, len(pool) + 1)
            success = pool[(pool["Type_Tag"] == "Challenger") & (pool["New_Rank"] <= 25)]
            results.append(len(success) / len(challengers) * 100)
    return results


def _gap_matrices_from_df(tier_df, etype):
    """Compute heatmap-gap matrices from master_df (same logic as the raw-file approach)."""
    sub = tier_df[tier_df["Eval_Type"] == etype]
    if sub.empty:
        return None, None, None, None

    # Mean score per (Evaluator, CV_Idx, Writer) across all jobs/runs
    pair_mean = (sub.groupby(["Evaluator", "CV_Idx", "Writer"])["Score"]
                   .mean().reset_index().rename(columns={"Score": "PairMean"}))
    # Mean score per (Evaluator, CV_Idx) across all writers/jobs/runs
    all_mean = (sub.groupby(["Evaluator", "CV_Idx"])["Score"]
                  .mean().reset_index().rename(columns={"Score": "AllMean"}))
    merged = pair_mean.merge(all_mean, on=["Evaluator", "CV_Idx"])
    merged["Gap"] = merged["PairMean"] - merged["AllMean"]

    data_rows, annot_rows = [], []
    for eval_name in UNIQUE_EVALUATORS:
        row_vals, row_annots = [], []
        for write_name in SORTED_WRITERS:
            gaps = merged.loc[
                (merged["Evaluator"] == eval_name) & (merged["Writer"] == write_name), "Gap"
            ].tolist()
            if gaps:
                mean_gap = np.mean(gaps)
                is_sig = False
                if len(gaps) > 1 and np.var(gaps) != 0:
                    try:
                        _, p = stats.ttest_1samp(gaps, 0)
                        is_sig = p < 0.05
                    except Exception:
                        pass
                row_vals.append(mean_gap)
                row_annots.append(f"{mean_gap:.2f}{'(*)' if is_sig else ''}")
            else:
                row_vals.append(0.0)
                row_annots.append("0.00")
        data_rows.append(row_vals)
        annot_rows.append(row_annots)

    df = pd.DataFrame(data_rows, index=UNIQUE_EVALUATORS, columns=SORTED_WRITERS)
    df_annot = pd.DataFrame(annot_rows, index=UNIQUE_EVALUATORS, columns=SORTED_WRITERS)
    col_means = df.mean(axis=0)
    ew  = [w for w in df.columns if w in UNIQUE_EVALUATORS]
    new = [w for w in df.columns if w not in UNIQUE_EVALUATORS]
    sc  = (sorted(ew,  key=lambda w: col_means[w], reverse=True)
           + sorted(new, key=lambda w: col_means[w], reverse=True))
    sr  = [w for w in sc if w in UNIQUE_EVALUATORS]
    df       = df.loc[sr, sc]
    df_annot = df_annot.loc[sr, sc]
    df.loc["Avg."]       = df.mean(axis=0)
    df_annot.loc["Avg."] = [f"{df.loc['Avg.', c]:.2f}" for c in df.columns]
    return df, df_annot, sr, sc


def plot_heatmap_gap(tier_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    etypes = ["cl_evaluations", "cv_cl_evaluations"]

    built = {}
    for etype in etypes:
        result = _gap_matrices_from_df(tier_df, etype)
        if result[0] is None:
            continue
        built[etype] = result

    # Symmetric scale: use max absolute value across both etypes (non-Average rows only)
    max_abs = max(
        built[et][0].loc[built[et][2], :].abs().max().max()
        for et in etypes
    )
    max_abs = np.ceil(max_abs * 20) / 20  # round up to nearest 0.05
    shared_vmin, shared_vmax = -max_abs, max_abs

    # --- Second pass: plot using shared scale ---
    fs = plt.rcParams["font.size"] + 3
    for etype in etypes:
        df, df_annot, sorted_rows, sorted_cols = built[etype]
        df.to_csv(os.path.join(save_dir, f"heatmap_gap_{etype}.csv"))

        df_plot      = df.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
        df_annot_plot = df_annot.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)

        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            df_plot, annot=df_annot_plot, fmt="", cmap="RdBu_r", center=0,
            vmin=shared_vmin, vmax=shared_vmax,
            linewidths=0.5, linecolor="gray",
            annot_kws={"size": fs},
            cbar_kws={"label": "Preference Gap (score units, 0–10 scale)"},
        )
        ax.axhline(len(sorted_rows), color="black", linewidth=2)
        # Thick border on diagonal cells (evaluator == writer → self-preference)
        for e in sorted_rows:
            if e in sorted_cols:
                r = list(sorted_rows).index(e)
                c = list(sorted_cols).index(e)
                ax.add_patch(plt.Rectangle(
                    (c, r), 1, 1, fill=False, edgecolor="black", linewidth=3
                ))
        plt.title(
            f"Relative Preference Gap | {TITLE_MAP[etype]}\n"
            "Columns sorted left → right by most preferred writer  |  (*) = p < 0.05  |  Bold border = self-evaluation",
            fontsize=fs + 5,
        )
        plt.xlabel("Writer Model", fontsize=fs + 4)
        plt.ylabel("Evaluator Model", fontsize=fs + 4)
        ax.tick_params(axis="y", labelsize=fs)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs, fontweight="bold")
        for tick in ax.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fs, fontweight="bold")
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ax.collections[0].colorbar.ax.tick_params(labelsize=fs)
        ax.collections[0].colorbar.ax.yaxis.label.set_size(fs + 4)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"heatmap_gap_{etype}.png"))
        plt.close()
        print(f"  Saved heatmap_gap_{etype}.png")


def plot_heatmap_gap_combined(tier_df, save_dir):
    """Both heatmaps side by side, each with its own ordering, shared colorbar."""
    os.makedirs(save_dir, exist_ok=True)
    etypes = ["cl_evaluations", "cv_cl_evaluations"]

    built = {et: _gap_matrices_from_df(tier_df, et) for et in etypes}
    if any(v[0] is None for v in built.values()):
        return

    # Shared symmetric scale
    max_abs = max(
        built[et][0].loc[built[et][2], :].abs().max().max()
        for et in etypes
    )
    max_abs = np.ceil(max_abs * 20) / 20
    shared_vmin, shared_vmax = -max_abs, max_abs

    fs = plt.rcParams["font.size"] + 7
    fig, axes = plt.subplots(1, 2, figsize=(24, 9), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.08)

    for ax_idx, (ax, etype) in enumerate(zip(axes, etypes)):
        df, df_annot, sorted_rows, sorted_cols = built[etype]
        df_plot       = df.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
        df_annot_plot = df_annot.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
        sns.heatmap(
            df_plot, annot=df_annot_plot, fmt="", ax=ax,
            cmap="RdBu_r", center=0, vmin=shared_vmin, vmax=shared_vmax,
            linewidths=0.5, linecolor="gray",
            annot_kws={"size": fs + 1},
            cbar=False,
        )
        for text in ax.texts:
            t = text.get_text()
            if "(*)" in t:
                text.set_text(t.replace("(*)", ""))
                text.set_fontweight("bold")
        # Separator before Average row
        ax.axhline(len(sorted_rows), color="black", linewidth=2)
        # Thick border on diagonal cells (self-evaluation) — positions are index-based
        for e in sorted_rows:
            if e in sorted_cols:
                r = list(sorted_rows).index(e)
                c = list(sorted_cols).index(e)
                ax.add_patch(plt.Rectangle(
                    (c, r), 1, 1, fill=False, edgecolor="black", linewidth=3
                ))
        ax.set_title(TITLE_MAP[etype], fontsize=fs + 5, pad=10)
        ax.set_xlabel("Writer Model", fontsize=fs + 4, labelpad=12)
        ax.set_ylabel("Evaluator Model" if ax_idx == 0 else "", fontsize=fs + 4)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs + 2, fontweight="bold")
        for tick in ax.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ytick_info = [(t.get_text(), DISPLAY_COLORS.get(t.get_text(), "black"))
                      for t in ax.get_yticklabels()]
        ax.set_yticklabels(
            ["Avg." if name == "Avg." else "■" for name, _ in ytick_info],
            rotation=0, fontweight="bold",
        )
        for tick, (name, color) in zip(ax.get_yticklabels(), ytick_info):
            tick.set_color("black" if tick.get_text() == "Avg." else color)
            tick.set_fontsize(fs + 2 if tick.get_text() == "Avg." else fs + 4)

    # Single shared colorbar to the right
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=plt.Normalize(vmin=shared_vmin, vmax=shared_vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("Relative Preference Gap, 0–10 scale", fontsize=fs + 4)
    cbar.ax.tick_params(labelsize=fs + 2)

    fig.suptitle(
        "Relative Preference Gap  |  Bold = p < 0.05",
        fontsize=fs + 5,
    )
    plt.savefig(os.path.join(save_dir, "heatmap_gap_combined.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved heatmap_gap_combined.png")


def plot_win_matrix(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    etype = "cv_cl_evaluations"
    writers = sorted([w for w in df["Writer"].unique() if w != "CV_ONLY"])
    n = len(writers)

    subset = df[df["Eval_Type"] == etype]
    if subset.empty:
        return

    win_matrix = np.zeros((n, n))
    match_count = np.zeros((n, n))
    for (job, cv, evaluator), group in subset.groupby(["Job_ID", "CV_Idx", "Evaluator"]):
        scores = dict(zip(group["Writer"], group["Score"]))
        for i, w1 in enumerate(writers):
            for j, w2 in enumerate(writers):
                if i == j or evaluator in (w1, w2):
                    continue
                s1, s2 = scores.get(w1), scores.get(w2)
                if s1 is not None and s2 is not None:
                    match_count[i][j] += 1
                    if s1 > s2:
                        win_matrix[i][j] += 1
                    elif s1 == s2:
                        win_matrix[i][j] += 0.5

    win_rate = np.full((n, n), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        win_rate = (win_matrix / match_count) * 100
    np.fill_diagonal(win_rate, np.nan)

    df_win = pd.DataFrame(win_rate, index=writers, columns=writers)
    df_win["Avg."] = df_win.mean(axis=1, skipna=True)
    # Sort model rows only, then append Avg. at the bottom
    df_win = df_win.sort_values("Avg.", ascending=False)
    sorted_models = list(df_win.index)
    df_win = df_win[sorted_models + ["Avg."]]
    avg_row = df_win.mean(axis=0, skipna=True)
    avg_row.name = "Avg."
    df_win = pd.concat([df_win, avg_row.to_frame().T])

    df_win.to_csv(os.path.join(save_dir, f"win_matrix_UNBIASED_{etype}.csv"))

    # Custom annotation: "%" suffix, "—" for NaN (diagonal)
    annot_df = df_win.copy().astype(object)
    for r in annot_df.index:
        for c in annot_df.columns:
            val = df_win.loc[r, c]
            annot_df.loc[r, c] = "—" if pd.isna(val) else f"{val:.0f}%"

    mask = pd.DataFrame(False, index=df_win.index, columns=df_win.columns)
    for w in writers:
        if w in df_win.index and w in df_win.columns:
            mask.loc[w, w] = True

    # Precompute diagonal positions (raw names) before renaming
    diag_pos = [(list(df_win.index).index(w), list(df_win.columns).index(w))
                for w in writers if w in df_win.index and w in df_win.columns]
    n_models = sum(1 for r in df_win.index if r != "Avg.")

    df_win   = df_win.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
    annot_df = annot_df.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
    mask     = mask.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)

    fs = plt.rcParams["font.size"] + 3
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        df_win, annot=annot_df, fmt="", cmap="RdBu_r", vmin=0, vmax=100, center=50,
        cbar_kws={"label": "Win Rate %"}, linewidths=0.5, linecolor="gray",
        mask=mask, ax=ax, annot_kws={"size": fs},
    )
    # Grey patches + "—" text for diagonal
    for ri, ci in diag_pos:
        ax.add_patch(plt.Rectangle((ci, ri), 1, 1, fill=True, color="lightgrey", lw=0))
        ax.text(ci + 0.5, ri + 0.5, "—", ha="center", va="center", fontsize=fs, color="grey")

    # Thick separator before Avg row/col
    ax.axhline(n_models, color="black", linewidth=2)
    ax.axvline(len(df_win.columns) - 1, color="black", linewidth=2)

    plt.title(f"Head-to-Head Win Rate | {TITLE_MAP[etype]}", fontsize=fs + 5)
    plt.xlabel("Opponent Writer Model", fontsize=fs + 4)
    plt.ylabel("Writer Model", fontsize=fs + 4)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs, fontweight="bold")
    for tick in ax.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fs, fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    ax.collections[0].colorbar.ax.tick_params(labelsize=fs)
    ax.collections[0].colorbar.ax.yaxis.label.set_size(fs + 4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"win_matrix_UNBIASED_{etype}.png"))
    plt.close()
    print(f"  Saved win_matrix_UNBIASED_{etype}.png")


def plot_net_advantage(delta_matrix, p_matrix, raw_matrix, writers, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    N = len(writers)
    ext_writers = list(writers) + ["Avg."]

    masked_delta = delta_matrix.copy()
    np.fill_diagonal(masked_delta, np.nan)
    row_means    = np.nanmean(masked_delta, axis=1)
    col_means    = np.nanmean(masked_delta, axis=0)
    overall_mean = np.nanmean(masked_delta)

    ext_delta = np.zeros((N + 1, N + 1))
    ext_delta[:N, :N] = delta_matrix
    ext_delta[:N, N]  = row_means
    ext_delta[N, :N]  = col_means
    ext_delta[N, N]   = overall_mean

    annot = []
    for i in range(N + 1):
        row = []
        for j in range(N + 1):
            val  = ext_delta[i, j]
            sign = "+" if val > 0 else ""
            if i == j and i < N:
                row.append("Control")
            elif i < N and j < N:
                star = "(*)" if p_matrix[i, j] < 0.05 else ""
                row.append(f"{sign}{val:.1f}%{star}")
            else:
                row.append(f"{sign}{val:.1f}%")
        annot.append(row)
    annot = np.array(annot)

    df_m = pd.DataFrame(ext_delta, index=ext_writers, columns=ext_writers)
    sort_vals     = pd.Series(col_means, index=writers).sort_values(ascending=False)
    sorted_models = sort_vals.index.tolist()
    df_m = df_m.loc[sorted_models + ["Avg."], sorted_models + ["Avg."]]
    old_idx = {w: i for i, w in enumerate(writers)}
    new_ri  = [old_idx[w] for w in sorted_models] + [N]
    new_ci  = [old_idx[w] for w in sorted_models] + [N]
    annot   = annot[np.ix_(new_ri, new_ci)]

    df_m.to_csv(os.path.join(save_dir, "net_advantage_matrix_ALL_COMBINED.csv"))

    df_m_plot = df_m.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)

    fs = plt.rcParams["font.size"] + 3
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        df_m_plot, annot=annot, fmt="", cmap="RdBu_r", center=0,
        cbar_kws={"label": "Net Advantage over Control Group (%)"},
        annot_kws={"size": fs},
    )
    ax.axhline(N, color="black", linewidth=2)
    ax.axvline(N, color="black", linewidth=2)
    plt.title(
        "Net Competitive Advantage\n"
        "Control = same model for both groups  |  (*) = p < 0.05",
        fontsize=fs + 5,
    )
    plt.ylabel("High-Fit Candidates | Cover Letter Writer", fontsize=fs + 4)
    plt.xlabel("Moderate-Fit Candidates | Cover Letter Writer", fontsize=fs + 4)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs, fontweight="bold")
    for tick in ax.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fs, fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    ax.collections[0].colorbar.ax.tick_params(labelsize=fs)
    ax.collections[0].colorbar.ax.yaxis.label.set_size(fs + 4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "net_advantage_matrix_ALL_COMBINED.png"))
    plt.close()
    print("  Saved net_advantage_matrix_ALL_COMBINED.png")


def plot_win_net_combined(tier_df, delta_matrix, p_matrix, raw_matrix, writers_net, save_dir):
    """Win-rate matrix (left) and net advantage matrix (right) side by side."""
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 8  # larger canvas (28×11) needs bigger base fs

    # ── Win matrix data ────────────────────────────────────────────────────────
    etype   = "cv_cl_evaluations"
    subset  = tier_df[tier_df["Eval_Type"] == etype]
    writers_win = sorted([w for w in subset["Writer"].unique() if w != "CV_ONLY"])
    n_win = len(writers_win)

    win_matrix  = np.zeros((n_win, n_win))
    match_count = np.zeros((n_win, n_win))
    for (job, cv, evaluator), group in subset.groupby(["Job_ID", "CV_Idx", "Evaluator"]):
        scores = dict(zip(group["Writer"], group["Score"]))
        for i, w1 in enumerate(writers_win):
            for j, w2 in enumerate(writers_win):
                if i == j or evaluator in (w1, w2):
                    continue
                s1, s2 = scores.get(w1), scores.get(w2)
                if s1 is not None and s2 is not None:
                    match_count[i][j] += 1
                    if s1 > s2:
                        win_matrix[i][j] += 1
                    elif s1 == s2:
                        win_matrix[i][j] += 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        win_rate = (win_matrix / match_count) * 100
    np.fill_diagonal(win_rate, np.nan)

    df_win = pd.DataFrame(win_rate, index=writers_win, columns=writers_win)
    df_win["Avg."] = df_win.mean(axis=1, skipna=True)
    df_win = df_win.sort_values("Avg.", ascending=False)
    sorted_win_models = list(df_win.index)
    df_win = df_win[sorted_win_models + ["Avg."]]
    avg_row_win = df_win.mean(axis=0, skipna=True)
    avg_row_win.name = "Avg."
    df_win = pd.concat([df_win, avg_row_win.to_frame().T])

    annot_win = df_win.copy().astype(object)
    for r in annot_win.index:
        for c in annot_win.columns:
            val = df_win.loc[r, c]
            annot_win.loc[r, c] = "—" if pd.isna(val) else f"{val:.0f}%"

    mask_win = pd.DataFrame(False, index=df_win.index, columns=df_win.columns)
    for w in writers_win:
        if w in df_win.index and w in df_win.columns:
            mask_win.loc[w, w] = True

    # ── Net advantage data ─────────────────────────────────────────────────────
    N = len(writers_net)
    masked_delta = delta_matrix.copy()
    np.fill_diagonal(masked_delta, np.nan)
    row_means    = np.nanmean(masked_delta, axis=1)
    col_means    = np.nanmean(masked_delta, axis=0)
    overall_mean = np.nanmean(masked_delta)

    ext_delta = np.zeros((N + 1, N + 1))
    ext_delta[:N, :N] = delta_matrix
    ext_delta[:N, N]  = row_means
    ext_delta[N, :N]  = col_means
    ext_delta[N, N]   = overall_mean

    annot_net = []
    for i in range(N + 1):
        row = []
        for j in range(N + 1):
            val  = ext_delta[i, j]
            sign = "+" if val > 0 else ""
            if i == j and i < N:
                row.append("Control")
            elif i < N and j < N:
                star = "(*)" if p_matrix[i, j] < 0.05 else ""
                row.append(f"{sign}{val:.1f}%{star}")
            else:
                row.append(f"{sign}{val:.1f}%")
        annot_net.append(row)
    annot_net = np.array(annot_net)

    ext_writers = list(writers_net) + ["Avg."]
    sort_vals     = pd.Series(col_means, index=writers_net).sort_values(ascending=False)
    sorted_net    = sort_vals.index.tolist()
    df_net        = pd.DataFrame(ext_delta, index=ext_writers, columns=ext_writers)
    df_net        = df_net.loc[sorted_net + ["Avg."], sorted_net + ["Avg."]]
    old_idx = {w: i for i, w in enumerate(writers_net)}
    new_ri  = [old_idx[w] for w in sorted_net] + [N]
    new_ci  = [old_idx[w] for w in sorted_net] + [N]
    annot_net = annot_net[np.ix_(new_ri, new_ci)]

    # Precompute diagonal positions (raw names) before renaming
    diag_win = [(list(df_win.index).index(w), list(df_win.columns).index(w))
                for w in writers_win if w in df_win.index and w in df_win.columns]
    n_win_models = len(sorted_win_models)

    df_win   = df_win.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
    annot_win = annot_win.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
    mask_win  = mask_win.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
    df_net    = df_net.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 11), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.08)

    # Left: win matrix
    sns.heatmap(
        df_win, annot=annot_win, fmt="", ax=ax1,
        cmap="RdBu_r", vmin=0, vmax=100, center=50,
        linewidths=0.5, linecolor="gray",
        mask=mask_win, cbar=False,
        annot_kws={"size": fs + 1},
    )
    for ri, ci in diag_win:
        ax1.add_patch(plt.Rectangle((ci, ri), 1, 1, fill=True, color="lightgrey", lw=0))
        ax1.text(ci + 0.5, ri + 0.5, "—", ha="center", va="center", fontsize=fs, color="grey")
    ax1.axhline(n_win_models, color="black", linewidth=2)
    ax1.axvline(len(df_win.columns) - 1, color="black", linewidth=2)
    ax1.set_title("Head-to-Head Win Rate", fontsize=fs + 5, pad=10)
    ax1.set_xlabel("Opponent Writer Model", fontsize=fs + 4, labelpad=12)
    ax1.set_ylabel("Writer Model", fontsize=fs + 4)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=fs, fontweight="bold")
    for tick in ax1.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    ytick_info1 = [(t.get_text(), DISPLAY_COLORS.get(t.get_text(), "black"))
                   for t in ax1.get_yticklabels()]
    ax1.set_yticklabels(
        ["Avg." if name == "Avg." else "■" for name, _ in ytick_info1],
        rotation=0, fontweight="bold",
    )
    for tick, (name, color) in zip(ax1.get_yticklabels(), ytick_info1):
        tick.set_color("black" if tick.get_text() == "Avg." else color)
        tick.set_fontsize(fs if tick.get_text() == "Avg." else fs + 4)
    sm1 = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 100))
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, ax=ax1, shrink=0.8)
    cb1.set_label("Win Rate, %", fontsize=fs + 4)
    cb1.ax.tick_params(labelsize=fs)

    # Right: net advantage
    net_vmax = np.nanmax(np.abs(ext_delta[:-1, :-1]))  # ignore AVERAGE row/col for scale
    sns.heatmap(
        df_net, annot=annot_net, fmt="", ax=ax2,
        cmap="RdBu_r", center=0,
        linewidths=0.5, linecolor="gray",
        cbar=False,
        annot_kws={"size": fs + 1},
    )
    for text in ax2.texts:
        t = text.get_text()
        if "(*)" in t:
            text.set_text(t.replace("(*)", ""))
            text.set_fontweight("bold")
    ax2.axhline(N, color="black", linewidth=2)
    ax2.axvline(N, color="black", linewidth=2)
    ax2.set_title("Net Competitive Advantage over Control", fontsize=fs + 5, pad=10)
    ax2.set_xlabel("Moderate-Fit Group — Writer Model", fontsize=fs + 4, labelpad=12)
    ax2.set_ylabel("High-Fit Group — Writer Model", fontsize=fs + 4)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=fs, fontweight="bold")
    for tick in ax2.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    ytick_info2 = [(t.get_text(), DISPLAY_COLORS.get(t.get_text(), "black"))
                   for t in ax2.get_yticklabels()]
    ax2.set_yticklabels(
        ["Avg." if name == "Avg." else "■" for name, _ in ytick_info2],
        rotation=0, fontweight="bold",
    )
    for tick, (name, color) in zip(ax2.get_yticklabels(), ytick_info2):
        tick.set_color("black" if tick.get_text() == "Avg." else color)
        tick.set_fontsize(fs if tick.get_text() == "Avg." else fs + 4)
    sm2 = plt.cm.ScalarMappable(
        cmap="RdBu_r",
        norm=plt.Normalize(vmin=-net_vmax, vmax=net_vmax),
    )
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax2, shrink=0.8)
    cb2.set_label("Net Advantage over Control, %", fontsize=fs + 4)
    cb2.ax.tick_params(labelsize=fs)

    fig.suptitle(
        "Writer Model Competitiveness — CV + Cover Letter  |  Bold = p < 0.05",
        fontsize=fs + 5,
    )
    plt.savefig(os.path.join(save_dir, "win_net_combined.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved win_net_combined.png")


def plot_evaluator_divergence(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    records = []
    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        subset = df[df["Eval_Type"] == etype].copy()
        if subset.empty:
            continue
        consensus = (
            subset.groupby(["Job_ID", "CV_Idx", "Writer"])["Score"]
            .mean().reset_index().rename(columns={"Score": "Consensus"})
        )
        subset = subset.merge(consensus, on=["Job_ID", "CV_Idx", "Writer"])
        subset["AbsDev"] = (subset["Score"] - subset["Consensus"]).abs()
        for evaluator, grp in subset.groupby("Evaluator"):
            records.append({"Evaluator": evaluator, "Eval_Type": TITLE_MAP[etype],
                             "Mean_AbsDev": grp["AbsDev"].mean()})

    results_df = pd.DataFrame(records)
    pivot = results_df.pivot(index="Evaluator", columns="Eval_Type", values="Mean_AbsDev")
    pivot["_sort"] = pivot.mean(axis=1)
    # Descending sort → lowest divergence at top of horizontal bar chart
    pivot = pivot.sort_values("_sort", ascending=True).drop(columns="_sort")
    mean_dev = results_df["Mean_AbsDev"].mean()
    pivot.to_csv(os.path.join(save_dir, "evaluator_divergence.csv"))

    fs = plt.rcParams["font.size"] + 3
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", ax=ax, color=["#4393C3", "#D6604D"], edgecolor="white")
    ax.axvline(mean_dev, color="black", linestyle="--", linewidth=1.2,
               label=f"Overall mean ({mean_dev:.2f})")
    # Value labels at tip of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=4, fontsize=fs - 2)
    ax.set_title("Evaluator Divergence from Group Consensus\n(Lower = Closer to Average)",
                 fontsize=fs + 4)
    ax.set_xlabel("Avg. Deviation from Group Score", fontsize=fs)
    ax.set_ylabel("Evaluator Model", fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.legend(fontsize=fs)   # no legend title
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "evaluator_divergence.png"))
    plt.close()
    print("  Saved evaluator_divergence.png")


def plot_stacked_score(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 3

    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        subset = df[df["Eval_Type"] == etype].copy()
        if subset.empty:
            continue
        evaluators = subset["Evaluator"].unique()
        writers    = subset["Writer"].unique()

        records = []
        for evaluator in evaluators:
            eval_data = subset[subset["Evaluator"] == evaluator]
            for writer in writers:
                gaps = []
                for (job, cv), group in eval_data.groupby(["Job_ID", "CV_Idx"]):
                    if writer not in group["Writer"].values:
                        continue
                    gaps.append(group[group["Writer"] == writer]["Score"].mean()
                                - group["Score"].mean())
                if gaps:
                    records.append({"Evaluator": evaluator, "Writer": writer,
                                    "Gap": np.mean(gaps)})
        if not records:
            continue

        gap_df    = pd.DataFrame(records)
        total_abs = gap_df.groupby("Evaluator")["Gap"].apply(lambda x: x.abs().sum())
        evaluator_order = total_abs.sort_values().index.tolist()

        ordered_writers = [w for w in SORTED_WRITERS if w in writers]

        fig, ax = plt.subplots(figsize=(14, 8))
        for x_pos, evaluator in enumerate(evaluator_order):
            eval_gaps = gap_df[gap_df["Evaluator"] == evaluator].set_index("Writer")
            bottom = 0.0
            # Pass 1: positive contributions (fixed writer order)
            for writer in ordered_writers:
                if writer not in eval_gaps.index:
                    continue
                gap = eval_gaps.loc[writer, "Gap"]
                if gap < 0:
                    continue
                ax.bar(x_pos, gap, bottom=bottom, color=WRITER_COLORS[writer],
                       edgecolor="white", linewidth=0.5)
                bottom += gap
            # Pass 2: negative contributions on top (fixed writer order, abs height)
            for writer in ordered_writers:
                if writer not in eval_gaps.index:
                    continue
                gap = eval_gaps.loc[writer, "Gap"]
                if gap >= 0:
                    continue
                ax.bar(x_pos, abs(gap), bottom=bottom, color=WRITER_COLORS[writer],
                       hatch="///", edgecolor="black", linewidth=0.5, alpha=0.85)
                bottom += abs(gap)

        ax.set_xticks(range(len(evaluator_order)))
        ax.set_xticklabels([MODEL_DISPLAY.get(e, e) for e in evaluator_order],
                           rotation=45, ha="right", fontsize=fs)
        for tick, ev in zip(ax.get_xticklabels(), evaluator_order):
            tick.set_color(WRITER_COLORS.get(ev, "black"))
            tick.set_fontweight("bold")

        ax.set_xlabel("Evaluator Model", fontsize=fs + 4)
        ax.set_ylabel("Cumulative |Score Gap| from Evaluator Mean", fontsize=fs + 4)
        ax.set_title(
            f"Cumulative Evaluator Bias (Score-Based) | {TITLE_MAP[etype]}\n"
            "Sorted left → right: least to most biased  |  Solid = over-scored, Hatched = under-scored",
            fontsize=fs + 5, pad=12,
        )
        ax.tick_params(axis="y", labelsize=fs)

        legend_handles = [mpatches.Patch(color=WRITER_COLORS[w], label=MODEL_DISPLAY.get(w, w))
                          for w in ordered_writers]
        legend_handles.append(mpatches.Patch(facecolor="white", hatch="///",
                                             edgecolor="black",
                                             label="Under-scored by this evaluator"))
        leg = ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                        fontsize=fs - 1, title="Writer Model", title_fontsize=fs + 4)

        plt.tight_layout(pad=1.5)
        plt.savefig(os.path.join(save_dir, f"fairness_stacked_{etype}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved fairness_stacked_{etype}.png")


def plot_stacked_rank(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 3

    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        subset = df[df["Eval_Type"] == etype].copy()
        if subset.empty:
            continue
        evaluators = subset["Evaluator"].unique()
        writers    = subset["Writer"].unique()
        n_writers  = len(writers)
        midpoint   = (n_writers + 1) / 2

        records = []
        for evaluator in evaluators:
            eval_data = subset[subset["Evaluator"] == evaluator]
            for (job_id, cv_idx), group in eval_data.groupby(["Job_ID", "CV_Idx"]):
                writer_ranks = group.groupby("Writer")["Score"].mean().rank(
                    ascending=False, method="average")
                for writer in writers:
                    if writer not in writer_ranks.index:
                        continue
                    records.append({"Evaluator": evaluator, "Writer": writer,
                                    "Gap": midpoint - writer_ranks[writer]})
        if not records:
            continue

        gap_df    = pd.DataFrame(records).groupby(["Evaluator", "Writer"])["Gap"].mean().reset_index()
        total_abs = gap_df.groupby("Evaluator")["Gap"].apply(lambda x: x.abs().sum())
        evaluator_order = total_abs.sort_values().index.tolist()

        ordered_writers = [w for w in SORTED_WRITERS if w in writers]

        fig, ax = plt.subplots(figsize=(14, 8))
        for x_pos, evaluator in enumerate(evaluator_order):
            eval_gaps = gap_df[gap_df["Evaluator"] == evaluator].set_index("Writer")
            bottom = 0.0
            # Pass 1: positive contributions (fixed writer order)
            for writer in ordered_writers:
                if writer not in eval_gaps.index:
                    continue
                gap = eval_gaps.loc[writer, "Gap"]
                if gap < 0:
                    continue
                ax.bar(x_pos, gap, bottom=bottom, color=WRITER_COLORS[writer],
                       edgecolor="white", linewidth=0.5)
                bottom += gap
            # Pass 2: negative contributions on top (fixed writer order, abs height)
            for writer in ordered_writers:
                if writer not in eval_gaps.index:
                    continue
                gap = eval_gaps.loc[writer, "Gap"]
                if gap >= 0:
                    continue
                ax.bar(x_pos, abs(gap), bottom=bottom, color=WRITER_COLORS[writer],
                       hatch="///", edgecolor="black", linewidth=0.5, alpha=0.85)
                bottom += abs(gap)

        ax.set_xticks(range(len(evaluator_order)))
        ax.set_xticklabels([MODEL_DISPLAY.get(e, e) for e in evaluator_order],
                           rotation=45, ha="right", fontsize=fs)
        for tick, ev in zip(ax.get_xticklabels(), evaluator_order):
            tick.set_color(WRITER_COLORS.get(ev, "black"))
            tick.set_fontweight("bold")

        ax.set_xlabel("Evaluator Model", fontsize=fs + 4)
        ax.set_ylabel("Cumulative |Rank Deviation| from Midpoint", fontsize=fs + 4)
        ax.set_title(
            f"Cumulative Evaluator Bias (Rank-Based) | {TITLE_MAP[etype]}",
            fontsize=fs + 5, pad=12,
        )
        ax.tick_params(axis="y", labelsize=fs)

        legend_handles = [mpatches.Patch(color=WRITER_COLORS[w], label=MODEL_DISPLAY.get(w, w))
                          for w in ordered_writers]
        legend_handles.append(mpatches.Patch(facecolor="white", hatch="///",
                                             edgecolor="black",
                                             label="Under-ranked by this evaluator"))
        leg = ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                        fontsize=fs - 1, title="Writer Model", title_fontsize=fs + 4)
        plt.tight_layout(pad=1.5)
        plt.savefig(os.path.join(save_dir, f"fairness_rank_stacked_{etype}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved fairness_rank_stacked_{etype}.png")


def plot_stacked_rank_combined(df, save_dir):
    """Side-by-side rank bias plot (CL Only | CV+CL) with a single shared legend."""
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 8  # bbox_inches=tight + external legend expands canvas
    etypes = ["cl_evaluations", "cv_cl_evaluations"]

    # Pre-compute gap data for both etypes
    computed = {}
    for etype in etypes:
        subset = df[df["Eval_Type"] == etype].copy()
        if subset.empty:
            continue
        writers   = subset["Writer"].unique()
        n_writers = len(writers)
        midpoint  = (n_writers + 1) / 2

        records = []
        for evaluator in subset["Evaluator"].unique():
            eval_data = subset[subset["Evaluator"] == evaluator]
            for (job_id, cv_idx), group in eval_data.groupby(["Job_ID", "CV_Idx"]):
                writer_ranks = group.groupby("Writer")["Score"].mean().rank(
                    ascending=False, method="average")
                for writer in writers:
                    if writer not in writer_ranks.index:
                        continue
                    records.append({"Evaluator": evaluator, "Writer": writer,
                                    "Gap": midpoint - writer_ranks[writer]})
        if not records:
            continue

        gap_df    = pd.DataFrame(records).groupby(["Evaluator", "Writer"])["Gap"].mean().reset_index()
        total_abs = gap_df.groupby("Evaluator")["Gap"].apply(lambda x: x.abs().sum())
        computed[etype] = {
            "gap_df":          gap_df,
            "total_abs":       total_abs,
            "evaluator_order": total_abs.sort_values().index.tolist(),
            "ordered_writers": [w for w in SORTED_WRITERS if w in writers],
            "n_writers":       n_writers,
        }

    if len(computed) < 2:
        print("  [v2] Skipping combined rank plot — data missing for one condition.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(24, 9))

    for ax, etype in zip(axes, etypes):
        d = computed[etype]
        gap_df, total_abs = d["gap_df"], d["total_abs"]
        evaluator_order, ordered_writers = d["evaluator_order"], d["ordered_writers"]

        for x_pos, evaluator in enumerate(evaluator_order):
            eval_gaps = gap_df[gap_df["Evaluator"] == evaluator].set_index("Writer")
            bottom = 0.0
            for writer in ordered_writers:
                if writer not in eval_gaps.index:
                    continue
                gap = eval_gaps.loc[writer, "Gap"]
                if gap < 0:
                    continue
                ax.bar(x_pos, gap, bottom=bottom, color=WRITER_COLORS[writer],
                       edgecolor="white", linewidth=0.5)
                bottom += gap
            for writer in ordered_writers:
                if writer not in eval_gaps.index:
                    continue
                gap = eval_gaps.loc[writer, "Gap"]
                if gap >= 0:
                    continue
                ax.bar(x_pos, abs(gap), bottom=bottom, color=WRITER_COLORS[writer],
                       hatch="///", edgecolor="black", linewidth=0.5, alpha=0.85)
                bottom += abs(gap)

        ax.set_xticks(range(len(evaluator_order)))
        ax.set_xticklabels([MODEL_DISPLAY.get(e, e) for e in evaluator_order],
                           rotation=45, ha="right", fontsize=fs)
        for tick, ev in zip(ax.get_xticklabels(), evaluator_order):
            tick.set_color(WRITER_COLORS.get(ev, "black"))
            tick.set_fontweight("bold")
        ax.set_xlabel("Evaluator Model", fontsize=fs + 4)
        ax.set_title(TITLE_MAP[etype], fontsize=fs + 5, pad=10)
        ax.tick_params(axis="y", labelsize=fs)

    axes[0].set_ylabel("Cumulative |Rank Deviation| from Midpoint", fontsize=fs + 4)

    # Single shared legend to the right of the second subplot
    ordered_writers = computed["cl_evaluations"]["ordered_writers"]
    legend_handles  = [mpatches.Patch(color=WRITER_COLORS[w], label=MODEL_DISPLAY.get(w, w))
                       for w in ordered_writers]
    legend_handles.append(mpatches.Patch(facecolor="white", hatch="///", edgecolor="black",
                                         label="Under-ranked by this evaluator"))
    fig.legend(handles=legend_handles, title="Writer Model", title_fontsize=fs + 4,
               bbox_to_anchor=(1.0, 0.5), loc="center left", fontsize=fs - 1)

    fig.suptitle(
        "Cumulative Evaluator Bias (Rank-Based)",
        fontsize=fs + 5,
    )
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(save_dir, "fairness_rank_combined.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fairness_rank_combined.png")


# ==========================
# AGREEMENT CORRELATION
# ==========================

def plot_agreement_corr(tier_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 3

    def _build_corr(vecs, method):
        df = pd.DataFrame(vecs).dropna()
        return df.corr(method=method)

    def _draw(corr, title, fname, cbar_label):
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        annot = corr.round(2).astype(str)
        sns.heatmap(corr, annot=annot, fmt="", ax=ax,
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="gray",
                    annot_kws={"size": fs + 1})
        ax.set_title(title, fontsize=fs + 7, pad=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                           fontsize=fs + 2, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=fs + 2, fontweight="bold")
        for tick in ax.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ax.set_xlabel("")
        ax.set_ylabel("")
        cbar = ax.collections[0].colorbar
        cbar.set_label(cbar_label, fontsize=fs + 6)
        cbar.ax.tick_params(labelsize=fs + 2)
        plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")

    # CV-only: raw score agreement (Pearson) + within-group rank Spearman
    sub_cv = tier_df[tier_df["Eval_Type"] == "cv_only"].copy()
    if not sub_cv.empty:
        vecs_cv = {
            MODEL_DISPLAY.get(ev, ev): (
                sub_cv[sub_cv["Evaluator"] == ev]
                .groupby(["Job_ID", "CV_Idx"])["Score"].mean()
            )
            for ev in UNIQUE_EVALUATORS
        }
        _draw(_build_corr(vecs_cv, "pearson"),
              "Evaluator Agreement — CV Only\n(Pearson r of raw scores)",
              "agreement_corr_cv_only.png", "Pearson r")
        _draw(_build_corr(vecs_cv, "spearman"),
              "Evaluator Agreement — CV Only\n(Spearman ρ of raw scores)",
              "agreement_corr_spearman_cv_only.png", "Spearman ρ")

    # CL and CV+CL: style residual agreement + within-group rank-based Spearman
    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        sub = tier_df[tier_df["Eval_Type"] == etype].copy()
        if sub.empty:
            continue

        # Residual vectors (for Pearson and residual-Spearman)
        sub["Residual"] = sub["Score"] - sub.groupby(
            ["Evaluator", "Job_ID", "CV_Idx"])["Score"].transform("mean")
        vecs_res = {
            MODEL_DISPLAY.get(ev, ev): (
                sub[sub["Evaluator"] == ev]
                .groupby(["Job_ID", "CV_Idx", "Writer"])["Residual"].mean()
            )
            for ev in UNIQUE_EVALUATORS
        }

        _draw(_build_corr(vecs_res, "pearson"),
              f"Evaluator Style Agreement — {TITLE_MAP[etype]}\n"
              "(Pearson r of within-candidate score residuals)",
              f"agreement_corr_STYLE_{etype}.png", "Pearson r")
        _draw(_build_corr(vecs_res, "spearman"),
              f"Evaluator Style Agreement — {TITLE_MAP[etype]}\n"
              "(Spearman ρ of within-candidate score residuals)",
              f"agreement_corr_spearman_STYLE_{etype}.png", "Spearman ρ")

        # Stylistic Agreement: rank writers within each (Job, CV, Evaluator) group
        sub["Style_Rank"] = sub.groupby(
            ["Job_ID", "CV_Idx", "Evaluator"])["Score"].rank(method="average")
        pivot_style = sub.pivot_table(
            index=["Job_ID", "CV_Idx", "Writer"], columns="Evaluator", values="Style_Rank")
        pivot_style.columns = [MODEL_DISPLAY.get(c, c) for c in pivot_style.columns]
        corr_style = pivot_style.corr(method="spearman")
        _draw(corr_style,
              f"Stylistic Agreement — {TITLE_MAP[etype]}\n"
              "(Spearman ρ of within-candidate writer ranks)",
              f"agreement_corr_spearman_STYLISTIC_{etype}.png", "Spearman ρ")

        # Merit Agreement: Spearman on raw scores per (Job, Writer, CV) — analogous to CV-only
        vecs_merit = {
            MODEL_DISPLAY.get(ev, ev): (
                sub[sub["Evaluator"] == ev]
                .groupby(["Job_ID", "Writer", "CV_Idx"])["Score"].mean()
            )
            for ev in UNIQUE_EVALUATORS
        }
        _draw(_build_corr(vecs_merit, "spearman"),
              f"Merit Agreement — {TITLE_MAP[etype]}\n"
              "(Spearman ρ of raw scores, fixed writer)",
              f"agreement_corr_spearman_MERIT_{etype}.png", "Spearman ρ")


def plot_merit_agreement_combined(tier_df, save_dir):
    """Three-panel heatmap: candidate quality agreement across all conditions."""
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 5

    def _corr(vecs):
        return pd.DataFrame(vecs).dropna().corr(method="spearman")

    # CV-only
    sub_cv = tier_df[tier_df["Eval_Type"] == "cv_only"]
    vecs_cv = {
        MODEL_DISPLAY.get(ev, ev): (
            sub_cv[sub_cv["Evaluator"] == ev]
            .groupby(["Job_ID", "CV_Idx"])["Score"].mean()
        )
        for ev in UNIQUE_EVALUATORS
    }

    # CL and CV+CL Merit: raw scores per (Job, Writer, CV)
    merit_corrs = {}
    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        sub = tier_df[tier_df["Eval_Type"] == etype]
        if sub.empty:
            continue
        vecs = {
            MODEL_DISPLAY.get(ev, ev): (
                sub[sub["Evaluator"] == ev]
                .groupby(["Job_ID", "Writer", "CV_Idx"])["Score"].mean()
            )
            for ev in UNIQUE_EVALUATORS
        }
        merit_corrs[etype] = _corr(vecs)

    panels = [
        (_corr(vecs_cv), "CV Only"),
        (merit_corrs.get("cl_evaluations"),   "Cover Letter Only\n(writer fixed)"),
        (merit_corrs.get("cv_cl_evaluations"), "CV + Cover Letter\n(writer fixed)"),
    ]
    panels = [(c, t) for c, t in panels if c is not None]
    n = len(panels)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 8), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.05)
    if n == 1:
        axes = [axes]

    for ax_idx, (ax, (corr, title)) in enumerate(zip(axes, panels)):
        is_last = ax_idx == n - 1
        sns.heatmap(corr, annot=corr.round(2).astype(str), fmt="", ax=ax,
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="gray",
                    annot_kws={"size": fs + 1}, cbar=False)
        ax.set_title(title, fontsize=fs + 5, pad=10)
        ax.set_xlabel("", fontsize=0)

        # X-axis: colored model names on all panels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                           fontsize=fs, fontweight="bold")
        for tick in ax.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))

        # Y-axis: colored model names only on leftmost panel
        if ax_idx == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                               fontsize=fs, fontweight="bold")
            for tick in ax.get_yticklabels():
                tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        else:
            ax.set_yticks([])
            ax.set_ylabel("")

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=fs + 4)
    cbar.ax.tick_params(labelsize=fs + 2)

    fig.suptitle("Candidate Quality Agreement across Conditions", fontsize=fs + 5)
    plt.savefig(os.path.join(save_dir, "merit_agreement_combined.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved merit_agreement_combined.png")


# ==========================
# STRICTNESS + AGREEMENT COMBINED
# ==========================

def _draw_strictness_left(ax_line, cv_sub, fs, variant):
    ax_line.cla()
    all_series = []
    all_stds = []
    for ev in UNIQUE_EVALUATORS:
        ev_data = cv_sub[cv_sub["Evaluator"] == ev].groupby("CV_Idx")["Score"].mean()
        if ev_data.empty:
            continue
        ev_std = cv_sub[cv_sub["Evaluator"] == ev].groupby("CV_Idx")["Score"].std().fillna(0)
        rolling = ev_data.rolling(window=3, min_periods=1).mean()
        rolling_std = ev_std.rolling(window=3, min_periods=1).mean()
        all_series.append(ev_data)
        all_stds.append(rolling_std)
        color = WRITER_COLORS[ev]
        ax_line.plot(
            rolling.index, rolling.values,
            label=MODEL_DISPLAY[ev],
            color=color,
            linewidth=2, alpha=0.85,
        )
    if all_series:
        all_rolling = pd.concat(all_series, axis=1).rolling(window=3, min_periods=1).mean()
        overall = all_rolling.mean(axis=1)
        ax_line.plot(overall.index, overall.values,
                     color="black", linewidth=3, label="Average", zorder=10)
        if variant == "avg_shade":
            overall_std = pd.concat(all_stds, axis=1).mean(axis=1)
            ax_line.fill_between(
                overall.index,
                overall.values - overall_std.values,
                overall.values + overall_std.values,
                color="black", alpha=0.12, label="Avg ± std",
            )
        elif variant == "envelope":
            ax_line.fill_between(
                overall.index,
                all_rolling.min(axis=1).values,
                all_rolling.max(axis=1).values,
                color="black", alpha=0.10, label="Model range",
            )
    ax_line.set_title("Evaluator Strictness by CV Rank", fontsize=fs + 5, pad=10)
    ax_line.set_xlabel("CV Rank", fontsize=fs + 4)
    ax_line.set_ylabel("Score", fontsize=fs + 4)
    ax_line.set_ylim(0, 10)
    ax_line.tick_params(axis="both", labelsize=fs)
    ax_line.grid(True, alpha=0.3)
    leg = ax_line.legend(fontsize=fs - 1, loc="upper right", framealpha=0.9)
    for text in leg.get_texts():
        text.set_color(DISPLAY_COLORS.get(text.get_text(), "black"))
        text.set_fontweight("bold")


def plot_strictness_agreement_combined(tier_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 5

    cv_sub = tier_df[tier_df["Eval_Type"] == "cv_only"]
    if cv_sub.empty:
        return

    # --- Right: evaluator agreement heatmap (shared across variants) ---
    vecs = {
        MODEL_DISPLAY.get(ev, ev): (
            cv_sub[cv_sub["Evaluator"] == ev]
            .groupby(["Job_ID", "CV_Idx"])["Score"].mean()
        )
        for ev in UNIQUE_EVALUATORS
    }
    corr_pearson  = pd.DataFrame(vecs).dropna().corr(method="pearson")
    corr_spearman = pd.DataFrame(vecs).dropna().corr(method="spearman")

    def _draw_heatmap(ax_heat, corr, cbar_label):
        sns.heatmap(corr, annot=corr.round(2).astype(str), fmt="", ax=ax_heat,
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="gray",
                    annot_kws={"size": fs - 1})
        ax_heat.set_title("Evaluator Agreement", fontsize=fs + 5, pad=10)
        ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha="right",
                                fontsize=fs, fontweight="bold")
        ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0,
                                fontsize=fs, fontweight="bold")
        for tick in ax_heat.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        for tick in ax_heat.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ax_heat.set_xlabel("", fontsize=fs + 4)
        ax_heat.set_ylabel("", fontsize=fs + 4)
        cbar = ax_heat.collections[0].colorbar
        cbar.set_label(cbar_label, fontsize=fs + 4)
        cbar.ax.tick_params(labelsize=fs)

    # --- Pearson: line plot combined with heatmap ---
    fig, (ax_line, ax_heat) = plt.subplots(
        1, 2, figsize=(20, 7),
        gridspec_kw={"width_ratios": [1.5, 1]},
        constrained_layout=True,
    )
    _draw_strictness_left(ax_line, cv_sub, fs, "avg_shade")
    _draw_heatmap(ax_heat, corr_pearson, "Pearson r")
    fig.savefig(os.path.join(save_dir, "strictness_agreement_cv_only.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved strictness_agreement_cv_only.png")

    # --- Spearman: line plot combined with heatmap ---
    fig, (ax_line, ax_heat) = plt.subplots(
        1, 2, figsize=(20, 7),
        gridspec_kw={"width_ratios": [1.5, 1]},
        constrained_layout=True,
    )
    _draw_strictness_left(ax_line, cv_sub, fs, "avg_shade")
    _draw_heatmap(ax_heat, corr_spearman, "Spearman ρ")
    fig.savefig(os.path.join(save_dir, "strictness_agreement_spearman_cv_only.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved strictness_agreement_spearman_cv_only.png")

    # --- Line plot standalone ---
    fig, ax_line = plt.subplots(figsize=(12, 7), constrained_layout=True)
    _draw_strictness_left(ax_line, cv_sub, fs, "avg_shade")
    fig.savefig(os.path.join(save_dir, "strictness_lineplot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved strictness_lineplot.png")

    # --- Box plot variant: High-Fit vs Moderate-Fit CV ranks ---
    all_ranks = sorted(cv_sub["CV_Idx"].unique())
    top25 = set(all_ranks[:25])
    bot25 = set(all_ranks[-25:])
    rows = []
    for ev in UNIQUE_EVALUATORS:
        ev_data = cv_sub[cv_sub["Evaluator"] == ev][["CV_Idx", "Score"]]
        for _, r in ev_data.iterrows():
            if r["CV_Idx"] in top25:
                rows.append({"Model": MODEL_DISPLAY[ev], "Tier": "High-Fit", "Score": r["Score"]})
            elif r["CV_Idx"] in bot25:
                rows.append({"Model": MODEL_DISPLAY[ev], "Tier": "Moderate-Fit", "Score": r["Score"]})
    box_df = pd.DataFrame(rows)

    fig, (ax_box, ax_heat) = plt.subplots(
        1, 2, figsize=(20, 7),
        gridspec_kw={"width_ratios": [1.5, 1]},
        constrained_layout=True,
    )
    model_order = [MODEL_DISPLAY[ev] for ev in UNIQUE_EVALUATORS if MODEL_DISPLAY[ev] in box_df["Model"].unique()]
    model_palette = {MODEL_DISPLAY[ev]: WRITER_COLORS[ev] for ev in UNIQUE_EVALUATORS}
    n_models = len(model_order)
    bar_width = 0.35 / n_models
    group_gap = 0.2
    tier_centers = []
    tier_means = {}
    for tier_idx, tier in enumerate(["High-Fit", "Moderate-Fit"]):
        x_center = tier_idx * (1 + group_gap)
        tier_centers.append(x_center)
        half_span = (n_models - 1) / 2 * (bar_width + 0.02) + bar_width / 2
        tier_means[tier] = (x_center - half_span, x_center + half_span,
                            box_df[box_df["Tier"] == tier]["Score"].mean())
        for model_idx, model in enumerate(model_order):
            subset = box_df[(box_df["Tier"] == tier) & (box_df["Model"] == model)]["Score"]
            if subset.empty:
                continue
            x_pos = x_center + (model_idx - (n_models - 1) / 2) * (bar_width + 0.02)
            ax_box.bar(x_pos, subset.mean(), width=bar_width,
                       color=model_palette[model], alpha=0.85,
                       yerr=subset.std(), error_kw={"capsize": 4, "linewidth": 1.5, "ecolor": "black"},
                       label=model if tier_idx == 0 else "_nolegend_")
    avg_line = None
    for tier, (x0, x1, mean_val) in tier_means.items():
        avg_line, = ax_box.plot([x0, x1], [mean_val, mean_val], color="black",
                                linestyle="dashed", linewidth=1.8, zorder=5)
        ax_box.text(x1 + 0.02, mean_val, f"{mean_val:.2f}", va="center",
                    ha="left", fontsize=fs - 1, fontweight="bold")
    ax_box.set_xticks(tier_centers)
    ax_box.set_xticklabels(["High-Fit", "Moderate-Fit"])
    ax_box.set_title("Evaluator Strictness: High-Fit vs Moderate-Fit CVs", fontsize=fs + 5, pad=10)
    ax_box.set_xlabel("", fontsize=fs + 4)
    ax_box.set_ylabel("Score", fontsize=fs + 4)
    ax_box.set_ylim(0, 10)
    ax_box.tick_params(axis="both", labelsize=fs)
    ax_box.grid(True, alpha=0.3, axis="y")
    for tick in ax_box.get_xticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(fs + 2)
    handles, labels = ax_box.get_legend_handles_labels()
    if avg_line is not None:
        handles.append(avg_line)
        labels.append("Average")
    leg = ax_box.legend(handles, labels, fontsize=fs - 1, framealpha=0.9, loc="upper right")
    for text in leg.get_texts():
        text.set_color(DISPLAY_COLORS.get(text.get_text(), "black"))
        text.set_fontweight("bold")
    _draw_heatmap(ax_heat, corr_pearson, "Pearson r")
    fig.savefig(os.path.join(save_dir, "strictness_agreement_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved strictness_agreement_boxplot.png")


def plot_strictness_by_condition(tier_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 5

    conditions = [
        ("cv_only",           "CV Only"),
        ("cl_evaluations",    "Cover Letter Only"),
        ("cv_cl_evaluations", "CV + Cover Letter"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    model_palette = {MODEL_DISPLAY[ev]: WRITER_COLORS[ev] for ev in UNIQUE_EVALUATORS}

    legend_handles, legend_labels = [], []

    for col, (etype, title) in enumerate(conditions):
        ax       = axes[col]
        sub      = tier_df[tier_df["Eval_Type"] == etype]
        is_first = col == 0

        if sub.empty:
            ax.set_visible(False)
            continue

        # --- Bar chart ---
        all_ranks = sorted(sub["CV_Idx"].unique())
        top25 = set(all_ranks[:25])
        bot25 = set(all_ranks[-25:])

        rows = []
        for ev in UNIQUE_EVALUATORS:
            ev_data = sub[sub["Evaluator"] == ev][["CV_Idx", "Score"]]
            for _, r in ev_data.iterrows():
                if r["CV_Idx"] in top25:
                    rows.append({"Model": MODEL_DISPLAY[ev], "Tier": "High-Fit", "Score": r["Score"]})
                elif r["CV_Idx"] in bot25:
                    rows.append({"Model": MODEL_DISPLAY[ev], "Tier": "Moderate-Fit", "Score": r["Score"]})
        box_df = pd.DataFrame(rows)

        model_order = [MODEL_DISPLAY[ev] for ev in UNIQUE_EVALUATORS if MODEL_DISPLAY[ev] in box_df["Model"].unique()]
        n_models    = len(model_order)
        bar_width   = 0.35 / n_models
        group_gap   = 0.2
        tier_centers = []
        tier_means   = {}

        for tier_idx, tier in enumerate(["High-Fit", "Moderate-Fit"]):
            x_center = tier_idx * (1 + group_gap)
            tier_centers.append(x_center)
            half_span = (n_models - 1) / 2 * (bar_width + 0.02) + bar_width / 2
            tier_means[tier] = (x_center - half_span, x_center + half_span,
                                box_df[box_df["Tier"] == tier]["Score"].mean())
            for model_idx, model in enumerate(model_order):
                subset = box_df[(box_df["Tier"] == tier) & (box_df["Model"] == model)]["Score"]
                if subset.empty:
                    continue
                x_pos = x_center + (model_idx - (n_models - 1) / 2) * (bar_width + 0.02)
                bar = ax.bar(x_pos, subset.mean(), width=bar_width,
                             color=model_palette[model], alpha=0.85,
                             yerr=subset.std(), error_kw={"capsize": 4, "linewidth": 1.5, "ecolor": "black"})
                if is_first and tier_idx == 0:
                    legend_handles.append(bar)
                    legend_labels.append(model)

        avg_line = None
        for tier, (x0, x1, mean_val) in tier_means.items():
            avg_line, = ax.plot([x0, x1], [mean_val, mean_val], color="black",
                                linestyle="dashed", linewidth=1.8, zorder=5)
            ax.text(x1 + 0.02, mean_val, f"{mean_val:.2f}", va="center",
                    ha="left", fontsize=fs - 1, fontweight="bold")

        if is_first and avg_line is not None:
            legend_handles.append(avg_line)
            legend_labels.append("Average")

        ax.set_xticks(tier_centers)
        ax.set_xticklabels(["High-Fit", "Moderate-Fit"])
        ax.set_title(title, fontsize=fs + 5, pad=10)
        ax.set_ylabel("Score" if is_first else "", fontsize=fs + 4)
        ax.set_ylim(0, 10)
        ax.tick_params(axis="both", labelsize=fs)
        ax.grid(True, alpha=0.3, axis="y")
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
            tick.set_fontsize(fs + 2)

    leg = axes[0].legend(legend_handles, legend_labels, fontsize=fs - 1,
                         framealpha=0.9, loc="upper right")
    for text in leg.get_texts():
        text.set_color(DISPLAY_COLORS.get(text.get_text(), "black"))
        text.set_fontweight("bold")

    fname = "strictness_by_condition.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def plot_intra_run_variance(tier_df, save_dir):
    """Score std across the 4 independent runs — measures evaluator reproducibility."""
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 5

    conditions = [
        ("cv_only",           "CV Only"),
        ("cl_evaluations",    "Cover Letter Only"),
        ("cv_cl_evaluations", "CV + Cover Letter"),
    ]
    cond_colors = ["#5C85C5", "#E07B4A", "#4CAF50"]

    def _run_std(sub, etype):
        grp = ["Job_ID", "CV_Idx", "Evaluator"] if etype == "cv_only" \
              else ["Job_ID", "CV_Idx", "Evaluator", "Writer"]
        return sub.groupby(grp)["Score"].std()

    # --- Figure 1: grouped bar chart per evaluator × condition ---
    n_ev   = len(UNIQUE_EVALUATORS)
    n_cond = len(conditions)
    bar_w  = 0.22
    x      = np.arange(n_ev)

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for ci, ((etype, label), color) in enumerate(zip(conditions, cond_colors)):
        sub  = tier_df[tier_df["Eval_Type"] == etype]
        stds = _run_std(sub, etype)
        means = [stds.xs(ev, level="Evaluator").mean() for ev in UNIQUE_EVALUATORS]
        offset = (ci - (n_cond - 1) / 2) * (bar_w + 0.03)
        bars = ax.bar(x + offset, means, width=bar_w, label=label,
                      color=color, alpha=0.85, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=fs - 3)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[ev] for ev in UNIQUE_EVALUATORS],
                       rotation=30, ha="right", fontsize=fs, fontweight="bold")
    for tick, ev in zip(ax.get_xticklabels(), UNIQUE_EVALUATORS):
        tick.set_color(DISPLAY_COLORS.get(MODEL_DISPLAY[ev], "black"))
    ax.set_ylabel("Mean score std across 4 runs", fontsize=fs + 2)
    ax.set_title("Scoring Reproducibility by Evaluator and Condition",
                 fontsize=fs + 5, pad=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.tick_params(axis="y", labelsize=fs)
    ax.grid(True, alpha=0.3, axis="y")
    leg = ax.legend(fontsize=fs, framealpha=0.9)
    for text in leg.get_texts():
        text.set_fontweight("bold")

    fig.savefig(os.path.join(save_dir, "intra_run_variance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved intra_run_variance.png")

    # --- Figure 2: heatmaps (evaluator × writer) for CL-only and CV+CL ---
    cl_conditions = [("cl_evaluations", "Cover Letter Only"),
                     ("cv_cl_evaluations", "CV + Cover Letter")]

    fig, axes = plt.subplots(1, 2, figsize=(24, 7), constrained_layout=True)
    all_vals = []
    matrices = {}
    for etype, _ in cl_conditions:
        sub  = tier_df[tier_df["Eval_Type"] == etype]
        stds = _run_std(sub, etype)
        mat  = pd.DataFrame(index=[MODEL_DISPLAY[ev] for ev in UNIQUE_EVALUATORS],
                            columns=[MODEL_DISPLAY.get(w, w) for w in RAW_WRITERS],
                            dtype=float)
        for ev in UNIQUE_EVALUATORS:
            for w in RAW_WRITERS:
                try:
                    mat.loc[MODEL_DISPLAY[ev], MODEL_DISPLAY.get(w, w)] = \
                        stds.xs((ev, w), level=("Evaluator", "Writer")).mean()
                except KeyError:
                    mat.loc[MODEL_DISPLAY[ev], MODEL_DISPLAY.get(w, w)] = np.nan
        matrices[etype] = mat
        all_vals.extend(mat.values.flatten())

    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    n_writers = len(RAW_WRITERS)
    n_evals   = len(UNIQUE_EVALUATORS)
    ylord_cmap = plt.cm.YlOrRd

    for ax, (etype, title) in zip(axes, cl_conditions):
        mat = matrices[etype]

        # append avg column to the matrix (masked from main colormap)
        mat_full = mat.copy()
        mat_full["Avg."] = mat.mean(axis=1)

        # mask the avg column so the main heatmap leaves it white
        avg_mask = np.zeros(mat_full.shape, dtype=bool)
        avg_mask[:, -1] = True

        annot_full = mat_full.round(3).astype(str).replace("nan", "")

        sns.heatmap(mat_full, annot=annot_full, fmt="", ax=ax,
                    cmap="YlOrRd", vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor="gray",
                    annot_kws={"size": fs - 1}, cbar=(ax is axes[-1]),
                    mask=avg_mask)

        # draw avg column cells with same YlOrRd colormap
        avg_vals_col = mat_full["Avg."].values
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for i, val in enumerate(avg_vals_col):
            color = ylord_cmap(norm(val))
            ax.add_patch(mpatches.Rectangle(
                (n_writers, i), 1, 1,
                facecolor=color, edgecolor="gray", linewidth=0.5, zorder=2))
            # pick text color by luminance, matching seaborn's logic
            rgb = np.array(color[:3])
            rgb_lin = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            lum = rgb_lin @ [0.2126, 0.7152, 0.0722]
            txt_color = ".15" if lum > 0.408 else "w"
            ax.text(n_writers + 0.5, i + 0.5, f"{val:.3f}",
                    ha="center", va="center", fontsize=fs - 1,
                    color=txt_color, zorder=3)

        # thick separator line between writer columns and avg column
        ax.axvline(n_writers, color="black", lw=2.5, zorder=4)

        ax.set_title(title, fontsize=fs + 5, pad=10)
        ax.set_xlabel("Writer Model", fontsize=fs + 2)
        ax.set_ylabel("Evaluator Model" if ax is axes[0] else "", fontsize=fs + 2)

        # rebuild xtick labels including "Avg"
        xtick_labels = [MODEL_DISPLAY.get(w, w) for w in RAW_WRITERS] + ["Avg."]
        ax.set_xticks(np.arange(len(xtick_labels)) + 0.5)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right",
                           fontsize=fs, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=fs, fontweight="bold")
        for tick in ax.get_xticklabels():
            if tick.get_text() == "Avg.":
                tick.set_color("black")
            else:
                tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))

    cbar = axes[-1].collections[0].colorbar
    cbar.set_label("Mean std across 4 runs", fontsize=fs + 2)
    cbar.ax.tick_params(labelsize=fs)

    fig.suptitle("Scoring Reproducibility by Evaluator and Writer Model",
                 fontsize=fs + 5)
    fig.savefig(os.path.join(save_dir, "intra_run_variance_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved intra_run_variance_heatmap.png")


# ==========================
# MAIN
# ==========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(BASE_DIR) and not os.path.exists(CACHE_PATH):
        print(f"ERROR: {BASE_DIR} does not exist.")
        return

    # Master dataframe (from parquet cache)
    print("\n=== Loading master dataframe ===")
    master_df = build_master_df(BASE_DIR)
    tier_df = pd.DataFrame()
    if not master_df.empty:
        start_cv, end_cv = TIER_RANGE
        tier_df = master_df[
            (master_df["CV_Idx"] >= start_cv) & (master_df["CV_Idx"] <= end_cv)
        ]

    # Net advantage (arena simulation)
    print("\n=== Loading data for net_advantage plot ===")
    comp_df = load_competitive_data(BASE_DIR)
    delta = p_mat = rounded_raw = writers = None
    if not comp_df.empty:
        writers = sorted(
            comp_df[comp_df["Type"] == "cv_cl_evaluations"]["Writer"].dropna().unique()
        )
        global_dist = {(b, t): [] for b in writers for t in writers}
        for evaluator in UNIQUE_EVALUATORS:
            print(f"  Simulating arena for {evaluator}...")
            for b in writers:
                for t in writers:
                    global_dist[(b, t)].extend(calculate_leapfrog(comp_df, evaluator, b, t))

        global_raw = np.zeros((len(writers), len(writers)))
        for i, b in enumerate(writers):
            for j, t in enumerate(writers):
                pcts = global_dist[(b, t)]
                global_raw[i, j] = np.mean(pcts) if pcts else np.nan

        rounded_raw = np.round(global_raw, 1)
        delta = np.zeros((len(writers), len(writers)))
        p_mat = np.full((len(writers), len(writers)), np.nan)
        for i, b in enumerate(writers):
            ctrl_pcts = global_dist[(b, b)]
            ctrl_val  = rounded_raw[i, i]
            for j, t in enumerate(writers):
                tgt_pcts = global_dist[(b, t)]
                if not tgt_pcts or not ctrl_pcts:
                    delta[i, j] = np.nan
                    continue
                delta[i, j] = rounded_raw[i, j] - ctrl_val
                if i == j:
                    p_mat[i, j] = 1.0
                elif np.array_equal(tgt_pcts, ctrl_pcts):
                    p_mat[i, j] = 1.0
                elif len(tgt_pcts) == len(ctrl_pcts):
                    _, p = stats.ttest_rel(tgt_pcts, ctrl_pcts)
                    p_mat[i, j] = p

    # Generate all plots
    print(f"\n=== Generating plots → {OUT_DIR} ===")

    if not tier_df.empty:
        plot_heatmap_gap(tier_df, OUT_DIR)
        plot_heatmap_gap_combined(tier_df, OUT_DIR)
        plot_win_matrix(tier_df, OUT_DIR)
        plot_evaluator_divergence(tier_df, OUT_DIR)
        plot_stacked_score(tier_df, OUT_DIR)
        plot_stacked_rank(tier_df, OUT_DIR)
        plot_stacked_rank_combined(tier_df, OUT_DIR)
        plot_agreement_corr(tier_df, OUT_DIR)
        plot_merit_agreement_combined(tier_df, OUT_DIR)
        plot_strictness_agreement_combined(tier_df, OUT_DIR)
        plot_strictness_by_condition(tier_df, OUT_DIR)
        plot_intra_run_variance(tier_df, OUT_DIR)
    else:
        print("  Skipping master-df plots (no data).")

    if comp_df is not None and not comp_df.empty:
        plot_net_advantage(delta, p_mat, rounded_raw, writers, OUT_DIR)
        if not tier_df.empty:
            plot_win_net_combined(tier_df, delta, p_mat, rounded_raw, writers, OUT_DIR)
    else:
        print("  Skipping net_advantage / win_net_combined (no competitive data).")

    print(f"\nDone. Plots → {OUT_DIR}/")


if __name__ == "__main__":
    main()
