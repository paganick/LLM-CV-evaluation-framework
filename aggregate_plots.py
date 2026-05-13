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
BASE_DIR  = "./output_eval"
OUT_DIR    = "./output_plots/paper_plots"
OUT_DIR_V2 = "./output_plots/paper_plots_v2"

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
# PLOTS 1 & 2: heatmap_gap
# ==========================

def plot_heatmap_gap(global_data, save_dir):
    start_cv, end_cv = TIER_RANGE

    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        data_rows, annot_rows = [], []

        for eval_name in UNIQUE_EVALUATORS:
            row_vals, row_annots = [], []
            for write_name in SORTED_WRITERS:
                pair = f"{eval_name}_{write_name}"
                gaps = []
                for cv_idx in range(start_cv, end_cv + 1):
                    pair_scores = global_data[etype].get(pair, {}).get(cv_idx, [])
                    if not pair_scores:
                        continue
                    avg_pair = np.mean(pair_scores)
                    all_scores = []
                    for w in RAW_WRITERS:
                        s = global_data[etype].get(f"{eval_name}_{w}", {}).get(cv_idx, [])
                        if s:
                            all_scores.extend(s)
                    if all_scores:
                        gaps.append(avg_pair - np.mean(all_scores))

                if gaps:
                    mean_gap = np.mean(gaps)
                    is_sig = False
                    if len(gaps) > 1:
                        if np.var(gaps) == 0:
                            is_sig = mean_gap != 0
                        else:
                            try:
                                _, p = stats.ttest_1samp(gaps, 0)
                                is_sig = p < 0.05
                            except Exception:
                                pass
                    row_vals.append(mean_gap)
                    row_annots.append(f"{mean_gap:.2f}" + ("\n(*)" if is_sig else ""))
                else:
                    row_vals.append(0.0)
                    row_annots.append("0.00")

            data_rows.append(row_vals)
            annot_rows.append(row_annots)

        df = pd.DataFrame(data_rows, index=UNIQUE_EVALUATORS, columns=SORTED_WRITERS)
        df_annot = pd.DataFrame(annot_rows, index=UNIQUE_EVALUATORS, columns=SORTED_WRITERS)

        col_means = df.mean(axis=0)
        eval_writers = [w for w in df.columns if w in UNIQUE_EVALUATORS]
        non_eval_writers = [w for w in df.columns if w not in UNIQUE_EVALUATORS]
        sorted_cols = (
            sorted(eval_writers, key=lambda w: col_means[w], reverse=True)
            + sorted(non_eval_writers, key=lambda w: col_means[w], reverse=True)
        )
        sorted_rows = [w for w in sorted_cols if w in UNIQUE_EVALUATORS]

        df = df.loc[sorted_rows, sorted_cols]
        df_annot = df_annot.loc[sorted_rows, sorted_cols]

        # Average row only (no Average column)
        avg_row = df.mean(axis=0)
        df.loc["Average"] = avg_row
        df_annot.loc["Average"] = [f"{df.loc['Average', c]:.2f}" for c in df.columns]

        df.to_csv(os.path.join(save_dir, f"heatmap_gap_{etype}.csv"))

        fs = plt.rcParams["font.size"] + 3
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            df, annot=df_annot, fmt="", cmap="RdBu_r", center=0,
            vmin=-0.6, vmax=0.6, linewidths=0.5, linecolor="gray",
            annot_kws={"size": fs},
        )
        ax.axhline(len(sorted_rows), color="black", linewidth=2)
        plt.title(
            f"Relative Preference Gap | {TITLE_MAP[etype]}\n(*) = p < 0.05",
            fontsize=fs + 4,
        )
        plt.xlabel("Writer Model", fontsize=fs)
        plt.ylabel("Evaluator Model", fontsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs)
        ax.collections[0].colorbar.ax.tick_params(labelsize=fs)
        ax.collections[0].colorbar.ax.yaxis.label.set_size(fs)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"heatmap_gap_{etype}.png"))
        plt.close()
        print(f"  Saved heatmap_gap_{etype}.png + .csv")


# ==========================
# DATA LOADING — Plot 3
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
                                "Eval_Type": etype, "Evaluator": evaluator,
                                "Writer": writer, "CV_Idx": cv_idx, "Score": score,
                            })
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE_PATH, index=False)
    print(f"  Cached master dataframe → {CACHE_PATH}")
    return df


# ==========================
# PLOT 3: win_matrix_UNBIASED
# ==========================

def plot_win_matrix(df, save_dir):
    etype = "cv_cl_evaluations"
    writers = sorted([w for w in df["Writer"].unique() if w != "CV_ONLY"])
    n = len(writers)

    subset = df[df["Eval_Type"] == etype]
    if subset.empty:
        print("  WARNING: no cv_cl_evaluations data, skipping win_matrix plot.")
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
    df_win["Avg"] = df_win.mean(axis=1, skipna=True)
    avg_row = df_win.mean(axis=0, skipna=True)
    avg_row.name = "Avg"
    df_win = pd.concat([df_win, avg_row.to_frame().T])
    df_win = df_win.sort_values("Avg", ascending=False)
    sorted_cols = [c for c in df_win.index if c != "Avg"]
    df_win = df_win[sorted_cols + ["Avg"]]

    df_win.to_csv(os.path.join(save_dir, f"win_matrix_UNBIASED_{etype}.csv"))

    mask = pd.DataFrame(False, index=df_win.index, columns=df_win.columns)
    for w in writers:
        if w in df_win.index and w in df_win.columns:
            mask.loc[w, w] = True

    fs = plt.rcParams["font.size"] + 3
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        df_win, annot=True, fmt=".0f", cmap="RdBu_r", vmin=0, vmax=100, center=50,
        cbar_kws={"label": "Win Rate %"}, linewidths=0.5, linecolor="gray",
        mask=mask, ax=ax, annot_kws={"size": fs},
    )
    for w in writers:
        if w in df_win.index and w in df_win.columns:
            ri = list(df_win.index).index(w)
            ci = list(df_win.columns).index(w)
            ax.add_patch(plt.Rectangle((ci, ri), 1, 1, fill=True, color="lightgrey", lw=0))

    plt.title(
        f"Head-to-Head Win Rate | {TITLE_MAP[etype]}",
        fontsize=fs + 4,
    )
    plt.xlabel("Opponent Writer Model", fontsize=fs)
    plt.ylabel("Row Writer Model", fontsize=fs)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs)
    ax.collections[0].colorbar.ax.tick_params(labelsize=fs)
    ax.collections[0].colorbar.ax.yaxis.label.set_size(fs)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"win_matrix_UNBIASED_{etype}.png"))
    plt.close()
    print(f"  Saved win_matrix_UNBIASED_{etype}.png + .csv")


# ==========================
# DATA LOADING — Plot 4
# ==========================

def load_competitive_data(base_dir):
    rows = []
    for job_folder in os.listdir(base_dir):
        job_path = os.path.join(base_dir, job_folder)
        if not os.path.isdir(job_path):
            continue
        for run_folder in os.listdir(job_path):
            m = re.search(r"run_(\d+)", run_folder)
            if not m:
                continue
            run_id = int(m.group(1))
            for etype in ["cv_only", "cv_cl_evaluations"]:
                eval_path = os.path.join(job_path, run_folder, etype)
                if not os.path.exists(eval_path):
                    continue
                for fname in os.listdir(eval_path):
                    if not fname.endswith(".txt"):
                        continue
                    evaluator, writer, cv_idx = parse_filename(fname, etype)
                    if evaluator and cv_idx:
                        score = extract_first_number(os.path.join(eval_path, fname))
                        if score is not None:
                            rows.append({
                                "Job": job_folder, "Run": run_id,
                                "Evaluator": evaluator, "Writer": writer,
                                "CV_Idx": cv_idx, "Score": score, "Type": etype,
                            })
    return pd.DataFrame(rows)


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


# ==========================
# PLOT 4: net_advantage_matrix
# ==========================

def plot_net_advantage(delta_matrix, p_matrix, raw_matrix, writers, save_dir):
    N = len(writers)
    ext_writers = list(writers) + ["AVERAGE"]

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

    mult_matrix = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            ctrl, tgt = raw_matrix[i, i], raw_matrix[i, j]
            if ctrl == 0 and tgt > 0:
                mult_matrix[i, j] = np.inf
            elif ctrl == 0:
                mult_matrix[i, j] = 1.0
            else:
                mult_matrix[i, j] = tgt / ctrl
    with np.errstate(invalid="ignore"):
        row_mult     = np.nanmean(mult_matrix, axis=1)
        col_mult     = np.nanmean(mult_matrix, axis=0)
        overall_mult = np.nanmean(mult_matrix)

    annot = []
    for i in range(N + 1):
        row = []
        for j in range(N + 1):
            val = ext_delta[i, j]
            if np.isnan(val):
                row.append("NaN")
                continue
            sign = "+" if val > 0 else ""
            if i == N and j == N:
                m = overall_mult
            elif i == N:
                m = col_mult[j]
            elif j == N:
                m = row_mult[i]
            else:
                m = mult_matrix[i, j]

            if i < N and j < N and i == j:
                ms = "(Control)"
            elif np.isinf(m):
                ms = "(Inf x)"
            elif np.isnan(m):
                ms = ""
            else:
                ms = f"({m:.1f}x)"

            if i == N or j == N:
                row.append(f"{sign}{val:.1f}%\n{ms}".strip())
            else:
                star = "(*)" if p_matrix[i, j] < 0.05 else ""
                row.append(f"{sign}{val:.1f}% {star}\n{ms}".strip())
        annot.append(row)
    annot = np.array(annot)

    df_m = pd.DataFrame(ext_delta, index=ext_writers, columns=ext_writers)
    sort_vals     = pd.Series(col_means, index=writers).sort_values(ascending=False)
    sorted_models = sort_vals.index.tolist()
    row_order = sorted_models + ["AVERAGE"]
    col_order = sorted_models + ["AVERAGE"]
    df_m = df_m.loc[row_order, col_order]

    old_idx = {w: i for i, w in enumerate(writers)}
    new_ri  = [old_idx[w] for w in sorted_models] + [N]
    new_ci  = [old_idx[w] for w in sorted_models] + [N]
    annot   = annot[np.ix_(new_ri, new_ci)]

    df_m.to_csv(os.path.join(save_dir, "net_advantage_matrix_ALL_COMBINED.csv"))

    fs = plt.rcParams["font.size"] + 3
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        df_m, annot=annot, fmt="", cmap="RdBu_r", center=0,
        cbar_kws={"label": "Net Advantage over Control Group (%)"},
        annot_kws={"size": fs - 2},
    )
    ax.axhline(N, color="black", linewidth=2)
    ax.axvline(N, color="black", linewidth=2)
    plt.title(
        "Net Competitive Advantage\n"
        "(Absolute Delta %) | (Relative Multiplier) | (*) = p < 0.05",
        fontsize=fs + 4,
    )
    plt.ylabel("Top-25 Candidates | Cover Letter Writer", fontsize=fs)
    plt.xlabel("Lower-25 Candidates | Cover Letter Writer", fontsize=fs)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs)
    ax.collections[0].colorbar.ax.tick_params(labelsize=fs)
    ax.collections[0].colorbar.ax.yaxis.label.set_size(fs)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "net_advantage_matrix_ALL_COMBINED.png"))
    plt.close()
    print("  Saved net_advantage_matrix_ALL_COMBINED.png + .csv")


# ==========================
# PLOT 5: evaluator divergence
# ==========================

def plot_evaluator_divergence(df, save_dir):
    """
    For each evaluator, compute mean absolute deviation from the group consensus
    (average score across all evaluators) over every (Job, CV, Writer) triple.
    Lower deviation = closer to the group average = most 'neutral' evaluator.
    """
    os.makedirs(save_dir, exist_ok=True)

    etypes = ["cl_evaluations", "cv_cl_evaluations"]
    records = []

    for etype in etypes:
        subset = df[df["Eval_Type"] == etype].copy()
        if subset.empty:
            continue

        # Consensus score = mean across all evaluators per (job, cv, writer)
        consensus = (
            subset.groupby(["Job_ID", "CV_Idx", "Writer"])["Score"]
            .mean()
            .reset_index()
            .rename(columns={"Score": "Consensus"})
        )
        subset = subset.merge(consensus, on=["Job_ID", "CV_Idx", "Writer"])
        subset["AbsDev"] = (subset["Score"] - subset["Consensus"]).abs()

        for evaluator, grp in subset.groupby("Evaluator"):
            records.append({
                "Evaluator": evaluator,
                "Eval_Type": TITLE_MAP[etype],
                "Mean_AbsDev": grp["AbsDev"].mean(),
            })

    results_df = pd.DataFrame(records)
    pivot = results_df.pivot(index="Evaluator", columns="Eval_Type", values="Mean_AbsDev")
    pivot["_sort"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_sort").drop(columns="_sort")

    pivot.to_csv(os.path.join(save_dir, "evaluator_divergence.csv"))

    fs = plt.rcParams["font.size"] + 3
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", ax=ax, color=["#4393C3", "#D6604D"], edgecolor="white")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(
        "Evaluator Divergence from Group Consensus\n(Lower = Closer to Average)",
        fontsize=fs + 4,
    )
    ax.set_xlabel("Mean Absolute Deviation from Ensemble Score", fontsize=fs)
    ax.set_ylabel("Evaluator Model", fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.legend(fontsize=fs, title="Eval Type", title_fontsize=fs)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "evaluator_divergence.png"))
    plt.close()
    print("  Saved evaluator_divergence.png + .csv")


# ==========================
# PLOTS 6 & 7: fairness stacked
# ==========================

def plot_stacked_score(df, save_dir):
    """
    For each (evaluator, writer): gap = writer_score minus evaluator's mean across all writers,
    averaged over all (job, CV) pairs. Stacked bars show signed per-writer contribution.
    """
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
                    writer_score   = group[group["Writer"] == writer]["Score"].mean()
                    baseline_score = group["Score"].mean()
                    gaps.append(writer_score - baseline_score)
                if gaps:
                    records.append({"Evaluator": evaluator, "Writer": writer, "Gap": np.mean(gaps)})

        if not records:
            continue

        gap_df    = pd.DataFrame(records)
        total_abs = gap_df.groupby("Evaluator")["Gap"].apply(lambda x: x.abs().sum())
        evaluator_order = total_abs.sort_values().index.tolist()

        fig, ax = plt.subplots(figsize=(12, 7))
        for x_pos, evaluator in enumerate(evaluator_order):
            eval_gaps  = gap_df[gap_df["Evaluator"] == evaluator].set_index("Writer")
            pos_bottom = neg_bottom = 0.0
            for writer in writers:
                if writer not in eval_gaps.index:
                    continue
                gap   = eval_gaps.loc[writer, "Gap"]
                color = WRITER_COLORS[writer]
                if gap >= 0:
                    ax.bar(x_pos, gap, bottom=pos_bottom, color=color, edgecolor="white", linewidth=0.5)
                    pos_bottom += gap
                else:
                    ax.bar(x_pos, gap, bottom=neg_bottom, color=color, hatch="///",
                           edgecolor="white", linewidth=0.5, alpha=0.85)
                    neg_bottom += gap
            ax.text(x_pos, pos_bottom + 0.02, f"{total_abs[evaluator]:.3f}",
                    ha="center", va="bottom", fontweight="bold", fontsize=fs - 1)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(evaluator_order)))
        ax.set_xticklabels(evaluator_order, rotation=45, ha="right", fontsize=fs)
        for tick, evaluator in zip(ax.get_xticklabels(), evaluator_order):
            tick.set_color(WRITER_COLORS.get(evaluator, "black"))
            tick.set_fontweight("bold")

        ax.set_xlabel("Evaluator Model", fontsize=fs)
        ax.set_ylabel("Relative Preference Gap (signed, summed across writers)", fontsize=fs)
        ax.set_title(
            f"Evaluator Bias Profile (Score-Based) | {TITLE_MAP[etype]}\n"
            "(Sorted by Total |Gap|; hatch = negative bias)",
            fontsize=fs + 4,
        )
        ax.tick_params(axis="y", labelsize=fs)

        legend_handles = [mpatches.Patch(color=WRITER_COLORS[w], label=w)
                          for w in RAW_WRITERS if w in writers]
        legend_handles.append(mpatches.Patch(facecolor="gray", hatch="///", edgecolor="gray",
                                             label="Negative bias", alpha=0.85))
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=fs - 1)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"fairness_stacked_{etype}.png"), dpi=150)
        plt.close()
        print(f"  Saved fairness_stacked_{etype}.png")


def plot_stacked_rank(df, save_dir):
    """
    For each (evaluator, job, CV): rank the writers by score (1=best).
    gap = (N+1)/2 - writer_rank. Positive = ranked in upper half.
    Stacked bars show signed per-writer rank-gap contribution.
    """
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
                writer_scores = group.groupby("Writer")["Score"].mean()
                writer_ranks  = writer_scores.rank(ascending=False, method="average")
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

        fig, ax = plt.subplots(figsize=(12, 7))
        for x_pos, evaluator in enumerate(evaluator_order):
            eval_gaps  = gap_df[gap_df["Evaluator"] == evaluator].set_index("Writer")
            pos_bottom = neg_bottom = 0.0
            for writer in writers:
                if writer not in eval_gaps.index:
                    continue
                gap   = eval_gaps.loc[writer, "Gap"]
                color = WRITER_COLORS[writer]
                if gap >= 0:
                    ax.bar(x_pos, gap, bottom=pos_bottom, color=color, edgecolor="white", linewidth=0.5)
                    pos_bottom += gap
                else:
                    ax.bar(x_pos, gap, bottom=neg_bottom, color=color, hatch="///",
                           edgecolor="white", linewidth=0.5, alpha=0.85)
                    neg_bottom += gap
            ax.text(x_pos, pos_bottom + 0.02, f"{total_abs[evaluator]:.2f}",
                    ha="center", va="bottom", fontweight="bold", fontsize=fs - 1)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(evaluator_order)))
        ax.set_xticklabels(evaluator_order, rotation=45, ha="right", fontsize=fs)
        for tick, evaluator in zip(ax.get_xticklabels(), evaluator_order):
            tick.set_color(WRITER_COLORS.get(evaluator, "black"))
            tick.set_fontweight("bold")

        ax.set_xlabel("Evaluator Model", fontsize=fs)
        ax.set_ylabel(
            f"Cumulative Rank Gap across Writers (scale 1–{n_writers})", fontsize=fs
        )
        ax.set_title(
            f"Evaluator Bias Profile (Rank-Based) | {TITLE_MAP[etype]}\n"
            f"(Sorted by Total |Gap|; hatch = under-ranked vs midpoint)",
            fontsize=fs + 4,
        )
        ax.tick_params(axis="y", labelsize=fs)

        legend_handles = [mpatches.Patch(color=WRITER_COLORS[w], label=w)
                          for w in RAW_WRITERS if w in writers]
        legend_handles.append(mpatches.Patch(facecolor="gray", hatch="///", edgecolor="gray",
                                             label="Under-ranked vs midpoint", alpha=0.85))
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=fs - 1)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"fairness_rank_stacked_{etype}.png"), dpi=150)
        plt.close()
        print(f"  Saved fairness_rank_stacked_{etype}.png")


# ==========================
# IMPROVED (V2) FUNCTIONS
# ==========================

def plot_heatmap_gap_v2(global_data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    start_cv, end_cv = TIER_RANGE
    etypes = ["cl_evaluations", "cv_cl_evaluations"]

    # --- First pass: build DataFrames for both etypes to determine shared scale ---
    built = {}
    for etype in etypes:
        data_rows, annot_rows = [], []
        for eval_name in UNIQUE_EVALUATORS:
            row_vals, row_annots = [], []
            for write_name in SORTED_WRITERS:
                pair = f"{eval_name}_{write_name}"
                gaps = []
                for cv_idx in range(start_cv, end_cv + 1):
                    pair_scores = global_data[etype].get(pair, {}).get(cv_idx, [])
                    if not pair_scores:
                        continue
                    avg_pair = np.mean(pair_scores)
                    all_scores = []
                    for w in RAW_WRITERS:
                        s = global_data[etype].get(f"{eval_name}_{w}", {}).get(cv_idx, [])
                        if s:
                            all_scores.extend(s)
                    if all_scores:
                        gaps.append(avg_pair - np.mean(all_scores))
                if gaps:
                    mean_gap = np.mean(gaps)
                    is_sig = False
                    if len(gaps) > 1:
                        if np.var(gaps) == 0:
                            is_sig = mean_gap != 0
                        else:
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
        eval_writers = [w for w in df.columns if w in UNIQUE_EVALUATORS]
        non_eval_writers = [w for w in df.columns if w not in UNIQUE_EVALUATORS]
        sorted_cols = (
            sorted(eval_writers, key=lambda w: col_means[w], reverse=True)
            + sorted(non_eval_writers, key=lambda w: col_means[w], reverse=True)
        )
        sorted_rows = [w for w in sorted_cols if w in UNIQUE_EVALUATORS]
        df = df.loc[sorted_rows, sorted_cols]
        df_annot = df_annot.loc[sorted_rows, sorted_cols]

        avg_row = df.mean(axis=0)
        df.loc["Average"] = avg_row
        df_annot.loc["Average"] = [f"{df.loc['Average', c]:.2f}" for c in df.columns]

        built[etype] = (df, df_annot, sorted_rows, sorted_cols)

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
        print(f"  [v2] Saved heatmap_gap_{etype}.png")


def plot_heatmap_gap_combined_v2(global_data, save_dir):
    """Both heatmaps side by side, each with its own ordering, shared colorbar."""
    os.makedirs(save_dir, exist_ok=True)
    start_cv, end_cv = TIER_RANGE
    etypes = ["cl_evaluations", "cv_cl_evaluations"]

    def _build(etype):
        data_rows, annot_rows = [], []
        for eval_name in UNIQUE_EVALUATORS:
            row_vals, row_annots = [], []
            for write_name in SORTED_WRITERS:
                pair = f"{eval_name}_{write_name}"
                gaps = []
                for cv_idx in range(start_cv, end_cv + 1):
                    pair_scores = global_data[etype].get(pair, {}).get(cv_idx, [])
                    if not pair_scores:
                        continue
                    avg_pair = np.mean(pair_scores)
                    all_sc = []
                    for w in RAW_WRITERS:
                        s = global_data[etype].get(f"{eval_name}_{w}", {}).get(cv_idx, [])
                        if s:
                            all_sc.extend(s)
                    if all_sc:
                        gaps.append(avg_pair - np.mean(all_sc))
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
        df       = pd.DataFrame(data_rows, index=UNIQUE_EVALUATORS, columns=SORTED_WRITERS)
        df_annot = pd.DataFrame(annot_rows, index=UNIQUE_EVALUATORS, columns=SORTED_WRITERS)
        # Sort columns by this etype's own average preference
        col_means = df.mean(axis=0)
        ew  = [w for w in df.columns if w in UNIQUE_EVALUATORS]
        new = [w for w in df.columns if w not in UNIQUE_EVALUATORS]
        sc  = (sorted(ew,  key=lambda w: col_means[w], reverse=True)
               + sorted(new, key=lambda w: col_means[w], reverse=True))
        sr  = [w for w in sc if w in UNIQUE_EVALUATORS]
        df       = df.loc[sr, sc]
        df_annot = df_annot.loc[sr, sc]
        df.loc["Average"]       = df.mean(axis=0)
        df_annot.loc["Average"] = [f"{df.loc['Average', c]:.2f}" for c in df.columns]
        return df, df_annot, sr, sc

    built = {et: _build(et) for et in etypes}

    # Shared symmetric scale
    max_abs = max(
        built[et][0].loc[built[et][2], :].abs().max().max()
        for et in etypes
    )
    max_abs = np.ceil(max_abs * 20) / 20
    shared_vmin, shared_vmax = -max_abs, max_abs

    fs = plt.rcParams["font.size"] + 3
    fig, axes = plt.subplots(1, 2, figsize=(24, 9), constrained_layout=True)

    for ax_idx, (ax, etype) in enumerate(zip(axes, etypes)):
        df, df_annot, sorted_rows, sorted_cols = built[etype]
        df_plot       = df.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
        df_annot_plot = df_annot.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
        sns.heatmap(
            df_plot, annot=df_annot_plot, fmt="", ax=ax,
            cmap="RdBu_r", center=0, vmin=shared_vmin, vmax=shared_vmax,
            linewidths=0.5, linecolor="gray",
            annot_kws={"size": fs - 1},
            cbar=False,
        )
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
        ax.set_xlabel("Writer Model", fontsize=fs + 4)
        ax.set_ylabel("Evaluator Model" if ax_idx == 0 else "", fontsize=fs + 4)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs, fontweight="bold")
        for tick in ax.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fs, fontweight="bold")
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))

    # Single shared colorbar to the right
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=plt.Normalize(vmin=shared_vmin, vmax=shared_vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("Preference Gap (score units, 0–10 scale)", fontsize=fs + 4)
    cbar.ax.tick_params(labelsize=fs)

    fig.suptitle(
        "Relative Preference Gap\n"
        "(*) = p < 0.05  |  Bold border = self-evaluation  |  "
        "Columns and rows sorted independently per condition by decreasing preference gap",
        fontsize=fs + 5,
    )
    plt.savefig(os.path.join(save_dir, "heatmap_gap_combined.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [v2] Saved heatmap_gap_combined.png")


def plot_win_matrix_v2(df, save_dir):
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
    df_win["Avg"] = df_win.mean(axis=1, skipna=True)
    # Sort model rows only, then append Avg at the bottom
    df_win = df_win.sort_values("Avg", ascending=False)
    sorted_models = list(df_win.index)
    df_win = df_win[sorted_models + ["Avg"]]
    avg_row = df_win.mean(axis=0, skipna=True)
    avg_row.name = "Avg"
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
    n_models = sum(1 for r in df_win.index if r != "Avg")

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
    print(f"  [v2] Saved win_matrix_UNBIASED_{etype}.png")


def plot_net_advantage_v2(delta_matrix, p_matrix, raw_matrix, writers, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    N = len(writers)
    ext_writers = list(writers) + ["AVERAGE"]

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
    df_m = df_m.loc[sorted_models + ["AVERAGE"], sorted_models + ["AVERAGE"]]
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
    plt.ylabel("Top-25 Candidates | Cover Letter Writer", fontsize=fs + 4)
    plt.xlabel("Lower-25 Candidates | Cover Letter Writer", fontsize=fs + 4)
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
    print("  [v2] Saved net_advantage_matrix_ALL_COMBINED.png")


def plot_win_net_combined_v2(tier_df, delta_matrix, p_matrix, raw_matrix, writers_net, save_dir):
    """Win-rate matrix (left) and net advantage matrix (right) side by side."""
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 6  # larger canvas (28×11) needs bigger base fs

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
    df_win["Avg"] = df_win.mean(axis=1, skipna=True)
    df_win = df_win.sort_values("Avg", ascending=False)
    sorted_win_models = list(df_win.index)
    df_win = df_win[sorted_win_models + ["Avg"]]
    avg_row_win = df_win.mean(axis=0, skipna=True)
    avg_row_win.name = "Avg"
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

    ext_writers = list(writers_net) + ["AVERAGE"]
    sort_vals     = pd.Series(col_means, index=writers_net).sort_values(ascending=False)
    sorted_net    = sort_vals.index.tolist()
    df_net        = pd.DataFrame(ext_delta, index=ext_writers, columns=ext_writers)
    df_net        = df_net.loc[sorted_net + ["AVERAGE"], sorted_net + ["AVERAGE"]]
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

    # Left: win matrix
    sns.heatmap(
        df_win, annot=annot_win, fmt="", ax=ax1,
        cmap="RdBu_r", vmin=0, vmax=100, center=50,
        linewidths=0.5, linecolor="gray",
        mask=mask_win, cbar=False,
        annot_kws={"size": fs - 1},
    )
    for ri, ci in diag_win:
        ax1.add_patch(plt.Rectangle((ci, ri), 1, 1, fill=True, color="lightgrey", lw=0))
        ax1.text(ci + 0.5, ri + 0.5, "—", ha="center", va="center", fontsize=fs, color="grey")
    ax1.axhline(n_win_models, color="black", linewidth=2)
    ax1.axvline(len(df_win.columns) - 1, color="black", linewidth=2)
    ax1.set_title("Head-to-Head Win Rate", fontsize=fs + 5, pad=10)
    ax1.set_xlabel("Opponent Writer Model", fontsize=fs + 4)
    ax1.set_ylabel("Writer Model", fontsize=fs + 4)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=fs, fontweight="bold")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=fs, fontweight="bold")
    for tick in ax1.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    for tick in ax1.get_yticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    sm1 = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 100))
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, ax=ax1, shrink=0.8)
    cb1.set_label("Win Rate (%)", fontsize=fs + 4)
    cb1.ax.tick_params(labelsize=fs)

    # Right: net advantage
    net_vmax = np.nanmax(np.abs(ext_delta[:-1, :-1]))  # ignore AVERAGE row/col for scale
    sns.heatmap(
        df_net, annot=annot_net, fmt="", ax=ax2,
        cmap="RdBu_r", center=0,
        linewidths=0.5, linecolor="gray",
        cbar=False,
        annot_kws={"size": fs - 1},
    )
    ax2.axhline(N, color="black", linewidth=2)
    ax2.axvline(N, color="black", linewidth=2)
    ax2.set_title("Net Competitive Advantage over Control", fontsize=fs + 5, pad=10)
    ax2.set_xlabel("Lower-25 Group — Writer Model", fontsize=fs + 4)
    ax2.set_ylabel("Top-25 Group — Writer Model", fontsize=fs + 4)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=fs, fontweight="bold")
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=fs, fontweight="bold")
    for tick in ax2.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    for tick in ax2.get_yticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    sm2 = plt.cm.ScalarMappable(
        cmap="RdBu_r",
        norm=plt.Normalize(vmin=-net_vmax, vmax=net_vmax),
    )
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax2, shrink=0.8)
    cb2.set_label("Net Advantage over Control (%)", fontsize=fs + 4)
    cb2.ax.tick_params(labelsize=fs)

    fig.suptitle(
        "CV + Cover Letter  |  Aggregated across all evaluator models\n"
        "(*) = p < 0.05  |  AVERAGE row/col: mean across all models  |  "
        "Control diagonal = same model for both groups",
        fontsize=fs + 5,
    )
    plt.savefig(os.path.join(save_dir, "win_net_combined.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [v2] Saved win_net_combined.png")


def plot_evaluator_divergence_v2(df, save_dir):
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
    print("  [v2] Saved evaluator_divergence.png")


def plot_stacked_score_v2(df, save_dir):
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
        print(f"  [v2] Saved fairness_stacked_{etype}.png")


def plot_stacked_rank_v2(df, save_dir):
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
            f"Cumulative Evaluator Bias (Rank-Based) | {TITLE_MAP[etype]}\n"
            "Sorted left → right: least to most biased  |  Solid = over-ranked, Hatched = under-ranked",
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
        print(f"  [v2] Saved fairness_rank_stacked_{etype}.png")


def plot_stacked_rank_combined_v2(df, save_dir):
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
        "Cumulative Evaluator Bias (Rank-Based)\n"
        "Sorted left → right: least to most biased  |  Solid = over-ranked, Hatched = under-ranked",
        fontsize=fs + 5,
    )
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(save_dir, "fairness_rank_combined.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [v2] Saved fairness_rank_combined.png")


# ==========================
# MAIN
# ==========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(BASE_DIR):
        print(f"ERROR: {BASE_DIR} does not exist.")
        return

    # --------------------------------------------------
    # Plots 1 & 2: heatmap_gap_cl / heatmap_gap_cv_cl
    # --------------------------------------------------
    print("=== Loading data for heatmap_gap plots ===")
    version_folders = []
    for folder_name in os.listdir(BASE_DIR):
        full_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.isdir(full_path):
            continue
        m = re.match(r"(job_\d+)", folder_name)
        if m:
            version_folders.append((m.group(1), folder_name, full_path))

    all_jobs_data = {}
    for job_id in sorted(set(v[0] for v in version_folders)):
        my_versions = [v for v in version_folders if v[0] == job_id]
        v_path = my_versions[0][2]
        runs = [d for d in os.listdir(v_path) if d.startswith("run_")]
        if not runs:
            continue
        cv_count = detect_cv_count(os.path.join(v_path, runs[0], "cl_evaluations"))
        if cv_count == 0:
            continue
        all_jobs_data[job_id] = collect_job_data(my_versions, cv_count)
        print(f"  Loaded {job_id} ({cv_count} CVs)")

    global_data = merge_jobs_data(all_jobs_data)
    print("Generating heatmap_gap plots...")
    plot_heatmap_gap(global_data, OUT_DIR)

    # --------------------------------------------------
    # Plot 3: win_matrix_UNBIASED_cv_cl_evaluations
    # --------------------------------------------------
    print("\n=== Loading master dataframe for win_matrix plot ===")
    master_df = build_master_df(BASE_DIR)
    if master_df.empty:
        print("  WARNING: master dataframe is empty, skipping win_matrix plot.")
    else:
        start_cv, end_cv = TIER_RANGE
        tier_df = master_df[
            (master_df["CV_Idx"] >= start_cv) & (master_df["CV_Idx"] <= end_cv)
        ]
        print("Generating win_matrix plot...")
        plot_win_matrix(tier_df, OUT_DIR)

        # --------------------------------------------------
        # Plot 5: evaluator_divergence
        # --------------------------------------------------
        print("\n=== Generating evaluator divergence plot ===")
        plot_evaluator_divergence(tier_df, OUT_DIR)

        # --------------------------------------------------
        # Plots 6 & 7: fairness stacked (score + rank)
        # --------------------------------------------------
        print("\n=== Generating fairness stacked plots ===")
        plot_stacked_score(tier_df, OUT_DIR)
        plot_stacked_rank(tier_df, OUT_DIR)

    # --------------------------------------------------
    # Plot 4: net_advantage_matrix_ALL_COMBINED
    # --------------------------------------------------
    print("\n=== Loading data for net_advantage_matrix plot ===")
    comp_df = load_competitive_data(BASE_DIR)
    if comp_df.empty:
        print("  WARNING: competitive data is empty, skipping net_advantage plot.")
    else:
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

        print("Generating net_advantage_matrix plot...")
        plot_net_advantage(delta, p_mat, rounded_raw, writers, OUT_DIR)

    # ==================================================
    # V2 IMPROVED PLOTS (saved to OUT_DIR_V2)
    # ==================================================
    print(f"\n=== Generating v2 (improved) plots → {OUT_DIR_V2} ===")

    # Heatmap v2 (uses global_data from earlier)
    plot_heatmap_gap_v2(global_data, OUT_DIR_V2)
    plot_heatmap_gap_combined_v2(global_data, OUT_DIR_V2)

    if not master_df.empty:
        plot_win_matrix_v2(tier_df, OUT_DIR_V2)
        plot_evaluator_divergence_v2(tier_df, OUT_DIR_V2)
        plot_stacked_score_v2(tier_df, OUT_DIR_V2)
        plot_stacked_rank_v2(tier_df, OUT_DIR_V2)
        plot_stacked_rank_combined_v2(tier_df, OUT_DIR_V2)
    else:
        print("  Skipping win_matrix/divergence/fairness v2 (no master data).")

    if not comp_df.empty:
        plot_net_advantage_v2(delta, p_mat, rounded_raw, writers, OUT_DIR_V2)
    else:
        print("  Skipping net_advantage v2 (no competitive data).")

    if not master_df.empty and not comp_df.empty:
        plot_win_net_combined_v2(tier_df, delta, p_mat, rounded_raw, writers, OUT_DIR_V2)
    else:
        print("  Skipping win_net_combined v2 (data missing).")

    print(f"\nDone. Originals → {OUT_DIR}/  |  Improved → {OUT_DIR_V2}/")


if __name__ == "__main__":
    main()
