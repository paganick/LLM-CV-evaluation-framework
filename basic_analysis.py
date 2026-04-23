import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "./output_eval"
OUT_PLOT_DIR = "./output_plots/basic_analysis"

# Tiers for analysis
TIERS = {
    "All_CVs": (1, 50),
    "Top_25": (1, 25),
    "Lower_25": (26, 50)
}

EVAL_TYPES = ["cl_evaluations", "cv_cl_evaluations", "cv_only_evaluations"]

# Presentation Mapping for Plot Titles
TITLE_MAP = {
    "cv_only_evaluations": "CV Only",
    "cl_evaluations": "CL Only",
    "cv_cl_evaluations": "CV + CL Combined"
}

# Full Matrix of Pairs
UNIQUE_EVALUATORS = [
    "gpt-4o-mini", "gpt-5-mini", "gemini-2.0-flash", 
    "gemini-3-flash-preview", "claude-haiku-4-5", "deepseek-chat"
]

RAW_WRITERS = [
    "gpt-4o-mini", "gpt-5-mini", "gemini-2.0-flash", 
    "gemini-3-flash-preview", "claude-haiku-4-5", 
    "deepseek-chat", "deepseek-r1-8b", "llama3.1-8b"
]

# This automatically creates all 48 combinations (evaluator_writer)
MODEL_PAIRS = [f"{evaluator}_{writer}" for evaluator in UNIQUE_EVALUATORS for writer in RAW_WRITERS]

# REORDER WRITERS: Put Evaluators first to form a diagonal
SORTED_WRITERS = [e for e in UNIQUE_EVALUATORS if e in RAW_WRITERS] + \
                 [w for w in RAW_WRITERS if w not in UNIQUE_EVALUATORS]

os.makedirs(OUT_PLOT_DIR, exist_ok=True)

# ==========================
# HELPERS
# ==========================

def detect_cv_count(path):
    max_cv = 0
    if not os.path.exists(path): return 0
    for root, _, files in os.walk(path):
        for f in files:
            match = re.search(r'cv(\d+)', f, re.IGNORECASE)
            if match:
                max_cv = max(max_cv, int(match.group(1)))
    return max_cv

def extract_scores(eval_folder):
    scores = {}
    if not os.path.exists(eval_folder): return scores
    
    for eval_file in os.listdir(eval_folder):
        if eval_file.endswith(".txt"):
            filepath = os.path.join(eval_folder, eval_file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Regex to find the absolute first number (with or without a decimal)
                    match = re.search(r'(\d+(?:\.\d+)?)', content)
                    
                    if match: 
                        scores[eval_file] = float(match.group(1))
                    else:
                        # If it physically cannot find a single digit, alert you!
                        print(f"⚠️ NO NUMBER FOUND IN: {eval_file}")
            except Exception as e:
                print(f"❌ ERROR reading file {eval_file}: {e}")
                
    return scores

def format_job_title(job_id, folder_name):
    """Extracts a clean title from folder name."""
    clean = folder_name.replace(job_id, "").strip("_")
    clean = re.sub(r'_\d+$', '', clean)
    clean = clean.replace("_", " ")
    return f"{clean} ({job_id.replace('job_', '')})"

# ==========================
# DATA COLLECTION
# ==========================

def collect_aggregated_job_data(version_folders, cv_count):
    """Collects data for a specific Job ID."""
    data = {etype: {pair: {i: [] for i in range(1, cv_count + 2)} 
            for pair in MODEL_PAIRS} for etype in EVAL_TYPES}

    for _, _, v_path in version_folders:
        if not os.path.exists(v_path): continue
        
        for run_folder in sorted(os.listdir(v_path)):
            run_path = os.path.join(v_path, run_folder)
            if not os.path.isdir(run_path): continue

            # 1. Process CL and CV+CL Evaluations
            for etype in ["cl_evaluations", "cv_cl_evaluations"]:
                etype_path = os.path.join(run_path, etype)
                run_scores = extract_scores(etype_path)
                for filename, score in run_scores.items():
                    for pair in MODEL_PAIRS:
                        if pair in filename:
                            match = re.search(r'cv(\d+)', filename, re.IGNORECASE)
                            if match:
                                idx = int(match.group(1))
                                if idx <= cv_count: data[etype][pair][idx].append(score)

            # 2. Process CV Only Evaluations
            cv_only_path = os.path.join(run_path, "cv_only")
            if os.path.exists(cv_only_path):
                run_scores = extract_scores(cv_only_path)
                for filename, score in run_scores.items():
                    evaluator = None
                    for eval_name in UNIQUE_EVALUATORS:
                        if filename.startswith(f"{eval_name}_"):
                            evaluator = eval_name
                            break
                    
                    if evaluator:
                        match = re.search(r'cv(\d+)', filename, re.IGNORECASE)
                        if match:
                            idx = int(match.group(1))
                            if idx <= cv_count:
                                for pair in MODEL_PAIRS:
                                    if pair.startswith(evaluator + "_"):
                                        data["cv_only_evaluations"][pair][idx].append(score)
    return data

def merge_all_jobs_data(all_jobs_data):
    """
    Merges data from ALL jobs into one massive data structure 
    to treat the whole experiment as one 'Global Job'.
    """
    # Initialize empty structure
    global_data = {etype: {pair: {} for pair in MODEL_PAIRS} for etype in EVAL_TYPES}
    
    # We don't know the exact max CV across all jobs, so we use a dict for indices
    for job_id, job_data in all_jobs_data.items():
        for etype in EVAL_TYPES:
            for pair in MODEL_PAIRS:
                for cv_idx, scores in job_data[etype][pair].items():
                    if cv_idx not in global_data[etype][pair]:
                        global_data[etype][pair][cv_idx] = []
                    global_data[etype][pair][cv_idx].extend(scores)
    return global_data

# ==========================
# PLOTTING FUNCTIONS
# ==========================

def plot_aggregated_heatmaps(job_data, job_id, job_title, tier_name, tier_range, save_dir):
    """
    Plot: Heatmaps showing RELATIVE BIAS (Gap from Evaluator Mean).
    Columns are sorted dynamically from highest average gap (red) to lowest (blue).
    """
    os.makedirs(save_dir, exist_ok=True)
    start_cv, end_cv = tier_range
    
    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        
        data_rows = []
        annot_rows = []

        for eval_name in UNIQUE_EVALUATORS:
            row_vals = []
            row_annots = []
            
            for write_name in SORTED_WRITERS:
                pair = f"{eval_name}_{write_name}"
                gaps = []
                
                for cv_idx in range(start_cv, end_cv + 1):
                    pair_scores = job_data[etype].get(pair, {}).get(cv_idx, [])
                    if not pair_scores: continue
                    avg_pair_score = np.mean(pair_scores)

                    all_writer_scores = []
                    for w in RAW_WRITERS:
                        p_check = f"{eval_name}_{w}"
                        s = job_data[etype].get(p_check, {}).get(cv_idx, [])
                        if s: all_writer_scores.extend(s)
                    
                    if all_writer_scores:
                        baseline = np.mean(all_writer_scores)
                        gaps.append(avg_pair_score - baseline)
                
                if gaps:
                    mean_gap = np.mean(gaps)
                    is_sig = False
                    if len(gaps) > 1:
                        if np.var(gaps) == 0:
                            if mean_gap != 0: is_sig = True
                        else:
                            try:
                                _, p_val = stats.ttest_1samp(gaps, 0)
                                if p_val < 0.05: is_sig = True
                            except: pass
                    
                    row_vals.append(mean_gap)
                    annot = f"{mean_gap:.2f}" + ("\n(*)" if is_sig else "")
                    row_annots.append(annot)
                else:
                    row_vals.append(0.0)
                    row_annots.append("0.00")

            data_rows.append(row_vals)
            annot_rows.append(row_annots)

        df = pd.DataFrame(data_rows, index=UNIQUE_EVALUATORS, columns=SORTED_WRITERS)
        df_annot = pd.DataFrame(annot_rows, index=UNIQUE_EVALUATORS, columns=SORTED_WRITERS)
        
        # ==========================================
        # SPLIT-SORT LOGIC (Preserves the Diagonal!)
        # ==========================================
        col_means = df.mean(axis=0)
        
        eval_writers = [w for w in df.columns if w in UNIQUE_EVALUATORS]
        non_eval_writers = [w for w in df.columns if w not in UNIQUE_EVALUATORS]
        
        sorted_eval_writers = sorted(eval_writers, key=lambda w: col_means[w], reverse=True)
        sorted_non_eval_writers = sorted(non_eval_writers, key=lambda w: col_means[w], reverse=True)
        
        sorted_cols = sorted_eval_writers + sorted_non_eval_writers
        sorted_rows = sorted_eval_writers 
        
        df = df.loc[sorted_rows, sorted_cols]
        df_annot = df_annot.loc[sorted_rows, sorted_cols]
        # ==========================================

        df['Average'] = df.mean(axis=1)
        avg_row = df.mean(axis=0)
        df.loc['Average'] = avg_row

        df_annot['Average'] = df['Average'].apply(lambda x: f"{x:.2f}")
        new_row_annot = []
        for col in df.columns:
            val = df.loc['Average', col]
            new_row_annot.append(f"{val:.2f}")
        df_annot.loc['Average'] = new_row_annot

        max_val = df.iloc[:-1, :-1].abs().max().max() 
        if pd.isna(max_val) or max_val == 0: max_val = 1

        plt.figure(figsize=(12, 8))
        
        ax = sns.heatmap(df, annot=df_annot, fmt="", cmap="RdBu_r", center=0,
                         vmin=-0.6, vmax=0.6, linewidths=0.5, linecolor='gray')
        
        ax.axhline(len(UNIQUE_EVALUATORS), color='black', linewidth=2)
        ax.axvline(len(SORTED_WRITERS), color='black', linewidth=2)
        
        # Updated Title
        clean_etype = TITLE_MAP.get(etype, etype)
        plt.title(f"Relative Preference Gap | {job_title}\n{clean_etype} | [{tier_name}]\n(*) = p < 0.05")
        
        plt.xlabel("Writer Model")
        plt.ylabel("Evaluator Model")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"heatmap_gap_{etype}.png"))
        plt.close()

def plot_global_job_bias_heatmap(all_jobs_data, target_etype, tier_name, tier_range, save_dir, job_titles_map):
    """
    GLOBAL Self-Preference Bias Heatmap (Jobs vs Evaluators).
    Sorted so the most biased models and jobs cluster in the top-left (red).
    """
    os.makedirs(save_dir, exist_ok=True)
    start_cv, end_cv = tier_range
    etype = target_etype 
    
    job_ids = sorted(list(all_jobs_data.keys()))
    job_labels = [job_titles_map.get(jid, jid) for jid in job_ids]

    data_gap_rows = []
    annot_rows = []

    for job in job_ids:
        row_gaps = []
        row_annots = []
        
        for evaluator in UNIQUE_EVALUATORS:
            cv_deltas = []
            for cv_idx in range(start_cv, end_cv + 1):
                pair_self = f"{evaluator}_{evaluator}"
                self_score_list = all_jobs_data[job][etype].get(pair_self, {}).get(cv_idx, [])
                if not self_score_list: continue
                score_self = np.mean(self_score_list)
                
                other_scores_for_this_cv = []
                for writer in RAW_WRITERS:
                    if writer != evaluator: 
                        pair_other = f"{evaluator}_{writer}"
                        s_list = all_jobs_data[job][etype].get(pair_other, {}).get(cv_idx, [])
                        if s_list: other_scores_for_this_cv.extend(s_list)
                
                if not other_scores_for_this_cv: continue
                score_other_avg = np.mean(other_scores_for_this_cv)
                cv_deltas.append(score_self - score_other_avg)
            
            if cv_deltas:
                avg_bias = np.mean(cv_deltas)
                is_sig = False
                if len(cv_deltas) > 1:
                    if np.var(cv_deltas) == 0:
                        if avg_bias != 0: is_sig = True
                    else:
                        try:
                            _, p_val = stats.ttest_1samp(cv_deltas, 0)
                            if p_val < 0.05: is_sig = True
                        except: pass
            else:
                avg_bias = 0
                is_sig = False
            
            row_gaps.append(avg_bias)
            row_annots.append(f"{avg_bias:.2f}" + ("\n(*)" if is_sig else ""))
        
        data_gap_rows.append(row_gaps)
        annot_rows.append(row_annots)

    df_gaps = pd.DataFrame(data_gap_rows, index=job_labels, columns=UNIQUE_EVALUATORS)
    df_annots = pd.DataFrame(annot_rows, index=job_labels, columns=UNIQUE_EVALUATORS)

    # ==========================================
    # NEW SORTING LOGIC (Red Top-Left, Blue Bottom-Right)
    # ==========================================
    col_means = df_gaps.mean(axis=0)
    sorted_evals = col_means.sort_values(ascending=False).index.tolist()
    
    row_means = df_gaps.mean(axis=1)
    sorted_jobs = row_means.sort_values(ascending=False).index.tolist()

    df_gaps = df_gaps.loc[sorted_jobs, sorted_evals]
    df_annots = df_annots.loc[sorted_jobs, sorted_evals]
    # ==========================================

    df_gaps['Average'] = df_gaps.mean(axis=1)
    avg_row = df_gaps.mean(axis=0)
    df_gaps.loc['Average'] = avg_row

    df_annots['Average'] = df_gaps['Average'].apply(lambda x: f"{x:.2f}")
    new_row_annot = [f"{val:.2f}" for val in df_gaps.loc['Average']]
    df_annots.loc['Average'] = new_row_annot

    max_val = df_gaps.iloc[:-1, :-1].abs().max().max()
    if pd.isna(max_val) or max_val == 0: max_val = 1
    
    fig_height = max(6, len(job_ids) * 0.8 + 2)

    plt.figure(figsize=(10, fig_height)) 
    
    ax = sns.heatmap(df_gaps, annot=df_annots, fmt="", cmap="RdBu_r", center=0,
                     linewidths=0.5, linecolor='gray', vmin=-0.75, vmax=0.75)

    ax.axhline(len(job_ids), color='black', linewidth=2)
    ax.axvline(len(UNIQUE_EVALUATORS), color='black', linewidth=2)

    # Updated Title
    clean_etype = TITLE_MAP.get(etype, etype)
    plt.title(f"Self-Preference Bias by Job | All Jobs Combined\n{clean_etype} | [{tier_name}]\n(*) = p < 0.05")
    
    plt.xlabel("Evaluator Model")
    plt.ylabel("Job")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"global_bias_heatmap_{etype}.png"))
    plt.close()

# ==========================
# MAIN EXECUTION
# ==========================

def main():
    # 1. Scan BASE_DIR
    print(f"📂 Scanning {BASE_DIR}...")
    version_folders = [] 
    JOB_TITLE_MAP = {}

    if os.path.exists(BASE_DIR):
        for folder_name in os.listdir(BASE_DIR):
            full_path = os.path.join(BASE_DIR, folder_name)
            if not os.path.isdir(full_path): continue
            match = re.match(r'(job_\d+)', folder_name)
            if match:
                job_id = match.group(1)
                version_folders.append((job_id, folder_name, full_path))
                if job_id not in JOB_TITLE_MAP:
                    JOB_TITLE_MAP[job_id] = format_job_title(job_id, folder_name)
    else:
        print(f"❌ Error: {BASE_DIR} does not exist.")
        exit()

    unique_jobs = sorted(list(set([v[0] for v in version_folders])))
    if not unique_jobs:
        print("❌ No job folders found.")
        exit()

    # MASTER DICTIONARY
    ALL_JOBS_DATA = {}

    # -------------------------------
    # PART A: Per-Job Data Collection & Analysis
    # -------------------------------
    for job_id in unique_jobs:
        current_job_title = JOB_TITLE_MAP.get(job_id, job_id)
        print(f"\n📊 Processing Job: {job_id} ({current_job_title})")
        
        my_versions = [v for v in version_folders if v[0] == job_id]
        
        # Detect CV count
        max_detected_cvs = 0
        if my_versions:
            v_path = my_versions[0][2]
            if os.path.exists(v_path):
                runs = [d for d in os.listdir(v_path) if d.startswith("run_")]
                if runs:
                    check_path = os.path.join(v_path, runs[0], "cl_evaluations")
                    max_detected_cvs = detect_cv_count(check_path)
        
        print(f"  -> Detected {max_detected_cvs} CVs total.")
        if max_detected_cvs == 0: continue

        # Collect Data
        aggregated_data = collect_aggregated_job_data(my_versions, max_detected_cvs)
        ALL_JOBS_DATA[job_id] = aggregated_data
        
        # Generate Plots for each Tier
        job_out_dir = os.path.join(OUT_PLOT_DIR, job_id)
        
        
        for tier_name, (start_cv, end_cv) in TIERS.items():
            print(f"  👉 Generating Plots for Tier: {tier_name}")
            tier_out_dir = os.path.join(job_out_dir, tier_name)
            
            plot_aggregated_heatmaps(aggregated_data, job_id, current_job_title, tier_name, (start_cv, end_cv), os.path.join(tier_out_dir, "heatmaps"))
            
    # -------------------------------
    # PART B: Global Bias Heatmap (Job Comparisons)
    # -------------------------------
    print("\n🌍 Generating Cross-Job Bias Summaries...")
    global_bias_out_dir = os.path.join(OUT_PLOT_DIR, "summary_global_bias/heatmaps")
    
    global_types = ["cv_cl_evaluations", "cl_evaluations"]

    for tier_name, (start_cv, end_cv) in TIERS.items():
        print(f"  👉 Processing Tier: {tier_name}")
        for etype in global_types:
            plot_global_job_bias_heatmap(
                ALL_JOBS_DATA, 
                etype, 
                tier_name, 
                (start_cv, end_cv), 
                os.path.join(global_bias_out_dir, tier_name),
                JOB_TITLE_MAP
            )

    # -------------------------------
    # PART C: GLOBAL AGGREGATE (Treating all data as one job)
    # -------------------------------
    print("\n🌐 Generating SUPER-GLOBAL Aggregate Plots (All Jobs Combined)...")
    
    # Merge all job data into one giant dictionary
    GLOBAL_DATA = merge_all_jobs_data(ALL_JOBS_DATA)
    global_agg_dir = os.path.join(OUT_PLOT_DIR, "summary_global_bias")
    
    for tier_name, (start_cv, end_cv) in TIERS.items():
        print(f"  👉 Global Aggregate Tier: {tier_name}")
        tier_out_dir = os.path.join(global_agg_dir, tier_name)
        
        plot_aggregated_heatmaps(GLOBAL_DATA, "GLOBAL", "All Jobs Combined", tier_name, (start_cv, end_cv), os.path.join(tier_out_dir, "heatmaps"))
        
    print("\n✨ Basic Analysis Complete.")

if __name__ == "__main__":
    main()