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
OUT_PLOT_DIR = "./output_plots/competitive_override_matrix"

UNIQUE_EVALUATORS = [
    "gpt-4o-mini", "gpt-5-mini", "gemini-2.0-flash", 
    "gemini-3-flash-preview", "claude-haiku-4-5", "deepseek-chat"
]

RAW_WRITERS = [
    "gpt-4o-mini", "gpt-5-mini", "gemini-2.0-flash", 
    "gemini-3-flash-preview", "claude-haiku-4-5", 
    "deepseek-chat", "deepseek-r1-8b", "llama3.1-8b"
]

sns.set_theme(style="whitegrid")
os.makedirs(OUT_PLOT_DIR, exist_ok=True)

# ==========================
# DATA LOADING (Standardized)
# ==========================
def parse_filename(filename, etype):
    evaluator = None
    for e in sorted(UNIQUE_EVALUATORS, key=len, reverse=True):
        if filename.startswith(e + "_"):
            evaluator = e
            break
    if not evaluator: return None, None, None

    if etype == "cv_only":
        writer = "CV_ONLY"
    else:
        remainder = filename[len(evaluator)+1:] 
        writer = None
        for w in sorted(RAW_WRITERS, key=len, reverse=True):
            if remainder.startswith(w + "_") or remainder.startswith(w + "."):
                writer = w
                break
            elif remainder == w: 
                writer = w
                break
    
    match_cv = re.search(r'cv(\d+)', filename, re.IGNORECASE)
    cv_idx = int(match_cv.group(1)) if match_cv else None

    return evaluator, writer, cv_idx

def extract_score(filepath):
    """Ultra-simple extraction: grabs the very first number in the file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Regex to find the first sequence of numbers (with or without a decimal)
            match = re.search(r'(\d+(?:\.\d+)?)', content)
            
            if match:
                return float(match.group(1))
            else:
                # If it physically cannot find a single digit, it yells at you
                print(f"⚠️ NO NUMBER FOUND IN: {os.path.basename(filepath)}")
                return None
                
    except Exception as e:
        print(f"❌ ERROR reading file {filepath}: {e}")
        return None

def load_data(base_dir):
    data = []
    if not os.path.exists(base_dir): return pd.DataFrame()

    for job_folder in os.listdir(base_dir):
        job_path = os.path.join(base_dir, job_folder)
        if not os.path.isdir(job_path): continue

        for run_folder in os.listdir(job_path):
            match_run = re.search(r'run_(\d+)', run_folder)
            if not match_run: continue
            run_id = int(match_run.group(1))
            
            for etype in ["cv_only", "cv_cl_evaluations"]:
                eval_path = os.path.join(job_path, run_folder, etype)
                if not os.path.exists(eval_path): continue
                
                for fname in os.listdir(eval_path):
                    if not fname.endswith(".txt"): continue
                    evaluator, writer, cv_idx = parse_filename(fname, etype)
                    
                    if evaluator and cv_idx:
                        score = extract_score(os.path.join(eval_path, fname))
                        if score is not None:
                            data.append({
                                "Job": job_folder,
                                "Run": run_id,
                                "Evaluator": evaluator,
                                "Writer": writer,
                                "CV_Idx": cv_idx,
                                "Score": score,
                                "Type": etype
                            })
    return pd.DataFrame(data)

# ==========================
# COMPETITIVE SIMULATION
# ==========================
def calculate_competitive_leapfrog(df, evaluator_name, baseline_writer, target_writer):
    """
    Returns the LIST of percentages so we can perform statistical T-Tests,
    rather than just returning the mean.
    """
    eval_df = df[df['Evaluator'] == evaluator_name].copy()
    if eval_df.empty: return []
    
    leapfrog_pcts = []

    for job_id in eval_df['Job'].unique():
        for run_id in eval_df['Run'].unique():
            env_df = eval_df[(eval_df['Job'] == job_id) & (eval_df['Run'] == run_id)].copy()
            if env_df.empty: continue
            
            cv_only = env_df[env_df['Type'] == 'cv_only'].copy()
            if len(cv_only) != 50: continue 
            
            cv_only = cv_only.sort_values(by=['Score', 'CV_Idx'], ascending=[False, True])
            cv_only['Baseline_Rank'] = range(1, 51)
            
            incumbent_cvs = cv_only[cv_only['Baseline_Rank'] <= 25]['CV_Idx'].tolist()
            challenger_cvs = cv_only[cv_only['Baseline_Rank'] > 25]['CV_Idx'].tolist()
            
            cv_cl = env_df[env_df['Type'] == 'cv_cl_evaluations'].copy()
            
            incumbents_cl = cv_cl[(cv_cl['Writer'] == baseline_writer) & (cv_cl['CV_Idx'].isin(incumbent_cvs))].copy()
            challengers_cl = cv_cl[(cv_cl['Writer'] == target_writer) & (cv_cl['CV_Idx'].isin(challenger_cvs))].copy()
            challengers_cl['Type_Tag'] = 'Challenger'
            
            if len(incumbents_cl) == 0 or len(challengers_cl) == 0: continue
                
            pool = pd.concat([incumbents_cl, challengers_cl])
            pool = pool.sort_values(by=['Score', 'CV_Idx'], ascending=[False, True])
            pool['New_Rank'] = range(1, len(pool) + 1)
            
            successful_challengers = pool[(pool['Type_Tag'] == 'Challenger') & (pool['New_Rank'] <= 25)]
            
            pct = (len(successful_challengers) / len(challengers_cl)) * 100
            leapfrog_pcts.append(pct)

    return leapfrog_pcts

# ==========================
# PLOTTING FUNCTIONS
# ==========================
def plot_raw_advantage_heatmap(raw_matrix, evaluator, writers, save_dir):
    """Plots the absolute raw percentages without any delta modifications."""
    df_matrix = pd.DataFrame(raw_matrix, index=writers, columns=writers)

    # Add average row and column, then sort by avg (row means) descending
    df_matrix['Avg'] = df_matrix.mean(axis=1, skipna=True)
    avg_row = df_matrix.mean(axis=0, skipna=True)
    avg_row.name = 'Avg'
    df_matrix = pd.concat([df_matrix, avg_row.to_frame().T])

    sorted_models = df_matrix.loc[df_matrix.index != 'Avg', 'Avg'].sort_values(ascending=True).index.tolist()
    row_order = sorted_models + ['Avg']
    col_order = sorted_models + ['Avg']
    df_matrix = df_matrix.loc[row_order, col_order]

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(11, 9))

    annot_matrix = np.array([
        [f"{val:.1f}%" if not np.isnan(val) else "NaN" for val in row]
        for row in df_matrix.values
    ])

    ax = sns.heatmap(df_matrix, annot=annot_matrix, fmt="", cmap="Reds",
                cbar_kws={'label': '% of CV-Only Rejects Stealing an Interview'})

    N = len(sorted_models)
    ax.axhline(N, color='black', linewidth=2)
    ax.axvline(N, color='black', linewidth=2)

    plt.title(f"Raw Competitive Override Matrix | Evaluator {evaluator.upper()}\n(% of Bottom-25 CVs displacing Top-25 CVs)")
    plt.ylabel("Top-25 CVs Model (Incumbents)")
    plt.xlabel("Lower-25 CVs Model (Challengers)")
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"raw_advantage_matrix_{evaluator}.png"))
    plt.close()

def plot_net_advantage_heatmap(delta_matrix, p_matrix, raw_matrix, evaluator, writers, save_dir):
    """Plots the delta advantage, relative multipliers, and an AVERAGE row/column for both."""
    N = len(writers)
    ext_writers = list(writers) + ["AVERAGE"]
    
    # ==========================================
    # 1. CALCULATE PERCENTAGE AVERAGES
    # ==========================================
    masked_delta = delta_matrix.copy()
    np.fill_diagonal(masked_delta, np.nan) # Ignore the 0.0 control cells
    
    row_means = np.nanmean(masked_delta, axis=1) # Avg vulnerability of incumbents
    col_means = np.nanmean(masked_delta, axis=0) # Avg offensive power of challengers
    overall_mean = np.nanmean(masked_delta)      # Grand average
    
    ext_delta = np.zeros((N + 1, N + 1))
    ext_delta[:N, :N] = delta_matrix
    ext_delta[:N, N] = row_means
    ext_delta[N, :N] = col_means
    ext_delta[N, N] = overall_mean

    # ==========================================
    # 2. CALCULATE MULTIPLIER AVERAGES
    # ==========================================
    mult_matrix = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(N):
            if i == j: continue # Skip diagonal for averaging
            
            control_raw = raw_matrix[i, i]
            target_raw = raw_matrix[i, j]
            
            if control_raw == 0 and target_raw > 0:
                mult_matrix[i, j] = np.inf
            elif control_raw == 0 and target_raw == 0:
                mult_matrix[i, j] = 1.0
            else:
                mult_matrix[i, j] = target_raw / control_raw
                
    # Averages of the multipliers (ignoring NaNs)
    with np.errstate(invalid='ignore'): # Suppress warnings if a row is all NaNs
        row_mult_means = np.nanmean(mult_matrix, axis=1) 
        col_mult_means = np.nanmean(mult_matrix, axis=0) 
        overall_mult_mean = np.nanmean(mult_matrix)

    # ==========================================
    # 3. BUILD ANNOTATION TEXT
    # ==========================================
    annot_matrix = []
    for i in range(N + 1):
        row = []
        for j in range(N + 1):
            val = ext_delta[i, j]
            
            if np.isnan(val):
                row.append("NaN")
                continue
                
            sign = "+" if val > 0 else ""
            
            # Figure out which multiplier to show
            if i == N and j == N:
                avg_mult = overall_mult_mean
            elif i == N:
                avg_mult = col_mult_means[j]
            elif j == N:
                avg_mult = row_mult_means[i]
            else:
                avg_mult = mult_matrix[i, j]
                
            # Format the Multiplier String
            if i < N and j < N and i == j:
                mult_str = "(Control)"
            elif np.isinf(avg_mult):
                mult_str = "(Inf x)"
            elif np.isnan(avg_mult):
                mult_str = ""
            else:
                mult_str = f"({avg_mult:.1f}x)"
            
            # Combine Percentage and Multiplier
            if i == N or j == N:
                # Average Cells (No p-values)
                cell_text = f"{sign}{val:.1f}%\n{mult_str}".strip()
            else:
                # Standard Cells (Include p-value stars)
                p_val = p_matrix[i, j]
                star = "(*)" if p_val < 0.05 else ""
                cell_text = f"{sign}{val:.1f}% {star}\n{mult_str}".strip()
                
            row.append(cell_text)
            
        annot_matrix.append(row)

    annot_matrix = np.array(annot_matrix)
    df_matrix = pd.DataFrame(ext_delta, index=ext_writers, columns=ext_writers)

    # Sort rows and columns by average net advantage (col_means = avg challenger power), AVERAGE last
    sort_vals = pd.Series(col_means, index=writers).sort_values(ascending=False)
    sorted_models = sort_vals.index.tolist()
    row_order = sorted_models + ['AVERAGE']
    col_order = sorted_models + ['AVERAGE']
    df_matrix = df_matrix.loc[row_order, col_order]

    # Reorder annot_matrix to match
    model_to_old_idx = {w: i for i, w in enumerate(writers)}
    new_row_order_idx = [model_to_old_idx[w] for w in sorted_models] + [N]
    new_col_order_idx = [model_to_old_idx[w] for w in sorted_models] + [N]
    annot_matrix = annot_matrix[np.ix_(new_row_order_idx, new_col_order_idx)]

    # ==========================================
    # 4. DRAW FIGURE
    # ==========================================
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 10)) 
    
    ax = sns.heatmap(df_matrix, annot=annot_matrix, fmt="", cmap="vlag", center=0, vmin=-5, vmax=5,
                     cbar_kws={'label': 'Net Advantage over Control Group (%)'})
    
    # Draw thick separator lines for the AVERAGE row and column
    ax.axhline(N, color='black', linewidth=2) 
    ax.axvline(N, color='black', linewidth=2)
    
    plt.title(f"Net Competitive Advantage | Evaluator {evaluator.upper()}\n(Absolute Delta %) | (Relative Multiplier) | (*) = p < 0.05")
    plt.ylabel("Top-25 CVs Model (Control Baseline)")
    plt.xlabel("Lower-25 CVs Model (Challenger Target)")
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f"net_advantage_matrix_{evaluator}.png"))
    plt.close()

# ==========================
# MAIN PIPELINE
# ==========================
def main():
    print("🔹 Loading Data...")
    df = load_data(BASE_DIR)
    if df.empty:
        print("❌ No data found.")
        return

    writers = sorted(df[df['Type'] == 'cv_cl_evaluations']['Writer'].dropna().unique())
    
    print(f"\n🌍 Generating Competitive Override Matrices (Raw & Net)...")
    
    # 1. Initialize a global bucket to hold data across ALL evaluators
    global_distributions = { (b, t): [] for b in writers for t in writers }
    
    for evaluator in UNIQUE_EVALUATORS:
        print(f"   -> Simulating arena for Evaluator: {evaluator}")
        
        # Gather all the lists of simulation results for this specific evaluator
        distributions = {}
        for baseline in writers:
            for target in writers:
                pcts = calculate_competitive_leapfrog(df, evaluator, baseline, target)
                distributions[(baseline, target)] = pcts
                
                # Add these percentages to the massive global bucket
                global_distributions[(baseline, target)].extend(pcts)
                
        # Calculate and Plot the INDIVIDUAL Raw Matrix
        raw_matrix = np.zeros((len(writers), len(writers)))
        for i, baseline in enumerate(writers):
            for j, target in enumerate(writers):
                target_pcts = distributions[(baseline, target)]
                raw_matrix[i, j] = np.mean(target_pcts) if target_pcts else np.nan
        
        plot_raw_advantage_heatmap(raw_matrix, evaluator, writers, OUT_PLOT_DIR)
        
        # Apply the ROUNDING FIX before calculating the Delta
        rounded_raw = np.round(raw_matrix, 1)

        # Calculate Delta and P-Values for INDIVIDUAL Evaluator
        delta_matrix = np.zeros((len(writers), len(writers)))
        p_matrix = np.full((len(writers), len(writers)), np.nan)
        
        for i, baseline in enumerate(writers):
            control_pcts = distributions[(baseline, baseline)]
            control_val = rounded_raw[i, i] 
            
            for j, target in enumerate(writers):
                target_pcts = distributions[(baseline, target)]
                
                if not target_pcts or not control_pcts:
                    delta_matrix[i, j] = np.nan
                    continue
                    
                delta_matrix[i, j] = rounded_raw[i, j] - control_val
                
                if i == j: 
                    p_matrix[i, j] = 1.0
                elif np.array_equal(target_pcts, control_pcts):
                    p_matrix[i, j] = 1.0
                elif len(target_pcts) == len(control_pcts): # Safe paired T-Test
                    stat, p_val = stats.ttest_rel(target_pcts, control_pcts)
                    p_matrix[i, j] = p_val
                
        plot_net_advantage_heatmap(delta_matrix, p_matrix, rounded_raw, evaluator, writers, OUT_PLOT_DIR)

    # ========================================================
    # 2. GENERATE THE GLOBAL AGGREGATE MATRIX (ALL EVALUATORS)
    # ========================================================
    print(f"\n🌎 Simulating GLOBAL Arena (All Evaluators Combined)...")
    
    global_raw_matrix = np.zeros((len(writers), len(writers)))
    for i, baseline in enumerate(writers):
        for j, target in enumerate(writers):
            target_pcts = global_distributions[(baseline, target)]
            global_raw_matrix[i, j] = np.mean(target_pcts) if target_pcts else np.nan
            
    plot_raw_advantage_heatmap(global_raw_matrix, "ALL_COMBINED", writers, OUT_PLOT_DIR)
    
    global_rounded_raw = np.round(global_raw_matrix, 1)
    global_delta_matrix = np.zeros((len(writers), len(writers)))
    global_p_matrix = np.full((len(writers), len(writers)), np.nan)
    
    for i, baseline in enumerate(writers):
        control_pcts = global_distributions[(baseline, baseline)]
        control_val = global_rounded_raw[i, i]
        
        for j, target in enumerate(writers):
            target_pcts = global_distributions[(baseline, target)]
            
            if not target_pcts or not control_pcts:
                global_delta_matrix[i, j] = np.nan
                continue
                
            global_delta_matrix[i, j] = global_rounded_raw[i, j] - control_val
            
            if i == j: 
                global_p_matrix[i, j] = 1.0
            elif np.array_equal(target_pcts, control_pcts):
                global_p_matrix[i, j] = 1.0
            elif len(target_pcts) == len(control_pcts):
                stat, p_val = stats.ttest_rel(target_pcts, control_pcts)
                global_p_matrix[i, j] = p_val
                
    plot_net_advantage_heatmap(global_delta_matrix, global_p_matrix, global_rounded_raw, "ALL_COMBINED", writers, OUT_PLOT_DIR)

    print(f"\n✅ Competitive Analysis Complete. Check {OUT_PLOT_DIR}")
if __name__ == "__main__":
    main()