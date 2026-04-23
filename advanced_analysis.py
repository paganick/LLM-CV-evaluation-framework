import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "./output_eval"
OUT_PLOT_DIR = "./output_plots/advanced_analysis"

# Tiers for analysis
TIERS = {
    "All_CVs": (1, 50),
    "Top_25": (1, 25),
    "Lower_25": (26, 50)
}

# Model Definitions (Updated to 2026 Naming Convention)
UNIQUE_EVALUATORS = [
    "gpt-4o-mini", "gpt-5-mini", "gemini-2.0-flash", 
    "gemini-3-flash-preview", "claude-haiku-4-5", "deepseek-chat"
]

RAW_WRITERS = [
    "gpt-4o-mini", "gpt-5-mini", "gemini-2.0-flash", 
    "gemini-3-flash-preview", "claude-haiku-4-5", 
    "deepseek-chat", "deepseek-r1-8b", "llama3.1-8b"
]

# We now include cv_only in our extraction pipeline
EVAL_TYPES = ["cv_only", "cl_evaluations", "cv_cl_evaluations"]

# Presentation Mapping for Plot Titles (Makes them look clean!)
TITLE_MAP = {
    "cv_only": "CV Only",
    "cl_evaluations": "CL Only",
    "cv_cl_evaluations": "CV + CL Combined"
}

# Visual Settings
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

os.makedirs(OUT_PLOT_DIR, exist_ok=True)

# ==========================
# DATA INGESTION (DataFrame Builder)
# ==========================

def format_job_title(job_id, folder_name):
    """Clean job title extraction."""
    clean = folder_name.replace(job_id, "").strip("_")
    clean = re.sub(r'_\d+$', '', clean)
    clean = clean.replace("_", " ")
    return f"{clean} ({job_id.replace('job_', '')})"

def parse_filename(filename, etype):
    """Strictly extracts Evaluator and Writer, handling CV_Only gracefully."""
    evaluator = None
    sorted_evals = sorted(UNIQUE_EVALUATORS, key=len, reverse=True)
    for e in sorted_evals:
        if filename.startswith(e + "_"):
            evaluator = e
            break
    
    if not evaluator: return None, None, None

    # For CV-only, there is no writer. We label it as CV_ONLY.
    if etype == "cv_only":
        writer = "CV_ONLY"
    else:
        remainder = filename[len(evaluator)+1:] 
        writer = None
        sorted_writers = sorted(RAW_WRITERS, key=len, reverse=True)
        for w in sorted_writers:
            if remainder.startswith(w + "_") or remainder.startswith(w + "."):
                writer = w
                break
            elif remainder == w: 
                writer = w
                break
        if not writer: return None, None, None

    match_cv = re.search(r'cv(\d+)', filename, re.IGNORECASE)
    if not match_cv: return None, None, None
    cv_idx = int(match_cv.group(1))

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

def build_master_dataframe(base_dir):
    """Scans all folders and builds a massive DataFrame."""
    rows = []
    print(f"📂 Scanning {base_dir} for data...")
    
    if not os.path.exists(base_dir):
        print("❌ Base dir not found.")
        return pd.DataFrame()

    for job_folder in sorted(os.listdir(base_dir)):
        job_path = os.path.join(base_dir, job_folder)
        if not os.path.isdir(job_path): continue

        match_job = re.match(r'(job_\d+)', job_folder)
        if not match_job: continue
        job_id = match_job.group(1)
        job_title = format_job_title(job_id, job_folder)

        for run_folder in os.listdir(job_path):
            if not run_folder.startswith("run_"): continue
            run_path = os.path.join(job_path, run_folder)
            
            for etype in EVAL_TYPES:
                eval_path = os.path.join(run_path, etype)
                if not os.path.exists(eval_path): continue
                
                for fname in os.listdir(eval_path):
                    if not fname.endswith(".txt"): continue
                    
                    evaluator, writer, cv_idx = parse_filename(fname, etype)
                    if evaluator and writer and cv_idx:
                        score = extract_score(os.path.join(eval_path, fname))
                        if score is not None:
                            rows.append({
                                "Job_ID": job_id,
                                "Job_Title": job_title,
                                "Eval_Type": etype,
                                "Evaluator": evaluator,
                                "Writer": writer,
                                "CV_Idx": cv_idx,
                                "Score": score
                            })
    
    return pd.DataFrame(rows)

# ==========================
# ADVANCED ANALYSIS FUNCTIONS
# ==========================
def plot_cv_only_agreement(df, title_prefix, tier_name, save_dir):
    """NEW 1: BASELINE CV AGREEMENT (How much do they agree on the raw resumes?)"""
    os.makedirs(save_dir, exist_ok=True)
    
    subset = df[df['Eval_Type'] == 'cv_only'].copy()
    if subset.empty: return

    # Rank the 50 CVs based on their raw score per evaluator
    subset['CV_Rank'] = subset.groupby(['Job_ID', 'Evaluator'])['Score'].rank(method='average')

    pivot = subset.pivot_table(index=['Job_ID', 'CV_Idx'], columns='Evaluator', values='CV_Rank')

    corr = pivot.corr(method='spearman')

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f")
    plt.title(f"Baseline Agreement (Raw CV Quality) | {title_prefix}\nCV Only | [{tier_name}]")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"agreement_corr_cv_only.png"))
    plt.close()

def plot_cv_only_rank_difference(df, title_prefix, tier_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    subset = df[df['Eval_Type'] == 'cv_only'].copy()
    if subset.empty: return

    subset = subset.groupby(['Job_ID', 'CV_Idx', 'Evaluator'])['Score'].mean().reset_index()
    
    subset['CV_Rank'] = subset.groupby(['Job_ID', 'Evaluator'])['Score'].rank(method='average')

    pivot = subset.pivot_table(index=['Job_ID', 'CV_Idx'], columns='Evaluator', values='CV_Rank')
    
    evaluators = pivot.columns
    diff_matrix = pd.DataFrame(index=evaluators, columns=evaluators, dtype=float)
    
    for e1 in evaluators:
        for e2 in evaluators:
            diff_matrix.loc[e1, e2] = (pivot[e1] - pivot[e2]).abs().mean()

    plt.figure(figsize=(8, 6))
    sns.heatmap(diff_matrix, annot=True, cmap="Reds", fmt=".1f", vmin=0, vmax=6,
                cbar_kws={'label': 'Avg Disagreement (Spots out of 50 Candidates)'})
    
    plt.title(f"Baseline Disagreement (Average Rank Difference) | {title_prefix}\nCV Only | [{tier_name}]\n(Lower = Higher Agreement)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"agreement_rank_diff_cv_only.png"))
    plt.close()


def plot_inter_annotator_agreement(df, title_prefix, tier_name, save_dir):
    """1. CORRELATION MATRICES (Stylistic Agreement vs. Candidate Merit Agreement)"""
    os.makedirs(save_dir, exist_ok=True)
    
    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        subset = df[df['Eval_Type'] == etype].copy()
        if subset.empty: continue

        # --- PLOT A: STYLISTIC AGREEMENT (Do they like the same AI Writer?) ---
        # Group by Candidate, rank the 8 Writers
        subset['Style_Rank'] = subset.groupby(['Job_ID', 'CV_Idx', 'Evaluator'])['Score'].rank(method='average')
        pivot_style = subset.pivot_table(index=['Job_ID', 'CV_Idx', 'Writer'], columns='Evaluator', values='Style_Rank')
        corr_style = pivot_style.corr(method='spearman')

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_style, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f")
        plt.title(f"Stylistic Agreement (Ranking Writers per Candidate) | {title_prefix}\n{TITLE_MAP[etype]} | [{tier_name}]")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"agreement_corr_STYLE_{etype}.png"))
        plt.close()

        # --- PLOT B: MERIT AGREEMENT (Do they like the same Candidates?) ---
        # Group by Writer, rank the 50 Candidates (CV_Idx)
        subset['Merit_Rank'] = subset.groupby(['Job_ID', 'Writer', 'Evaluator'])['Score'].rank(method='average')
        pivot_merit = subset.pivot_table(index=['Job_ID', 'Writer', 'CV_Idx'], columns='Evaluator', values='Merit_Rank')
        corr_merit = pivot_merit.corr(method='spearman')

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_merit, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f")
        plt.title(f"Merit Agreement (Ranking Candidates 1-50) | {title_prefix}\n{TITLE_MAP[etype]} | [{tier_name}]")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"agreement_corr_MERIT_{etype}.png"))
        plt.close()

def plot_inter_annotator_rank_difference(df, title_prefix, tier_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        subset = df[df['Eval_Type'] == etype].copy()
        if subset.empty: continue

        # THE FIX: Average the 4 runs into a single consensus score FIRST
        subset = subset.groupby(['Job_ID', 'CV_Idx', 'Writer', 'Evaluator'])['Score'].mean().reset_index()

        # ==========================================
        # PLOT A: STYLISTIC AGREEMENT
        # ==========================================
        subset['Style_Rank'] = subset.groupby(['Job_ID', 'CV_Idx', 'Evaluator'])['Score'].rank(method='average')
        pivot_style = subset.pivot_table(index=['Job_ID', 'CV_Idx', 'Writer'], columns='Evaluator', values='Style_Rank')
        
        evaluators = pivot_style.columns
        diff_style = pd.DataFrame(index=evaluators, columns=evaluators, dtype=float)
        
        for e1 in evaluators:
            for e2 in evaluators:
                diff_style.loc[e1, e2] = (pivot_style[e1] - pivot_style[e2]).abs().mean()

        plt.figure(figsize=(8, 6))
        sns.heatmap(diff_style, annot=True, cmap="Reds", fmt=".1f", vmin=0, vmax=2,
                    cbar_kws={'label': 'Avg Disagreement (Spots out of ~8 Writers)'})
        plt.title(f"Stylistic Disagreement (Average Rank Difference) | {title_prefix}\n{TITLE_MAP[etype]} | [{tier_name}]\n(Lower = Higher Agreement)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"agreement_rank_diff_STYLE_{etype}.png"))
        plt.close()

        # ==========================================
        # PLOT B: MERIT AGREEMENT
        # ==========================================
        subset['Merit_Rank'] = subset.groupby(['Job_ID', 'Writer', 'Evaluator'])['Score'].rank(method='average')
        pivot_merit = subset.pivot_table(index=['Job_ID', 'Writer', 'CV_Idx'], columns='Evaluator', values='Merit_Rank')
        
        diff_merit = pd.DataFrame(index=evaluators, columns=evaluators, dtype=float)
        
        for e1 in evaluators:
            for e2 in evaluators:
                diff_merit.loc[e1, e2] = (pivot_merit[e1] - pivot_merit[e2]).abs().mean()

        plt.figure(figsize=(8, 6))
        sns.heatmap(diff_merit, annot=True, cmap="Reds", fmt=".1f", vmin=0, vmax=6,
                    cbar_kws={'label': 'Avg Disagreement (Spots out of 50 Candidates)'})
        plt.title(f"Merit Disagreement (Average Rank Difference) | {title_prefix}\n{TITLE_MAP[etype]} | [{tier_name}]\n(Lower = Higher Agreement)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"agreement_rank_diff_MERIT_{etype}.png"))
        plt.close()

def plot_score_distributions(df, title_prefix, tier_name, save_dir):
    """2. SCORE DISTRIBUTIONS (Discrete Grouped Bar Plots + Single Trendline)"""
    os.makedirs(save_dir, exist_ok=True)

    for etype in EVAL_TYPES:
        subset = df[df['Eval_Type'] == etype].copy()
        if subset.empty: continue

        plt.figure(figsize=(12, 6))
        
        # 1. Create a strict color map so the bars and the legend match perfectly
        evals = sorted(subset['Evaluator'].unique())
        color_palette = sns.color_palette("muted", n_colors=len(evals))
        color_dict = dict(zip(evals, color_palette))
        
        # 2. Plot the side-by-side evaluator bars (turning off Seaborn's auto-legend)
        ax = sns.histplot(data=subset, x='Score', hue='Evaluator', multiple='dodge', 
                          discrete=True, shrink=0.8, palette=color_dict, alpha=0.6, legend=False)
        
        # 3. Calculate the single average trend across all evaluators
        counts = subset.groupby(['Evaluator', 'Score']).size().reset_index(name='Count')
        avg_counts = counts.groupby('Score')['Count'].mean().reset_index()
        
        # 4. Plot the single black trendline
        sns.lineplot(data=avg_counts, x='Score', y='Count', color='black', 
                     linewidth=3, marker='o', markersize=8, ax=ax)
        
        plt.title(f"Evaluator Score Distribution | {title_prefix}\n{TITLE_MAP[etype]} | [{tier_name}]")
        plt.xlabel("Given Score")
        plt.ylabel("Frequency")
        plt.xticks(range(0, 11)) 
        
        # 5. Build a bulletproof custom legend
        handles = [mpatches.Patch(color=color_dict[ev], alpha=0.6, label=ev) for ev in evals]
        handles.append(mlines.Line2D([], [], color='black', linewidth=3, marker='o', markersize=8, label='Average Trend'))
        
        # 6. Apply the custom legend to the outside of the plot
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Evaluators & Trend")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"dist_bars_{etype}.png"))
        plt.close()

def plot_head_to_head_matrix_unbiased(df, title_prefix, tier_name, save_dir):
    """3B. HEAD-TO-HEAD WIN RATE (UNBIASED)"""
    os.makedirs(save_dir, exist_ok=True)
    writers = sorted([w for w in df['Writer'].unique() if w != "CV_ONLY"])
    n = len(writers)

    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        subset = df[df['Eval_Type'] == etype]
        if subset.empty: continue

        win_matrix = np.zeros((n, n))
        match_count_matrix = np.zeros((n, n))
        groups = subset.groupby(['Job_ID', 'CV_Idx', 'Evaluator'])

        for (job, cv, evaluator), group in groups:
            scores = dict(zip(group['Writer'], group['Score']))
            for i, w1 in enumerate(writers):
                for j, w2 in enumerate(writers):
                    if i == j: continue
                    if evaluator == w1 or evaluator == w2: continue
                    
                    s1 = scores.get(w1)
                    s2 = scores.get(w2)
                    if s1 is not None and s2 is not None:
                        match_count_matrix[i][j] += 1
                        if s1 > s2: win_matrix[i][j] += 1
                        elif s1 == s2: win_matrix[i][j] += 0.5 
        
        win_rate_pct = np.full((n, n), np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            win_rate_pct = (win_matrix / match_count_matrix) * 100
        np.fill_diagonal(win_rate_pct, np.nan)
        if np.all(np.isnan(win_rate_pct)): continue

        df_win = pd.DataFrame(win_rate_pct, index=writers, columns=writers)

        # Add average column and row, then sort by average win rate (descending)
        df_win['Avg'] = df_win.mean(axis=1, skipna=True)
        avg_row = df_win.mean(axis=0, skipna=True)
        avg_row.name = 'Avg'
        df_win = pd.concat([df_win, avg_row.to_frame().T])
        df_win = df_win.sort_values('Avg', ascending=False)
        sorted_cols = [c for c in df_win.index if c != 'Avg']
        df_win = df_win[sorted_cols + ['Avg']]

        # Mask diagonal (self vs self) cells
        mask = pd.DataFrame(False, index=df_win.index, columns=df_win.columns)
        for w in writers:
            if w in df_win.index and w in df_win.columns:
                mask.loc[w, w] = True

        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(df_win, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100, center=50,
                    cbar_kws={'label': 'Win Rate %'}, linewidths=0.5, linecolor='gray',
                    mask=mask, ax=ax)
        # Grey out diagonal NaN cells
        for w in writers:
            if w in df_win.index and w in df_win.columns:
                ri = list(df_win.index).index(w)
                ci = list(df_win.columns).index(w)
                ax.add_patch(plt.Rectangle((ci, ri), 1, 1, fill=True, color='lightgrey', lw=0))
        plt.title(f"Head-to-Head Win Rate (Unbiased) | {title_prefix}\n{TITLE_MAP[etype]} | [{tier_name}]")
        plt.xlabel("Opponent")
        plt.ylabel("Winner")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"win_matrix_UNBIASED_{etype}.png"))
        plt.close()

def plot_strictness_evolution(df, title_prefix, tier_name, save_dir):
    """5. STRICTNESS EVOLUTION (Line Plot)"""
    os.makedirs(save_dir, exist_ok=True)
    if df['CV_Idx'].nunique() < 5: return

    for etype in EVAL_TYPES: # Added CV_Only here too to see baseline strictness drop
        data = df[df['Eval_Type'] == etype]
        if data.empty: continue

        plt.figure(figsize=(12, 6))
        all_model_scores = []
        for eval_name in UNIQUE_EVALUATORS:
            eval_data = data[data['Evaluator'] == eval_name]
            if eval_data.empty: continue
            per_rank_scores = eval_data.groupby('CV_Idx')['Score'].mean()
            rolling_scores = per_rank_scores.rolling(window=3, min_periods=1).mean()
            plt.plot(rolling_scores.index, rolling_scores.values, label=eval_name, linewidth=2, alpha=0.4)
            all_model_scores.append(per_rank_scores)

        # Overall average trendline (black) across all models, window=5
        if all_model_scores:
            combined = pd.concat(all_model_scores, axis=1).mean(axis=1)
            overall_rolling = combined.rolling(window=3, min_periods=1).mean()
            plt.plot(overall_rolling.index, overall_rolling.values,
                     color='black', linewidth=3, label='Overall Average', zorder=10)

        plt.title(f"Strictness Evolution over CV Rank | {title_prefix}\n{TITLE_MAP[etype]} | [{tier_name}]")
        plt.xlabel("CV Rank")
        plt.ylabel("Average Score Given")
        plt.ylim(0, 10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"strictness_evolution_{etype}.png"))
        plt.close()

def plot_fairness_leaderboard(df, title_prefix, tier_name, save_dir):
    """THE 3-FACTOR PENALTY (Now with Candidate Ranking)
    Ranks evaluators by their Total Rank Deviation across three dimensions:
    1. HR Competence (Deviation on Candidate Ranking)
    2. General Bias (Deviation on others' writing)
    3. Self-Bias (Deviation on its own writing)
    *Uses strict deterministic tie-breaking (Score DESC, ID ASC) to match competitive arena*
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        subset = df[df['Eval_Type'] == etype].copy()
        if subset.empty: continue
        
        evaluators = subset['Evaluator'].unique()
        bias_data = []
        
        for evaluator in evaluators:
            eval_data = subset[subset['Evaluator'] == evaluator]
            if eval_data.empty: continue
            
            self_errors = []
            other_errors = []
            candidate_errors = []
            
            # ==========================================
            # DIMENSION 1 & 2: WRITER RANKING (Stylistic Bias)
            # ==========================================
            for (job, cv), group in subset.groupby(['Job_ID', 'CV_Idx']):
                if evaluator not in group['Evaluator'].values: continue
                
                # Consensus Rank (Strict Tie-Breaking: Score DESC, Writer ASC)
                cons_scores = group.groupby('Writer')['Score'].mean().reset_index()
                cons_scores = cons_scores.sort_values(by=['Score', 'Writer'], ascending=[False, True])
                cons_scores['Cons_Rank'] = range(1, len(cons_scores) + 1)
                cons_scores = cons_scores.set_index('Writer')
                
                # Evaluator Rank (Strict Tie-Breaking)
                eval_scores = group[group['Evaluator'] == evaluator].groupby('Writer')['Score'].mean().reset_index()
                eval_scores = eval_scores.sort_values(by=['Score', 'Writer'], ascending=[False, True])
                eval_scores['Eval_Rank'] = range(1, len(eval_scores) + 1)
                eval_scores = eval_scores.set_index('Writer')
                
                merged_writers = pd.merge(cons_scores[['Cons_Rank']], eval_scores[['Eval_Rank']], left_index=True, right_index=True)
                
                if not merged_writers.empty:
                    if evaluator in merged_writers.index:
                        self_err = abs(merged_writers.loc[evaluator, 'Cons_Rank'] - merged_writers.loc[evaluator, 'Eval_Rank'])
                        other_err = (merged_writers.drop(index=evaluator)['Cons_Rank'] - merged_writers.drop(index=evaluator)['Eval_Rank']).abs().mean()
                        if pd.isna(other_err): other_err = 0 
                    else:
                        self_err = 0
                        other_err = (merged_writers['Cons_Rank'] - merged_writers['Eval_Rank']).abs().mean()
                        
                    self_errors.append(self_err)
                    other_errors.append(other_err)

            # ==========================================
            # DIMENSION 3: CANDIDATE RANKING (HR Competence)
            # ==========================================
            for (job, writer), group in subset.groupby(['Job_ID', 'Writer']):
                if evaluator not in group['Evaluator'].values: continue
                
                # Consensus Candidate Rank (Strict Tie-Breaking: Score DESC, CV_Idx ASC)
                cons_cand = group.groupby('CV_Idx')['Score'].mean().reset_index()
                cons_cand = cons_cand.sort_values(by=['Score', 'CV_Idx'], ascending=[False, True])
                cons_cand['Cons_Rank'] = range(1, len(cons_cand) + 1)
                cons_cand = cons_cand.set_index('CV_Idx')
                
                # Evaluator Candidate Rank (Strict Tie-Breaking)
                eval_cand = group[group['Evaluator'] == evaluator].groupby('CV_Idx')['Score'].mean().reset_index()
                eval_cand = eval_cand.sort_values(by=['Score', 'CV_Idx'], ascending=[False, True])
                eval_cand['Eval_Rank'] = range(1, len(eval_cand) + 1)
                eval_cand = eval_cand.set_index('CV_Idx')
                
                merged_cand = pd.merge(cons_cand[['Cons_Rank']], eval_cand[['Eval_Rank']], left_index=True, right_index=True)
                
                if not merged_cand.empty:
                    mae = (merged_cand['Cons_Rank'] - merged_cand['Eval_Rank']).abs().mean()
                    candidate_errors.append(mae)
            
            # ==========================================
            # AGGREGATION
            # ==========================================
            avg_self_bias = np.mean(self_errors) if self_errors else 0
            avg_other_bias = np.mean(other_errors) if other_errors else 0
            avg_cand_error = np.mean(candidate_errors) if candidate_errors else 0
            
            total_penalty = avg_self_bias + avg_other_bias + avg_cand_error
            
            bias_data.append({
                'Evaluator': evaluator,
                'Self_Bias_Penalty': avg_self_bias,
                'General_Bias_Penalty': avg_other_bias,
                'Candidate_Ranking_Penalty': avg_cand_error,
                'Total_Penalty': total_penalty
            })
            
        if not bias_data: continue
        fdf = pd.DataFrame(bias_data)
        
        # Sort ascending so the MOST FAIR (Lowest Total Penalty) is on the left
        fdf = fdf.sort_values('Total_Penalty', ascending=True)
        
        # --- Visualization: Stacked Penalty Chart ---
        plt.figure(figsize=(12, 7))
        
        # Plot stacked components: HR (Green), General Bias (Blue), Self Bias (Red)
        fdf.set_index('Evaluator')[['Candidate_Ranking_Penalty', 'General_Bias_Penalty', 'Self_Bias_Penalty']].plot(
            kind='bar', stacked=True, color=['#2ECC71', '#3498DB', '#E74C3C'], ax=plt.gca()
        )
        
        # Overlay total raw score on top of the bars
        for i, idx in enumerate(fdf.index):
            score = fdf.loc[idx, 'Total_Penalty']
            plt.text(i, score + 0.1, f"{score:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.title(f"Evaluator Fairness Ranking (3-Factor Penalty Method) | {title_prefix}\n{TITLE_MAP[etype]} | [{tier_name}]\n(Lowest Total Bar = Most Fair)")
        plt.ylabel("Total Rank Deviation Penalty")
        plt.xlabel("Evaluator Model")
        plt.xticks(rotation=45, ha='right')
        
        plt.legend([
            "HR Competence (Candidate Rank Deviation)", 
            "General Bias (Writer Rank Deviation)", 
            "Self-Bias (Deviation on itself)"
        ], bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"fairness_ranking_three_factor_{etype}.png"))
        plt.close()

# ==========================
# MAIN EXECUTION PIPELINE
# ==========================

def run_analysis_suite(df, title, tier_name, save_dir):
    plot_cv_only_agreement(df, title, tier_name, save_dir)
    plot_cv_only_rank_difference(df, title, tier_name, save_dir)
    plot_inter_annotator_agreement(df, title, tier_name, save_dir)
    plot_inter_annotator_rank_difference(df, title, tier_name, save_dir)
    plot_score_distributions(df, title, tier_name, save_dir)
    plot_head_to_head_matrix_unbiased(df, title, tier_name, save_dir)
    plot_strictness_evolution(df, title, tier_name, save_dir)
    plot_fairness_leaderboard(df, title, tier_name, save_dir)


def main():
    master_df = build_master_dataframe(BASE_DIR)
    
    if master_df.empty:
        print("❌ No data found. Check BASE_DIR.")
        return

    
    unique_job_ids = master_df['Job_ID'].unique()
    print(f"\n📊 Found {len(unique_job_ids)} Jobs. Starting Tiered Analysis...\n")

    for job_id in unique_job_ids:
        job_df = master_df[master_df['Job_ID'] == job_id]
        job_title = job_df['Job_Title'].iloc[0]
        
        print(f"👉 Processing Job: {job_title}")
        
        for tier_name, (start_cv, end_cv) in TIERS.items():
            tier_df = job_df[(job_df['CV_Idx'] >= start_cv) & (job_df['CV_Idx'] <= end_cv)]
            if tier_df.empty: continue

            save_path = os.path.join(OUT_PLOT_DIR, job_id, tier_name)
            run_analysis_suite(tier_df, job_title, tier_name, save_path)
    

    print("\n🌍 Generating GLOBAL Aggregated Analysis (Tiered)...")
    global_base_dir = os.path.join(OUT_PLOT_DIR, "GLOBAL_AGGREGATE")
    
    for tier_name, (start_cv, end_cv) in TIERS.items():
        print(f"   👉 Global Tier: {tier_name}")
        
        tier_df = master_df[(master_df['CV_Idx'] >= start_cv) & (master_df['CV_Idx'] <= end_cv)]
        if tier_df.empty: continue
            
        save_path = os.path.join(global_base_dir, tier_name)
        run_analysis_suite(tier_df, "All Jobs Combined", tier_name, save_path)

    print("\n✨ Advanced Analysis Complete.")

if __name__ == "__main__":
    main()