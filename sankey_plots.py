import os
import re
import pandas as pd
import plotly.graph_objects as go

BASE_DIR = "./output_eval"
OUT_PLOT_DIR = "./output_plots/sankey_diagrams"

UNIQUE_EVALUATORS = [
    "gpt-4o-mini", "gpt-5-mini", "gemini-2.0-flash", 
    "gemini-3-flash-preview", "claude-haiku-4-5", "deepseek-chat"
]

RAW_WRITERS = [
    "gpt-4o-mini", "gpt-5-mini", "gemini-2.0-flash", 
    "gemini-3-flash-preview", "claude-haiku-4-5", 
    "deepseek-chat", "deepseek-r1-8b", "llama3.1-8b"
]

os.makedirs(OUT_PLOT_DIR, exist_ok=True)

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
            match = re.search(r'(\d+(?:\.\d+)?)', content)
            if match:
                return float(match.group(1))
            else:
                print(f"⚠️ NO NUMBER FOUND IN: {os.path.basename(filepath)}")
                return None
    except Exception as e:
        print(f"❌ ERROR reading file {filepath}: {e}")
        return None

def load_data(base_dir):
    data = []
    print(f"📂 Scanning {base_dir} for data...")
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

def get_clean_job_title(job_id):
    """Helper to extract 'Primary School Teacher' from 'job_1169_Primary_school_teacher'"""
    return job_id.split('_', 2)[-1].replace('_', ' ').title()

def plot_single_job_consensus_sankey(df, evaluator_name, writer_name, job_id, save_dir):
    job_df = df[(df['Job'] == job_id) & (df['Evaluator'] == evaluator_name)].copy()
    
    if job_df.empty:
        print(f"❌ No data found for Job: {job_id}, Evaluator: {evaluator_name}")
        return

    # 1. BASELINE (CV Only)
    cv_only = job_df[job_df['Type'] == 'cv_only']
    if cv_only.empty: return
    
    baseline_avg = cv_only.groupby('CV_Idx')['Score'].mean().reset_index()
    baseline_avg = baseline_avg.sort_values(by=['Score', 'CV_Idx'], ascending=[False, True])
    baseline_avg['Baseline_Rank'] = range(1, len(baseline_avg) + 1)
    
    # 2. OUTCOME (CV + CL)
    cv_cl = job_df[(job_df['Type'] == 'cv_cl_evaluations') & (job_df['Writer'] == writer_name)]
    if cv_cl.empty: return
    
    outcome_avg = cv_cl.groupby('CV_Idx')['Score'].mean().reset_index()
    outcome_avg = outcome_avg.sort_values(by=['Score', 'CV_Idx'], ascending=[False, True])
    outcome_avg['New_Rank'] = range(1, len(outcome_avg) + 1)
    
    # 3. Merge
    merged = pd.merge(baseline_avg[['CV_Idx', 'Baseline_Rank']], 
                      outcome_avg[['CV_Idx', 'New_Rank']], 
                      on='CV_Idx')
    
    if len(merged) == 0: return

    # 4. BUCKETING
    merged['Base_Group'] = (merged['Baseline_Rank'] - 1) // 5
    merged['New_Group'] = (merged['New_Rank'] - 1) // 5
    flows = merged.groupby(['Base_Group', 'New_Group']).size().reset_index(name='Count')

    # 5. NODE COORDINATES
    y_coords = [0.05 + (i * 0.1) for i in range(10)]
    node_x = [0.01] * 10 + [0.99] * 10
    node_y = y_coords + y_coords

    group_labels = ["Ranks 1-5", "Ranks 6-10", "Ranks 11-15", "Ranks 16-20", "Ranks 21-25",
                    "Ranks 26-30", "Ranks 31-35", "Ranks 36-40", "Ranks 41-45", "Ranks 46-50"]
    node_labels = [f"Base: {label}" for label in group_labels] + [f"New: {label}" for label in group_labels]
    node_colors = (['#1f77b4'] * 5 + ['#7f7f7f'] * 5) * 2

    # 6. BUILD LINKS
    sources = flows['Base_Group'].tolist()
    targets = (flows['New_Group'] + 10).tolist()
    values = flows['Count'].tolist()
    
    link_colors = []
    for _, row in flows.iterrows():
        b, n = row['Base_Group'], row['New_Group']
        if b >= 5 and n < 5:
            link_colors.append('rgba(46, 204, 113, 0.7)') # Leapfrog (Green)
        elif b < 5 and n >= 5:
            link_colors.append('rgba(231, 76, 60, 0.7)')  # Displaced (Red)
        elif n < b:
            link_colors.append('rgba(46, 204, 113, 0.15)') # Upward
        elif n > b:
            link_colors.append('rgba(231, 76, 60, 0.15)')  # Downward
        else:
            link_colors.append('rgba(189, 195, 199, 0.2)') # Stable

    # 7. FIGURE DRAWING
    fig = go.Figure(data=[go.Sankey(
        arrangement='perpendicular', 
        node=dict(pad=30, thickness=20, line=dict(color="black", width=1), label=node_labels, color=node_colors, x=node_x, y=node_y),
        link=dict(source=sources, target=targets, value=values, color=link_colors, hovertemplate='Candidates: %{value}<extra></extra>')
    )])
    
    clean_title = get_clean_job_title(job_id)
    fig.update_layout(
        title_text=f"Candidate Tier Movement | {clean_title}<br>Evaluator: {evaluator_name} | Writer: {writer_name}",
        font_size=12, width=1000, height=800, margin=dict(t=100, l=150, r=150, b=50)
    )
    
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"fixed_sankey_{job_id}_{writer_name}.png")
    fig.write_image(filepath, scale=2)
    print(f"   ✅ Single Combo saved to: {filepath}")

def plot_competitive_leapfrog_sankey(df, evaluator_name, writer_top, writer_low, job_id, save_dir):
    job_df = df[(df['Job'] == job_id) & (df['Evaluator'] == evaluator_name)].copy()
    
    if job_df.empty: return

    # 1. BASELINE
    cv_only = job_df[job_df['Type'] == 'cv_only']
    if cv_only.empty: return
    
    baseline_avg = cv_only.groupby('CV_Idx')['Score'].mean().reset_index()
    baseline_avg = baseline_avg.sort_values(by=['Score', 'CV_Idx'], ascending=[False, True])
    baseline_avg['Baseline_Rank'] = range(1, len(baseline_avg) + 1)
    
    # 2. COMPETITIVE OUTCOME
    top_indices = baseline_avg.head(25)['CV_Idx'].tolist()
    low_indices = baseline_avg.tail(25)['CV_Idx'].tolist()

    scores_top = job_df[(job_df['CV_Idx'].isin(top_indices)) & (job_df['Writer'] == writer_top)]
    scores_low = job_df[(job_df['CV_Idx'].isin(low_indices)) & (job_df['Writer'] == writer_low)]
    
    competitive_df = pd.concat([scores_top, scores_low])
    if competitive_df.empty: return
    
    outcome_avg = competitive_df.groupby('CV_Idx')['Score'].mean().reset_index()
    outcome_avg = outcome_avg.sort_values(by=['Score', 'CV_Idx'], ascending=[False, True])
    outcome_avg['New_Rank'] = range(1, len(outcome_avg) + 1)
    
    # 3. Merge & Buckets
    merged = pd.merge(baseline_avg[['CV_Idx', 'Baseline_Rank']], 
                      outcome_avg[['CV_Idx', 'New_Rank']], on='CV_Idx')
    
    merged['Base_Group'] = (merged['Baseline_Rank'] - 1) // 5
    merged['New_Group'] = (merged['New_Rank'] - 1) // 5
    flows = merged.groupby(['Base_Group', 'New_Group']).size().reset_index(name='Count')

    # 4. COORDINATES
    y_coords = [0.05 + (i * 0.1) for i in range(10)]
    node_x = [0.01] * 10 + [0.99] * 10
    node_y = y_coords + y_coords

    group_labels = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"]
    node_labels = [f"Base Ranks {l}" for l in group_labels] + [f"Final Ranks {l}" for l in group_labels]
    node_colors = (['#1f77b4'] * 5 + ['#7f7f7f'] * 5) * 2

    # 5. LINK COLORING
    link_colors = []
    for _, row in flows.iterrows():
        b, n = row['Base_Group'], row['New_Group']
        if b >= 5 and n < 5:
            link_colors.append('rgba(46, 204, 113, 0.8)')
        elif b < 5 and n >= 5:
            link_colors.append('rgba(231, 76, 60, 0.8)')
        elif n < b:
            link_colors.append('rgba(46, 204, 113, 0.15)')
        elif n > b:
            link_colors.append('rgba(231, 76, 60, 0.15)')
        else:
            link_colors.append('rgba(189, 195, 199, 0.2)')

    # 6. DRAW
    fig = go.Figure(data=[go.Sankey(
        arrangement='perpendicular',
        node=dict(pad=30, thickness=20, label=node_labels, color=node_colors, x=node_x, y=node_y),
        link=dict(source=flows['Base_Group'], target=flows['New_Group']+10, value=flows['Count'], color=link_colors)
    )])
    
    clean_title = get_clean_job_title(job_id)
    fig.update_layout(
        title_text=f"Competitive Leapfrog | {clean_title}<br>Evaluator: {evaluator_name} | Top 25: {writer_top} | Lower 25: {writer_low}",
        width=1000, height=800, margin=dict(t=120, l=150, r=150, b=50)
    )
    
    os.makedirs(save_dir, exist_ok=True)
    fname = f"leapfrog_{job_id}_eva_{evaluator_name}_vs_{writer_top}_vs_{writer_low}.png"
    fig.write_image(os.path.join(save_dir, fname), scale=2)
    print(f"   ✅ Competitive Sankey saved to: {fname}")


def plot_overall_average_sankey(df, job_id, save_dir):
    """
    NEW: Visualizes the global average movement by aggregating ALL evaluators 
    and ALL writers together for a specific job.
    """
    job_df = df[df['Job'] == job_id].copy()
    if job_df.empty: return

    # 1. BASELINE: Average across ALL evaluators and ALL runs for cv_only
    cv_only = job_df[job_df['Type'] == 'cv_only']
    if cv_only.empty: return
    
    baseline_avg = cv_only.groupby('CV_Idx')['Score'].mean().reset_index()
    baseline_avg = baseline_avg.sort_values(by=['Score', 'CV_Idx'], ascending=[False, True])
    baseline_avg['Baseline_Rank'] = range(1, len(baseline_avg) + 1)
    
    # 2. OUTCOME: Average across ALL evaluators, ALL writers, and ALL runs for cv_cl_evaluations
    cv_cl = job_df[job_df['Type'] == 'cv_cl_evaluations']
    if cv_cl.empty: return
    
    outcome_avg = cv_cl.groupby('CV_Idx')['Score'].mean().reset_index()
    outcome_avg = outcome_avg.sort_values(by=['Score', 'CV_Idx'], ascending=[False, True])
    outcome_avg['New_Rank'] = range(1, len(outcome_avg) + 1)
    
    # 3. Merge & Bucket
    merged = pd.merge(baseline_avg[['CV_Idx', 'Baseline_Rank']], 
                      outcome_avg[['CV_Idx', 'New_Rank']], on='CV_Idx')
    
    merged['Base_Group'] = (merged['Baseline_Rank'] - 1) // 5
    merged['New_Group'] = (merged['New_Rank'] - 1) // 5
    flows = merged.groupby(['Base_Group', 'New_Group']).size().reset_index(name='Count')

    # 4. Coordinates
    y_coords = [0.05 + (i * 0.1) for i in range(10)]
    node_x = [0.01] * 10 + [0.99] * 10
    node_y = y_coords + y_coords

    group_labels = ["Ranks 1-5", "Ranks 6-10", "Ranks 11-15", "Ranks 16-20", "Ranks 21-25",
                    "Ranks 26-30", "Ranks 31-35", "Ranks 36-40", "Ranks 41-45", "Ranks 46-50"]
    node_labels = [f"Global Base: {label}" for label in group_labels] + [f"Global Final: {label}" for label in group_labels]
    node_colors = (['#1f77b4'] * 5 + ['#7f7f7f'] * 5) * 2

    # 5. Link Colors
    link_colors = []
    for _, row in flows.iterrows():
        b, n = row['Base_Group'], row['New_Group']
        if b >= 5 and n < 5:
            link_colors.append('rgba(46, 204, 113, 0.8)') # Leapfrog!
        elif b < 5 and n >= 5:
            link_colors.append('rgba(231, 76, 60, 0.8)') # Displaced!
        elif n < b:
            link_colors.append('rgba(46, 204, 113, 0.15)') # Upward
        elif n > b:
            link_colors.append('rgba(231, 76, 60, 0.15)')  # Downward
        else:
            link_colors.append('rgba(189, 195, 199, 0.2)') # Stable

    # 6. Draw
    fig = go.Figure(data=[go.Sankey(
        arrangement='perpendicular',
        node=dict(pad=30, thickness=20, line=dict(color="black", width=1), label=node_labels, color=node_colors, x=node_x, y=node_y),
        link=dict(source=flows['Base_Group'], target=flows['New_Group']+10, value=flows['Count'], color=link_colors)
    )])
    
    clean_title = get_clean_job_title(job_id)
    fig.update_layout(
        title_text=f"Global Average Candidate Movement | {clean_title}<br>All Evaluators & All Writers Aggregated",
        font_size=12, width=1000, height=800, margin=dict(t=100, l=150, r=150, b=50)
    )
    
    os.makedirs(save_dir, exist_ok=True)
    fname = f"overall_average_sankey_{job_id}.png"
    fig.write_image(os.path.join(save_dir, fname), scale=2)
    print(f"   ✅ Overall Average Sankey saved to: {fname}")


# ==========================
# MAIN EXECUTION
# ==========================
def main():
    df = load_data(BASE_DIR)
    if df.empty:
        print("❌ No data found.")
        return

    unique_jobs = df['Job'].unique()
    if len(unique_jobs) == 0:
        print("❌ No jobs found in data.")
        return
        
    # Variables for the specific plots
    target_evaluator = "gemini-3-flash-preview"
    target_writer = "llama3.1-8b"
    writer_top = "llama3.1-8b"
    writer_low = "gemini-3-flash-preview"

    print(f"\n🌍 Generating Sankey diagrams for ALL {len(unique_jobs)} JOBS...\n")

    for job_id in unique_jobs:
        print(f"👉 Processing Job: {job_id}")
        
        # 1. Define specific save directories
        eval_save_dir = os.path.join(OUT_PLOT_DIR, job_id, target_evaluator)
        overall_save_dir = os.path.join(OUT_PLOT_DIR, job_id, "overall_average")
        
        os.makedirs(eval_save_dir, exist_ok=True)
        os.makedirs(overall_save_dir, exist_ok=True)
        
        # 2. Generate the 3 plots
        plot_single_job_consensus_sankey(df, target_evaluator, target_writer, job_id, eval_save_dir)
        plot_competitive_leapfrog_sankey(df, target_evaluator, writer_top, writer_low, job_id, eval_save_dir)
        #plot_overall_average_sankey(df, job_id, overall_save_dir)
        
        print("-" * 40)

    print("✨ All Sankey diagrams generated successfully!")

if __name__ == "__main__":
    main()