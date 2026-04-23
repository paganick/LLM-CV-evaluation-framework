import os
import re
import pandas as pd
import numpy as np

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "./output_eval"

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

EVAL_TYPES = ["cv_only", "cl_evaluations", "cv_cl_evaluations"]

# ==========================
# DATA INGESTION (Identical to your main script)
# ==========================
def format_job_title(job_id, folder_name):
    clean = folder_name.replace(job_id, "").strip("_")
    clean = re.sub(r'_\d+$', '', clean)
    clean = clean.replace("_", " ")
    return f"{clean} ({job_id.replace('job_', '')})"

def parse_filename(filename, etype):
    evaluator = None
    sorted_evals = sorted(UNIQUE_EVALUATORS, key=len, reverse=True)
    for e in sorted_evals:
        if filename.startswith(e + "_"):
            evaluator = e
            break
    
    if not evaluator: return None, None, None

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
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'(\d+(?:\.\d+)?)', content)
            if match:
                return float(match.group(1))
            else:
                return None
    except Exception:
        return None

def build_master_dataframe(base_dir):
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
# VARIANCE CALCULATION
# ==========================
def calculate_intra_model_variance(df):
    """Calculates the average standard deviation across the 4 runs."""
    print("\n📊 Calculating Intra-Model Variance (Standard Deviation across runs)...")
    
    group_cols = ['Job_ID', 'Evaluator', 'Writer', 'CV_Idx', 'Eval_Type']
    
    # Calculate std dev for the identical groupings (the 4 runs)
    std_df = df.groupby(group_cols)['Score'].std().reset_index()
    std_df.rename(columns={'Score': 'Score_StdDev'}, inplace=True)
    
    # Average it out per evaluator
    avg_std_per_evaluator = std_df.groupby('Evaluator')['Score_StdDev'].mean().reset_index()
    avg_std_per_evaluator = avg_std_per_evaluator.sort_values(by='Score_StdDev')
    
    print("\n✅ Average Standard Deviation per Evaluator (Lower = More Consistent):")
    print(avg_std_per_evaluator.to_string(index=False, float_format="%.4f"))
    
    overall_avg = avg_std_per_evaluator['Score_StdDev'].mean()
    print(f"\n🌍 Overall Average Standard Deviation across all models: {overall_avg:.4f}\n")

if __name__ == "__main__":
    df = build_master_dataframe(BASE_DIR)
    
    if df.empty:
        print("❌ No data found. Check BASE_DIR.")
    else:
        calculate_intra_model_variance(df)