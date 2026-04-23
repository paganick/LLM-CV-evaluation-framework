import os
import glob
import re

# ==========================================
# CONFIGURATION
# ==========================================
BASE_CL_DIR = "./output_cl"
BASE_EVAL_DIR = "./output_eval"

# The models as defined in your prompt
WRITERS = [
    "gpt-4o-mini", "gpt-5-mini", "gemini-2.0-flash", 
    "gemini-3-flash-preview", "claude-haiku-4-5", 
    "deepseek-chat", "deepseek-r1-8b", "llama3.1-8b"
]

EVALUATORS = [
    "gpt-4o-mini", "gpt-5-mini", "gemini-2.0-flash", 
    "gemini-3-flash-preview", "claude-haiku-4-5", "deepseek-chat"
]

NUM_JOBS = 10
NUM_CVS = 50
RUN_AMOUNT = 4

def check_dataset_integrity():
    print("🔍 Starting Dataset Integrity Check...\n")
    
    # 1. Discover Jobs
    job_folders_cl = [d for d in os.listdir(BASE_CL_DIR) if os.path.isdir(os.path.join(BASE_CL_DIR, d))]
    
    if len(job_folders_cl) != NUM_JOBS:
        print(f"⚠️  WARNING: Found {len(job_folders_cl)} job folders in {BASE_CL_DIR}, expected {NUM_JOBS}.")
    
    total_expected_cls = NUM_JOBS * NUM_CVS * len(WRITERS)
    total_expected_evals = NUM_JOBS * NUM_CVS * RUN_AMOUNT * len(EVALUATORS) * (1 + 2 * len(WRITERS))  # CV Only + CL Only + CV+CL
    
    actual_cls_found = 0
    actual_evals_found = 0
    missing_cls = []
    missing_evals = []
    
    # Sets to track exact file paths
    expected_cl_files = set()
    expected_eval_files = set()
    
    # ==========================================
    # VERIFY COVER LETTERS
    # ==========================================
    print("📄 Checking Cover Letters...")
    for job_folder in job_folders_cl:
        cl_dir = os.path.join(BASE_CL_DIR, job_folder)
        
        for cv_num in range(1, NUM_CVS + 1):
            for writer in WRITERS:
                expected_filename = f"{writer}_cover_letter_cv{cv_num}.txt"
                file_path = os.path.normpath(os.path.join(cl_dir, expected_filename))
                
                expected_cl_files.add(file_path)
                
                if os.path.exists(file_path):
                    actual_cls_found += 1
                else:
                    missing_cls.append(file_path)

    # Check for junk/duplicate files in CL
    all_cl_files_on_disk = set(os.path.normpath(p) for p in glob.glob(f"{BASE_CL_DIR}/*/*.txt"))
    junk_cls = list(all_cl_files_on_disk - expected_cl_files)

    # ==========================================
    # VERIFY EVALUATIONS
    # ==========================================
    print("📊 Checking Evaluations...")
    for job_folder in job_folders_cl:
        for run_id in range(1, RUN_AMOUNT + 1):
            eval_run_dir = os.path.join(BASE_EVAL_DIR, job_folder, f"run_{run_id}")
            
            for cv_num in range(1, NUM_CVS + 1):
                for evaluator in EVALUATORS:
                    
                    # 1. CV ONLY
                    cv_only_file = f"{evaluator}_cv_only_eval_cv{cv_num}.txt"
                    path_1 = os.path.normpath(os.path.join(eval_run_dir, "cv_only", cv_only_file))
                    expected_eval_files.add(path_1)
                    if os.path.exists(path_1):
                        actual_evals_found += 1
                    else:
                        missing_evals.append(path_1)
                    
                    for writer in WRITERS:
                        # 2. CL ONLY
                        cl_only_file = f"{evaluator}_{writer}_evaluation_cv{cv_num}.txt"
                        path_2 = os.path.normpath(os.path.join(eval_run_dir, "cl_evaluations", cl_only_file))
                        expected_eval_files.add(path_2)
                        if os.path.exists(path_2):
                            actual_evals_found += 1
                        else:
                            missing_evals.append(path_2)
                        
                        # 3. CV + CL
                        cv_cl_file = f"{evaluator}_{writer}_cv_cl_eval_cv{cv_num}.txt"
                        path_3 = os.path.normpath(os.path.join(eval_run_dir, "cv_cl_evaluations", cv_cl_file))
                        expected_eval_files.add(path_3)
                        if os.path.exists(path_3):
                            actual_evals_found += 1
                        else:
                            missing_evals.append(path_3)

    # Check for junk/duplicate files in Eval
    all_eval_files_on_disk = set(os.path.normpath(p) for p in glob.glob(f"{BASE_EVAL_DIR}/*/*/*/*.txt"))
    junk_evals = list(all_eval_files_on_disk - expected_eval_files)

    # ==========================================
    # REPORTING
    # ==========================================
    print("\n" + "="*50)
    print("📈 INTEGRITY REPORT")
    print("="*50)
    
    # Cover Letters
    print(f"\n[COVER LETTERS]")
    print(f"Expected: {total_expected_cls:,}")
    print(f"Found:    {actual_cls_found:,}")
    if missing_cls:
        print(f"❌ Missing: {len(missing_cls):,}")
    else:
        print(f"✅ No missing Cover Letters!")
        
    if junk_cls:
        print(f"⚠️  Warning: Found {len(junk_cls):,} extra/junk files:")
        for j in junk_cls[:10]:  # Print first 10 junk files
            print(f"     - {j}")
        if len(junk_cls) > 10:
            print("     ... and more.")

    # Evaluations
    print(f"\n[EVALUATIONS]")
    print(f"Expected: {total_expected_evals:,}")
    print(f"Found:    {actual_evals_found:,}")
    if missing_evals:
        print(f"❌ Missing: {len(missing_evals):,}")
        print("   (Showing first 5 missing evaluations as examples):")
        for m in missing_evals[:5]:
            print(f"     - {m}")
    else:
        print(f"✅ No missing Evaluations!")
        
    if junk_evals:
        print(f"⚠️  Warning: Found {len(junk_evals):,} extra/junk files:")
        for j in junk_evals[:10]:  # Print first 10 junk files
            print(f"     - {j}")
        if len(junk_evals) > 10:
            print("     ... and more.")

    print("\n" + "="*50)
    if not missing_cls and not missing_evals and not junk_cls and not junk_evals:
        print("🎉 PERFECT DATASET! You are ready for analysis.")
    else:
        print("🛠️ Dataset requires attention before running analysis.")

if __name__ == "__main__":
    check_dataset_integrity()