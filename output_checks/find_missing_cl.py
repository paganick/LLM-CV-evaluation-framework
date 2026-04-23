import os
import glob
import re

# ==========================
# CONFIGURATION
# ==========================
DATASET_FOLDER = "./dataset"
JOBS_FOLDER = os.path.join(DATASET_FOLDER, "jobs_done") # Or "jobs" if you haven't moved them
RESUMES_FOLDER = os.path.join(DATASET_FOLDER, "resumes")
OUTPUT_CL_BASE = "./output_cl"

# The 6 models you used
MODELS = ["gpt", "claude", "gemini", "llama", "deep_local", "deep_api"]

def find_missing_files():
    print("🔍 Auditing Cover Letter Generation...\n")
    
    missing_files = []
    total_expected = 0
    total_found = 0

    # 1. Get List of Jobs
    job_files = glob.glob(os.path.join(JOBS_FOLDER, "*.txt"))
    
    for job_path in job_files:
        filename = os.path.basename(job_path)
        # Extract Job ID (e.g., job_1169)
        match = re.match(r'(job_\d+)', filename)
        
        if not match: continue
        job_id = match.group(1)
        job_folder_name = filename.replace(".txt", "") # e.g. job_1169_Teacher
        
        # 2. Find associated Resumes
        # Look for folder starting with job_id in resumes folder
        resume_subfolders = glob.glob(os.path.join(RESUMES_FOLDER, f"{job_id}_*"))
        
        if not resume_subfolders:
            print(f"⚠️  No resume folder found for {job_id}")
            continue
            
        resume_folder = resume_subfolders[0]
        resume_files = glob.glob(os.path.join(resume_folder, "*.txt"))
        cv_count = len(resume_files)
        
        # 3. Check for Expected Files
        cl_output_folder = os.path.join(OUTPUT_CL_BASE, job_folder_name)
        
        if not os.path.exists(cl_output_folder):
            print(f"❌ Missing Folder: {cl_output_folder}")
            # Add all expected files to missing list
            for i in range(1, cv_count + 1):
                for model in MODELS:
                    missing_files.append(f"{job_folder_name}/{model}_cover_letter_cv{i}.txt")
            continue

        for i in range(1, cv_count + 1):
            for model in MODELS:
                expected_filename = f"{model}_cover_letter_cv{i}.txt"
                expected_path = os.path.join(cl_output_folder, expected_filename)
                
                total_expected += 1
                
                if os.path.exists(expected_path):
                    total_found += 1
                else:
                    missing_files.append(expected_path)

    # --- REPORT ---
    print(f"📊 Stats:")
    print(f"   Total Expected: {total_expected}")
    print(f"   Total Found:    {total_found}")
    print(f"   Missing:        {len(missing_files)}\n")

    if missing_files:
        print("❌ MISSING FILES:")
        for f in missing_files:
            print(f"   - {f}")
    else:
        print("✅ All files accounted for!")

if __name__ == "__main__":
    find_missing_files()