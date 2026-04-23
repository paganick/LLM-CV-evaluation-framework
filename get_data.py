import pandas as pd
import os

# Open data files
job_df = pd.read_csv("../../dataset/job_desc_data.csv")
resume_df = pd.read_csv("../../dataset/resume_data.csv")

# Adjust number of resumes
num_selected_resumes = 25

# Open data file for sorted similarity matrix
matrix_df = pd.read_csv("../../dataset/exp_v2_job_matrix.csv")

# Get specific rows from job_df CHECK FOR CORRECT ORDER
selected_jobs = job_df[job_df['ID'].isin([
    1169, # Primary School Teacher
    4165, # Sales Manager
    3373, # Web Developer
    1079, # HR Manager
    3946, # CFO
    2561, # Accountant
    3701, # UX Designer
    615,  # Construction Planner
    4142, # Sales Director
    3697  # UX Designer
])]

# Save each job description to a separate text file in the jobs folder
for _, row in selected_jobs.iterrows():
    job_id = row['ID']
    job_title = row['position'].replace(" ", "_").replace("/", "_")
    description = row['description'] if pd.notna(row['description']) else ""

    #remove [ and ] from description
    description = description.replace("[", "").replace("]", "")
    
    filename = f"job_{job_id}_{job_title}.txt"
    with open(f"dataset/jobs/{filename}", "w", encoding="utf-8") as f:
        f.write(description)
        print(f"✅ Saved job description to dataset/jobs/{filename}")



def get_tiered_resumes(matrix_df):
    selected_map = {}
    
    # We iterate through the dataframe in steps of 2 (Row 0=IDs, Row 1=Scores, Row 2=IDs, etc.)
    for i in range(0, len(matrix_df), 2):
        
        # 1. Extract Data
        # Assuming Column 0 is the Job Name, and Columns 1+ are the data
        ids = matrix_df.iloc[i, 1:].dropna().values
        scores = matrix_df.iloc[i+1, 1:].dropna().values
        
        # Ensure scores are floats for comparison
        scores = scores.astype(float)
        
        # 2. Pick Top N (Highest Similarity)
        # We assume the matrix is already sorted by similarity descending, as per your image
        top_n = ids[:num_selected_resumes].tolist()
        
        # 3. Find the "Cutoff" index where similarity <= 0.4
        low_sim_indices = [idx for idx, score in enumerate(scores) if score <= 0.4]
        
        if low_sim_indices:
            start_index = low_sim_indices[0]
            # Pick the next N resumes starting from that cutoff
            low_n = ids[start_index : start_index + num_selected_resumes].tolist()
        else:
            # Fallback: If no resume is below 0.4, take the last N available
            print(f"Warning: No resumes below 0.4 found for row {i}. Taking bottom {num_selected_resumes}.")
            low_n = ids[-num_selected_resumes:].tolist()
            
        # 4. Combine
        # Use a Set to avoid duplicates if the list is short, then convert back to list
        combined = list(dict.fromkeys(top_n + low_n))
        
        # Store in our results list (or map directly if you have the job IDs ready)
        # We'll use the row index 'i' (0, 2, 4...) as a temporary key
        selected_map[i] = combined

    return selected_map

# Run the function
results_map = get_tiered_resumes(matrix_df)

# Assign to your specific Job IDs (based on your previous snippet)
resume_map = {
    615: results_map[0],   # Construction Planner (Row 0 IDs)
    1079: results_map[2],  # HR Manager (Row 2 IDs)
    1169: results_map[4],  # Primary School Teacher (Row 4 IDs)
    2561: results_map[6],  # Accountant (Row 6 IDs)
    3373: results_map[8],  # Web Developer (Row 8 IDs)
    3697: results_map[10], # UX Designer (Row 10 IDs)
    3701: results_map[12], # UX Designer (Row 12 IDs)
    3946: results_map[14], # CFO (Row 14 IDs)
    4142: results_map[16], # Sales Director (Row 16 IDs)
    4165: results_map[18]  # Sales Manager (Row 18 IDs)
}

# Check if directories exist, else create them

for job_id in resume_map.keys():
    job_title = selected_jobs[selected_jobs['ID'] == job_id].iloc[0]['position'].replace(" ", "_").replace("/", "_")
    os.makedirs(f"dataset/resumes/job_{job_id}_{job_title}", exist_ok=True)



for job_id, resume_ids in resume_map.items():
    job_row = selected_jobs[selected_jobs['ID'] == job_id].iloc[0]
    job_title = job_row['position'].replace(" ", "_").replace("/", "_")
    
    for rank, resume_id in enumerate(resume_ids, start=1):
        try:
            rid = int(resume_id)
        except (ValueError, TypeError):
            print(f"⚠️ Invalid resume ID {resume_id}")
            continue

        resume_row = resume_df[resume_df['ID'] == rid]
        if not resume_row.empty:
            resume_text = resume_row.iloc[0]['Resume_str'] if pd.notna(resume_row.iloc[0]['Resume_str']) else ""

            filename = f"{rank:02d}_resume_{rid}.txt"  # prefix with rank in list
            with open(f"dataset/resumes/job_{job_id}_{job_title}/{filename}", "w", encoding="utf-8") as f:
                f.write(resume_text)
                print(f"✅ Saved rank {rank} resume to dataset/resumes/job_{job_id}_{job_title}/{filename}")
        else:
            print(f"⚠️ Resume ID {rid} not found in resume_df")

# Save resume map for reference
import json
with open("dataset/resume_selection_map.json", "w", encoding="utf-8") as f:
    json.dump(resume_map, f, indent=4)
    print("✅ Saved resume selection map to dataset/resume_selection_map.json")