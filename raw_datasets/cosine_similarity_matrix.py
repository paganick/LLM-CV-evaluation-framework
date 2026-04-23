from sentence_transformers import SentenceTransformer, util
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Load data
job_df = pd.read_csv("../job_desc_data.csv")
resume_df = pd.read_csv("../resume_data.csv")

# Clean data
job_df['description'] = job_df['description'].fillna("").astype(str)
resume_df['Resume_str'] = resume_df['Resume_str'].fillna("").astype(str)

# SELECT THE JOBS TO PROCESS
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

# Encode texts
model = SentenceTransformer("all-MiniLM-L6-v2")

job_emb = model.encode(selected_jobs['description'].tolist(), convert_to_tensor=True)
resume_emb = model.encode(resume_df['Resume_str'].tolist(), convert_to_tensor=True)

cos_scores = util.cos_sim(job_emb, resume_emb)

similarity_matrix = cos_scores.cpu().numpy()

# Label columns with resume IDs (if available)
resume_labels = resume_df['ID'] if 'ID' in resume_df.columns else [f"Resume_{i}" for i in range(len(resume_df))]

resume_labels = list(resume_labels)

# Create the sorted matrix DataFrame
sorted_rows = []

for i, job in enumerate(selected_jobs.itertuples(index=False)):
    job_title = getattr(job, "position", f"Job_{i}")
    job_id = getattr(job, "ID", f"Job_{i}")
    
    # Get cosine similarity scores for this job (corrected indexing)
    scores = similarity_matrix[i]
    
    # Pair resume IDs and scores, sort descending
    paired = sorted(zip(resume_labels, scores), key=lambda x: x[1], reverse=True)
    
    # Separate IDs and scores into two ordered lists
    sorted_ids = [p[0] for p in paired]
    sorted_scores = [round(p[1], 3) for p in paired]
    
    # Add 2 rows: one for IDs, one for scores
    sorted_rows.append([f"{job_title}/{job_id} (CV IDs)"] + sorted_ids)
    sorted_rows.append([f"{job_title}/{job_id} (Scores)"] + sorted_scores)

# Convert to DataFrame
sim_df = pd.DataFrame(sorted_rows)

# Plot the results using density plots
results = pd.DataFrame(sim_df)
results.to_excel("../exp_v2_job_matrix.xlsx", index=False)
results.to_csv("../exp_v2_job_matrix.csv", index=False)

print("✅ Results saved to exp_v2_job_matrix.xlsx")
