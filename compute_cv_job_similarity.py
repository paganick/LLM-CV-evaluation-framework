"""
compute_cv_job_similarity.py — CV-to-job-ad cosine similarity, independent of
any writer/cover-letter.

This is the "objective" candidate-fit signal: embeds each raw CV and its job
description directly (same model/approach as extract_cl_features.py's
cv_cosine_sim / job_cosine_sim, which are computed on the *cover letter*
instead), so it varies by (Job_ID, CV_Idx) only — never by Writer.

Output: output_eval/cv_job_similarity.parquet  (Job_ID, CV_Idx, cv_job_cosine_sim)
"""

import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

JOB_DIR  = "dataset/jobs"
CV_DIR   = "dataset/resumes"
OUT_PATH = "output_eval/cv_job_similarity.parquet"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH = 64


def _job_id_from_folder(folder_name: str) -> str:
    m = re.match(r"(job_\d+)", folder_name)
    return m.group(1) if m else folder_name


def _read_cv_texts() -> dict:
    texts = {}
    for job_folder in sorted(os.listdir(CV_DIR)):
        job_dir = os.path.join(CV_DIR, job_folder)
        if not os.path.isdir(job_dir):
            continue
        job_id = _job_id_from_folder(job_folder)
        for fname in sorted(os.listdir(job_dir)):
            if not fname.endswith(".txt"):
                continue
            cv_idx = int(fname.split("_")[0])
            with open(os.path.join(job_dir, fname)) as f:
                texts[(job_id, cv_idx)] = f.read().strip()
    return texts


def _read_job_texts() -> dict:
    texts = {}
    for fname in sorted(os.listdir(JOB_DIR)):
        if not fname.endswith(".txt"):
            continue
        job_id = _job_id_from_folder(os.path.splitext(fname)[0])
        with open(os.path.join(JOB_DIR, fname)) as f:
            texts[job_id] = f.read().strip()
    return texts


def main():
    print("Reading CV and job texts...")
    cv_texts  = _read_cv_texts()
    job_texts = _read_job_texts()
    print(f"  {len(cv_texts)} CVs | {len(job_texts)} jobs")

    model = SentenceTransformer(EMBED_MODEL)

    print("Embedding jobs...")
    job_ids = sorted(job_texts)
    job_embs = model.encode([job_texts[j] for j in job_ids], convert_to_numpy=True, show_progress_bar=False)
    job_emb_map = dict(zip(job_ids, job_embs))

    print("Embedding CVs...")
    cv_keys = sorted(cv_texts)
    cv_embs = model.encode([cv_texts[k] for k in cv_keys], convert_to_numpy=True,
                            show_progress_bar=True, batch_size=EMBED_BATCH)

    rows = []
    for (job_id, cv_idx), emb in zip(cv_keys, cv_embs):
        job_emb = job_emb_map[job_id]
        sim = float(cosine_similarity(emb.reshape(1, -1), job_emb.reshape(1, -1))[0, 0])
        rows.append({"Job_ID": job_id, "CV_Idx": cv_idx, "cv_job_cosine_sim": sim})

    df = pd.DataFrame(rows)
    os.makedirs("output_eval", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(df["cv_job_cosine_sim"].describe())


if __name__ == "__main__":
    main()
