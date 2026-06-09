"""
extract_cl_features.py — extract text features and embeddings from all cover letters.

Output: output_eval/cl_features.parquet
  One row per (Job_ID, Writer, CV_Idx) with:
  - Basic stats : char_count, word_count, avg_word_length, sentence_count,
                  paragraph_count, comma_count, exclaim_count, question_count
  - Lexical     : ttr, flesch_reading_ease, flesch_kincaid_grade
  - Sentiment   : vader_compound  (range −1 … +1)
  - Embedding   : list[float32], 384-d  (all-MiniLM-L6-v2)
  - Similarity  : job_cosine_sim  (cosine between CL embedding and job-description embedding)
"""

import os
import re
import glob

import nltk
import numpy as np
import pandas as pd
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt",         quiet=True)
nltk.download("punkt_tab",     quiet=True)

CL_DIR   = "output_cl"
JOB_DIR  = "dataset/jobs"
OUT_PATH = "output_eval/cl_features.parquet"

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE  = 64


def _job_id_from_folder(folder_name: str) -> str:
    """'job_1079_HR_Manager' → 'job_1079'"""
    m = re.match(r"(job_\d+)", folder_name)
    return m.group(1) if m else folder_name


def text_features(text: str) -> dict:
    words = [w for w in word_tokenize(text.lower()) if w.isalpha()]
    n_words = len(words)
    return dict(
        char_count          = len(text),
        word_count          = n_words,
        avg_word_length     = float(np.mean([len(w) for w in words])) if words else 0.0,
        sentence_count      = len(sent_tokenize(text)),
        paragraph_count     = len([p for p in text.split("\n\n") if p.strip()]),
        comma_count         = text.count(","),
        exclaim_count       = text.count("!"),
        question_count      = text.count("?"),
        ttr                 = len(set(words)) / n_words if n_words > 0 else 0.0,
        flesch_reading_ease = textstat.flesch_reading_ease(text),
        flesch_kincaid_grade= textstat.flesch_kincaid_grade(text),
    )


def main():
    sia   = SentimentIntensityAnalyzer()
    model = SentenceTransformer(EMBED_MODEL)

    # --- load and embed job descriptions ---
    print("Loading job descriptions...")
    job_texts, job_ids = {}, []
    for jpath in sorted(glob.glob(os.path.join(JOB_DIR, "*.txt"))):
        folder_like = os.path.splitext(os.path.basename(jpath))[0]
        job_id = _job_id_from_folder(folder_like)
        with open(jpath) as f:
            job_texts[job_id] = f.read()
        job_ids.append(job_id)

    print(f"  {len(job_ids)} jobs found: {job_ids}")
    job_emb_map = dict(zip(
        job_ids,
        model.encode([job_texts[j] for j in job_ids], convert_to_numpy=True,
                     show_progress_bar=False),
    ))

    # --- walk cover letters ---
    records, cl_texts, cl_job_ids = [], [], []

    for job_folder in sorted(os.listdir(CL_DIR)):
        job_dir = os.path.join(CL_DIR, job_folder)
        if not os.path.isdir(job_dir):
            continue
        job_id = _job_id_from_folder(job_folder)

        for fname in sorted(os.listdir(job_dir)):
            if not fname.endswith(".txt"):
                continue
            m = re.match(r"(.+)_cover_letter_cv(\d+)\.txt", fname)
            if not m:
                continue
            writer = m.group(1)
            cv_idx = int(m.group(2))

            with open(os.path.join(job_dir, fname)) as f:
                text = f.read().strip()

            rec = dict(
                Job_ID = job_id,
                Writer = writer,
                CV_Idx = cv_idx,
                vader_compound = sia.polarity_scores(text)["compound"],
                **text_features(text),
            )
            records.append(rec)
            cl_texts.append(text)
            cl_job_ids.append(job_id)

    # --- batch-embed all CLs ---
    print(f"\nEmbedding {len(cl_texts)} cover letters...")
    cl_embeddings = model.encode(
        cl_texts, convert_to_numpy=True,
        show_progress_bar=True, batch_size=BATCH_SIZE,
    )

    # --- attach embedding + cosine sim ---
    for rec, emb, job_id in zip(records, cl_embeddings, cl_job_ids):
        job_emb = job_emb_map.get(job_id)
        rec["job_cosine_sim"] = (
            float(cosine_similarity(emb.reshape(1, -1), job_emb.reshape(1, -1))[0, 0])
            if job_emb is not None else float("nan")
        )
        rec["embedding"] = emb.astype(np.float32).tolist()

    df = pd.DataFrame(records)
    df.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(df)} rows → {OUT_PATH}")
    print(df.dtypes)
    print("\nSample:")
    print(df[["Job_ID", "Writer", "CV_Idx", "char_count", "vader_compound",
              "flesch_reading_ease", "job_cosine_sim"]].head(5).to_string())


if __name__ == "__main__":
    main()
