"""
extract_cl_features.py — extract text features and embeddings from all cover letters.

Output: output_eval/cl_features.parquet
  One row per (Job_ID, Writer, CV_Idx) with:
  - Basic stats   : char_count, word_count, avg_word_length, sentence_count,
                    paragraph_count, comma_count, exclaim_count, question_count
  - Lexical       : ttr, flesch_reading_ease, flesch_kincaid_grade
  - Sentiment     : vader_compound  (range −1 … +1)
  - VAD           : vad_valence, vad_arousal, vad_dominance  (NRC-VAD, range 0-1)
  - Emotions      : emo_joy, emo_neutral, emo_surprise, emo_fear,
                    emo_anger, emo_disgust, emo_sadness  (j-hartmann distilroberta)
  - Embedding     : list[float32], 384-d  (all-MiniLM-L6-v2)
  - Similarity    : job_cosine_sim  (cosine between CL embedding and job-description embedding)

Re-running the script is idempotent: if the parquet already exists the embedding step is
skipped and only new/missing columns are recomputed.
"""

import io
import os
import re
import glob
import zipfile
import urllib.request

import nltk
import numpy as np
import pandas as pd
import textstat
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from tqdm import tqdm

nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt",         quiet=True)
nltk.download("punkt_tab",     quiet=True)

CL_DIR   = "output_cl"
JOB_DIR  = "dataset/jobs"
OUT_PATH = "output_eval/cl_features.parquet"

EMBED_MODEL   = "all-MiniLM-L6-v2"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMBED_BATCH   = 64
EMOTION_BATCH = 32

VAD_URL   = "https://saifmohammad.com/WebDocs/NRC-VAD-Lexicon.zip"
VAD_CACHE = "raw_datasets/NRC-VAD-Lexicon-v2.1.txt"


# ── helpers ──────────────────────────────────────────────────────────────────

def _job_id_from_folder(folder_name: str) -> str:
    m = re.match(r"(job_\d+)", folder_name)
    return m.group(1) if m else folder_name


def _read_cl_texts() -> dict:
    """Return {(job_id, writer, cv_idx): text} for all cover letters."""
    texts = {}
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
            with open(os.path.join(job_dir, fname)) as f:
                texts[(job_id, m.group(1), int(m.group(2)))] = f.read().strip()
    return texts


# ── text features ─────────────────────────────────────────────────────────────

def text_features(text: str) -> dict:
    words = [w for w in word_tokenize(text.lower()) if w.isalpha()]
    n_words = len(words)
    return dict(
        char_count           = len(text),
        word_count           = n_words,
        avg_word_length      = float(np.mean([len(w) for w in words])) if words else 0.0,
        sentence_count       = len(sent_tokenize(text)),
        paragraph_count      = len([p for p in text.split("\n\n") if p.strip()]),
        comma_count          = text.count(","),
        exclaim_count        = text.count("!"),
        question_count       = text.count("?"),
        ttr                  = len(set(words)) / n_words if n_words > 0 else 0.0,
        flesch_reading_ease  = textstat.flesch_reading_ease(text),
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text),
    )


# ── NRC VAD ───────────────────────────────────────────────────────────────────

def _load_vad_lexicon() -> dict:
    """Download (once) and parse NRC-VAD lexicon → {word: (valence, arousal, dominance)}."""
    if not os.path.exists(VAD_CACHE):
        print("Downloading NRC-VAD lexicon...")
        try:
            req = urllib.request.Request(VAD_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                fname = next(n for n in z.namelist() if n.endswith(".txt") and "Lexicon" in n)
                content = z.read(fname).decode("utf-8")
            os.makedirs(os.path.dirname(VAD_CACHE), exist_ok=True)
            with open(VAD_CACHE, "w") as f:
                f.write(content)
            print(f"  Saved to {VAD_CACHE}")
        except Exception as e:
            print(f"  WARNING: could not download NRC-VAD lexicon ({e}). VAD features will be NaN.")
            return {}

    lexicon = {}
    with open(VAD_CACHE) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            parts = line.strip().split("\t")
            if len(parts) == 4:
                term, v, a, d = parts
                if " " in term:
                    continue  # skip multi-word entries
                try:
                    lexicon[term.lower()] = (float(v), float(a), float(d))
                except ValueError:
                    pass
    print(f"  NRC-VAD loaded: {len(lexicon):,} entries")
    return lexicon


def vad_features(text: str, lexicon: dict) -> dict:
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    scores = [lexicon[w] for w in words if w in lexicon]
    if not scores:
        return {"vad_valence": np.nan, "vad_arousal": np.nan, "vad_dominance": np.nan}
    arr = np.array(scores)
    return {
        "vad_valence":   float(arr[:, 0].mean()),
        "vad_arousal":   float(arr[:, 1].mean()),
        "vad_dominance": float(arr[:, 2].mean()),
    }


# ── emotion model ─────────────────────────────────────────────────────────────

def _emotion_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return -1


def compute_emotion_features(texts: list) -> list:
    print(f"Loading emotion model ({EMOTION_MODEL})...")
    device = _emotion_device()
    print(f"  device: {device}")
    pipe = pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        top_k=None,
        device=device,
        truncation=True,
        max_length=512,
    )
    print(f"Running emotion inference on {len(texts)} texts...")
    results = []
    for output in tqdm(pipe(texts, batch_size=EMOTION_BATCH,
                            truncation=True, max_length=512),
                       total=len(texts)):
        results.append({f"emo_{item['label'].lower()}": item["score"] for item in output})
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    sia = SentimentIntensityAnalyzer()

    # ── read all cover letter texts ──
    print("Reading cover letter texts...")
    texts_map = _read_cl_texts()
    print(f"  {len(texts_map)} cover letters found")

    # ── load job description embeddings (needed for new entries) ──
    def _load_job_emb_map():
        model_ = SentenceTransformer(EMBED_MODEL)
        job_texts_, job_ids_ = {}, []
        for jpath in sorted(glob.glob(os.path.join(JOB_DIR, "*.txt"))):
            jid = _job_id_from_folder(os.path.splitext(os.path.basename(jpath))[0])
            with open(jpath) as f:
                job_texts_[jid] = f.read()
            job_ids_.append(jid)
        embs = model_.encode([job_texts_[j] for j in job_ids_], convert_to_numpy=True,
                             show_progress_bar=False)
        return model_, dict(zip(job_ids_, embs))

    # ── check if parquet already exists; detect new entries ──
    if os.path.exists(OUT_PATH):
        print(f"\nLoading existing parquet ({OUT_PATH})...")
        df_existing = pd.read_parquet(OUT_PATH)
        existing_keys = set(zip(df_existing["Job_ID"], df_existing["Writer"], df_existing["CV_Idx"]))
        new_keys = {k for k in texts_map if k not in existing_keys}
        print(f"  {len(existing_keys)} existing | {len(new_keys)} new")

        if not new_keys:
            df    = df_existing
            texts = [texts_map[k] for k in zip(df["Job_ID"], df["Writer"], df["CV_Idx"])]
        else:
            # Process only new entries
            model, job_emb_map = _load_job_emb_map()
            new_records, new_texts, new_job_ids = [], [], []
            for key in sorted(new_keys):
                job_id, writer, cv_idx = key
                text = texts_map[key]
                rec = dict(
                    Job_ID         = job_id,
                    Writer         = writer,
                    CV_Idx         = cv_idx,
                    vader_compound = sia.polarity_scores(text)["compound"],
                    **text_features(text),
                )
                new_records.append(rec)
                new_texts.append(text)
                new_job_ids.append(job_id)

            print(f"\nEmbedding {len(new_texts)} new cover letters...")
            new_embs = model.encode(new_texts, convert_to_numpy=True,
                                    show_progress_bar=True, batch_size=EMBED_BATCH)
            for rec, emb, jid in zip(new_records, new_embs, new_job_ids):
                job_emb = job_emb_map.get(jid)
                rec["job_cosine_sim"] = (
                    float(cosine_similarity(emb.reshape(1, -1), job_emb.reshape(1, -1))[0, 0])
                    if job_emb is not None else float("nan")
                )
                rec["embedding"] = emb.astype(np.float32).tolist()

            df_new = pd.DataFrame(new_records)
            df     = pd.concat([df_existing, df_new], ignore_index=True)
            # texts list must match df row order for VAD/emotion below
            old_keys  = list(zip(df_existing["Job_ID"], df_existing["Writer"], df_existing["CV_Idx"]))
            texts     = [texts_map[k] for k in old_keys] + new_texts
    else:
        model, job_emb_map = _load_job_emb_map()
        records, texts, cl_job_ids = [], [], []
        for (job_id, writer, cv_idx), text in sorted(texts_map.items()):
            rec = dict(
                Job_ID         = job_id,
                Writer         = writer,
                CV_Idx         = cv_idx,
                vader_compound = sia.polarity_scores(text)["compound"],
                **text_features(text),
            )
            records.append(rec)
            texts.append(text)
            cl_job_ids.append(job_id)

        print(f"\nEmbedding {len(texts)} cover letters...")
        cl_embeddings = model.encode(texts, convert_to_numpy=True,
                                     show_progress_bar=True, batch_size=EMBED_BATCH)
        for rec, emb, job_id in zip(records, cl_embeddings, cl_job_ids):
            job_emb = job_emb_map.get(job_id)
            rec["job_cosine_sim"] = (
                float(cosine_similarity(emb.reshape(1, -1), job_emb.reshape(1, -1))[0, 0])
                if job_emb is not None else float("nan")
            )
            rec["embedding"] = emb.astype(np.float32).tolist()
        df = pd.DataFrame(records)
    # ── NRC-VAD ──
    print("\n=== NRC-VAD features ===")
    vad_lex = _load_vad_lexicon()
    vad_rows = [vad_features(t, vad_lex) for t in tqdm(texts, desc="VAD")]
    for col in ("vad_valence", "vad_arousal", "vad_dominance"):
        df[col] = [r[col] for r in vad_rows]

    # ── emotions ──
    print("\n=== Emotion features ===")
    emo_rows = compute_emotion_features(texts)
    emo_cols = sorted({k for r in emo_rows for k in r})
    for col in emo_cols:
        df[col] = [r.get(col, np.nan) for r in emo_rows]

    # ── save ──
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_PATH}")
    new_cols = ["vad_valence", "vad_arousal", "vad_dominance"] + emo_cols
    print(df[["Job_ID", "Writer", "CV_Idx"] + new_cols].head(3).to_string())


if __name__ == "__main__":
    main()
