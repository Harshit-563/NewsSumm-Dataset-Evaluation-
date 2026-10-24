# This script computes semantic similarity and ROUGE metrics for the datasets
# and saves both per-row results and an aggregate comparison table.
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# ==============================
# CONFIG
# ==============================

SAMPLE_SIZE = 1000   # 🔥 REDUCED (important)
RANDOM_STATE = 42

# ==============================
# LOAD DATA
# ==============================

def load_data(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "article_text": "article",
        "human_summary": "summary"
    })

    df = df[["article", "summary"]].copy()

    df["article"] = df["article"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)

    df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)

    return df.reset_index(drop=True)

# ==============================
# LIGHTWEIGHT SEMANTIC SIMILARITY
# ==============================

def compute_semantic(df):
    texts = df["article"].tolist() + df["summary"].tolist()

    vectorizer = TfidfVectorizer(max_features=5000)
    vectors = vectorizer.fit_transform(texts)

    article_vecs = vectors[:len(df)]
    summary_vecs = vectors[len(df):]

    sims = []

    for i in range(len(df)):
        sim = cosine_similarity(article_vecs[i], summary_vecs[i])[0][0]
        sims.append(sim)

    return sims

# ==============================
# ROUGE
# ==============================

def compute_rouge(df):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    r1, r2, rl = [], [], []

    for a, s in tqdm(zip(df["article"], df["summary"]),
                     total=len(df),
                     desc="ROUGE"):

        scores = scorer.score(a, s)

        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    return r1, r2, rl

# ==============================
# EVALUATION
# ==============================

def evaluate(df):
    df = df.copy()

    print("🔹 Semantic similarity (TF-IDF)...")
    df["semantic_similarity"] = compute_semantic(df)

    print("🔹 ROUGE...")
    r1, r2, rl = compute_rouge(df)

    df["rouge1"] = r1
    df["rouge2"] = r2
    df["rougeL"] = rl

    metrics = {
        "semantic": df["semantic_similarity"].mean(),
        "rouge1": df["rouge1"].mean(),
        "rouge2": df["rouge2"].mean(),
        "rougeL": df["rougeL"].mean(),
    }

    return df, metrics

# ==============================
# COMPARISON
# ==============================

def create_comparison(m1, m2):
    return pd.DataFrame({
        "Metric": ["Semantic", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "NewsSumm": [
            m1["semantic"], m1["rouge1"], m1["rouge2"], m1["rougeL"]
        ],
        "NewsSumm++": [
            m2["semantic"], m2["rouge1"], m2["rouge2"], m2["rougeL"]
        ],
    })

# ==============================
# MAIN
# ==============================

def main():
    print("📂 Loading datasets...")
    df1 = load_data("Data/Newssumm.xlsx")
    df2 = load_data("Data/NewsSumm++.xlsx")

    print("📊 Evaluating NewsSumm...")
    eval1, m1 = evaluate(df1)

    print("📊 Evaluating NewsSumm++...")
    eval2, m2 = evaluate(df2)

    print("💾 Saving results...")
    eval1.to_excel("Outputs/rougue+sentiment/NewsSumm_evaluated.xlsx", index=False)
    eval2.to_excel("Outputs/rougue+sentiment/NewsSumm++_evaluated.xlsx", index=False)

    comp = create_comparison(m1, m2)
    comp.to_excel("Outputs/rougue+sentiment/comparison_metrics.xlsx", index=False)

    print("✅ DONE")

if __name__ == "__main__":
    main()
