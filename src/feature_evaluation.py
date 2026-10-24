# This script computes additional lightweight dataset features such as length,
# lexical, overlap, entity, and readability metrics for comparison.
import re
import string
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

MODE = "fast"  # "fast" or "full"
SAMPLE_SIZE = 5000
CHUNK_SIZE = 1000
USE_ENTITIES = True
USE_READABILITY = True

if MODE == "full":
    USE_ENTITIES = True
    USE_READABILITY = True

# ==============================
# SETUP
# ==============================

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
TOKEN_PATTERN = r"\b\w+\b"
PUNCTUATION = set(string.punctuation)
KEY_FEATURES = [
    "doc_length_words",
    "summary_length_words",
    "compression_ratio",
    "sentence_count",
    "avg_sentence_length",
    "vocabulary_size",
    "type_token_ratio",
    "stopword_ratio",
    "keyword_density",
    "entity_count",
    "entity_density",
    "flesch_reading_ease",
    "repetition_ratio",
    "word_overlap_ratio",
    "coverage_score",
]

NLP = None
TEXTSTAT = None


# ==============================
# HELPERS
# ==============================

def safe_divide(a, b):
    return a / b if b else 0.0


def safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def safe_variance(values):
    return float(np.var(values)) if values else 0.0


def get_nlp():
    global NLP
    if NLP is None:
        import spacy

        NLP = spacy.load(
            "en_core_web_sm",
            disable=["tagger", "parser", "lemmatizer", "attribute_ruler"],
        )
    return NLP


def get_textstat():
    global TEXTSTAT
    if TEXTSTAT is None:
        from textstat import textstat

        TEXTSTAT = textstat
    return TEXTSTAT


def safe_textstat(func, text):
    try:
        return float(func(text)) if text.strip() else 0.0
    except Exception:
        return 0.0


def count_paragraphs(text):
    parts = text.split("\n\n")
    return float(sum(bool(part.strip()) for part in parts))


def compute_sentence_features(text):
    sentences = [s for s in sent_tokenize(text) if s.strip()]
    sentence_lengths = [len(re.findall(TOKEN_PATTERN, sentence.lower())) for sentence in sentences]
    sentence_count = len(sentence_lengths)
    avg_sentence_length = safe_mean(sentence_lengths)
    sentence_length_variance = safe_variance(sentence_lengths)
    duplicate_sentence_ratio = safe_divide(
        sentence_count - len({s.strip().lower() for s in sentences}),
        sentence_count,
    )
    return sentence_count, avg_sentence_length, sentence_length_variance, duplicate_sentence_ratio


def compute_token_features(tokens):
    alpha = [token for token in tokens if token.isalpha()]
    total_tokens = len(tokens)
    total_alpha = len(alpha)
    freq = Counter(alpha)
    freq_values = list(freq.values())
    unique_tokens = len(freq)

    avg_word_length = safe_mean([len(token) for token in alpha])
    unique_word_ratio = safe_divide(unique_tokens, total_tokens)
    hapax_legomena_ratio = safe_divide(sum(count == 1 for count in freq_values), unique_tokens)
    stopword_ratio = safe_divide(sum(token in STOP_WORDS for token in alpha), total_alpha)
    non_stopword_ratio = 1.0 - stopword_ratio
    keyword_density = safe_divide(sum(count for _, count in freq.most_common(5)), sum(freq_values))
    keyword_diversity = sum(1 for _, count in freq.most_common(5) if count > 0)
    repetition_ratio = safe_divide(sum(count - 1 for count in freq_values if count > 1), sum(freq_values))

    return {
        "alpha_tokens": alpha,
        "freq": freq,
        "vocabulary_size": unique_tokens,
        "unique_word_ratio": unique_word_ratio,
        "type_token_ratio": unique_word_ratio,
        "lexical_diversity": unique_word_ratio,
        "hapax_legomena_ratio": hapax_legomena_ratio,
        "stopword_ratio": stopword_ratio,
        "non_stopword_ratio": non_stopword_ratio,
        "keyword_density": keyword_density,
        "keyword_diversity": keyword_diversity,
        "avg_word_length": avg_word_length,
        "repetition_ratio": repetition_ratio,
        "numeric_token_ratio": safe_divide(sum(token.isdigit() for token in tokens), total_tokens),
    }


def extract_entities(texts, desc):
    nlp = get_nlp()
    entity_counts = []
    entity_densities = []
    unique_entity_ratios = []
    entity_sets = []

    docs = nlp.pipe(texts, batch_size=128)
    for doc in tqdm(docs, total=len(texts), desc=desc):
        entities = [ent.text.strip().lower() for ent in doc.ents if ent.text.strip()]
        entity_set = set(entities)
        entity_count = len(entities)

        entity_counts.append(entity_count)
        entity_sets.append(entity_set)
        unique_entity_ratios.append(safe_divide(len(entity_set), entity_count))
        entity_densities.append(entity_count)

    return entity_counts, entity_densities, unique_entity_ratios, entity_sets


# ==============================
# LOAD DATA
# ==============================

def load_data(path, sample_size=SAMPLE_SIZE):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    df = df.rename(
        columns={
            "article_text": "article",
            "human_summary": "summary",
        }
    )

    df = df[["article", "summary"]].copy()
    df["article"] = df["article"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)

    df = df.sample(min(sample_size, len(df)), random_state=42)

    return df.reset_index(drop=True)


# ==============================
# BASIC FEATURES
# ==============================

def compute_basic_features(chunk):
    chunk = chunk.copy()

    article_tokens = chunk["article"].str.lower().str.findall(TOKEN_PATTERN).tolist()
    summary_tokens = chunk["summary"].str.lower().str.findall(TOKEN_PATTERN).tolist()

    chunk["doc_length_words"] = np.fromiter((len(tokens) for tokens in article_tokens), dtype=float)
    chunk["summary_length_words"] = np.fromiter((len(tokens) for tokens in summary_tokens), dtype=float)
    chunk["doc_length_chars"] = chunk["article"].str.len().astype(float)
    chunk["summary_length_chars"] = chunk["summary"].str.len().astype(float)
    chunk["compression_ratio"] = np.divide(
        chunk["summary_length_words"],
        chunk["doc_length_words"],
        out=np.zeros(len(chunk), dtype=float),
        where=chunk["doc_length_words"].to_numpy() != 0,
    )
    chunk["paragraph_count"] = chunk["article"].apply(count_paragraphs)
    chunk["punctuation_density"] = np.fromiter(
        (
            safe_divide(sum(char in PUNCTUATION for char in text), len(text))
            for text in chunk["article"].tolist()
        ),
        dtype=float,
    )
    chunk["uppercase_ratio"] = np.fromiter(
        (
            safe_divide(sum(char.isupper() for char in text), sum(char.isalpha() for char in text))
            for text in chunk["article"].tolist()
        ),
        dtype=float,
    )
    chunk["whitespace_ratio"] = np.fromiter(
        (
            safe_divide(sum(char.isspace() for char in text), len(text))
            for text in chunk["article"].tolist()
        ),
        dtype=float,
    )

    sentence_metrics = [compute_sentence_features(text) for text in chunk["article"].tolist()]
    chunk["sentence_count"] = np.array([item[0] for item in sentence_metrics], dtype=float)
    chunk["avg_sentence_length"] = np.array([item[1] for item in sentence_metrics], dtype=float)
    chunk["sentence_length_variance"] = np.array([item[2] for item in sentence_metrics], dtype=float)
    chunk["duplicate_sentence_ratio"] = np.array([item[3] for item in sentence_metrics], dtype=float)

    return chunk, article_tokens, summary_tokens


# ==============================
# VOCAB FEATURES
# ==============================

def compute_vocab_features(chunk, article_tokens):
    chunk = chunk.copy()

    token_metrics = [compute_token_features(tokens) for tokens in article_tokens]
    alpha_tokens = [item["alpha_tokens"] for item in token_metrics]
    freqs = [item["freq"] for item in token_metrics]

    chunk["avg_word_length"] = np.array([item["avg_word_length"] for item in token_metrics], dtype=float)
    chunk["vocabulary_size"] = np.array([item["vocabulary_size"] for item in token_metrics], dtype=float)
    chunk["unique_word_ratio"] = np.array([item["unique_word_ratio"] for item in token_metrics], dtype=float)
    chunk["type_token_ratio"] = np.array([item["type_token_ratio"] for item in token_metrics], dtype=float)
    chunk["lexical_diversity"] = np.array([item["lexical_diversity"] for item in token_metrics], dtype=float)
    chunk["hapax_legomena_ratio"] = np.array([item["hapax_legomena_ratio"] for item in token_metrics], dtype=float)
    chunk["stopword_ratio"] = np.array([item["stopword_ratio"] for item in token_metrics], dtype=float)
    chunk["non_stopword_ratio"] = np.array([item["non_stopword_ratio"] for item in token_metrics], dtype=float)
    chunk["keyword_density"] = np.array([item["keyword_density"] for item in token_metrics], dtype=float)
    chunk["keyword_diversity"] = np.array([item["keyword_diversity"] for item in token_metrics], dtype=float)
    chunk["repetition_ratio"] = np.array([item["repetition_ratio"] for item in token_metrics], dtype=float)
    chunk["numeric_token_ratio"] = np.array([item["numeric_token_ratio"] for item in token_metrics], dtype=float)

    return chunk, alpha_tokens, freqs


# ==============================
# ENTITY FEATURES
# ==============================

def compute_entity_features(chunk, article_texts, summary_texts):
    chunk = chunk.copy()

    if not USE_ENTITIES:
        zero_array = np.zeros(len(chunk), dtype=float)
        chunk["entity_count"] = zero_array
        chunk["entity_density"] = zero_array
        chunk["unique_entity_ratio"] = zero_array
        chunk["entity_coverage"] = zero_array
        chunk["missing_entity_ratio"] = zero_array
        return chunk

    article_counts, article_densities, article_unique_ratios, article_sets = extract_entities(
        article_texts,
        "Article entities",
    )
    summary_counts, _, _, summary_sets = extract_entities(
        summary_texts,
        "Summary entities",
    )

    doc_lengths = chunk["doc_length_words"].to_numpy()
    chunk["entity_count"] = np.array(article_counts, dtype=float)
    chunk["entity_density"] = np.divide(
        np.array(article_densities, dtype=float),
        doc_lengths,
        out=np.zeros(len(chunk), dtype=float),
        where=doc_lengths != 0,
    )
    chunk["unique_entity_ratio"] = np.array(article_unique_ratios, dtype=float)
    chunk["entity_coverage"] = np.array(
        [
            safe_divide(len(article_set & summary_set), len(article_set))
            for article_set, summary_set in zip(article_sets, summary_sets)
        ],
        dtype=float,
    )
    chunk["missing_entity_ratio"] = 1.0 - chunk["entity_coverage"]

    return chunk


# ==============================
# READABILITY FEATURES
# ==============================

def compute_readability_features(chunk):
    chunk = chunk.copy()

    if not USE_READABILITY:
        zero_array = np.zeros(len(chunk), dtype=float)
        chunk["flesch_reading_ease"] = zero_array
        chunk["smog_index"] = zero_array
        chunk["automated_readability_index"] = zero_array
        return chunk

    textstat = get_textstat()
    articles = chunk["article"].tolist()

    chunk["flesch_reading_ease"] = np.array(
        [safe_textstat(textstat.flesch_reading_ease, text) for text in articles],
        dtype=float,
    )
    chunk["smog_index"] = np.array(
        [safe_textstat(textstat.smog_index, text) for text in articles],
        dtype=float,
    )
    chunk["automated_readability_index"] = np.array(
        [safe_textstat(textstat.automated_readability_index, text) for text in articles],
        dtype=float,
    )

    return chunk


# ==============================
# OVERLAP FEATURES
# ==============================

def compute_overlap_features(chunk, article_alpha_tokens, summary_tokens):
    chunk = chunk.copy()

    summary_alpha_tokens = [[token for token in tokens if token.isalpha()] for tokens in summary_tokens]
    article_sets = [set(tokens) for tokens in article_alpha_tokens]
    summary_sets = [set(tokens) for tokens in summary_alpha_tokens]

    chunk["word_overlap_ratio"] = np.array(
        [
            safe_divide(len(article_set & summary_set), len(summary_set))
            for article_set, summary_set in zip(article_sets, summary_sets)
        ],
        dtype=float,
    )
    chunk["jaccard_similarity"] = np.array(
        [
            safe_divide(len(article_set & summary_set), len(article_set | summary_set))
            for article_set, summary_set in zip(article_sets, summary_sets)
        ],
        dtype=float,
    )
    chunk["coverage_score"] = np.array(
        [
            safe_divide(sum(token in article_set for token in summary_alpha), len(summary_alpha))
            for article_set, summary_alpha in zip(article_sets, summary_alpha_tokens)
        ],
        dtype=float,
    )

    return chunk


# ==============================
# AGGREGATION
# ==============================

def aggregate_metrics(df):
    metrics = {
        "avg_doc_length_words": float(df["doc_length_words"].mean()),
        "avg_summary_length_words": float(df["summary_length_words"].mean()),
        "avg_doc_length_chars": float(df["doc_length_chars"].mean()),
        "avg_summary_length_chars": float(df["summary_length_chars"].mean()),
        "compression_ratio": float(df["compression_ratio"].mean()),
        "length_ratio_variance": float(np.var(df["compression_ratio"].to_numpy(), ddof=0)),
        "avg_word_length": float(df["avg_word_length"].mean()),
        "sentence_count": float(df["sentence_count"].mean()),
        "avg_sentence_length": float(df["avg_sentence_length"].mean()),
        "sentence_length_variance": float(df["sentence_length_variance"].mean()),
        "vocabulary_size": float(df["vocabulary_size"].mean()),
        "unique_word_ratio": float(df["unique_word_ratio"].mean()),
        "type_token_ratio": float(df["type_token_ratio"].mean()),
        "lexical_diversity": float(df["lexical_diversity"].mean()),
        "hapax_legomena_ratio": float(df["hapax_legomena_ratio"].mean()),
        "stopword_ratio": float(df["stopword_ratio"].mean()),
        "non_stopword_ratio": float(df["non_stopword_ratio"].mean()),
        "keyword_density": float(df["keyword_density"].mean()),
        "keyword_diversity": float(df["keyword_diversity"].mean()),
        "avg_entity_count": float(df["entity_count"].mean()),
        "entity_density": float(df["entity_density"].mean()),
        "unique_entity_ratio": float(df["unique_entity_ratio"].mean()),
        "entity_coverage": float(df["entity_coverage"].mean()),
        "missing_entity_ratio": float(df["missing_entity_ratio"].mean()),
        "paragraph_count": float(df["paragraph_count"].mean()),
        "punctuation_density": float(df["punctuation_density"].mean()),
        "numeric_token_ratio": float(df["numeric_token_ratio"].mean()),
        "uppercase_ratio": float(df["uppercase_ratio"].mean()),
        "flesch_reading_ease": float(df["flesch_reading_ease"].mean()),
        "smog_index": float(df["smog_index"].mean()),
        "automated_readability_index": float(df["automated_readability_index"].mean()),
        "repetition_ratio": float(df["repetition_ratio"].mean()),
        "duplicate_sentence_ratio": float(df["duplicate_sentence_ratio"].mean()),
        "whitespace_ratio": float(df["whitespace_ratio"].mean()),
        "word_overlap_ratio": float(df["word_overlap_ratio"].mean()),
        "jaccard_similarity": float(df["jaccard_similarity"].mean()),
        "coverage_score": float(df["coverage_score"].mean()),
        "total_samples": float(len(df)),
        "duplicate_percentage": float(
            safe_divide(df.duplicated(subset=["article", "summary"]).sum() * 100, len(df))
        ),
    }

    numeric_columns = [
        "doc_length_words",
        "summary_length_words",
        "doc_length_chars",
        "summary_length_chars",
        "compression_ratio",
        "avg_word_length",
        "sentence_count",
        "avg_sentence_length",
        "sentence_length_variance",
        "vocabulary_size",
        "unique_word_ratio",
        "type_token_ratio",
        "lexical_diversity",
        "hapax_legomena_ratio",
        "stopword_ratio",
        "non_stopword_ratio",
        "keyword_density",
        "keyword_diversity",
        "entity_count",
        "entity_density",
        "unique_entity_ratio",
        "entity_coverage",
        "missing_entity_ratio",
        "paragraph_count",
        "punctuation_density",
        "numeric_token_ratio",
        "uppercase_ratio",
        "flesch_reading_ease",
        "smog_index",
        "automated_readability_index",
        "repetition_ratio",
        "duplicate_sentence_ratio",
        "whitespace_ratio",
        "word_overlap_ratio",
        "jaccard_similarity",
        "coverage_score",
    ]

    metrics["avg_all_features"] = float(df[numeric_columns].to_numpy(dtype=float).mean())

    key_feature_array = np.column_stack(
        [
            df["doc_length_words"].to_numpy(dtype=float),
            df["summary_length_words"].to_numpy(dtype=float),
            df["compression_ratio"].to_numpy(dtype=float),
            df["sentence_count"].to_numpy(dtype=float),
            df["avg_sentence_length"].to_numpy(dtype=float),
            df["vocabulary_size"].to_numpy(dtype=float),
            df["type_token_ratio"].to_numpy(dtype=float),
            df["stopword_ratio"].to_numpy(dtype=float),
            df["keyword_density"].to_numpy(dtype=float),
            df["entity_count"].to_numpy(dtype=float),
            df["entity_density"].to_numpy(dtype=float),
            df["flesch_reading_ease"].to_numpy(dtype=float),
            df["repetition_ratio"].to_numpy(dtype=float),
            df["word_overlap_ratio"].to_numpy(dtype=float),
            df["coverage_score"].to_numpy(dtype=float),
        ]
    )
    metrics["std_dev_key_features"] = float(np.std(key_feature_array, axis=0, ddof=0).mean())

    return metrics


# ==============================
# COMPARISON
# ==============================

def create_comparison(m1, m2):
    metrics = list(m1.keys())
    return pd.DataFrame(
        {
            "Metric": metrics,
            "NewsSumm": [m1[metric] for metric in metrics],
            "NewsSumm++": [m2[metric] for metric in metrics],
        }
    )


# ==============================
# DATASET PIPELINE
# ==============================

def process_dataset(input_path, output_path, sample_size=SAMPLE_SIZE, chunk_size=CHUNK_SIZE):
    df = load_data(input_path, sample_size=sample_size)
    feature_frames = []

    for start in tqdm(range(0, len(df), chunk_size), desc=f"Processing {output_path}"):
        chunk = df.iloc[start:start + chunk_size].copy()

        chunk, article_tokens, summary_tokens = compute_basic_features(chunk)
        chunk, article_alpha_tokens, _ = compute_vocab_features(chunk, article_tokens)
        chunk = compute_entity_features(chunk, chunk["article"].tolist(), chunk["summary"].tolist())
        chunk = compute_readability_features(chunk)
        chunk = compute_overlap_features(chunk, article_alpha_tokens, summary_tokens)

        feature_frames.append(chunk)

        del article_tokens
        del summary_tokens
        del article_alpha_tokens

    final_df = pd.concat(feature_frames, ignore_index=True)
    metrics = aggregate_metrics(final_df)
    final_df.to_excel(output_path, index=False)

    return final_df, metrics


# ==============================
# MAIN
# ==============================

def main():
    _, metrics_1 = process_dataset(
        "Data/NewsSumm.xlsx",
        "Data/NewsSumm_features.xlsx",
        sample_size=SAMPLE_SIZE,
        chunk_size=CHUNK_SIZE,
    )
    _, metrics_2 = process_dataset(
        "Data/NewsSumm++.xlsx",
        "Data/NewsSumm++_features.xlsx",
        sample_size=SAMPLE_SIZE,
        chunk_size=CHUNK_SIZE,
    )

    comparison = create_comparison(metrics_1, metrics_2)
    comparison.to_excel("Outputs/raw_features/comparison.xlsx", index=False)


if __name__ == "__main__":
    main()
