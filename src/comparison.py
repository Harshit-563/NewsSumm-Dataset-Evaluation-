# This script merges semantic, feature-based, and BERTScore outputs into one
# formatted comparison table for NewsSumm and NewsSumm++.
from pathlib import Path

import pandas as pd


CATEGORY_ORDER = {
    "Basic Stats": 1,
    "Quality Metrics": 2,
    "Structure Features": 3,
    "Entity Features": 4,
    "Lexical Features": 5,
    "Semantic Features": 6,
    "Model Impact": 7,
    "Dataset Engineering": 8,
    "Others": 9,
}


PARAMETER_MAP = {
    "semantic": "Semantic Similarity",
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougel": "ROUGE-L",
    "bertscore": "BERTScore",
    "avg_doc_length_words": "Avg Document Length (words)",
    "avg_summary_length_words": "Avg Summary Length (words)",
    "avg_doc_length_chars": "Avg Document Length (chars)",
    "avg_summary_length_chars": "Avg Summary Length (chars)",
    "compression_ratio": "Compression Ratio",
    "length_ratio_variance": "Length Ratio Variance",
    "avg_word_length": "Avg Word Length",
    "sentence_count": "Avg Sentences per Doc",
    "avg_sentence_length": "Avg Sentence Length",
    "sentence_length_variance": "Sentence Length Variance",
    "vocabulary_size": "Vocabulary Size",
    "unique_word_ratio": "Unique Word Ratio",
    "type_token_ratio": "Type-Token Ratio",
    "lexical_diversity": "Lexical Diversity",
    "hapax_legomena_ratio": "Hapax Legomena Ratio",
    "stopword_ratio": "Stopword Ratio",
    "non_stopword_ratio": "Non-Stopword Ratio",
    "keyword_density": "Keyword Density",
    "keyword_diversity": "Keyword Diversity",
    "avg_entity_count": "Avg Entity Count",
    "entity_density": "Entity Density",
    "unique_entity_ratio": "Unique Entity Ratio",
    "entity_coverage": "Entity Coverage in Summary (%)",
    "missing_entity_ratio": "Missing Entities (%)",
    "paragraph_count": "Paragraph Count",
    "punctuation_density": "Punctuation Density",
    "numeric_token_ratio": "Numeric Token Ratio",
    "uppercase_ratio": "Uppercase Ratio",
    "flesch_reading_ease": "Readability Score",
    "smog_index": "SMOG Index",
    "automated_readability_index": "Automated Readability Index",
    "repetition_ratio": "Redundancy Score",
    "duplicate_sentence_ratio": "Duplicate Sentence Ratio",
    "whitespace_ratio": "Whitespace Ratio",
    "word_overlap_ratio": "Word Overlap Ratio",
    "jaccard_similarity": "Jaccard Similarity",
    "coverage_score": "Coverage Score",
    "total_samples": "Total Samples",
    "duplicate_percentage": "Duplicate Entries (%)",
    "avg_all_features": "Average of All Features",
    "std_dev_key_features": "Std Dev of Key Features",
}


def load_data(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    return df


def extract_bertscore(path):
    df = load_data(path)

    score_column = None
    for column in df.columns:
        if "bertscore" in column.lower():
            score_column = column
            break

    if score_column is None:
        return 0.0

    values = pd.to_numeric(df[score_column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def merge_metrics(comparison_metrics, feature_metrics, bertscore_newssumm, bertscore_newssummpp):
    bertscore_row = pd.DataFrame(
        {
            "Metric": ["BERTScore"],
            "NewsSumm": [bertscore_newssumm],
            "NewsSumm++": [bertscore_newssummpp],
        }
    )

    merged = pd.concat(
        [
            comparison_metrics[["Metric", "NewsSumm", "NewsSumm++"]],
            bertscore_row,
            feature_metrics[["Metric", "NewsSumm", "NewsSumm++"]],
        ],
        ignore_index=True,
    )

    merged["Metric"] = merged["Metric"].astype(str).str.strip()
    merged["metric_key"] = merged["Metric"].str.lower().str.strip()
    merged["NewsSumm"] = pd.to_numeric(merged["NewsSumm"], errors="coerce")
    merged["NewsSumm++"] = pd.to_numeric(merged["NewsSumm++"], errors="coerce")

    merged = merged.drop_duplicates(subset=["metric_key"], keep="first").reset_index(drop=True)

    return merged


def add_categories(df):
    def classify(metric_key):
        if metric_key in {"semantic", "bertscore", "word_overlap_ratio", "jaccard_similarity", "coverage_score"}:
            return "Semantic Features"
        if "rouge" in metric_key:
            return "Model Impact"
        if metric_key in {
            "total_samples",
            "avg_doc_length_words",
            "avg_summary_length_words",
            "avg_doc_length_chars",
            "avg_summary_length_chars",
            "duplicate_percentage",
        }:
            return "Basic Stats"
        if metric_key in {
            "compression_ratio",
            "length_ratio_variance",
            "sentence_count",
            "avg_sentence_length",
            "sentence_length_variance",
            "paragraph_count",
            "punctuation_density",
            "numeric_token_ratio",
            "uppercase_ratio",
            "whitespace_ratio",
        }:
            return "Structure Features"
        if "entity" in metric_key:
            return "Entity Features"
        if metric_key in {
            "vocabulary_size",
            "unique_word_ratio",
            "type_token_ratio",
            "lexical_diversity",
            "hapax_legomena_ratio",
            "stopword_ratio",
            "non_stopword_ratio",
            "keyword_density",
            "keyword_diversity",
            "avg_word_length",
            "repetition_ratio",
            "duplicate_sentence_ratio",
        }:
            return "Lexical Features"
        if metric_key in {
            "flesch_reading_ease",
            "smog_index",
            "automated_readability_index",
        }:
            return "Quality Metrics"
        if metric_key in {"avg_all_features", "std_dev_key_features"}:
            return "Dataset Engineering"
        return "Others"

    df = df.copy()
    df["Category"] = df["metric_key"].apply(classify)
    df["Parameter"] = df["metric_key"].map(PARAMETER_MAP).fillna(df["Metric"])
    df["_sort_order"] = df["Category"].map(CATEGORY_ORDER).fillna(999)
    df = df.sort_values(["_sort_order", "Parameter"], kind="stable").drop(columns=["_sort_order", "Metric", "metric_key"])

    return df[["Category", "Parameter", "NewsSumm", "NewsSumm++"]]


def main():
    root = Path(__file__).resolve().parent.parent

    comparison_metrics = load_data(root / "Outputs" / "rougue+sentiment" / "comparison_metrics.xlsx")
    feature_metrics = load_data(root / "Outputs" / "raw_features" / "comparison.xlsx")

    bertscore_newssumm = extract_bertscore(root / "Outputs" / "bert" / "NewsSumm_bertscore.xlsx")
    bertscore_newssummpp = extract_bertscore(root / "Outputs" / "bert" / "NewsSumm++_bertscore.xlsx")

    merged = merge_metrics(
        comparison_metrics,
        feature_metrics,
        bertscore_newssumm,
        bertscore_newssummpp,
    )

    final_df = add_categories(merged)
    final_df["NewsSumm"] = final_df["NewsSumm"].round(4)
    final_df["NewsSumm++"] = final_df["NewsSumm++"].round(4)
    final_df.to_excel(root / "final_comparison.xlsx", index=False)


if __name__ == "__main__":
    main()
