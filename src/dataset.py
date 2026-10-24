# This script cleans the raw news dataset, extracts lightweight text features,
# and saves the processed output for downstream evaluation.
import re
import pandas as pd
import nltk
import spacy
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from textstat import flesch_reading_ease

# ==============================
# SETUP
# ==============================

nltk.download("punkt", quiet=True)

nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "tagger", "lemmatizer", "attribute_ruler"],
)

# ==============================
# CLEANING
# ==============================

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned_paragraphs = []

    for paragraph in paragraphs:
        paragraph = re.sub(r"[ \t]+", " ", paragraph)
        paragraph = re.sub(r"\s*\n\s*", " ", paragraph)
        paragraph = re.sub(r"\s([?.!,])", r"\1", paragraph)
        paragraph = paragraph.strip()
        if paragraph:
            cleaned_paragraphs.append(paragraph)

    return "\n\n".join(cleaned_paragraphs)


def remove_noise(text):
    text = re.sub(r"(subscribe now|click here|buy now)", "", text, flags=re.I)
    text = re.sub(r"By [A-Z][a-z]+ [A-Z][a-z]+", "", text)
    return text.strip()


def remove_short_entries(df):
    return df[(df["article"].str.len() > 50) & (df["summary"].str.len() > 20)]


def remove_duplicates(df):
    df = df.drop_duplicates(subset=["article"])
    df = df.drop_duplicates(subset=["summary"])
    return df


# ==============================
# FEATURES
# ==============================

def batch_extract_entities(texts):
    entities = []
    for doc in nlp.pipe(texts, batch_size=128):
        entities.append([(ent.text, ent.label_) for ent in doc.ents])
    return entities


def sentence_count(text):
    return len(nltk.sent_tokenize(text))


def keyword_density(text):
    words = [
        w for w in text.lower().split()
        if w not in ENGLISH_STOP_WORDS and w.isalpha()
    ]

    if not words:
        return 0

    freq = Counter(words)
    top = sum(v for _, v in freq.most_common(5))
    return top / len(words)


def readability(text):
    try:
        return flesch_reading_ease(text)
    except:
        return 0


def compression_ratio(article, summary):
    return len(summary) / len(article) if len(article) > 0 else 0


def enhance_summary(entities, summary):
    if not summary:
        return summary

    entity_texts = [ent[0] for ent in entities]

    for ent in entity_texts[:3]:
        if ent not in summary:
            summary += f" {ent}"

    return summary.strip()


# ==============================
# DATA PREP
# ==============================

def load_data(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "article_text": "article",
        "human_summary": "summary",
        "news_category": "topic_label",
    })

    return df[["article", "summary", "topic_label"]]


# ==============================
# CHUNK GENERATOR
# ==============================

def chunk_generator(df, chunk_size):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]


# ==============================
# PROCESS ONE CHUNK
# ==============================

def process_chunk(chunk):
    chunk = chunk.copy()

    chunk["article"] = chunk["article"].astype(str)
    chunk["summary"] = chunk["summary"].astype(str)

    # Cleaning
    chunk["article"] = chunk["article"].apply(clean_text)
    chunk["summary"] = chunk["summary"].apply(clean_text)

    chunk = remove_short_entries(chunk)
    chunk = remove_duplicates(chunk)

    # Noise
    chunk["article"] = chunk["article"].apply(remove_noise)
    chunk["summary"] = chunk["summary"].apply(remove_noise)

    # ==============================
    # FEATURES
    # ==============================

    chunk["Entities"] = batch_extract_entities(chunk["article"].tolist())
    chunk["entity_count"] = chunk["Entities"].apply(len)

    # FAST FEATURES (lightweight)
    chunk["keyword_density"] = chunk["article"].apply(keyword_density)
    chunk["readability_score"] = chunk["article"].apply(readability)
    chunk["compression_ratio"] = chunk.apply(
        lambda x: compression_ratio(x["article"], x["summary"]), axis=1
    )

    # OPTIONAL (comment if slow)
    # chunk["sentence_count"] = chunk["article"].apply(sentence_count)

    # Enhance summary
    chunk["summary"] = chunk.apply(
        lambda x: enhance_summary(x["Entities"], x["summary"]), axis=1
    )

    chunk["Length"] = chunk["article"].str.len()

    return chunk


# ==============================
# MAIN PIPELINE
# ==============================

def process_dataset(input_path, output_path, chunk_size=5000):
    df = load_data(input_path)

    print(f"📊 Total rows: {len(df)}")

    results = []

    for i, chunk in enumerate(chunk_generator(df, chunk_size), start=1):
        print(f"🚀 Processing chunk {i}")

        processed = process_chunk(chunk)
        results.append(processed)

    final_df = pd.concat(results, ignore_index=True)

    final_df["ID"] = final_df.index + 1

    final_df = final_df[
        [
            "ID",
            "article",
            "summary",
            "Entities",
            "topic_label",
            "Length",
            "compression_ratio",
            "readability_score",
            "entity_count",
            "keyword_density",
        ]
    ]

    final_df.to_excel(output_path, index=False)

    print("✅ Done. Saved to:", output_path)


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    process_dataset("Newssumm.xlsx", "NewsSumm++.xlsx")
