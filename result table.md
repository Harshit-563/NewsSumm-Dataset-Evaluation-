# Comparative Evaluation Report

## 1. Methodology

This document compares `NewsSumm` and `NewsSumm++` using semantic, lexical, structural, readability, and model-impact metrics derived from the final merged results in `final_comparison.xlsx`.

The evaluation pipeline used the following stages:

1. Dataset loading and normalization of article-summary pairs.
2. Feature extraction for document length, compression, sentence structure, paragraph count, lexical diversity, entity coverage, and readability.
3. Semantic evaluation using `Semantic Similarity`, `BERTScore`, `Coverage Score`, `Jaccard Similarity`, and `Word Overlap Ratio`.
4. Model-impact evaluation using `ROUGE-1`, `ROUGE-2`, and `ROUGE-L`.
5. Final merging of all metrics into a unified comparison table using [comparison.py](/d:/suvidha/src/comparison.py).

For statistical checking, a paired t-test was computed on the saved per-sample ROUGE outputs from the evaluated files.

## 2. Dataset Changes

`NewsSumm++` represents the processed or engineered version of the original `NewsSumm` dataset. Based on the current pipeline and outputs, the main changes include:

- improved summary coverage of source content
- improved entity retention in summaries
- slightly better readability
- reduced duplicate entries
- altered paragraph structure, with many documents in `NewsSumm++` appearing as a single paragraph

### Important preprocessing note

Paragraph structure should be preserved during preprocessing.

- Do not collapse documents into a single paragraph.
- Preserve structural boundaries before feature extraction.
- Use `text.split("\n\n")` style paragraph boundaries in the pipeline.

## 3. Tables

### 3.1 Overall Highlights

| Key Metric | NewsSumm | NewsSumm++ | Change |
| :-- | --: | --: | --: |
| BERTScore | 0.8878 | 0.8904 | +0.0026 |
| Coverage Score | 0.7747 | 0.7849 | +0.0102 |
| Semantic Similarity | 0.6178 | 0.6258 | +0.0080 |
| ROUGE-1 | 0.4159 | 0.4283 | +0.0124 |
| ROUGE-2 | 0.2472 | 0.2557 | +0.0085 |
| ROUGE-L | 0.3181 | 0.3213 | +0.0032 |
| Entity Coverage in Summary (%) | 0.3611 | 0.4065 | +0.0454 |
| Readability Score | 40.2661 | 41.4140 | +1.1479 |

### 3.2 Relative Improvement (%)

| Metric | Gain (%) |
| :-- | --: |
| ROUGE-1 | +2.98% |
| ROUGE-2 | +3.44% |
| ROUGE-L | +1.01% |
| BERTScore | +0.29% |
| Entity Coverage | +12.57% |

### 3.3 Significance Test

Paired t-test on the current per-sample ROUGE outputs:

| Metric | p-value | Result |
| :-- | --: | :-- |
| ROUGE-1 | 0.1700 | Not significant at `p < 0.05` |
| ROUGE-2 | 0.3306 | Not significant at `p < 0.05` |
| ROUGE-L | 0.7125 | Not significant at `p < 0.05` |

### 3.4 Basic Stats

| Parameter | NewsSumm | NewsSumm++ |
| :-- | --: | --: |
| Avg Document Length (chars) | 2051.2360 | 2095.8524 |
| Avg Document Length (words) | 343.5824 | 352.8454 |
| Avg Summary Length (chars) | 625.7334 | 651.5990 |
| Avg Summary Length (words) | 97.5894 | 101.9940 |
| Duplicate Entries (%) | 0.8800 | 0.0200 |
| Total Samples | 5000 | 5000 |

### 3.5 Quality Metrics

| Parameter | NewsSumm | NewsSumm++ |
| :-- | --: | --: |
| Automated Readability Index | 15.6733 | 15.3188 |
| Readability Score | 40.2661 | 41.4140 |
| SMOG Index | 13.9660 | 13.9954 |

### 3.6 Structure Features

| Parameter | NewsSumm | NewsSumm++ |
| :-- | --: | --: |
| Avg Sentence Length | 26.8381 | 26.9002 |
| Avg Sentences per Doc | 14.3356 | 14.7454 |
| Compression Ratio | 0.5081 | 0.5296 |
| Length Ratio Variance | 0.6938 | 0.7940 |
| Numeric Token Ratio | 0.0280 | 0.0279 |
| Paragraph Count | 5.0758 | 1.0000 |
| Punctuation Density | 0.0240 | 0.0239 |
| Sentence Length Variance | 250.1085 | 224.6306 |
| Uppercase Ratio | 0.0479 | 0.0481 |
| Whitespace Ratio | 0.1636 | 0.1615 |

### 3.7 Entity Features

| Parameter | NewsSumm | NewsSumm++ |
| :-- | --: | --: |
| Avg Entity Count | 34.4080 | 35.4096 |
| Entity Coverage in Summary (%) | 0.3611 | 0.4065 |
| Entity Density | 0.1059 | 0.1071 |
| Missing Entities (%) | 0.6389 | 0.5935 |
| Unique Entity Ratio | 0.7863 | 0.7812 |

### 3.8 Lexical Features

| Parameter | NewsSumm | NewsSumm++ |
| :-- | --: | --: |
| Avg Word Length | 4.9498 | 4.9442 |
| Duplicate Sentence Ratio | 0.0031 | 0.0032 |
| Hapax Legomena Ratio | 0.7180 | 0.7158 |
| Keyword Density | 0.2090 | 0.2067 |
| Keyword Diversity | 4.9978 | 5.0000 |
| Lexical Diversity | 0.5626 | 0.5603 |
| Non-Stopword Ratio | 0.5948 | 0.5957 |
| Redundancy Score | 0.4197 | 0.4224 |
| Stopword Ratio | 0.4052 | 0.4043 |
| Type-Token Ratio | 0.5626 | 0.5603 |
| Unique Word Ratio | 0.5626 | 0.5603 |
| Vocabulary Size | 169.2750 | 172.3240 |

### 3.9 Semantic Features

| Parameter | NewsSumm | NewsSumm++ |
| :-- | --: | --: |
| BERTScore | 0.8878 | 0.8904 |
| Coverage Score | 0.7747 | 0.7849 |
| Jaccard Similarity | 0.3075 | 0.3185 |
| Semantic Similarity | 0.6178 | 0.6258 |
| Word Overlap Ratio | 0.7166 | 0.7262 |

### 3.10 Model Impact

| Parameter | NewsSumm | NewsSumm++ |
| :-- | --: | --: |
| ROUGE-1 | 0.4159 | 0.4283 |
| ROUGE-2 | 0.2472 | 0.2557 |
| ROUGE-L | 0.3181 | 0.3213 |

### 3.11 Dataset Engineering

| Parameter | NewsSumm | NewsSumm++ |
| :-- | --: | --: |
| Average of All Features | 102.9593 | 104.6253 |
| Std Dev of Key Features | 38.9969 | 40.1985 |

### 3.12 Error Analysis

| Error Type | NewsSumm | NewsSumm++ |
| :-- | --: | --: |
| Missing Info | 28% | 19% |
| Entity Errors | 22% | 14% |

## 4. Observations

The comparison indicates that `NewsSumm++` performs better than `NewsSumm` across the main semantic and ROUGE-based evaluation metrics. Gains are visible in `ROUGE-1`, `ROUGE-2`, `ROUGE-L`, `BERTScore`, and `Coverage Score`, which suggests stronger overlap with source content and better summary relevance.

Entity-related metrics also favor `NewsSumm++`. Entity coverage in the summary increases from `0.3611` to `0.4065`, while missing entities decrease from `0.6389` to `0.5935`. This aligns with the manual error-analysis table, where entity errors drop from `22%` to `14%`.

Readability is slightly improved in `NewsSumm++`, with a higher readability score and a lower automated readability index. Duplicate entries are also sharply reduced, which is a positive sign of dataset cleanup quality.

The main structural concern is paragraph preservation. `NewsSumm` shows an average paragraph count of `5.0758`, while `NewsSumm++` shows `1.0000`, which strongly suggests paragraph boundaries were collapsed during preprocessing. This should be treated as a pipeline issue rather than a quality improvement.

Finally, while the average ROUGE scores are higher for `NewsSumm++`, the paired t-test on the currently saved evaluation outputs does not show statistical significance at `p < 0.05`. Therefore, the practical gains are visible, but the statistical claim should be stated carefully in any formal write-up.
