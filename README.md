# RAG vs. Long-Context Prompting for Document Question Answering

**KTH Royal Institute of Technology · II2202 Scientific Method · Autumn 2026**  
**Group 6:** Qiaoyu Wan and Yijia Liu

This project compares **Retrieval-Augmented Generation (RAG)** and **Long-Context Prompting (LCP)** for question answering over scientific papers.

## Research questions

1. **RQ1 — Document length:** How does document length affect RAG/LCP answer quality?
2. **RQ2 — Reasoning type:** How does answer quality differ between single-hop and multi-hop questions?
3. **RQ3 — Efficiency:** How do latency and token consumption change with document length?

## Experiment

- **Dataset:** QASPER v0.3; 300 questions from **300 distinct scientific papers** (272 development, 28 training).
- **Document length:** short (≤ 3,631 proxy tokens), medium (3,632–5,534), long (5,535–16,000), with 100 questions in each group. The proxy tokenizer is `cl100k_base`, not the Gemma model tokenizer.
- **Model:** Gemma 3 4B running locally through Ollama, identical for both methods.
- **Generation:** temperature 0, context window 32,768 tokens, maximum output 160 tokens.
- **RAG:** BM25, 240-word chunks, 40-word overlap, top 5 retrieved chunks.
- **LCP:** entire extracted paper supplied as context without retrieval.
- **Metrics:** token-level answer F1; end-to-end response latency; model input + output tokens.
- **Statistics:** paired bootstrap with 4,000 resamples and 95% percentile confidence intervals for LCP minus RAG F1 differences.

## Results

| Method | Questions | Overall mean F1 | Mean latency | Mean total tokens |
| --- | ---: | ---: | ---: | ---: |
| RAG | 300 | 0.318 | 2.58 s | 1,698 |
| LCP | 300 | 0.340 | 3.16 s | 4,869 |

**RQ1:** The LCP–RAG mean F1 differences were +0.0456 (short), −0.0025 (medium), and +0.0237 (long). **All three 95% confidence intervals include zero**; an observed change in ranking is not evidence of a statistically established document-length crossover.

**RQ2:** The final labels consist of 206 single-hop, 32 multi-hop, and 62 ambiguous questions. The ambiguous group is excluded from the RQ2 comparison, but retained for RQ1 and RQ3. For multi-hop questions, the LCP–RAG mean F1 difference was +0.0722, 95% CI [+0.0010, +0.1520]. This exploratory subgroup has only 32 questions and should be interpreted cautiously.

**RQ3:** Under the tested configuration, RAG used fewer model tokens and achieved lower mean latency, while LCP had modestly higher mean answer F1.

## Repository contents

| Path | Description |
| --- | --- |
| `data/results_300_papers.csv` | **Final 600 model responses** with question/paper IDs, method, predictions, F1/EM, latency, and token counts |
| `data/questions_300_index.csv` | **All 300 selected questions**, paper IDs, document groups, and reference answers; extracted paper texts and supporting passages omitted |
| `data/rq2_labels_300_papers.csv` | Final 300 question-type labels and annotation reasons |
| `data/rq2_summary_300_papers.csv` | Single-hop and multi-hop descriptive statistics |
| `data/rq2_bootstrap_ci.csv` | Final RQ2 paired bootstrap confidence intervals |
| `data/sampling_metadata_300_papers.json` | Sampling and document-length boundary metadata |
| `analysis/analyze_final.py` | Self-contained post-experiment verifier; recomputes paired CIs and regenerates four charts |
| `figures/` | Four **actual final-experiment** F1, latency, and token-usage charts |
| `src/experiment.py` | Initial QASPER / Ollama inference, scoring, and visualization code |
| `src/extend_to_300_papers.py` | Original final 300-distinct-paper extension workflow |
| `src/merge_rq2_final.py` | Original code used to merge manually reviewed RQ2 annotations |

The `analysis/analyze_final.py` utility was prepared from the final released CSV/labels to simplify independent **verification of the reported analysis**. The original inference scripts are preserved separately.

## Reproduce the final analysis

Requires Python 3.10 or newer:

```bash
pip install -r requirements.txt
python analysis/analyze_final.py
```

This script validates all 300 question pairs and question-type assignments, reproduces the overall means and the RQ1/RQ2 paired-bootstrap confidence intervals, and regenerates four charts under `figures/reproduced/`.

**Scope of reproducibility:** The included final data are sufficient to reproduce the published aggregate statistics, paired CIs, and figures. **Rerunning the 600 Ollama inferences from scratch is not yet a one-command operation:** the original 300-paper extension script depends on separate intermediate outputs from earlier experiment phases that are not included here. It also requires the QASPER source data and the specified local model environment. The precise Ollama model build/digest and complete package version lockfile are not archived.

## Dataset attribution and redistribution

QASPER is distributed by the Allen Institute for AI under **CC BY 4.0**. See the [QASPER dataset page](https://huggingface.co/datasets/allenai/qasper), the [license](https://creativecommons.org/licenses/by/4.0/), and its paper:

> Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner (2021). *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers.* NAACL-HLT 2021.

The provided `questions_300_index.csv` is an adapted subset of QASPER annotations, identified by their original question and paper IDs. It omits extracted full-paper texts and the original supporting-evidence passages. The original dataset and individual research papers remain attributable to their respective sources. Do not commit the raw QASPER archive, extracted full-paper files, peer-review materials, credentials, or private student submissions.
