# RAG vs. Long-Context Prompting for Document Question Answering

A scientific-method course project comparing **retrieval-augmented generation (RAG)** and **long-context prompting (LCP)** on questions about research papers.

**Course:** II2202 Scientific Method, KTH Royal Institute of Technology, Autumn 2026  
**Authors:** Qiaoyu Wan and Yijia Liu

## Research questions

1. How does document length affect answer quality for RAG and LCP?
2. How does answer quality differ between single-hop and multi-hop questions?
3. How do response latency and token usage change with document length?

## Final experimental configuration

- **Dataset:** QASPER v0.3; 300 questions from 300 distinct scientific papers (272 development, 28 training papers)
- **Model:** Gemma 3 4B running locally via Ollama for both approaches
- **RAG:** BM25; 240-word chunks; 40-word overlap; top 5 passages
- **LCP:** full extracted paper as the context
- **Controls:** same model, questions, prompt instructions, and generation settings
- **Metrics:** word/token-overlap F1, end-to-end latency, input + output tokens
- **Statistical analysis:** paired bootstrap confidence intervals for differences in F1

## Summary of final results

| Approach | Mean F1 | Mean latency | Mean tokens |
| --- | ---: | ---: | ---: |
| RAG | 0.318 | 2.58 s | 1,698 |
| LCP | 0.340 | 3.16 s | 4,869 |

LCP achieved a slightly higher mean F1 in the tested configuration, while RAG required less time and fewer tokens. Document-length-group confidence intervals included zero; therefore, the observed ranking reversal does **not** establish a statistically reliable crossover threshold. The multi-hop subset contained only 32 questions, so interpretation is cautious.

See `data/summary_statistics.csv` for aggregate results.

## Files

- `src/experiment.py`: initial local QASPER preparation, retrieval, Ollama inference, scoring, and summaries (150-question phase).
- `src/extend_to_300_papers.py`: later extension workflow combining prior experiment outputs and 28 additional training-split papers to form 300 distinct-paper evaluation instances.
- `requirements.txt`: Python packages used by the scripts.
- `data/summary_statistics.csv`: aggregate results only.

**Reproducibility note:** These scripts reflect the research workflow and are provided for inspection. The 300-paper extension expects intermediate files produced in additional experimental phases. Those intermediate scripts and full per-question data are **not** included here, so the public repository is not yet a one-command reproduction of the complete final experiment.

## Setup

1. Install Python dependencies with `pip install -r requirements.txt`.
2. Install [Ollama](https://ollama.com/), then fetch the model: `ollama pull gemma3:4b`.
3. From the `src/` folder run `python experiment.py prepare` to retrieve the QASPER v0.3 train/dev archive and prepare the initial sample.
4. See the docstrings of each script for available commands and expected input files.

Raw QASPER paper texts, full model outputs, student peer-review documents, and private course submissions are intentionally not published in this repository. Obtain the original QASPER dataset from its official source and respect its license and citation requirements.
