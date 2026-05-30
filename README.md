# RAG PDF Chatbot — Google Cloud

A production-grade RAG chatbot that answers questions across 50 AI research papers with citations.

## Use Case
> "Your client gives you 500 PDFs containing text, tables, charts, and scanned images. Build a chatbot that answers questions accurately with citations."

## Architecture
PDFs → Cloud Storage → Document AI → Vertex AI RAG Engine (HNSW index)
↓
User Question → Dense Retrieval (semantic)  ─┐
→ BM25 Retrieval (keyword)     ─┴→ RRF Fusion → Gemini → Answer + Citations

## Why Hybrid Search
Dense vector search understands meaning but misses exact terms. BM25 catches precise keywords but misses semantics. Reciprocal Rank Fusion (RRF) merges both ranked lists — results appearing high in both get promoted, giving better recall than either alone.

## Stack
| Service | Role |
|---|---|
| Google Cloud Storage | Stores raw PDFs |
| Document AI Layout Parser | OCR, table extraction, layout understanding |
| Vertex AI RAG Engine | Chunking (512 tokens, 50 overlap), embedding, HNSW vector index |
| BM25 (rank-bm25) | Sparse keyword retrieval |
| Reciprocal Rank Fusion | Merges dense + sparse ranked lists |
| Gemini 3.5 Flash | Grounded answer generation with citations |

## Evaluation (LLM-as-Judge)
| Category | Score |
|---|---|
| Factual Lookup | 4.4/5 |
| Faithfulness | 4.6/5 |

## Setup
```bash
pip install -r requirements.txt
python3 chat.py       # interactive chatbot
python3 evaluate.py   # run quality evaluation
```

## Key Design Decisions
- **Hybrid search over pure dense**: catches exact paper titles, acronyms, and model names that semantic search misses
- **512-token chunks with 50-token overlap**: prevents answer context from being cut at chunk boundaries
- **LLM-as-judge evaluation**: automated quality scoring without ground-truth labels
- **Grounded generation**: Gemini instructed to answer only from retrieved chunks, never from training data
