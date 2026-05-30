import os, json, pickle
import vertexai
from vertexai.preview import rag
from google import genai

PROJECT_ID  = "rag-pdf-demo-497822"
REGION      = "us-west4"
CORPUS_NAME = "projects/882160541460/locations/us-west4/ragCorpora/4611686018427387904"
GEMINI_KEY  = os.environ.get("GEMINI_API_KEY", "")
TOP_K       = 10
RRF_K       = 60

with open(os.path.expanduser("~/rag-chatbot/bm25_index.pkl"), "rb") as f:
    bm25_data = pickle.load(f)
bm25        = bm25_data["bm25"]
bm25_chunks = bm25_data["chunks"]
bm25_meta   = bm25_data["meta"]

with open(os.path.expanduser("~/rag-chatbot/metadata.json")) as f:
    metadata = json.load(f)

def get_citation(filename):
    return metadata.get(filename, {}).get("citation", filename)

vertexai.init(project=PROJECT_ID, location=REGION)
gemini = genai.Client(api_key=GEMINI_KEY)

def rrf(dense_results, bm25_results, k=RRF_K):
    scores, texts, files = {}, {}, {}
    for rank, r in enumerate(dense_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + 1 / (k + rank + 1)
        texts[rid] = r["text"]
        files[rid] = r["filename"]
    for rank, r in enumerate(bm25_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + 1 / (k + rank + 1)
        texts[rid] = r["text"]
        files[rid] = r["filename"]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": rid, "text": texts[rid], "filename": files[rid]}
            for rid, _ in ranked[:TOP_K]]

def ask(question):
    dense_response = rag.retrieval_query(
        rag_corpora=[CORPUS_NAME],
        text=question,
        similarity_top_k=TOP_K,
    )
    dense_results = []
    for i, ctx in enumerate(dense_response.contexts.contexts):
        filename = ctx.source_uri.split("/")[-1] if ctx.source_uri else "unknown"
        dense_results.append({"id": f"dense_{i}_{filename}", "text": ctx.text, "filename": filename})

    query_tokens = question.lower().split()
    bm25_scores  = bm25.get_scores(query_tokens)
    top_idx      = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]
    bm25_results = [{"id": f"bm25_{idx}_{bm25_meta[idx][0]}", "text": " ".join(bm25_chunks[idx]), "filename": bm25_meta[idx][0]}
                    for idx in top_idx]

    fused = rrf(dense_results, bm25_results)
    context_block = "\n\n---\n\n".join(f"[Source: {r['filename']}]\n{r['text']}" for r in fused)

    prompt = f"""You are a research assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the indexed papers."
Always cite which paper(s) your answer comes from.

CONTEXT:
{context_block}

QUESTION: {question}

ANSWER:"""

    response = gemini.models.generate_content(model="gemini-3.5-flash", contents=prompt)
    sources = list(dict.fromkeys(r["filename"] for r in fused))

    print(f"\nAnswer:\n{response.text}")
    print(f"\nSources ({len(sources)} papers):")
    for s in sources:
        print(f"  • {get_citation(s)}")
    print(f"\n[Dense: {len(dense_results)} | BM25: {len(bm25_results)} → fused: {len(fused)}]")

print("\nHybrid RAG Chatbot ready (Dense + BM25 + RRF)")
print("Type your question or 'quit' to exit\n")
while True:
    try:
        q = input("Your question: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q or q.lower() in ["quit", "exit"]:
        break
    ask(q)
    print()
