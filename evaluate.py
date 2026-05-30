import os, json, pickle, time
import vertexai
from vertexai.preview import rag
from google import genai

PROJECT_ID  = "rag-pdf-demo-497822"
REGION      = "us-west4"
CORPUS_NAME = "projects/882160541460/locations/us-west4/ragCorpora/4611686018427387904"
GEMINI_KEY  = os.environ.get("GEMINI_API_KEY", "")
TOP_K       = 10
RRF_K       = 60

# Load indexes
with open(os.path.expanduser("~/rag-chatbot/bm25_index.pkl"), "rb") as f:
    bm25_data = pickle.load(f)
bm25        = bm25_data["bm25"]
bm25_chunks = bm25_data["chunks"]
bm25_meta   = bm25_data["meta"]

vertexai.init(project=PROJECT_ID, location=REGION)
gemini = genai.Client(api_key=GEMINI_KEY)

def rrf(dense_results, bm25_results, k=RRF_K):
    scores, texts, files = {}, {}, {}
    for rank, r in enumerate(dense_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + 1 / (k + rank + 1)
        texts[rid] = r["text"]; files[rid] = r["filename"]
    for rank, r in enumerate(bm25_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + 1 / (k + rank + 1)
        texts[rid] = r["text"]; files[rid] = r["filename"]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": rid, "text": texts[rid], "filename": files[rid]} for rid, _ in ranked[:TOP_K]]

def get_answer(question):
    dense_response = rag.retrieval_query(
        rag_corpora=[CORPUS_NAME], text=question, similarity_top_k=TOP_K)
    dense_results = [{"id": f"d_{i}", "text": ctx.text,
                      "filename": ctx.source_uri.split("/")[-1] if ctx.source_uri else "unknown"}
                     for i, ctx in enumerate(dense_response.contexts.contexts)]
    query_tokens = question.lower().split()
    bm25_scores  = bm25.get_scores(query_tokens)
    top_idx      = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]
    bm25_results = [{"id": f"b_{idx}", "text": " ".join(bm25_chunks[idx]),
                     "filename": bm25_meta[idx][0]} for idx in top_idx]
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
    return response.text

def score_answer(question, answer):
    prompt = f"""You are evaluating a RAG chatbot answer. Score strictly 1-5 on each dimension.

QUESTION: {question}
ANSWER: {answer}

1. RELEVANCE (1-5): Does the answer directly address the question?
2. FAITHFULNESS (1-5): Does it stay within retrieved content with no hallucination?
3. COMPLETENESS (1-5): Is it thorough enough to be useful?

Return ONLY valid JSON: {{"relevance": X, "faithfulness": X, "completeness": X, "comment": "one sentence"}}"""
    try:
        r = gemini.models.generate_content(model="gemini-3.5-flash", contents=prompt)
        text = r.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"): text = text[4:]
        scores = json.loads(text.strip())
        scores["avg"] = round((scores["relevance"] + scores["faithfulness"] + scores["completeness"]) / 3, 1)
        return scores
    except Exception as e:
        return {"relevance": 0, "faithfulness": 0, "completeness": 0, "avg": 0, "comment": str(e)}

# Load questions
with open(os.path.expanduser("~/rag-chatbot/eval_questions.json")) as f:
    questions_data = json.load(f)

results = {}
all_scores = []

print("=" * 60)
print("RAG Evaluation — LLM-as-Judge Scoring")
print("=" * 60)

for category, cat_data in questions_data["categories"].items():
    if category == "out_of_scope":
        continue
    questions = cat_data["questions"]
    print(f"\nCategory: {category.upper()} ({len(questions)} questions)")
    cat_results = []

    for i, question in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {question[:55]}...")
        answer  = get_answer(question)
        time.sleep(15)
        scores  = score_answer(question, answer)
        cat_results.append({"question": question, "answer": answer[:200], "scores": scores})
        all_scores.append(scores["avg"])
        print(f"    → Relevance:{scores['relevance']} Faithful:{scores['faithfulness']} Complete:{scores['completeness']} | Avg:{scores['avg']}")
        time.sleep(15)

    avg = round(sum(r["scores"]["avg"] for r in cat_results) / len(cat_results), 1)
    results[category] = {"avg_score": avg, "results": cat_results}

# Print summary table
overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
for cat, data in results.items():
    avg   = data["avg_score"]
    flag  = "✅" if avg >= 4.0 else "⚠" if avg >= 3.0 else "❌"
    print(f"  {flag} {cat:<20} {avg}/5")
print(f"\n  Overall Score: {overall}/5")
print("  ✅ Production-ready" if overall >= 4.0 else "  ⚠ Good baseline" if overall >= 3.0 else "  ❌ Needs improvement")

# Save results
out_path = os.path.expanduser("~/rag-chatbot/eval_results.json")
with open(out_path, "w") as f:
    json.dump({"overall": overall, "categories": results}, f, indent=2)
print(f"\nFull results saved → {out_path}")
