import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient

HF_API_TOKEN = "API_KEY"  #<-- Use your own api(free but rate limit applies)
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

CSV_PATH = "examples/wiki_movie_plots_deduped.csv"
QUERY = "Which movies feature an artificial intelligence antagonist?"
SAMPLE_SIZE = 300
K = 3
CHUNK_WORDS = 300
CHUNK_OVERLAP = 50
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_PATH = "output.json"


def load_and_sample(csv_path, sample_size):
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8")
    cols = {c.lower().strip(): c for c in df.columns}
    title_col = cols.get("title")
    plot_col = cols.get("plot")
    if not title_col or not plot_col:
        raise ValueError("CSV must contain 'Title' and 'Plot' columns (case-insensitive).")
    df = df[[title_col, plot_col]].dropna().rename(columns={title_col: "Title", plot_col: "Plot"})
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    return df


def chunk_text(text, words_per_chunk, overlap):
    words = str(text).split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_corpus(df):
    texts, metadatas = [], []
    for idx, row in df.iterrows():
        title = str(row["Title"])
        plot = str(row["Plot"]).replace("\r\n", " ").replace("\n", " ")
        for i, c in enumerate(chunk_text(plot, CHUNK_WORDS, CHUNK_OVERLAP)):
            texts.append(f"{title} — {c}")
            metadatas.append({"title": title, "doc_index": int(idx), "chunk_index": i})
    return texts, metadatas


def embed_texts(texts, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    return embeddings, model


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def generate_with_hf(question, contexts):
    client = InferenceClient(model=HF_MODEL, token=HF_API_TOKEN)
    context_block = "\n\n---\n\n".join(contexts)
    system = (
        "You are a helpful assistant that answers questions about movie plots using ONLY the provided contexts. "
        "Do NOT invent facts. Return a JSON object with exactly two fields: 'answer' and 'reasoning'. "
        "Respond in JSON only."
    )
    user = (
        f"Question: {question}\n\n"
        f"Contexts:\n{context_block}\n\n"
        "Now reply in JSON."
    )
    resp = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=512,
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    try:
        parsed = json.loads(text)
        answer = parsed.get("answer", "")
    except Exception:
        answer = text
    return answer


def main():
    df = load_and_sample(CSV_PATH, SAMPLE_SIZE)
    texts, metadatas = build_corpus(df)
    embeddings, embed_model = embed_texts(texts, EMBED_MODEL_NAME)
    index = build_faiss_index(embeddings)
    q_emb = embed_model.encode([QUERY], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, K)
    contexts = [texts[int(idx)] for idx in I[0] if idx >= 0]

    try:
        answer = generate_with_hf(QUERY, contexts)
    except Exception as e:
        print("HF generation failed:", e)
        answer = " ".join(contexts[:3])

    output = {"question": QUERY, "answer": answer, "contexts": contexts}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Question:\n" + QUERY)
    print("\nAnswer:\n" + answer)


if __name__ == "__main__":
    main()
