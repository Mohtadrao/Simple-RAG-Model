# 🎬 Movie Plot Question Answering with FAISS and Hugging Face

This project demonstrates how to build a **retrieval-augmented question-answering system (RAG)** using movie plot data.  
It retrieves the most relevant movie plots from a dataset, embeds them using **Sentence Transformers**, and generates an answer using a **Hugging Face LLM (Mistral-7B-Instruct)**.

---

## 🚀 Features

- Loads and samples movie plot data from CSV  
- Splits long plots into overlapping text chunks  
- Generates embeddings with **SentenceTransformer** (`all-MiniLM-L6-v2`)  
- Builds a **FAISS similarity index** for retrieval  
- Uses **Mistral-7B-Instruct** via **Hugging Face Inference API** for context-based answering  
- Saves the response (question, answer, and retrieved contexts) to a JSON file  

---

## 🧠 Project Workflow

1. **Load Data**  
   Reads movie plot data from a CSV file (`examples/wiki_movie_plots_deduped.csv`).

2. **Chunk Text**  
   Splits long plots into overlapping 300-word chunks (50-word overlap).

3. **Generate Embeddings**  
   Creates semantic vector representations of all chunks using Sentence Transformers.

4. **FAISS Indexing**  
   Builds a FAISS index for fast similarity search.

5. **Retrieve Contexts**  
   Searches for the top-3 relevant chunks related to the query.

6. **LLM Reasoning**  
   Sends contexts and question to Mistral-7B on Hugging Face for JSON-formatted reasoning.

7. **Output**  
   Writes results (answer, contexts) to `output.json`.

---

## 🧩 Requirements

Install dependencies via pip:

```bash
pip install numpy pandas sentence-transformers faiss-cpu huggingface-hub
```

> 🧠 If you have a GPU, you can install `faiss-gpu` instead of `faiss-cpu`.

---

## ⚙️ Configuration

All key parameters are defined at the top of the script:

| Variable | Description | Example |
|-----------|--------------|----------|
| `HF_API_TOKEN` | Hugging Face access token | `"hf_XXXXXXXXXXXXXXXX"` |
| `HF_MODEL` | LLM model to query | `"mistralai/Mistral-7B-Instruct-v0.3"` |
| `CSV_PATH` | Path to movie plot dataset | `"examples/wiki_movie_plots_deduped.csv"` |
| `QUERY` | Question to ask the model | `"Which movies feature an artificial intelligence antagonist?"` |
| `SAMPLE_SIZE` | Number of rows to sample | `300` |
| `K` | Number of contexts to retrieve | `3` |
| `CHUNK_WORDS` | Words per chunk | `300` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `EMBED_MODEL_NAME` | SentenceTransformer model name | `"all-MiniLM-L6-v2"` |
| `OUTPUT_PATH` | Output JSON file path | `"output.json"` |

---

## 🧾 Example Output

```json
{
  "question": "Which movies feature an artificial intelligence antagonist?",
  "answer": "Movies like '2001: A Space Odyssey' and 'The Terminator' feature AI antagonists.",
  "contexts": [
    "The Terminator — In the future, an AI called Skynet becomes self-aware...",
    "2001: A Space Odyssey — The HAL 9000 computer turns against the crew...",
    "Ex Machina — A young programmer tests the intelligence of a humanoid robot..."
  ]
}
```

---

## ▶️ How to Run

1. Place your CSV dataset (e.g. `wiki_movie_plots_deduped.csv`) in the `examples/` directory.  
2. Add your **Hugging Face API token** to `HF_API_TOKEN`.  
3. Run the script:

```bash
python main.py
```

4. The output (`output.json`) will contain the question, model’s answer, and top relevant contexts.

---

## 📁 File Structure

```
project/
├── examples/
│   └── wiki_movie_plots_deduped.csv
├── output.json
├── main.py
└── README.md
```

---

## 🧰 Dependencies Used

- **Python 3.8+**
- **numpy**
- **pandas**
- **sentence-transformers**
- **faiss**
- **huggingface-hub**

---

## ⚠️ Notes

- Make sure your Hugging Face account has **access to Mistral-7B-Instruct** or replace with another open model.
- Large embeddings or datasets can require significant RAM — adjust `SAMPLE_SIZE` as needed.
- Do not share your `HF_API_TOKEN` publicly.

---

## 🧑‍💻 Author

**Muhammad Mohtad Younus**  
AI Engineer & Researcher | FAST NUCES | PAF-IAST  
📧 [GitHub](https://github.com/Mohtadrao) | [LinkedIn](https://linkedin.com/in/mohtad)

---
