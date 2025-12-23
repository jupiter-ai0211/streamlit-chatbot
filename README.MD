# E-commerce Customer Support Chatbot (Streamlit + FAISS)

This repository contains a simple customer-support chatbot that retrieves answers from a small **e-commerce knowledge base** using **Sentence-Transformers embeddings** and a **FAISS** similarity index.

It includes:
- A script to **build** the knowledge base + FAISS index.
- A **Streamlit** chat UI that loads those artifacts and answers user questions via semantic search.

## Project structure

- `create_knowledge_base.py` — Builds `knowledge_base.pkl`, `ecommerce_index.faiss`, and `model_name.txt`. fileciteturn0file0  
- `app.py` — Streamlit app that loads the saved artifacts and serves a chat interface. fileciteturn0file1  

Generated artifacts (created after running the build script):
- `knowledge_base.pkl` — List of `{question, answer, intent, category}` entries.
- `ecommerce_index.faiss` — FAISS vector index for similarity search.
- `model_name.txt` — Name of the embedding model to reload at runtime.

## How it works

1. The build script downloads the Bitext customer support dataset, normalizes a few template placeholders in responses, and stores items as a knowledge base. fileciteturn0file0  
2. It encodes each question with `sentence-transformers/all-MiniLM-L6-v2`, normalizes embeddings, and creates a FAISS `IndexFlatIP` index (cosine similarity via L2-normalization + inner product). fileciteturn0file0  
3. The Streamlit app embeds the user query, searches the top matches, and returns the best answer unless the similarity score is below a threshold (then it uses a keyword-based fallback). fileciteturn0file1  

## Requirements

- Python 3.9+ recommended
- Internet access is required the first time you build the knowledge base (dataset download + model download).

Key Python packages used:
- `streamlit`
- `sentence-transformers`
- `faiss` (CPU)
- `datasets`
- `numpy`

## Setup

Create and activate a virtual environment (recommended), then install dependencies.

### Option A: pip

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -U pip
pip install streamlit sentence-transformers datasets numpy pandas
# FAISS install varies by OS:
pip install faiss-cpu
```

### Option B: conda (often easiest for FAISS)

```bash
conda create -n ecommerce-bot python=3.10 -y
conda activate ecommerce-bot
pip install streamlit sentence-transformers datasets numpy pandas
conda install -c conda-forge faiss-cpu -y
```

## Build the knowledge base

Run:

```bash
python create_knowledge_base.py
```

Expected output files:
- `knowledge_base.pkl`
- `ecommerce_index.faiss`
- `model_name.txt`

If these files do not exist, the Streamlit app will show an error indicating which file is missing. fileciteturn0file1  

## Run the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

## Configuration notes

- Similarity threshold: the app uses a minimum score of `0.3` to decide whether to use the retrieval answer vs. fallback logic. fileciteturn0file1  
- Confidence labels:
  - `High` if score > `0.7`
  - `Medium` otherwise (above threshold)
  - `Low` for fallback/errors fileciteturn0file1  

## Troubleshooting

### “Missing: knowledge_base.pkl / ecommerce_index.faiss / model_name.txt”
You likely haven’t run the build step yet. Run:

```bash
python create_knowledge_base.py
```

### FAISS install issues
- If `pip install faiss-cpu` fails on your platform, use conda-forge as shown above.
- Ensure you are using a supported Python version for the wheels available for your OS.

### Model or dataset download problems
The first run downloads:
- The Bitext dataset via `datasets.load_dataset(...)`. fileciteturn0file0  
- The Sentence-Transformers model `sentence-transformers/all-MiniLM-L6-v2`. fileciteturn0file0  

If you are behind a proxy or have restricted network access, configure your environment accordingly.

## Security and data

This project uses a public dataset and performs local embedding + retrieval. No user queries are sent to external services by default, but your environment may download model/dataset assets during setup.
