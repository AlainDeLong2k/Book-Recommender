# 📚 Semantic Book Recommender

An end-to-end semantic book recommendation system built with **Python**, **LangChain**, **Gradio**, and **Hugging Face**, designed to explore and enrich book metadata using modern NLP techniques and deploy a fully interactive app.

🔗 **Live Demo**: [Hugging Face Space](https://alaindelong-demo-book.hf.space/)  
💻 **Dataset**: [7K Books with Metadata on Kaggle](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)

---

## 🚀 Features

- **Vector-based semantic search** using book descriptions
- **Zero-shot category classification** to simplify book genres
- **Emotion detection** based on book summaries
- **Interactive app** with live recommendation via Gradio
- **Deployed on Hugging Face Spaces**

---

## 🧠 Project Workflow

All steps are reproducible through Jupyter notebooks and scripts.

### 1. `1-data-exploration.ipynb`
- Load dataset directly via `kagglehub`
- Clean, merge, and preprocess data into `books_cleaned.csv`

### 2. `2-vector-search.ipynb` → replaced by `build_vector_db.py`
- Builds a vector database using **LangChain** + **Chroma**
- Embeds the `description` column and adds metadata for filtering

### 3. `3-text-classification.ipynb`
- Applies **zero-shot classification** (HuggingFace model)  
- Creates a simplified `simple_categories` column

### 4. `4-sentiment-analysis.ipynb`
- Performs **multi-label emotion detection** (anger, joy, fear, etc.)  
- Scores are aggregated from sentence-level predictions for each book

### 5. `app.py`
- Builds the **Gradio UI** for searching similar books
- Uses **LangChain Embeddings**, **Chroma**, and Hugging Face API

### 6. ✅ Deploy
- App is deployed to Hugging Face Spaces for instant interaction  
- No setup needed — try it live [here](https://alaindelong-demo-book.hf.space/)

---

## 🛠️ Tech Stack

- Python, Pandas, Gradio
- LangChain, ChromaDB
- Hugging Face Transformers & Inference API
- Open-source zero-shot and emotion models