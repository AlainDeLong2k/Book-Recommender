import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd

# from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma

import gradio as gr


load_dotenv()

DEFAULT_IMAGE_URL = "https://placehold.co/180x270/cccccc/000000.png?text=No+Cover"

books = pd.read_csv("datasets/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    DEFAULT_IMAGE_URL,
    books["large_thumbnail"],
)


def load_vector_db():
    # ollama_embedding = OllamaEmbeddings(model="bge-m3", num_thread=4, num_gpu=-1)

    hf_embedding = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        task="feature-extraction",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        provider="hf-inference",
    )

    db_books = Chroma(
        collection_name="books",
        # embedding_function=ollama_embedding,
        embedding_function=hf_embedding,
        persist_directory="./chroma_db",
    )
    return db_books


db_books = load_vector_db()


def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k=16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query=query, k=initial_top_k)
    books_list = [rec.metadata["isbn13"] for rec in recs]

    book_recs = books.set_index("isbn13").loc[books_list].reset_index()

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(
            n=final_top_k
        )
    else:
        book_recs = book_recs.head(n=final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    # print(book_recs["description"])
    return book_recs


def recommend_books(query: str, category: str = None, tone: str = None):
    recommendations = retrieve_semantic_recommendations(
        query=query, category=category, tone=tone, final_top_k=14
    )

    html_output = "<div style='display: flex; flex-wrap: wrap; gap: 16px;'>"

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = str(row["authors"]).split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        html_output += f"""
        <div style='width: 180px; font-family: sans-serif;'>
            <img src="{row['large_thumbnail']}" style='width: 100%; height: auto; border-radius: 8px;' />
            <div style='font-weight: bold; margin-top: 6px; font-size: 14px;'>{row['title']}</div>
            <div style='font-style: italic; color: #666; font-size: 13px;'>{authors_str}</div>
            <div style='font-size: 12px; margin-top: 4px; color: #333;'>{truncated_description}</div>
        </div>
        """

    html_output += "</div>"
    return html_output


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness",
        )
        category_dropdown = gr.Dropdown(
            choices=categories, label="Select a category:", value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, label="Select an emotional tone:", value="All"
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommended Books")
    output = gr.HTML(label="Recommended Books")

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch()
