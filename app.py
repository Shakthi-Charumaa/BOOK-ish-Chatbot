# app.py
import os
import openai
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = """
You are BookBuddy — a friendly, concise book assistant.
- Help users find books, summarize plots, give author info, suggest reading plans, and recommend books by genre or similar titles.
- When you use items from the catalog, present up to 3 concise recommendations with 1-sentence reasons and a link if available.
- Keep replies short, readable, and friendly.
"""

# Load optional books catalog: books.csv (columns: id,title,author,genre,description,url)
def load_books(csv_path="books.csv"):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    df = df.rename(columns=str.lower)  # normalize
    # ensure common columns
    for c in ["title", "author", "genre", "description", "url"]:
        if c not in df.columns:
            df[c] = ""
    texts = (df["title"].astype(str) + " — " + df["author"].astype(str) + " — " + df["genre"].astype(str) + "\n" + df["description"].astype(str)).tolist()
    return df, texts

df, book_texts = load_books()

# Build embeddings if possible (cost: one embedding per book on startup)
embeddings = None
if df is not None and OPENAI_API_KEY:
    try:
        # batch embeddings (adjust model or batch size as desired)
        BATCH = 50
        embs = []
        for i in range(0, len(book_texts), BATCH):
            batch = book_texts[i:i+BATCH]
            resp = openai.Embedding.create(model="text-embedding-3-small", input=batch)
            embs.extend([d["embedding"] for d in resp["data"]])
        embeddings = np.array(embs)
    except Exception as e:
        print("Embedding build failed:", e)
        embeddings = None

def retrieve_books(query, top_k=3):
    if df is None:
        return []
    if OPENAI_API_KEY and embeddings is not None:
        q_emb = openai.Embedding.create(model="text-embedding-3-small", input=[query])["data"][0]["embedding"]
        sims = cosine_similarity([q_emb], embeddings)[0]
        idx = sims.argsort()[::-1][:top_k]
        return df.iloc[idx].to_dict(orient="records")
    else:
        # simple keyword scoring fallback
        q = query.lower()
        df_local = df.copy()
        df_local["score"] = df_local[["title","author","genre","description"]].apply(
            lambda r: sum(q in str(r[c]).lower() for c in ["title","author","genre","description"]), axis=1
        )
        res = df_local[df_local["score"]>0].sort_values("score", ascending=False).head(top_k)
        return res.to_dict(orient="records")

def format_book_list(books):
    if not books:
        return "No matching books found in catalog."
    lines = []
    for i,b in enumerate(books, start=1):
        lines.append(f"{i}. {b.get('title','')} — {b.get('author','')} ({b.get('genre','')})")
        if b.get("description"):
            lines.append("   " + (b.get("description")[:200] + ("..." if len(b.get("description"))>200 else "")))
        if b.get("url"):
            lines.append("   " + b.get("url"))
    return "\n".join(lines)

def detect_intent(message):
    m = message.lower()
    if any(w in m for w in ["recommend", "suggest", "books like", "best", "looking for"]):
        return "recommend"
    if any(w in m for w in ["summary", "summarize", "plot", "what happens"]):
        return "summary"
    if any(w in m for w in ["who wrote", "author", "written by"]):
        return "author"
    if any(w in m for w in ["reading plan", "plan", "schedule", "30 days", "weekly"]):
        return "plan"
    return "general"

def bookbuddy(message, history):
    intent = detect_intent(message)
    # build base conversation
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    for u,b in history:
        messages.append({"role":"user","content":u})
        messages.append({"role":"assistant","content":b})

    if intent == "recommend":
        # use catalog if possible
        books = retrieve_books(message, top_k=5)
        context = format_book_list(books)
        if books and OPENAI_API_KEY:
            messages.append({"role":"user","content":f"User asked for recommendations: {message}\n\nCatalog results:\n{context}\n\nPlease produce up to 3 recommendations with one-sentence reasons."})
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, temperature=0.7)
            return resp["choices"][0]["message"]["content"]
        elif books:
            return "Catalog matches:\n\n" + context
        else:
            # fallback to general model suggestion if key available
            if OPENAI_API_KEY:
                messages.append({"role":"user","content":f"User asked for recommendations: {message}. No catalog matches. Suggest books generally."})
                resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, temperature=0.7)
                return resp["choices"][0]["message"]["content"]
            else:
                return "No catalog loaded and OpenAI key missing. Upload a books.csv or add OPENAI_API_KEY to the Space secrets."

    elif intent in ("summary", "author"):
        books = retrieve_books(message, top_k=1)
        if books:
            b = books[0]
            if OPENAI_API_KEY:
                messages.append({"role":"user","content":f"User asked: {message}\nUse this book info: Title: {b.get('title')} Author: {b.get('author')}\nDescription: {b.get('description')}"})
                resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, temperature=0.3)
                return resp["choices"][0]["message"]["content"]
            else:
                if intent == "summary":
                    return f"{b.get('title')} — {b.get('description')}"
                else:
                    return f"{b.get('title']} is written by {b.get('author')}."
        else:
            if OPENAI_API_KEY:
                messages.append({"role":"user","content":f"User asked: {message}. No catalog match — answer generally."})
                resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, temperature=0.3)
                return resp["choices"][0]["message"]["content"]
            else:
                return "Couldn't find the book in the catalog. Add books.csv or enable OpenAI key for general answers."

    elif intent == "plan":
        if OPENAI_API_KEY:
            messages.append({"role":"user","content":f"User asked: {message}. Provide a 4-week reading plan relevant to the request."})
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, temperature=0.6)
            return resp["choices"][0]["message"]["content"]
        else:
            return "Reading-plan feature needs the OpenAI key. Add OPENAI_API_KEY in Space settings to enable."

    else:
        # general fallback
        if OPENAI_API_KEY:
            messages.append({"role":"user","content":message})
            resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, temperature=0.8)
            return resp["choices"][0]["message"]["content"]
        else:
            return "BookBuddy offline: upload books.csv to repo for catalog-based answers or add OPENAI_API_KEY in Secrets for general AI responses."

demo = gr.ChatInterface(
    fn=bookbuddy,
    title="📚 BookBuddy",
    description="Ask for book recommendations, summaries, author info, or a reading plan. (Upload books.csv to the Space for catalog-aware replies.)"
)

if __name__ == "__main__":
    demo.launch()
