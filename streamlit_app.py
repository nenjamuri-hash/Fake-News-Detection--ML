import os
import re
import requests
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ============================================================
# MODEL
# ============================================================
MODEL_PATH = r"C:\Users\nikhi\Desktop\fake news new\artifacts\model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

LABELS = ["FAKE", "REAL"]

STOPWORDS = {
    "the","a","an","on","in","to","for","and","new","latest",
    "breaking","issue","issues","says","reports","update","talks",
    "is","are","was","were","be","been","has","had","have","at","from"
}

def clean_words(text):
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text.lower())
    return [w for w in text.split() if w not in STOPWORDS]

# ============================================================
# GOOGLE VERIFICATION
# ============================================================
def google_news_search(query):
    import feedparser
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    return [e.title for e in feed.entries][:10]

def word_overlap(headline, title):
    h = set(clean_words(headline))
    t = set(clean_words(title))
    overlap = h.intersection(t)
    return len(overlap), list(overlap)

def google_verify(headline):

    queries = [
        headline,
        headline + " news",
        " ".join(clean_words(headline)[:5])
    ]

    all_titles = []
    for q in queries:
        all_titles.extend(google_news_search(q))

    best_overlap = 0
    best_title = ""
    best_words = []

    for title in all_titles:
        count, words = word_overlap(headline, title)
        if count > best_overlap:
            best_overlap = count
            best_title = title
            best_words = words

    # Google scoring
    if best_overlap >= 4:
        g_conf = 0.80
    elif best_overlap == 3:
        g_conf = 0.55
    elif best_overlap == 2:
        g_conf = 0.30
    elif best_overlap == 1:
        g_conf = 0.25
    else:
        g_conf = 0.0

    return best_title, best_overlap, best_words, g_conf

# ============================================================
# MODEL PREDICTION
# ============================================================
def predict_news(headline):
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    label = LABELS[torch.argmax(probs).item()]
    confidence = float(probs.max())
    return label, confidence

# ============================================================
# FINAL DECISION
# ============================================================
def final_verdict(ml_label, ml_score, google_score):

    # Convert ML output to REAL probability
    if ml_label == "REAL":
        ml_real = ml_score
    else:
        ml_real = 1 - ml_score

    combined = (ml_real * 0.7) + (google_score * 0.3)

    if combined >= 0.70:
        return "REAL", combined, "🟩 Strong evidence for REAL"
    elif combined >= 0.50:
        return "REAL", combined, "🟢 Moderate evidence for REAL"
    elif combined >= 0.40:
        return "CONFLICT", combined, "🟧 Mixed signals"
    else:
        return "FAKE", combined, "🟥 Likely FAKE"

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detection (ML + Google Verification)")
st.subheader("RoBERTa Model + Google RSS Matching + 70/30 Combined Scoring")

# -----------------------------
# Single Headline Input
# -----------------------------
st.header("🔍 Single Headline Check")
headline = st.text_input("Enter news headline:")

if headline:

    ml_label, ml_score = predict_news(headline)
    st.write(f"### 🤖 Model: **{ml_label} ({ml_score*100:.2f}%)**")

    g_title, g_overlap, g_words, g_score = google_verify(headline)

    st.write("### 🌍 Google Verification")
    st.write(f"Best Match Title: **{g_title}**")
    st.write(f"Overlap Words: **{g_words}**")
    st.write(f"Google Score: **{g_score*100:.1f}%**")

    final_label, combined_score, explanation = final_verdict(ml_label, ml_score, g_score)

    st.write(f"### 🔢 Combined Score (70% ML + 30% Google): **{combined_score*100:.1f}%**")

    if final_label == "REAL":
        st.success(f"### ✅ FINAL VERDICT: REAL\n{explanation}")
    elif final_label == "FAKE":
        st.error(f"### ❌ FINAL VERDICT: FAKE\n{explanation}")
    else:
        st.warning(f"### ⚠ FINAL VERDICT: CONFLICT\n{explanation}")

# -----------------------------
# Batch Prediction (CSV + Excel)
# -----------------------------
st.header("📁 CSV / Excel Batch Prediction")

uploaded = st.file_uploader("Upload file with a 'headline' column", type=["csv", "xlsx"])

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    if "headline" not in df.columns:
        st.error("File must contain a 'headline' column.")
    else:

        st.write("Processing entries...")

        progress = st.progress(0)
        results = []

        total = len(df)

        for i, h in enumerate(df["headline"]):

            ml_label, ml_score = predict_news(h)
            g_title, g_overlap, g_words, g_score = google_verify(h)
            final_label, combined_score, explanation = final_verdict(ml_label, ml_score, g_score)

            results.append([
                h, ml_label, ml_score, g_score, combined_score, final_label, explanation
            ])

            progress.progress((i + 1) / total)

        result_df = pd.DataFrame(results, columns=[
            "headline", "model_label", "model_score",
            "google_score", "combined_score",
            "final_verdict", "reason"
        ])

        st.success("Batch processing complete!")
        st.dataframe(result_df, use_container_width=True)

        st.download_button(
            "📥 Download Results",
            result_df.to_csv(index=False),
            "fake_news_results.csv"
        )
