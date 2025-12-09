# =============================
# GOOGLE NEWS INTENT CHECKER
# Reliable RSS-based headline matching
# =============================

import feedparser
import re

# ----------------------------
# CLEAN WORDS
# ----------------------------
def clean_words(text):
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = text.lower().split()

    stop = {
        "the","a","an","and","or","to","of","in","on","for","with","as",
        "is","are","was","were","be","been","has","had","have","by","from",
        "that","this","it","at","news","latest","update"
    }

    return [w for w in text if w not in stop]


# ----------------------------
# FETCH GOOGLE NEWS RESULTS
# Using RSS (never blocked)
# ----------------------------
def google_news_search(query, limit=15):
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
    feed = feedparser.parse(url)

    results = []
    for entry in feed.entries[:limit]:
        results.append({
            "title": entry.title,
            "summary": entry.summary if "summary" in entry else ""
        })
    return results


# ----------------------------
# WORD OVERLAP
# ----------------------------
def overlap_score(headline, text):
    h = set(clean_words(headline))
    t = set(clean_words(text))

    overlap = h.intersection(t)
    return len(overlap), overlap


# ----------------------------
# CONFIDENCE RULES
# ----------------------------
def compute_confidence(overlap):
    if overlap >= 5:
        return 0.90
    elif overlap == 4:
        return 0.75
    elif overlap == 3:
        return 0.55
    elif overlap == 2:
        return 0.30
    elif overlap == 1:
        return 0.20       
    else:
        return 0.00


# ----------------------------
# MAIN FUNCTION
# ----------------------------
def google_verify(headline):

    print("\nSearching Google News...\n")

    queries = [
        headline,
        " ".join(clean_words(headline)),
        headline.split(" ")[0] + " " + headline.split(" ")[1]  # key entities
    ]

    all_results = []
    for q in queries:
        all_results.extend(google_news_search(q))

    best_title = None
    best_overlap_count = 0
    best_overlap_words = set()

    for item in all_results:
        combined = item["title"] + " " + item["summary"]
        count, words = overlap_score(headline, combined)

        if count > best_overlap_count:
            best_overlap_count = count
            best_overlap_words = words
            best_title = item["title"]

    confidence = compute_confidence(best_overlap_count)

    return {
        "best_title": best_title,
        "overlap_count": best_overlap_count,
        "overlap_words": list(best_overlap_words),
        "confidence": confidence
    }



