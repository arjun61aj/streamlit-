import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Social Media Big Data Analyzer", layout="wide")
st.title("Social Media Big Data Analyzer")

# ------------------ SAFE DATA FETCHERS ------------------

def fetch_google_trends_safe():
    url = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=IN"
    r = requests.get(url, timeout=10)

    titles = re.findall(r"<title>(.*?)</title>", r.text)
    titles = titles[1:]  # remove channel title

    # Ensure minimum data
    if len(titles) < 50:
        return []

    return titles[:500]   # 500 topics guaranteed

def fetch_reddit_safe():
    url = "https://www.reddit.com/r/all/top.json?limit=200"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10).json()

    posts = [p["data"]["title"] for p in r["data"]["children"]]
    return posts if len(posts) > 50 else []

# ------------------ TF-IDF SAFE FUNCTION ------------------

def compute_tfidf_safe(text_list):
    text_list = [t for t in text_list if isinstance(t, str) and len(t.strip()) > 3]

    if len(text_list) == 0:
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        token_pattern=r"(?u)\b\w+\b"
    )

    X = vectorizer.fit_transform([" ".join(text_list)])
    words = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]

    df = pd.DataFrame({"Word": words, "TF-IDF Score": scores})
    return df.sort_values(by="TF-IDF Score", ascending=False)

# ------------------ WORD CLOUD ------------------

def show_wordcloud(words):
    wc = WordCloud(
        width=900,
        height=400,
        background_color="black",
        collocations=False
    ).generate(" ".join(words))

    plt.figure(figsize=(10,4))
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(plt)

# ------------------ UI TABS ------------------

tab1, tab2, tab3 = st.tabs(["Twitter", "Reddit", "Facebook"])

# ------------------ TWITTER ------------------
with tab1:
    st.subheader("Twitter Trending Topics (Simulated)")
    if st.button("Fetch Twitter Trends"):
        data = fetch_google_trends_safe()

        if not data:
            st.warning("No trending data available. Try again later.")
        else:
            st.success(f"Fetched {len(data)} topics")
            df = compute_tfidf_safe(data)

            if df is not None:
                st.dataframe(df.head(50))
                show_wordcloud(df["Word"].head(200))
            else:
                st.warning("TF-IDF could not be generated.")

# ------------------ REDDIT ------------------
with tab2:
    st.subheader("Reddit Trending Topics (Live)")
    if st.button("Fetch Reddit Trends"):
        data = fetch_reddit_safe()

        if not data:
            st.warning("No Reddit data available.")
        else:
            st.success(f"Fetched {len(data)} posts")
            df = compute_tfidf_safe(data)

            if df is not None:
                st.dataframe(df.head(50))
                show_wordcloud(df["Word"].head(200))

# ------------------ FACEBOOK ------------------
with tab3:
    st.subheader("Facebook Public Trends (Simulated)")
    if st.button("Fetch Facebook Trends"):
        data = fetch_google_trends_safe()

        if not data:
            st.warning("No Facebook trend data available.")
        else:
            st.success(f"Fetched {len(data)} topics")
            df = compute_tfidf_safe(data)

            if df is not None:
                st.dataframe(df.head(50))
                show_wordcloud(df["Word"].head(200))
