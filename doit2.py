import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Social Media Big Data Analyzer", layout="wide")

st.title("Social Media Big Data Analyzer")

# ---------- DATA FETCHERS ----------

def fetch_reddit():
    url = "https://www.reddit.com/r/all/top.json?limit=100"
    headers = {"User-Agent": "Mozilla/5.0"}
    data = requests.get(url, headers=headers).json()
    return [post["data"]["title"] for post in data["data"]["children"]]

def fetch_google_trends():
    url = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=IN"
    xml = requests.get(url).text
    return xml.split("<title>")[2:502]  # 500 words approx

def compute_tfidf(text_list):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform([" ".join(text_list)])
    scores = X.toarray()[0]
    words = vectorizer.get_feature_names_out()
    return pd.DataFrame({"Word": words, "TF-IDF Score": scores}).sort_values(
        by="TF-IDF Score", ascending=False
    )

def generate_wordcloud(words):
    wc = WordCloud(width=800, height=400, background_color="black").generate(
        " ".join(words)
    )
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(plt)

# ---------- TABS ----------

tab1, tab2, tab3 = st.tabs(["Twitter", "Reddit", "Facebook"])

# ---------- TWITTER TAB ----------
with tab1:
    st.subheader("Twitter Trending Topics (Simulated)")
    if st.button("Fetch Twitter Trends"):
        twitter_data = fetch_google_trends()
        st.success(f"Fetched {len(twitter_data)} topics")
        tfidf_df = compute_tfidf(twitter_data)
        st.dataframe(tfidf_df.head(50))
        generate_wordcloud(tfidf_df["Word"].head(200))

# ---------- REDDIT TAB ----------
with tab2:
    st.subheader("Reddit Trending Topics (Live Data)")
    if st.button("Fetch Reddit Trends"):
        reddit_data = fetch_reddit()
        st.success(f"Fetched {len(reddit_data)} posts")
        tfidf_df = compute_tfidf(reddit_data)
        st.dataframe(tfidf_df.head(50))
        generate_wordcloud(tfidf_df["Word"].head(200))

# ---------- FACEBOOK TAB ----------
with tab3:
    st.subheader("Facebook Public Trends (Simulated)")
    if st.button("Fetch Facebook Trends"):
        fb_data = fetch_google_trends()
        st.success(f"Fetched {len(fb_data)} topics")
        tfidf_df = compute_tfidf(fb_data)
        st.dataframe(tfidf_df.head(50))
        generate_wordcloud(tfidf_df["Word"].head(200))
