import streamlit as st
import feedparser
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# ------------------------------------
# PAGE CONFIG
# ------------------------------------
st.set_page_config(
    page_title="Word Cloud Analyzer",
    layout="wide"
)

st.title("📊 Reddit Trend Word Cloud")
st.caption("TF-IDF based word cloud from live Reddit RSS data")

# ------------------------------------
# FUNCTIONS
# ------------------------------------
def fetch_reddit_text(query, min_words):
    """
    Fetch Reddit RSS content until min_words is reached
    """
    url = f"https://www.reddit.com/search.rss?q={query}&sort=hot"
    feed = feedparser.parse(url)

    texts = []
    word_count = 0

    for entry in feed.entries:
        content = entry.title + " " + entry.get("summary", "")
        texts.append(content)
        word_count += len(content.split())

        if word_count >= min_words:
            break

    return texts, word_count


def generate_wordcloud(texts):
    """
    Generate TF-IDF WordCloud
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    scores = tfidf_matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    frequencies = dict(zip(words, scores))

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate_from_frequencies(frequencies)

    return wc


def render_tab(topic_name, search_query):
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader(topic_name)
        word_limit = st.slider(
            "Words to analyze",
            min_value=500,
            max_value=5000,
            step=500,
            value=1500,
            key=topic_name
        )

        run = st.button(
            f"Analyze {topic_name}",
            use_container_width=True
        )

    if run:
        with st.spinner("Fetching data & building word cloud..."):
            texts, total_words = fetch_reddit_text(search_query, word_limit)

            if not texts:
                st.error("No data fetched.")
                return

            wc = generate_wordcloud(texts)

        st.success(f"Processed ~{total_words} words")

        with col2:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

# ------------------------------------
# TABS
# ------------------------------------
tab1, tab2, tab3 = st.tabs([
    "💻 Technology",
    "🤖 Artificial Intelligence",
    "🌍 Global News"
])

with tab1:
    render_tab("Technology", "technology")

with tab2:
    render_tab("Artificial Intelligence", "artificial intelligence")

with tab3:
    render_tab("Global News", "world news")
