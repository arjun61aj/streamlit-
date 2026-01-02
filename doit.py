import streamlit as st
import feedparser
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Trend WordCloud Analyzer",
    layout="wide"
)

st.title("📊 Trend Word Cloud Analyzer")
st.caption("TF-IDF based word cloud from live Reddit RSS data")

# -------------------------------
# Helper Functions
# -------------------------------
def fetch_reddit_data(query, min_words):
    """
    Fetch Reddit RSS posts until min_words is reached
    """
    url = f"https://www.reddit.com/search.rss?q={query}&sort=hot"
    feed = feedparser.parse(url)

    texts = []
    word_count = 0

    for entry in feed.entries:
        text = entry.title + " " + entry.get("summary", "")
        wc = len(text.split())
        texts.append(text)
        word_count += wc

        if word_count >= min_words:
            break

    return texts, word_count


def generate_tfidf_wordcloud(texts):
    """
    Generate WordCloud using TF-IDF scores
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    scores = tfidf_matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    tfidf_dict = dict(zip(words, scores))

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate_from_frequencies(tfidf_dict)

    return wc


def render_tab(topic, query):
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader(f"📝 Topic: {topic}")
        word_limit = st.slider(
            "Number of words to analyze",
            min_value=500,
            max_value=5000,
            step=500,
            value=1500
        )

        if st.button(f"🔍 Analyze {topic}", use_container_width=True):
            with st.spinner("Fetching live data & generating word cloud..."):
                texts, total_words = fetch_reddit_data(query, word_limit)

                if len(texts) == 0:
                    st.error("No data fetched.")
                    return

                wc = generate_tfidf_wordcloud(texts)

            st.success(f"Analyzed ~{total_words} words")

            with col2:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)


# -------------------------------
# Tabs
# -------------------------------
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
