import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import xml.etree.ElementTree as ET

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Trend Word Cloud",
    layout="wide"
)

st.title("📊 Trend Word Cloud Analyzer")
st.caption("Live Reddit trends | TF-IDF | Streamlit")

# ----------------------------------
# FUNCTIONS
# ----------------------------------
def fetch_reddit_rss(query, min_words):
    url = f"https://www.reddit.com/search.rss?q={query}&sort=hot"

    response = urllib.request.urlopen(url)
    xml_data = response.read()

    root = ET.fromstring(xml_data)

    texts = []
    word_count = 0

    for item in root.findall(".//item"):
        title = item.findtext("title", "")
        desc = item.findtext("description", "")

        text = f"{title} {desc}"
        texts.append(text)
        word_count += len(text.split())

        if word_count >= min_words:
            break

    return texts, word_count


def create_wordcloud(texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf = vectorizer.fit_transform(texts)
    scores = tfidf.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    freq = dict(zip(words, scores))

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate_from_frequencies(freq)

    return wc


def tab_ui(topic, query):
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader(topic)
        word_limit = st.slider(
            "Number of words",
            500, 5000, 1500, 500,
            key=topic
        )

        run = st.button(
            f"Analyze {topic}",
            use_container_width=True
        )

    if run:
        with st.spinner("Fetching live data..."):
            texts, total_words = fetch_reddit_rss(query, word_limit)

            if not texts:
                st.error("No data found")
                return

            wc = create_wordcloud(texts)

        st.success(f"Processed ~{total_words} words")

        with col2:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

# ----------------------------------
# TABS
# ----------------------------------
tab1, tab2, tab3 = st.tabs([
    "💻 Technology",
    "🤖 Artificial Intelligence",
    "🌍 World News"
])

with tab1:
    tab_ui("Technology", "technology")

with tab2:
    tab_ui("Artificial Intelligence", "artificial intelligence")

with tab3:
    tab_ui("World News", "world news")
