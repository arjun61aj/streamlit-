import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- API Function ---
def fetch_trends(country='india'):
    url = f"https://trendstools.net/json/twitter/{country}"
    resp = requests.get(url).json()
    # Return list of trend names
    return [item['name'] for item in resp]

# --- Fetch 500+ Words ---
def get_corpus(trends):
    return " ".join(trends)

# --- TF-IDF Table ---
def compute_tfidf(corpus):
    vect = TfidfVectorizer(stop_words='english')
    tfidf = vect.fit_transform([corpus])
    df = pd.DataFrame(tfidf.toarray(), columns=vect.get_feature_names_out()).T
    df.columns = ['tfidf']
    df = df.sort_values(by='tfidf', ascending=False)
    return df

# --- Streamlit App ---
st.title("Social Media Big Data Analyzer")

tab1, tab2, tab3 = st.tabs(["Trending API", "TF-IDF Table", "Word Cloud"])

with tab1:
    country = st.text_input("Country for trending data", "india")
    if st.button("Fetch Trending Topics"):
        trends = fetch_trends(country)
        st.write(trends)

with tab2:
    if st.button("Generate TF-IDF Table"):
        trends = fetch_trends("india")
        corpus = get_corpus(trends)
        df_tfidf = compute_tfidf(corpus)
        st.dataframe(df_tfidf.head(50))

with tab3:
    if st.button("Generate Word Cloud"):
        trends = fetch_trends("india")
        corpus = get_corpus(trends)
        wordcloud = WordCloud(width=800, height=400).generate(corpus)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
