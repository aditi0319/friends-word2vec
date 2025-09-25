import streamlit as st
import pandas as pd
from gensim.models import Word2Vec


# Load trained model
model = Word2Vec.load("friends_word2vec.model")


# App title + sidebar
st.set_page_config(page_title="Friends Word Similarity Explorer", layout="wide")

st.sidebar.title("🎬 Friends Word2Vec")
st.sidebar.markdown("Find the most related words from Friends dialogues.")

# Sidebar input
word = st.sidebar.text_input("Enter a word:", value="love")
topn = st.sidebar.slider("Number of similar words:", 3, 15, 5)
btn = st.sidebar.button("🔍 Search")

# Main content
st.markdown("<h1 style='text-align:center; color:white;'>💜 F.R.I.E.N.D.S </h1>", unsafe_allow_html=True)

if btn:
    try:
        results = model.wv.most_similar(word, topn=topn)
        df = pd.DataFrame(results, columns=["Word", "Similarity"])

        # Show as chips
        st.subheader(f"Top {topn} words similar to **{word}**:")
        chip_html = " ".join([f"<span style='background-color:black; padding:8px; margin:4px; border-radius:20px; display:inline-block;'>{w}</span>" for w in df["Word"]])
        st.markdown(chip_html, unsafe_allow_html=True)

        # Bar chart
        st.bar_chart(df.set_index("Word"))

    except KeyError:
        st.error(f"'{word}' not in vocabulary!")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>🚀 Powered by Word2Vec • Dataset: Friends Script</p>", unsafe_allow_html=True)
