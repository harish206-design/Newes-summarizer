import streamlit as st
from processor import process_pdf

st.set_page_config(page_title="Top 5 News Extractor", layout="centered")

st.title("🗞️ Top 5 News Extractor from Newspaper PDF")
st.markdown("Upload a newspaper PDF. The app will show the **Top 5 most important news articles** based on content, keywords, and entities.")

uploaded_file = st.file_uploader("Upload your newspaper PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        top_articles = process_pdf(uploaded_file)

    st.success("Top 5 articles extracted:")
    for i, article in enumerate(top_articles, 1):
        st.subheader(f"#{i}. {article['headline']}")
        st.write(article['summary'])
        st.markdown(f"**Tags**: `{', '.join(article['keywords'])}`")

