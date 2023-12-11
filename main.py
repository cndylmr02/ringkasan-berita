import streamlit as st
from gensim.summarization import summarize

# Title
st.title('Aplikasi Ringkas Berita')

# Input Text Area
berita_input = st.text_area("Masukkan teks berita:")

# Button Summarize
if st.button("Ringkas"):
    if not berita_input:
        st.warning("Masukkan teks berita terlebih dahulu.")
    else:
        # Summarize the input text using Gensim's summarize function
        summary = summarize(berita_input)

        # Display the summarized text
        st.subheader("Ringkasan:")
        st.write(summary)
