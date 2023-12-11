import streamlit as st
from summarizer import Summarizer

# Title
st.title('Aplikasi Ringkas Berita')

# Input Text Area
berita_input = st.text_area("Masukkan teks berita:")

# Button Summarize
if st.button("Ringkas"):
    if not berita_input:
        st.warning("Masukkan teks berita terlebih dahulu.")
    else:
        # Load pre-trained BERT model for extractive summarization
        model = Summarizer()

        # Summarize the input text
        summary = model(berita_input)

        # Display the summarized text
        st.subheader("Ringkasan:")
        st.write(summary)
