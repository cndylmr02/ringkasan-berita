import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt

# Download data untuk NLTK (dibutuhkan hanya pada instalasi pertama kali)
nltk.download('punkt')

# Title
st.title('Aplikasi Ringkas Berita dengan Cosine Similarity dan Closeness Centrality')

# Input Text Area
berita_input = st.text_area("Masukkan teks berita:")

# Button Summarize
if st.button("Ringkas"):
    if not berita_input:
        st.warning("Masukkan teks berita terlebih dahulu.")
    else:
        # Tokenisasi berita menjadi kalimat
        kalimat = sent_tokenize(berita_input)

        # Vektorisasi TF-IDF
        vectorizer = TfidfVectorizer()
        matriks_tfidf = vectorizer.fit_transform(kalimat)

        # Hitung cosine similarity
        cosine_sim = (matriks_tfidf * matriks_tfidf.T).toarray()

        # Bangun graf dari cosine similarity
        G = nx.Graph()
        for i in range(len(kalimat)):
            for j in range(i+1, len(kalimat)):
                similarity_score = cosine_sim[i][j]
                if similarity_score > 0.2:  # Ambil hanya similarity yang cukup tinggi
                    G.add_edge(i, j, weight=similarity_score)

        # Hitung closeness centrality
        closeness_centrality = nx.closeness_centrality(G)

        # Tampilkan grafik
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=100)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
        plt.title('Graf Cosine Similarity')
        st.pyplot(plt.show())

        # Tampilkan closeness centrality
        st.subheader("Closeness Centrality:")
        for i, centrality_score in closeness_centrality.items():
            st.write(f"Kalimat {i+1}: {centrality_score:.4f}")

        # Ringkasan berdasarkan kalimat yang memiliki cosine similarity tinggi dan closeness centrality rendah
        top_sentences = [kalimat[i] for i in range(len(kalimat)) if closeness_centrality[i] < 0.5]
        summary = ' '.join(top_sentences)

        # Tampilkan ringkasan
        st.subheader("Ringkasan:")
        st.write(summary)
