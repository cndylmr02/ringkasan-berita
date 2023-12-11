import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import networkx as nx
import matplotlib.pyplot as plt

# Download data untuk NLTK (hanya diperlukan pada instalasi pertama kali)
nltk.download('punkt')

# Title
st.title('Aplikasi Ringkas Berita dengan Cosine Similarity, Closeness Centrality, dan Ekstraksi Kata Kunci')

# Input Text Area
berita_input = st.text_area("Masukkan teks berita:")

# Jumlah kata kunci yang akan ditampilkan
jumlah_kata_kunci = st.slider("Jumlah Kata Kunci yang Ditampilkan", min_value=1, max_value=10, value=5)

# Button Summarize
if st.button("Ringkas"):
    if not berita_input:
        st.warning("Masukkan teks berita terlebih dahulu.")
    else:
        # Tokenisasi berita menjadi kalimat
        kalimat = sent_tokenize(berita_input)

        # Vektorisasi TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
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
        for node, centrality_score in closeness_centrality.items():
            st.write(f"Kalimat {node+1}: {centrality_score:.4f}")

        # Ekstraksi kata kunci berdasarkan TF-IDF
        feature_names = vectorizer.get_feature_names()
        tfidf_scores = matriks_tfidf.sum(axis=0).A1
        top_keywords_indices = tfidf_scores.argsort()[-jumlah_kata_kunci:][::-1]
        top_keywords = [feature_names[idx] for idx in top_keywords_indices]

        # Ringkasan berdasarkan kalimat yang memiliki cosine similarity tinggi dan closeness centrality rendah
        top_nodes = [node for node, score in closeness_centrality.items() if score < 0.5]
        top_sentences = [kalimat[node] for node in top_nodes]
        summary = ' '.join(top_sentences)

        # Tampilkan ringkasan, kata kunci, dan nilai kata kunci
        if len(top_sentences) > 0:
            st.subheader("Ringkasan:")
            st.write(summary)
            st.subheader(f"Top {jumlah_kata_kunci} Kata Kunci:")
            for keyword, idx in zip(top_keywords, top_keywords_indices):
                st.write(f"{keyword}: {tfidf_scores[idx]:.4f}")
        else:
            st.warning("Semua kalimat terkoneksi satu sama lain. Tidak ada yang dapat di-ringkas.")
