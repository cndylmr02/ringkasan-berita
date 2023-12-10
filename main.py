import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data-uas.csv')  # Ganti dengan nama file dataset Anda

# Train the model
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Calculate cosine similarity matrix
tfidf_matrix = TfidfVectorizer().fit_transform(df['text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create graph using cosine similarity
G = nx.Graph()
for i in range(len(df)):
    for j in range(i+1, len(df)):
        if cosine_sim[i, j] > 0.7:  # Adjust the threshold as needed
            G.add_edge(df['text'][i], df['text'][j])

# Calculate closeness centrality
closeness_centrality = nx.closeness_centrality(G)

# Streamlit App
st.title("Aplikasi untuk Mengklasifikasikan Kategori pada Suatu Berita")

# User input
user_input = st.text_area("Masukkan Berita:", "")

# Predict category on button click
if st.button("Prediksi Kategori"):
    if user_input:
        # Predict category
        prediction = model.predict([user_input])[0]
        st.write(f"Predicted Category: {prediction}")

        # Display cosine similarity graph
        st.subheader("Cosine Similarity Graph:")
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, font_size=8, node_size=700, node_color='skyblue', font_color='black', font_weight='bold', edge_color='gray')
        st.pyplot(plt)

        # Display closeness centrality
        st.subheader("Closeness Centrality:")
        st.write(closeness_centrality)
    else:
        st.warning("Please enter a news headline.")
