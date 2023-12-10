import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

def load_dataset():
    # Load your dataset
    # Modify the path accordingly
    df = pd.read_csv('data-uas.csv')
    return df

def train_model(df):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df['Artikel'], df['Kategori'], test_size=0.2, random_state=42)

    # Create a pipeline with a TF-IDF vectorizer and a Naive Bayes classifier
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    st.write(f"Akurasi Model: {accuracy:.2f}")

    return model

def predict_category(model, sentence):
    # Predict the category for the input sentence
    category = model.predict([sentence])[0]
    return category

def main():
    st.title("Aplikasi Klasisifikasi Kategori pada Kalimat")
    st.write("Masukkan Kalimat berita untuk Memprediksi Kategorinya:")

    # Load the dataset
    df = load_dataset()

    # Train the model
    model = train_model(df)

    # User input
    sentence = st.text_input("Input Kalimat Berita:", "")

    if st.button("Prediksi Kategori"):
        # Predict and display the category
        category = predict_category(model, sentence)
        st.write(f"Hasil Prediksi: {category}")

if __name__ == "__main__":
    main()
