import matplotlib
matplotlib.use("Agg")  # Tambahkan baris ini

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler

# Title
st.title('Aplikasi Klasifikasi Kategori Berita')

# Load your news dataset
# Pastikan dataset Anda memiliki kolom 'text' untuk konten berita dan 'category' untuk kategori berita
news_dataset = pd.read_csv('data-uas.csv')  # Ganti 'your_news_dataset.csv' dengan jalur file aktual Anda

# Tampilkan Dataset
st.write('## Dataset Berita')
st.dataframe(data=news_dataset)

# Missing Value
st.write('## Nilai yang Hilang')
st.write(news_dataset.isna().sum())

# Fitur dan Target
fitur = news_dataset['text']
target = news_dataset['category']

# Vektorisasi TF-IDF
st.write("## Vektorisasi TF-IDF")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(fitur).toarray()
X = pd.DataFrame(X, columns=vectorizer.get_feature_names_out())

# Tampilkan DataFrame TF-IDF
st.dataframe(X)

# Tambahkan Klasifikasi
algorithm = st.sidebar.selectbox(
    'Pilih Algoritma',
    ('Multinomial Naive Bayes',)
)

# Tambahkan Parameter
def tambah_parameter(algorithm):
    params = dict()
    if algorithm == 'Multinomial Naive Bayes':
        alpha = st.sidebar.slider('Alpha', 0.1, 1.0, step=0.1)
        params['alpha'] = alpha
    return params

parameters = tambah_parameter(algorithm)

# Pilih Klasifikasi
def pilih_klasifikasi(algorithm, parameters):
    classifier = None
    if algorithm == 'Multinomial Naive Bayes':
        classifier = MultinomialNB(alpha=parameters['alpha'])
    return classifier

clf = pilih_klasifikasi(algorithm, parameters)

# Proses Klasifikasi
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
pred_labels = clf.predict(X_test)
acc = accuracy_score(y_test, pred_labels)

col1, col2 = st.columns(2)
with col1:
    st.write(f'#### Algoritma = {algorithm}')
with col2:
    st.write('#### Akurasi = {}%'.format(round(acc*100)))

precision, recall, threshold = precision_recall_curve(y_test, pred_labels)

with col1:
    st.write(f"Presisi  =  {precision[0]}")
with col2:
    st.write(f"Recall  =  {recall[0]}")

# Grafik Metrik Kinerja
st.write("## Grafik Metrik Kinerja")
plt.plot(threshold, precision[:-1], label='Presisi')
plt.plot(threshold, recall[:-1], label='Recall')
plt.plot(acc*np.ones_like(threshold), label='Akurasi')
plt.xlabel('Threshold')
plt.ylabel('Skor')
plt.title('Metrik Kinerja')
plt.legend()
st.pyplot(plt.show())

# Matriks Confusion
st.write("## Matriks Confusion")
f, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(confusion_matrix(y_test, pred_labels), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Kategori Terprediksi")
plt.ylabel("Kategori Sebenarnya")
st.pyplot(plt.show())

# Prediksi untuk Data Baru
st.write('# Prediksi Kategori Berita untuk Data Baru')

berita_input = st.text_area("Masukkan konten berita:")

# Tombol Prediksi
prediksi = ''
if st.button("Prediksi"):
    data_proses = vectorizer.transform([berita_input]).toarray()
    prediksi_kategori = clf.predict(data_proses)
    prediksi = prediksi_kategori[0]
    st.success(f'Kategori Terprediksi: {prediksi}')
