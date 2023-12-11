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
st.title('News Category Classification App')

# Load your news dataset
# Make sure your dataset has 'text' for news content and 'category' for news category
news_dataset = pd.read_csv(r'data-uas.csv')  # Replace 'your_news_dataset.csv' with your actual file path

# Display Dataset
st.write('## News Dataset')
st.dataframe(data=news_dataset)

# Missing Value
st.write('## Missing Value')
st.write(news_dataset.isna().sum())

# Features and Target
features = news_dataset['text']
target = news_dataset['category']

# TF-IDF Vectorization
st.write("## TF-IDF Vectorization")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(features).toarray()
X = pd.DataFrame(X, columns=vectorizer.get_feature_names_out())

# Display TF-IDF DataFrame
st.dataframe(X)

# Add Klasifikasi
algorithm = st.sidebar.selectbox(
    'Choose Algorithm',
    ('Multinomial Naive Bayes',)
)

# Add Parameter
def add_parameters(algorithm):
    params = dict()
    if algorithm == 'Multinomial Naive Bayes':
        alpha = st.sidebar.slider('Alpha', 0.1, 1.0, step=0.1)
        params['alpha'] = alpha
    return params

parameters = add_parameters(algorithm)

# Choose Classification
def choose_classification(algorithm, parameters):
    classifier = None
    if algorithm == 'Multinomial Naive Bayes':
        classifier = MultinomialNB(alpha=parameters['alpha'])
    return classifier

clf = choose_classification(algorithm, parameters)

# Classification Process
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
pred_labels = clf.predict(X_test)
acc = accuracy_score(y_test, pred_labels)

col1, col2 = st.columns(2)
with col1:
    st.write(f'#### Algorithm = {algorithm}')
with col2:
    st.write('#### Accuracy = {}%'.format(round(acc*100)))

precision, recall, threshold = precision_recall_curve(y_test, pred_labels)

with col1:
    st.write(f"Precision  =  {precision[0]}")
with col2:
    st.write(f"Recall  =  {recall[0]}")

# Performance Metrics Graph
st.write("## Performance Metrics Graph")
plt.plot(threshold, precision[:-1], label='Precision')
plt.plot(threshold, recall[:-1], label='Recall')
plt.plot(acc*np.ones_like(threshold), label='Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Performance Metrics')
plt.legend()
st.pyplot(plt.show())

# Confusion Matrix
st.write("## Confusion Matrix")
f, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(confusion_matrix(y_test, pred_labels), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted Category")
plt.ylabel("True Category")
st.pyplot(plt.show())

# Prediction for New Data
st.write('# Predict News Category for New Data')

news_input = st.text_area("Enter the news content:")

# Button Predict
prediction = ''
if st.button("Predict"):
    process_data = vectorizer.transform([news_input]).toarray()
    predict_category = clf.predict(process_data)
    prediction = predict_category[0]
    st.success(f'Predicted Category: {prediction}')
