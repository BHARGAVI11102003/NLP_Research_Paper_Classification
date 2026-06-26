import streamlit as st
import joblib
from keybert import KeyBERT

# Load models
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

kw_model = KeyBERT()

st.title("AI-Based Scientific Paper Classification")

text = st.text_area("Enter Research Paper Title + Abstract")

if st.button("Predict"):

    X = vectorizer.transform([text])

    prediction = model.predict(X)[0]

    st.subheader("Predicted Domain")
    st.write(prediction)

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1,3),
        stop_words="english",
        top_n=10
    )

    st.subheader("Extracted Keywords")

    for k, score in keywords:
        st.write(k)