import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load model
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = " ".join([w for w in text.split() if w not in ENGLISH_STOP_WORDS])
    return text

# Prediction
def predict_job(text):
    text = clean_text(text)
    vector = tfidf.transform([text])
    result = model.predict(vector)
    return "Fake Job" if result[0] == 1 else "Real Job"

# UI
st.title("Fake Job Detection System")

user_input = st.text_area("Enter Job Description")

if st.button("Predict"):
    if user_input.strip() != "":
        result = predict_job(user_input)
        st.success(f"Result: {result}")
    else:
        st.warning("Please enter text")