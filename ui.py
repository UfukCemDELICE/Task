import streamlit as st
import requests

st.title("IMDB Sentiment Predictor")

text = st.text_area("Enter your review:")

if st.button("Predict"):
    if text.strip():
        response = requests.post(
            "http://localhost:8000/predict",
            json={"text": text}
        )
        if response.status_code == 200:
            data = response.json()
            st.write(f"**Prediction:** {data['label']}")
            st.write(f"**Confidence:** {data['confidence']:.2f}")
        else:
            st.error("API error!")
