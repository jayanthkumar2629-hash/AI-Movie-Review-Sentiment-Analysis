import streamlit as st
from src.predict import predict_sentiment

st.set_page_config(
    page_title="AI Movie Review Sentiment Analyzer",
    page_icon="🎬"
)

st.title("🎬 AI Movie Review Sentiment Analyzer")

st.write(
    "Analyze the sentiment of a movie review using "
    "a fine-tuned DistilBERT model."
)

review = st.text_area(
    "Enter your movie review:",
    height=180,
    placeholder="Example: The movie was excellent and the acting was amazing."
)

if st.button("🔍 Analyze Sentiment"):

    if not review.strip():

        st.warning("Please enter a review.")

    else:

        label, confidence = predict_sentiment(review)

        st.subheader("Prediction")

        if label.upper() == "POSITIVE":
            st.success("😊 POSITIVE")

        else:
            st.error("😞 NEGATIVE")

        st.write(
            f"Confidence: **{confidence * 100:.2f}%**"
        )