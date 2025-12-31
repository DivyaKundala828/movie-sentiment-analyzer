import streamlit as st
from transformers import pipeline

# Set page title and layout
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="centered")

# Title and description
st.title("Movie Review Sentiment Analyzer 🚀")
st.write("Type a movie review below and get instant sentiment prediction (Positive or Negative) using AI!")

# Load the sentiment analysis model (runs only once thanks to caching)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

classifier = load_sentiment_model()

# User input box
review = st.text_area(
    "Enter your movie review here...",
    height=150,
    placeholder="Example: This movie was absolutely amazing and full of surprises!"
)

# Analyze button
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text to analyze!")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(review)[0]
            label = result['label']
            score = result['score'] * 100  # Convert to percentage
            
            if label == "POSITIVE":
                st.success(f"**Positive** sentiment! Confidence: {score:.1f}% 😊🎉")
            else:
                st.error(f"**Negative** sentiment! Confidence: {score:.1f}% 😔")
            
            # Show the review for reference
            st.markdown("**Your review:**")
            st.write(review)