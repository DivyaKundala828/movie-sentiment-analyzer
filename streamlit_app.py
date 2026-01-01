import streamlit as st
from transformers import pipeline
import time  # for loading spinner simulation

# Page config
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="centered")

# Title & description
st.title("Movie Review Sentiment Analyzer 🎬✨")
st.markdown("Enter any movie review below — get instant sentiment: **Positive 😊**, **Negative 😔**, or **Neutral 😐**!")

# Load model with caching (runs only once)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

classifier = load_model()

# Input box
review = st.text_area(
    "Type your movie review here...",
    height=120,
    placeholder="Example: This movie was okay, nothing special but enjoyable."
)

# Analyze button
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        # Show loading spinner
        with st.spinner("Analyzing..."):
            time.sleep(1)  # small delay to show spinner (optional)
            result = classifier(review)[0]
            label_id = result['label']  # e.g. 'LABEL_2'
            score = result['score'] * 100

            # Map label to human-readable
            if label_id == 'LABEL_0':
                sentiment = "Negative"
                emoji = "😔"
                color = "error"
            elif label_id == 'LABEL_1':
                sentiment = "Neutral"
                emoji = "😐"
                color = "info"
            else:  # LABEL_2
                sentiment = "Positive"
                emoji = "😊"
                color = "success"

            # Display result
            if color == "success":
                st.success(f"**{sentiment}** {emoji}  Confidence: {score:.1f}%")
            elif color == "error":
                st.error(f"**{sentiment}** {emoji}  Confidence: {score:.1f}%")
            else:
                st.info(f"**{sentiment}** {emoji}  Confidence: {score:.1f}%")

            st.markdown("**Your review:**")
            st.write(review)
