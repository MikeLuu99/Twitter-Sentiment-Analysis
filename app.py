import streamlit as st
from predict import (
    download_nltk_data,
    load_model,
    predict_sentiment,
)

# Download required NLTK data at startup
download_nltk_data(quiet=True)

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

st.title("Twitter Sentiment Analysis 🐦")

# Load the model at startup
@st.cache_resource
def get_model():
    model = load_model()
    if model is None:
        st.error("Model file not found. Please ensure the model is trained and saved as 'sentiment_model.pkl'")
    return model

model = get_model()

if model is None:
    st.stop()

# Create the text input and store it in session state
if 'tweet_text' not in st.session_state:
    st.session_state.tweet_text = ''

tweet_text = st.text_area(
    "Enter your tweet:",
    height=100,
    key='tweet_input',
    on_change=lambda: setattr(st.session_state, 'tweet_text', st.session_state.tweet_input)
)

# Analysis button
if st.button("Analyze", key='analyze_button'):
    if tweet_text.strip():
        with st.spinner("Analyzing..."):
            prediction, confidence = predict_sentiment(tweet_text, model)

            # Display results with emojis and styling
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sentiment")
                if prediction == "positive":
                    st.markdown("### 😊 Positive")
                elif prediction == "negative":
                    st.markdown("### 😔 Negative")
                else:
                    st.markdown("### 😐 Neutral")

            with col2:
                st.subheader("Confidence")
                if confidence is not None:
                    st.progress(confidence)
                    st.text(f"{confidence:.2%}")
                else:
                    st.text("Confidence: N/A")
    else:
        st.warning("Please enter some text to analyze")
