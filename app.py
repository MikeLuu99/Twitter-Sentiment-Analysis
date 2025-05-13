import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

@st.cache_resource
def load_model():
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is trained and saved as 'sentiment_model.pkl'")
        return None

def predict_sentiment(text, model):
    if model is None:
        return None, None

    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Make prediction
    probabilities = model.predict_proba([preprocessed_text])[0]

    # Get probability for each class
    # Note: First probability is for negative class, second for positive
    neg_prob, pos_prob = probabilities * 100

    print(neg_prob, "Negative")
    print(pos_prob, "Postive")

    # Determine sentiment based on probability ranges
    if 35 <= pos_prob <= 65:
        prediction = "neutral"
        confidence = 1 - abs(0.5 - pos_prob/100)  # Convert confidence to be centered around 0.5
    else:
        prediction = "positive" if pos_prob > 65 else "negative"
        confidence = max(probabilities)

    return prediction, confidence

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

st.title("Twitter Sentiment Analysis 🐦")

# Load the model at startup
model = load_model()

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
                # st.text(f"{confidence:.2%}")
    else:
        st.warning("Please enter some text to analyze")
