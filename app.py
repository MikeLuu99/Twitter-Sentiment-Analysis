import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Download required NLTK data silently
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove usernames
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags but keep the text
    text = re.sub(r'#', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back into text
    processed_text = ' '.join(tokens)

    return processed_text

@st.cache_resource
def load_model_and_vectorizer():
    """Load the pre-trained model and vectorizer"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None

def train_model_from_data(data_path):
    """Train the model from the CSV data file"""
    try:
        # Try to read the CSV file directly
        df = pd.read_csv(data_path)

        # Check if the expected columns exist
        if 'sentiment' in df.columns and 'text' in df.columns:
            st.success(f"Successfully loaded {len(df)} records from {data_path}")
        else:
            # If CSV has different column names, attempt to map them
            if len(df.columns) >= 2:
                # Assume first column is sentiment and second is text
                df.columns = ['sentiment', 'text'] + list(df.columns[2:])
                st.success(f"Mapped columns in {data_path} to ['sentiment', 'text']")
            else:
                st.error(f"CSV file doesn't have enough columns. Found: {df.columns.tolist()}")
                return None, None
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None, None

    # Map sentiment labels if they are numeric
    if df['sentiment'].dtype != 'object':
        sentiment_mapping = {
            4: 'positive',
            0: 'negative',
            2: 'neutral'
        }
        df['sentiment'] = df['sentiment'].map(sentiment_mapping)

    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Split data
    from sklearn.model_selection import train_test_split
    X = df['processed_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial')
    model.fit(X_train_vec, y_train)

    # Save model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for a given text"""
    if not text or not model or not vectorizer:
        return None, None

    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = model.predict(text_vec)[0]
    probs = model.predict_proba(text_vec)[0]

    # Get confidence for the predicted class
    class_idx = list(model.classes_).index(prediction)
    confidence = probs[class_idx]

    # Get probabilities for all classes
    class_probs = {cls: prob for cls, prob in zip(model.classes_, probs)}

    return prediction, confidence, class_probs

def get_sentiment_emoji(sentiment):
    """Return emoji based on sentiment"""
    if sentiment == 'positive':
        return '😊'
    elif sentiment == 'negative':
        return '😞'
    elif sentiment == 'neutral':
        return '😐'
    return ''

def get_sentiment_color(sentiment):
    """Return color based on sentiment"""
    if sentiment == 'positive':
        return '#28a745'  # Green
    elif sentiment == 'negative':
        return '#dc3545'  # Red
    elif sentiment == 'neutral':
        return '#6c757d'  # Gray
    return 'white'

def main():
    """Main function for the Streamlit app"""
    st.title("Sentiment Analyzer")
    st.markdown("Enter text to analyze its sentiment - positive, negative, or neutral.")

    # Initialize the model
    if not (os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl')):
        with st.spinner("Training model for the first time..."):
            model, vectorizer = train_model_from_data('data.csv')
            if model is not None:
                st.success("Model trained successfully!")
    else:
        model, vectorizer = load_model_and_vectorizer()

    # User input text area
    user_input = st.text_area("Enter your text:", height=150,
                              placeholder="Type or paste text here for sentiment analysis...")

    if st.button("Analyze Sentiment"):
        if user_input:
            with st.spinner("Analyzing sentiment..."):
                result = predict_sentiment(user_input, model, vectorizer)

                # Fix for the tuple unpacking issue
                if len(result) == 3:
                    prediction, confidence, class_probs = result
                else:
                    prediction, confidence = result
                    class_probs = {}

                if prediction:
                    # Display result with styling
                    emoji = get_sentiment_emoji(prediction)
                    color = get_sentiment_color(prediction)

                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white;">
                        <h2 style="margin: 0; display: flex; align-items: center; justify-content: center;">
                            {emoji} {prediction.upper()} {emoji}
                        </h2>
                        <p style="text-align: center; margin-top: 10px;">
                            Confidence: {confidence:.2%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display probability breakdown
                    st.markdown("### Probability Breakdown")

                    for sentiment, prob in class_probs.items():
                        sentiment_emoji = get_sentiment_emoji(sentiment)
                        st.markdown(f"**{sentiment.capitalize()}** {sentiment_emoji}: {prob:.2%}")
                        # Create a progress bar
                        st.progress(float(prob))

                    # Add some examples for the user to try
                    st.markdown("---")
                    st.markdown("### Try these examples:")
                    examples = [
                        "I absolutely love this product! It works perfectly.",
                        "This is the worst experience I've had. Very disappointed.",
                        "The package arrived on time as expected."
                    ]

                    for example in examples:
                        if st.button(example, key=example):
                            st.session_state['user_input'] = example
                            st.experimental_rerun()

                else:
                    st.error("Error analyzing sentiment. Please try again.")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
