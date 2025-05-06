import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

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

def load_data(file_path):
    """Load the data from the CSV file"""
    # Read data directly from CSV
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from CSV: {file_path}")
        print(f"Columns in CSV: {df.columns.tolist()}")

        # Check if the expected columns exist
        if 'sentiment' in df.columns and 'text' in df.columns:
            print(f"Loaded {len(df)} records from the data file")
            print(f"Sentiment distribution: \n{df['sentiment'].value_counts()}")
            return df
        else:
            # If CSV has different column names, attempt to map them
            if len(df.columns) >= 2:
                # Assume first column is sentiment and second is text
                df.columns = ['sentiment', 'text'] + list(df.columns[2:])
                print(f"Renamed columns to: {df.columns.tolist()}")
                print(f"Loaded {len(df)} records from the data file")
                print(f"Sentiment distribution: \n{df['sentiment'].value_counts()}")
                return df
            else:
                raise ValueError(f"CSV file doesn't have the expected columns. Found: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        print("Attempting fallback method...")

        # Fallback to manually parsing the file
        data = []
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    # Skip header
                    next(f, None)
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            sentiment = parts[0]
                            text = ','.join(parts[1:])
                            data.append((sentiment, text))

                print(f"Successfully read file with encoding: {encoding}")
                break
            except Exception as e:
                print(f"Failed with encoding {encoding}: {str(e)}")
                continue

        if not data:
            raise ValueError("Failed to read file with any method")

        df = pd.DataFrame(data, columns=['sentiment', 'text'])
        print(f"Loaded {len(df)} records using fallback method")
        print(f"Sentiment distribution: \n{df['sentiment'].value_counts()}")
        return df

def build_model(data):
    """Build and train the sentiment analysis model"""
    # Preprocess text
    data['processed_text'] = data['text'].apply(preprocess_text)

    # Split data
    X = data['processed_text']
    y = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial')
    model.fit(X_train_vec, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    return model, vectorizer

def save_model(model, vectorizer, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    """Save the trained model and vectorizer"""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_trained_model(model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    """Load the trained model and vectorizer"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for a given text"""
    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = model.predict(text_vec)[0]
    probs = model.predict_proba(text_vec)[0]
    confidence = max(probs)

    return prediction, confidence

if __name__ == "__main__":
    # Define file path
    file_path = 'data/training.1600000.processed.noemoticon.csv'

    # Check if model already exists
    if not (os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl')):
        # Load data
        data = load_data(file_path)

        # Build and train model
        model, vectorizer = build_model(data)

        # Save model and vectorizer
        save_model(model, vectorizer)
        print("Model and vectorizer saved successfully.")
    else:
        print("Model and vectorizer already exist. Loading them...")
        model, vectorizer = load_trained_model()

    # Test with a few examples
    test_texts = [
        "I love this product, it's amazing!",
        "This is the worst experience I've ever had.",
        "The product arrived on time."
    ]

    for text in test_texts:
        sentiment, confidence = predict_sentiment(text, model, vectorizer)
        print(f"Text: {text}")
        print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})")
        print()
