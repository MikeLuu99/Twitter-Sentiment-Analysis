import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
def download_nltk_data(quiet=True):
    nltk.download('punkt', quiet=quiet)
    nltk.download('stopwords', quiet=quiet)
    nltk.download('wordnet', quiet=quiet)
    nltk.download('omw-1.4', quiet=quiet)

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

def load_model(model_path='sentiment_model.pkl'):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

def get_sentiment_from_probabilities(probabilities):
    """Convert model probabilities into sentiment prediction and confidence score."""
    neg_prob, pos_prob = probabilities * 100
    
    # Determine sentiment based on probability ranges
    if 40 <= pos_prob <= 60:
        prediction = "neutral"
        confidence = 1 - abs(0.5 - pos_prob/100)  # Convert confidence to be centered around 0.5
    else:
        prediction = "positive" if pos_prob > 60 else "negative"
        confidence = max(probabilities)
    
    return prediction, confidence

def predict_sentiment(text, model):
    """Predict sentiment for a single piece of text."""
    if model is None:
        return None, None

    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Make prediction
    probabilities = model.predict_proba([preprocessed_text])[0]
    return get_sentiment_from_probabilities(probabilities)

def predict_batch(texts, model):
    """Predict sentiment for multiple texts."""
    if model is None:
        return [], []

    # Preprocess all texts
    preprocessed_texts = [preprocess_text(text) for text in texts]

    # Make predictions
    all_probabilities = model.predict_proba(preprocessed_texts)
    
    predictions = []
    confidences = []
    
    for probs in all_probabilities:
        prediction, confidence = get_sentiment_from_probabilities(probs)
        predictions.append(prediction)
        confidences.append(confidence)

    return predictions, confidences

def process_csv_file(file_path, model):
    """Process a CSV file containing text data and return predictions."""
    try:
        df = pd.read_csv(file_path)
        if 'text' not in df.columns:
            raise ValueError("CSV file must contain a 'text' column")

        predictions, confidences = predict_batch(df['text'].tolist(), model)

        # Add predictions to DataFrame
        df['predicted_sentiment'] = predictions
        df['confidence'] = confidences

        return df
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")

def main():
    # Download NLTK data
    download_nltk_data()

    # Load the trained model
    model = load_model()
    if model is None:
        print("Error: Model file 'sentiment_model.pkl' not found. Please ensure you have trained the model first.")
        return

    print("Sentiment Analysis Predictor")
    print("Enter 'q' to quit")
    print("Enter 'f' to predict from a file")
    print("Or type your text for prediction")

    while True:
        user_input = input("\nEnter your choice: ").strip()

        if user_input.lower() == 'q':
            break

        elif user_input.lower() == 'f':
            file_path = input("Enter the path to your CSV file (should have a 'text' column): ")
            try:
                df = process_csv_file(file_path, model)
                
                # Save results
                output_path = file_path.rsplit('.', 1)[0] + '_predictions.csv'
                df.to_csv(output_path, index=False)
                print(f"Predictions saved to: {output_path}")

            except Exception as e:
                print(str(e))

        else:
            # Single text prediction
            prediction, confidence = predict_sentiment(user_input, model)
            if prediction:
                print(f"\nPredicted Sentiment: {prediction}")
                print(f"Confidence: {confidence:.2f}")
            else:
                print("Error: Could not make prediction")

if __name__ == "__main__":
    main()
