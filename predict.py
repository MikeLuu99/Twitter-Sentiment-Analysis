import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
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
        print(f"Error: Model file '{model_path}' not found. Please ensure you have trained the model first.")
        return None

def predict_sentiment(text, model):
    if model is None:
        return None
    
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    
    # Make prediction
    prediction = model.predict([preprocessed_text])[0]
    probability = max(model.predict_proba([preprocessed_text])[0])
    
    return prediction, probability

def predict_batch(texts, model):
    if model is None:
        return None
    
    # Preprocess all texts
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # Make predictions
    predictions = model.predict(preprocessed_texts)
    probabilities = [max(proba) for proba in model.predict_proba(preprocessed_texts)]
    
    return predictions, probabilities

def main():
    # Load the trained model
    model = load_model()
    if model is None:
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
                df = pd.read_csv(file_path)
                if 'text' not in df.columns:
                    print("Error: CSV file must contain a 'text' column")
                    continue
                    
                predictions, probabilities = predict_batch(df['text'].tolist(), model)
                
                # Add predictions to DataFrame
                df['predicted_sentiment'] = predictions
                df['confidence'] = probabilities
                
                # Save results
                output_path = file_path.rsplit('.', 1)[0] + '_predictions.csv'
                df.to_csv(output_path, index=False)
                print(f"Predictions saved to: {output_path}")
                
            except Exception as e:
                print(f"Error processing file: {str(e)}")
                
        else:
            # Single text prediction
            prediction, probability = predict_sentiment(user_input, model)
            print(f"\nPredicted Sentiment: {prediction}")
            print(f"Confidence: {probability:.2f}")

if __name__ == "__main__":
    main()