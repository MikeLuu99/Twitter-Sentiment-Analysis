import pickle
import re
from datetime import datetime

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

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

def load_dataset(file_path):
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, header=None)
                print(f"Successfully loaded the dataset with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise Exception("Could not read the file with any of the attempted encodings")

        # Rename columns
        df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

        # Convert target to binary (0 for negative, 1 for positive)
        df['sentiment'] = df['target'].map({0: 'negative', 4: 'positive'})

        # Keep only necessary columns
        df = df[['text', 'sentiment']]

        return df

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def save_metrics(y_test, y_pred, model, X_test, model_name="Logistic Regression"):
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Get feature importance (for LogisticRegression)
    vectorizer = model.named_steps['vectorizer']
    classifier = model.named_steps['classifier']
    feature_importance = pd.DataFrame({
        'feature': vectorizer.get_feature_names_out(),
        'importance': abs(classifier.coef_[0])
    })
    top_features = feature_importance.nlargest(10, 'importance')

    # Create metrics report
    report = f"""
===========================================
Model Evaluation Metrics
===========================================
Timestamp: {timestamp}
Model: {model_name}

Accuracy: {accuracy:.4f}

Classification Report:
{class_report}

Confusion Matrix:
{conf_matrix}

Top 10 Most Important Features:
{top_features.to_string()}

Note:
- Confusion Matrix Format:
    [[TN, FP],
     [FN, TP]]
- TN: True Negatives
- FP: False Positives
- FN: False Negatives
- TP: True Positives
===========================================
"""

    # Save metrics to file
    with open('model_evaluation_metrics.txt', 'w') as f:
        f.write(report)

    print("\nMetrics have been saved to 'model_evaluation_metrics.txt'")

def main():
    try:
        # Load the dataset
        print("Loading dataset...")
        df = load_dataset('data/training.1600000.processed.noemoticon.csv')

        # Preprocess the tweets
        print("Preprocessing tweets...")
        df['cleaned_text'] = [preprocess_text(text) for text in df['text'].astype(str)]

        # Split features and target
        X = df['cleaned_text']
        y = df['sentiment']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create the pipeline
        print("Creating and training the model pipeline...")
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(max_features=5000)),
            ('tfidf', TfidfTransformer()),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        print("Evaluating the model...")
        y_pred = pipeline.predict(X_test)

        # Save evaluation metrics
        save_metrics(y_test, y_pred, pipeline, X_test)

        # Save the model
        print("\nSaving the model...")
        with open('sentiment_model.pkl', 'wb') as f:
            pickle.dump(pipeline, f)

        print("Model has been trained and saved as 'sentiment_model.pkl'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
