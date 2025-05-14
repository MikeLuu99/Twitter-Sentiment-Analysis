import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sentiment_model import preprocess_text
from src.utils.data_processing import load_dataset, save_metrics

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

        # Calculate metrics
        metrics = {
            'total': len(y_test),
            'correct': sum(y_pred == y_test),
            'incorrect': sum(y_pred != y_test),
            'accuracy': accuracy_score(y_test, y_pred),
            'predictions': {
                'positive': sum(y_pred == 'positive'),
                'negative': sum(y_pred == 'negative'),
                'neutral': sum(y_pred == 'neutral')
            },
            'incorrect_cases': []
        }

        # Save evaluation metrics
        save_metrics(metrics, 'model_evaluation_metrics.txt')

        # Save the model
        print("\nSaving the model...")
        with open('sentiment_model.pkl', 'wb') as f:
            pickle.dump(pipeline, f)

        print("Model has been trained and saved as 'sentiment_model.pkl'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
