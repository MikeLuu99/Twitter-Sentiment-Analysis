import pandas as pd
from datetime import datetime
from predict import load_model, predict_batch

def load_test_data(file_path):
    """Load and prepare test data."""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded the dataset with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise Exception("Could not read the file with any of the attempted encodings")

        # Ensure required columns exist
        required_cols = {'text', 'sentiment'}
        if not required_cols.issubset(df.columns):
            raise ValueError("CSV file must contain 'text' and 'sentiment' columns")

        # Convert sentiment to lowercase
        df['sentiment'] = df['sentiment'].str.lower()

        return df

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def evaluate_model(model, test_df):
    """Evaluate model performance and return metrics."""
    # Get predictions
    predictions, confidences = predict_batch(test_df['text'].tolist(), model)
    
    # Calculate basic accuracy metrics
    total = len(predictions)
    correct = sum(1 for pred, true in zip(predictions, test_df['sentiment']) if pred == true)
    incorrect = total - correct
    accuracy = correct / total

    # Get incorrect predictions for analysis
    incorrect_cases = []
    for i, (pred, true, text) in enumerate(zip(predictions, test_df['sentiment'], test_df['text'])):
        if pred != true:
            incorrect_cases.append({
                'text': text,
                'predicted': pred,
                'actual': true,
                'confidence': confidences[i]
            })

    # Count predictions by class
    pred_counts = {
        'positive': predictions.count('positive'),
        'negative': predictions.count('negative'),
        'neutral': predictions.count('neutral')
    }

    return {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'predictions': pred_counts,
        'incorrect_cases': incorrect_cases
    }

def save_evaluation_results(metrics, output_file):
    """Save evaluation results to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
===========================================
Model Evaluation Results on Test Data
===========================================
Timestamp: {timestamp}

Basic Accuracy Metrics:
- Total predictions: {metrics['total']}
- Correct predictions: {metrics['correct']}
- Incorrect predictions: {metrics['incorrect']}
- Accuracy: {metrics['accuracy']:.2%}

Prediction Distribution:
- Positive: {metrics['predictions']['positive']} ({metrics['predictions']['positive']/metrics['total']*100:.1f}%)
- Negative: {metrics['predictions']['negative']} ({metrics['predictions']['negative']/metrics['total']*100:.1f}%)
- Neutral:  {metrics['predictions']['neutral']} ({metrics['predictions']['neutral']/metrics['total']*100:.1f}%)

Incorrect Predictions:
"""
    
    # Add incorrect predictions
    for case in metrics['incorrect_cases']:
        report += f"""
Text: "{case['text']}"
- Predicted: {case['predicted']}
- Actual: {case['actual']}
- Confidence: {case['confidence']:.2%}
"""
    
    report += "\n==========================================="
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\nEvaluation results have been saved to '{output_file}'")

def main():
    # Load the model
    print("Loading model...")
    model = load_model()
    if model is None:
        print("Error: Could not load model. Please ensure 'sentiment_model.pkl' exists.")
        return

    # Load test data
    print("\nLoading test data...")
    try:
        test_df = load_test_data('test.csv')
    except Exception as e:
        print(f"Error: {str(e)}")
        return

    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_df)

    # Save results
    save_evaluation_results(metrics, 'test_evaluation_results.txt')

if __name__ == "__main__":
    main()
