import pandas as pd
from datetime import datetime

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

def save_metrics(metrics_dict, output_file):
    """Save evaluation metrics to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
===========================================
Model Evaluation Results
===========================================
Timestamp: {timestamp}

Basic Accuracy Metrics:
- Total predictions: {metrics_dict['total']}
- Correct predictions: {metrics_dict['correct']}
- Incorrect predictions: {metrics_dict['incorrect']}
- Accuracy: {metrics_dict['accuracy']:.2%}

Prediction Distribution:
- Positive: {metrics_dict['predictions']['positive']} ({metrics_dict['predictions']['positive']/metrics_dict['total']*100:.1f}%)
- Negative: {metrics_dict['predictions']['negative']} ({metrics_dict['predictions']['negative']/metrics_dict['total']*100:.1f}%)
- Neutral:  {metrics_dict['predictions']['neutral']} ({metrics_dict['predictions']['neutral']/metrics_dict['total']*100:.1f}%)

Incorrect Predictions:
"""
    
    # Add incorrect predictions
    for case in metrics_dict['incorrect_cases']:
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
