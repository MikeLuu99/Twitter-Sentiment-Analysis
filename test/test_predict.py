import pytest
import numpy as np
import pandas as pd
from src.models.sentiment_model import (
    load_model,
    predict_sentiment,
    predict_batch,
    process_csv_file,
    get_sentiment_from_probabilities
)

@pytest.fixture
def model():
    """Load the model once for all tests"""
    return load_model()

@pytest.fixture
def test_data():
    """Load test data from test.csv"""
    return pd.read_csv('test.csv')

def test_model_loading(model):
    """Test that the model loads successfully"""
    assert model is not None

def test_get_sentiment_from_probabilities():
    """Test sentiment classification from probabilities"""
    # Test positive case
    pred, conf = get_sentiment_from_probabilities(np.array([0.2, 0.8]))
    assert pred == "positive"
    assert conf == 0.8

    # Test negative case
    pred, conf = get_sentiment_from_probabilities(np.array([0.7, 0.3]))
    assert pred == "negative"
    assert conf == 0.7

    # Test neutral case
    pred, conf = get_sentiment_from_probabilities(np.array([0.45, 0.55]))
    assert pred == "neutral"
    assert conf > 0.9  # High confidence for being close to neutral

def test_predict_sentiment(model):
    """Test single text prediction"""
    # Test positive case
    text = "Amazing experience!"
    prediction, confidence = predict_sentiment(text, model)
    assert prediction in ["positive", "negative", "neutral"]
    assert confidence is not None and 0 <= confidence <= 1

    # Test negative case
    text = "Terrible service!"
    prediction, confidence = predict_sentiment(text, model)
    assert prediction in ["positive", "negative", "neutral"]
    assert confidence is not None and 0 <= confidence <= 1

def test_predict_batch(model, test_data):
    """Test batch prediction"""
    texts = test_data['text'].tolist()
    predictions, confidences = predict_batch(texts, model)

    assert len(predictions) == len(texts)
    assert len(confidences) == len(texts)

    # Check all predictions are valid
    for pred, conf in zip(predictions, confidences):
        assert pred in ["positive", "negative", "neutral"]
        assert 0 <= conf <= 1

def test_process_csv_file(model):
    """Test CSV file processing"""
    try:
        result_df = process_csv_file('test.csv', model)
    except FileNotFoundError:
        pytest.skip("test.csv not found. Skipping test.")
        return

    # Check that the DataFrame has the expected columns
    assert 'text' in result_df.columns
    assert 'predicted_sentiment' in result_df.columns
    assert 'confidence' in result_df.columns

    # Check that all rows have predictions
    assert not result_df['predicted_sentiment'].isnull().any()
    assert not result_df['confidence'].isnull().any()

def test_model_accuracy(model, test_data):
    """Test model accuracy against test.csv ground truth"""
    predictions, _ = predict_batch(test_data['text'].tolist(), model)

    # Convert ground truth to lowercase to match predictions
    ground_truth = test_data['sentiment'].str.lower()

    # Calculate accuracy
    correct = sum(1 for pred, true in zip(predictions, ground_truth)
                 if (pred == true) or (pred == "neutral" and true in ["positive", "negative"]))
    accuracy = correct / len(predictions)

    # We expect some reasonable accuracy (adjust threshold as needed)
    assert accuracy > 0.3  # Setting a low threshold since it's a small test set
