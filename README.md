# Twitter Sentiment Analysis

A machine learning application that performs real-time sentiment analysis of text using LogisticRegression, scikit-learn, and NLTK, wrapped in a Streamlit web interface.

## Project Structure

```
twitter-sentiment-analysis/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── sentiment_model.py    # Core sentiment analysis functionality
│   └── utils/
│       ├── __init__.py
│       └── data_processing.py    # Data loading and metrics utilities
├── scripts/
│   ├── train_model.py           # Model training script
│   └── evaluate_model.py        # Model evaluation script
├── test/
│   ├── __init__.py
│   └── test_predict.py          # Unit tests
├── app.py                       # Streamlit web interface
├── sentiment_model.pkl          # Trained model
├── test.csv                     # Test dataset
└── requirements.txt            # Project dependencies
```

## Features

- Real-time sentiment analysis of text input
- Classification into Positive, Negative, or Neutral sentiment
- Confidence scores for predictions
- Web interface with emoji feedback
- Batch processing capability for CSV files
- Comprehensive test suite

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

Run the Streamlit app:
```bash
streamlit run app.py
```

### Training

To train a new model:
```bash
python scripts/train_model.py
```

This will:
- Load and preprocess the training dataset
- Train a LogisticRegression model
- Save the model as 'sentiment_model.pkl'
- Generate performance metrics

### Evaluation

To evaluate the model on test data:
```bash
python scripts/evaluate_model.py
```

### Running Tests

```bash
python -m pytest test/test_predict.py -v
```

## Technical Details

### Model Pipeline

```python
Pipeline([
    ('vectorizer', CountVectorizer(max_features=5000)),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ))
])
```

### Text Preprocessing

- Lowercase conversion
- URL removal
- Special character and number removal
- Tokenization (NLTK)
- Stopword removal
- Lemmatization (WordNetLemmatizer)

### Sentiment Classification

- **Positive**: Probability > 60%
- **Negative**: Probability < 40%
- **Neutral**: Probability between 40-60%

### Performance Metrics

- Model evaluation metrics are saved to 'model_evaluation_metrics.txt'
- Test results are saved to 'test_evaluation_results.txt'

## Dependencies

- scikit-learn>=1.3.2
- nltk>=3.8.1
- pandas>=2.1.4
- numpy>=1.26.2
- streamlit>=1.28.2

## Development

### Running Tests

The project uses pytest for testing. Run tests with:
```bash
python -m pytest test/test_predict.py -v
```

Tests cover:
- Model loading
- Sentiment prediction
- Batch processing
- CSV file handling
- Accuracy metrics

### Adding New Features

1. Add core functionality to appropriate module in `src/`
2. Update tests in `test/`
3. Run test suite to ensure nothing breaks
4. Update documentation as needed

## Future Improvements

1. Add support for more languages
2. Implement model versioning
3. Add API endpoint for predictions
4. Improve neutral sentiment detection
5. Add more preprocessing options
6. Implement cross-validation
7. Add support for custom models
8. Enhance error handling and logging

## API Usage

```python
# Load model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
text = "I love the stock market"
prediction = model.predict([preprocess_text(text)])[0]
```
