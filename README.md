# Twitter Sentiment Analysis with LogisticRegression

A machine learning application for real-time sentiment analysis of tweets using LogisticRegression, scikit-learn, and NLTK, wrapped in a Streamlit web interface.

## Architecture Overview

### Training Pipeline
```
Raw Tweet → Text Preprocessing → Feature Extraction → Model Training → Evaluation → Serialization
```

#### 1. Text Preprocessing
- Lowercase conversion
- URL removal using regex
- Special character and number removal
- Tokenization using NLTK
- Stopword removal (NLTK English stopwords)
- Lemmatization with WordNetLemmatizer

#### 2. Feature Engineering Pipeline
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

- **CountVectorizer**: Converts text to token counts (max 5000 features)
- **TF-IDF Transformer**: Converts counts to TF-IDF representations
- **LogisticRegression**: Binary classifier with balanced class weights

### Model Details

- **Algorithm**: Logistic Regression
- **Features**: TF-IDF vectors (max 5000 features)
- **Training Split**: 80% training, 20% testing (stratified)
- **Class Weights**: Balanced
- **Parallel Processing**: Enabled (n_jobs=-1)
- **Max Iterations**: 1000

### Prediction Process

1. **Model Loading**:
   ```python
   with open('sentiment_model.pkl', 'rb') as f:
       model = pickle.load(f)
   ```

2. **Single Tweet Prediction**:
   ```python
   # Preprocess tweet
   preprocessed_text = preprocess_text(tweet)
   
   # Get prediction and confidence
   prediction = model.predict([preprocessed_text])[0]
   confidence = max(model.predict_proba([preprocessed_text])[0])
   ```

## Project Structure

```
twitter-sentiment-analysis/
├── train_model.py          # Training script
├── app.py                  # Streamlit interface
├── sentiment_model.pkl     # Serialized model
├── model_evaluation_metrics.txt  # Performance metrics
└── requirements.txt        # Dependencies
```

## Installation & Usage

1. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Train Model**:
   ```bash
   uv run train_model.py
   ```
   This creates:
   - `sentiment_model.pkl`: Trained model
   - `model_evaluation_metrics.txt`: Performance metrics

3. **Run Web Interface**:
   ```bash
   streamlit run app.py
   ```

## Model Artifacts

### sentiment_model.pkl
Contains the serialized scikit-learn Pipeline including:
- Vocabulary from CountVectorizer
- IDF values from TfidfTransformer
- LogisticRegression coefficients and intercepts

### model_evaluation_metrics.txt
Includes:
- Accuracy score
- Classification report (precision, recall, F1)
- Confusion matrix
- Top 10 most important features based on coefficient magnitudes

## Dependencies

- scikit-learn>=1.3.2
- nltk>=3.8.1
- pandas>=2.1.4
- numpy>=1.26.2
- streamlit>=1.28.2

## Data Requirements

Training data should be a CSV with columns:
- target (0 for negative, 4 for positive)
- text (tweet content)
- Additional columns will be filtered out

## Performance Monitoring

The system logs:
- Training accuracy
- Feature importance rankings
- Confusion matrix metrics
- Per-class precision and recall

## Error Handling

- Robust text preprocessing with fallbacks
- Multiple encoding attempts for data loading
- Stratified sampling for balanced evaluation
- Confidence scores for prediction reliability

## Future Improvements

1. Hyperparameter tuning via grid search
2. Cross-validation for more robust evaluation
3. Custom tokenization for Twitter-specific text
4. Ensemble methods for improved accuracy
5. Model versioning and performance tracking

## API Usage

```python
# Load model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
text = "I love this new feature!"
prediction = model.predict([preprocess_text(text)])[0]
```

For more detailed implementation examples, refer to `app.py` and `train_model.py`.