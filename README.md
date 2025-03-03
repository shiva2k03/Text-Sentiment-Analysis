# Sentiment Analysis for E-commerce Reviews

## Overview

This project implements a sentiment analysis model using **Naïve Bayes (MultinomialNB)** to classify e-commerce reviews as positive or negative. The dataset consists of text reviews and their corresponding sentiments. The model utilizes **CountVectorizer** for feature extraction, **SMOTE** for handling class imbalance, and **Scikit-learn** for training and evaluation.

## Features

- Preprocesses text using **bigram** tokenization.
- Uses a **Naïve Bayes** classifier for sentiment prediction.
- Handles **class imbalance** using **SMOTE**.
- Saves and loads trained models for future use.
- Supports **real-time sentiment analysis** via user input.

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install pandas scikit-learn imbalanced-learn joblib
```

## Dataset

The dataset should be in CSV format with the following columns:

- **review**: The text of the review.
- **sentiment**: The corresponding sentiment label (e.g., positive/negative or 1/0).

## Usage

### 1. Prepare the Dataset

Ensure your dataset (`ecommerce_reviews.csv`) is in the same directory as the script and contains the required columns.

### 2. Run the Script

Execute the script to train the sentiment analysis model:

```bash
python sentiment_analysis.py
```

### 3. Analyze Sentiment of New Reviews

Once the model is trained, you can analyze new reviews interactively:

```bash
Enter a review to analyze its sentiment (or type 'exit' to quit): This product is amazing!
The sentiment of the review is: Positive
```

## Model Training & Evaluation

- The dataset is **split (80-20)** into training and testing sets.
- **CountVectorizer** extracts text features using **bigrams**.
- **SMOTE** is applied to handle imbalanced classes.
- The **Naïve Bayes model** is trained and evaluated.
- Performance metrics such as **accuracy score** and **classification report** are displayed.

## Saving and Loading the Model

The trained model and vectorizer are saved for future use:

```bash
sentiment_model.pkl
vectorizer.pkl
```

To load the model later:

```python
analyzer.load_model('sentiment_model.pkl', 'vectorizer.pkl')
```

## Example Predictions

```python
new_reviews = ["This product is great!", "Very poor quality", "It's okay, not too bad."]
sentiments = [analyzer.analyze(review)[0] for review in new_reviews]
```

Expected output:

```
Review: This product is great! -> Sentiment: Positive
Review: Very poor quality -> Sentiment: Negative
Review: It's okay, not too bad. -> Sentiment: Neutral
```

## License

This project is open-source and available for educational purposes.

## Author

Developed by **Shiva Shankaran S**. Feel free to contribute or enhance this project!
