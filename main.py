import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# SentimentAnalyzer Class
class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))  # Using bigrams
        self.model = MultinomialNB(alpha=0.1)  # Adjusting smoothing parameter
    
    def preprocess(self, text_data):
        return self.vectorizer.transform(text_data)
    
    def train(self, data, labels):
        self.model.fit(data, labels)
    
    def analyze(self, new_text_data):
        if isinstance(new_text_data, str):
            processed_data = self.vectorizer.transform([new_text_data])
        else:
            processed_data = new_text_data
        return self.model.predict(processed_data)
    
    def save_model(self, model_path, vectorizer_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
    
    def load_model(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

# Load e-commerce review data
data_path = 'ecommerce_reviews.csv'
df = pd.read_csv(data_path)

# Check for NaN values and handle them
df.dropna(inplace=True)

# Assume the CSV has 'review' and 'sentiment' columns
text_data = df['review']
labels = df['sentiment']

# Check class distribution
print(df['sentiment'].value_counts())

# If the dataset is too small, consider augmenting it or collecting more data
if df['sentiment'].value_counts().min() < 2:
    print("Not enough data to split. Consider collecting more data.")
else:
    # Initialize the SentimentAnalyzer
    analyzer = SentimentAnalyzer()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)

    # Preprocess text data
    X_train_processed = analyzer.vectorizer.fit_transform(X_train)
    X_test_processed = analyzer.vectorizer.transform(X_test)

    # Balance the dataset using SMOTE
    smote = SMOTE(sampling_strategy='auto', k_neighbors=1)  # Adjust k_neighbors as needed
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

    # Train the model
    analyzer.train(X_train_balanced, y_train_balanced)

    # Evaluate on training set
    train_predictions = analyzer.analyze(X_train_processed)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f'Training Accuracy: {train_accuracy}')
    print(classification_report(y_train, train_predictions, zero_division=0))

    # Evaluate on test set
    test_predictions = analyzer.analyze(X_test_processed)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f'Test Accuracy: {test_accuracy}')
    print(classification_report(y_test, test_predictions, zero_division=0))

    # Save the trained model and vectorizer
    analyzer.save_model('sentiment_model.pkl', 'vectorizer.pkl')

    # Analyze sentiment of new reviews
    new_reviews = ["This product is great!", "Very poor quality", "It's okay, not too bad."]
    sentiments = [analyzer.analyze(review)[0] for review in new_reviews]
    print("Predicted Sentiments:")
    for review, sentiment in zip(new_reviews, sentiments):
        print(f"Review: {review} -> Sentiment: {sentiment}")

    # Interactive user input for sentiment analysis
    while True:
        user_input = input("Enter a review to analyze its sentiment (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        sentiment = analyzer.analyze(user_input)
        print(f'The sentiment of the review is: {sentiment[0]}')
