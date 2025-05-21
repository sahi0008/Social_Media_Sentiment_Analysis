import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import joblib
from collections import Counter
from wordcloud import WordCloud

# Load data
data_path = r"E:\final year project\imdb_top_1000.csv"
data_df = pd.read_csv(data_path)

print("Column names in the dataset:")
print(data_df.columns)

print("\nFirst few rows of the dataset:")
print(data_df.head())

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Ensure text is string
    text = text.lower().strip()
    return text

# Clean 'Overview' column
data_df['cleaned_review'] = data_df['Overview'].apply(clean_text)

# Create sentiment labels (adjusted threshold for more balance)
data_df['sentiment_label'] = data_df['IMDB_Rating'].apply(lambda x: 1 if x >= 8 else 0)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data_df['cleaned_review'])
y = data_df['sentiment_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show label distributions
print("\nSentiment label distribution in the training set:")
print(y_train.value_counts())

print("\nSentiment label distribution in the test set:")
print(y_test.value_counts())

# Apply SMOTE if classes are imbalanced
if y_train.value_counts().min() / y_train.value_counts().max() < 0.5:
    print("\nHandling class imbalance using SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("\nResampled Sentiment label distribution in the training set:")
    print(pd.Series(y_train).value_counts())

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
train_predictions = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"\nTraining Accuracy: {train_accuracy:.2f}")

test_predictions = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"\nTesting Accuracy: {test_accuracy:.2f}")

print(f"Unique classes in y_test: {y_test.unique()}")
print(f"Unique classes in predictions: {set(test_predictions)}")

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, test_predictions, labels=[0, 1], target_names=['negative', 'positive'], zero_division=1))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, test_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Show basic dataset statistics
print("\nBasic statistics of the dataset:")
print(data_df.describe())

# Display sample cleaned reviews
print("\nSample of cleaned reviews with sentiment labels:")
print(data_df[['cleaned_review', 'sentiment_label']].head())

# Word frequency analysis
word_freq = Counter(" ".join(data_df['cleaned_review']).split())
most_common_words = word_freq.most_common(10)
print("\nMost frequent words in the cleaned reviews:")
print(most_common_words)

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Most Frequent Words in the Reviews')
plt.axis('off')
plt.show()

# Save model and vectorizer
joblib.dump(rf_model, 'sentiment_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("\nModel and vectorizer saved successfully.")
