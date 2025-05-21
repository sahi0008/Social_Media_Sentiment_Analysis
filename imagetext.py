# train_image_sentiment_model.py
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pytesseract
import pytesseract
nltk.download('punkt')
nltk.download('stopwords')
df = pd.read_csv(r'E:\final year project\vaccination_tweets.csv')
stop_words = set(stopwords.words("english"))
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    return " ".join([w for w in tokens if w not in stop_words])
df["processed_text"] = df["text"].astype(str).apply(preprocess_text)
df["label"] = df["processed_text"].apply(lambda x: 1 if "good" in x or "thank" in x else 0)  # crude sentiment labels
X = df["processed_text"]
y = df["label"]
vectorizer = CountVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)
from sklearn.model_selection import train_test_split
vectorizer = CountVectorizer(max_features=3000)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer(max_features=3000)
X_vectorized = vectorizer.fit_transform(df["processed_text"])
from sklearn.metrics import classification_report, accuracy_score
model = joblib.load("img_sentiment_model.pkl")
vectorizer = joblib.load("img_vectorizer.pkl")

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

model = joblib.load("img_sentiment_model.pkl")
vectorizer = joblib.load("img_vectorizer.pkl")
from joblib import load

model = load("img_sentiment_model.pkl")
vectorizer = load("img_vectorizer.pkl")

# Example usage
sample_text = ["Your sample text here"]
sample_vector = vectorizer.transform(sample_text)
prediction = model.predict(sample_vector)
print("Prediction:", prediction)
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
import pytesseract

image = Image.open(r'C:\Users\banik\Downloads\1709264998975.jpg')  # Replace with your file
ocr_text = pytesseract.image_to_string(image)
from PIL import Image
import pytesseract

# Optional: For Windows, point to Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"OCR Error: {e}"

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image).strip()
import streamlit as st
option = st.sidebar.selectbox(
    "Choose Sentiment Analysis Type",
    ("✍️ Tweet Sentiment", "🖼 Image Sentiment")
)
st.title("🖼 Image Sentiment Analysis")
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
     # Extract text
extracted_text = pytesseract.image_to_string(image)
st.write("Extracted Text")
st.write(extracted_text)
preprocessed = preprocess_text(extracted_text)
st.write("Preprocessed Text:")
st.write(preprocessed)
if preprocessed.strip():
    vectorized = vectorizer.transform([preprocessed])
    prediction = model.predict(vectorized)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    st.success(f"Predicted Sentiment: {sentiment}")
else:
    st.warning("Could not extract any meaningful text from the image.")
from PIL import Image, ImageEnhance, ImageFilter

# OCR text extraction
extracted_text = extract_text(image)
st.text(f"OCR Raw Output (Length: {len(extracted_text)}):")
st.code(repr(extracted_text))
image = Image.open(r'D:\steamin mugs\IMG20231124165532.jpg')
text = pytesseract.image_to_string(image)
print(repr(text))