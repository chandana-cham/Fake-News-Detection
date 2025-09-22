import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import nltk
import re
import string

def preprocess_text(text):
    # lowercase
    text = text.lower()
    # remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # remove punctuation and digits
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # remove extra white spaces
    text = ' '.join(text.split())
    return text

# Load dataset
df_fake = pd.read_csv('Fake.csv.zip')
df_real = pd.read_csv('True.csv.zip')

df_fake['label'] = 0
df_real['label'] = 1

df = pd.concat([df_fake, df_real], axis=0).reset_index(drop=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df.sample(frac=0.1, random_state=42)

df['text_clean'] = df['text'].apply(preprocess_text)

X = df['text_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2), stop_words='english')),
    ('clf', MultinomialNB()),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(pipeline, 'fake_news_detector_nb.joblib')

def predict_news(text):
    text = preprocess_text(text)
    pred = pipeline.predict([text])[0]
    return 'Real News' if pred == 1 else 'Fake News'

if __name__ == "__main__":
    sample_news = "Local community organizes charity event to support homeless families"
    print("News Text:", sample_news)
    print("Prediction:", predict_news(sample_news))
