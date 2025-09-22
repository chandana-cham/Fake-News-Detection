# Fake-News-Detection
Fake News Detection System using Python and Machine Learning. This project preprocesses news text with NLP techniques, vectorizes it using TF-IDF, and classifies news as real or fake using efficient ML models like Naive Bayes. Includes model training, evaluation, and prediction features for practical fake news detection.
Features
Data loading and preprocessing: text cleaning, tokenization, stopwords removal.

Text vectorization with TF-IDF (including n-grams).

Training and evaluation with metrics: accuracy, precision, recall, and confusion matrix.

Save and load trained models for prediction on new news articles.

Includes a prediction function for real-time testing.

Technologies Used
Python 3.6+

pandas

scikit-learn

nltk

joblib

Setup and Installation
Clone this repository:

text
git clone https://github.com/yourusername/fake-news-detection.git
Install dependencies:

text
pip install pandas scikit-learn nltk joblib
Download the ISOT Fake News Dataset and place Fake.csv and True.csv into the project folder.

Run the main script:

text
python fake_news_detection.py
Usage
Train the model on the dataset.

Use the predict_news() function to test new news headlines for classification.

Future Improvements
Integrate transformer-based models like BERT for higher accuracy.

Build a web interface or API for real-time fake news detection.

Add multimodal analysis incorporating images and social engagement data.
