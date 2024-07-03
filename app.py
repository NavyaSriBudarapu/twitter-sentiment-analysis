# Library imports
import pandas as pd
import numpy as np
import spacy
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import string
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
import nltk

# Load trained Pipeline
vectorizer = joblib.load('vectoriser-ngram-(1,2).pickle')
model = joblib.load('Sentiment-LR.pickle')

stopwords = list(STOP_WORDS)

# Create the app object
app = Flask(__name__)


# creating a function for data cleaning
from custom_tokenizer_function import CustomTokenizer


# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])


def predict():
    text = [str(x) for x in request.form.values()]
    text_vectors = vectorizer.transform(text)
    text_vectors = text_vectors.toarray()


    # Reshape the input array
    text_vectors = np.array(text_vectors)

    predictions = model.predict(text_vectors)
    if predictions==0:
        return render_template('index.html', prediction_text='Negative')
    else:
        return render_template('index.html', prediction_text='Positive')


if __name__ == "__main__":
    app.run(debug=True)
