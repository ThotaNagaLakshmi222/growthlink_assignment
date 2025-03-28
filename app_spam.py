# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:35:32 2025

@author: NAGA LAKSHMI
"""

from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
with open('spam_classifier_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    cv = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return "Welcome to the Spam Classifier API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the message from the request
    data = request.get_json()
    message = data.get("message", "")

    # Transform the message using the vectorizer
    message_transformed = cv.transform([message])

    # Predict using the loaded model
    prediction = loaded_model.predict(message_transformed)
    
    # Return the result as JSON
    response = {
        "message": message,
        "prediction": prediction[0]
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
