# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:20:37 2025

@author: NAGA LAKSHMI
"""

import pickle

# Load the vectorizer
with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    cv = pickle.load(vectorizer_file)

# Load the model
with open('spam_classifier_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Function to predict messages
def predict_message(message):
    # Transform the input message using the loaded vectorizer
    message_transformed = cv.transform([message])
    prediction = loaded_model.predict(message_transformed)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    sample_message = input("Enter a message: ")
    result = predict_message(sample_message)
    print(f"The message is classified as: {result}")
