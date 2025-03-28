# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:59:37 2025

@author: NAGA LAKSHMI
"""
#importing the libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#loading the dataset
data=pd.read_csv("F:\ml\spam.csv",encoding="latin-1")

#check whether dataset is loaded or not
data.head()
data.info()

#renaming the column names to class and message
data["class"]= data["v1"]
data["message"]=data["v2"]

#store column data in form of array
x = np.array(data["message"])
y = np.array(data["class"])
#calling countvectorizer
cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
#split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#call the multibnomial naive bayes classifier
clf = MultinomialNB()
#fit to training data
clf.fit(X_train,y_train)

#prediction
y_pred=clf.predict((X_test))

#accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


#check whether the model is predicting or not 
sample = input('Enter a message:')
data = cv.transform([sample]).toarray()
print(clf.predict(data))
            
