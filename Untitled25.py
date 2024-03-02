#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the diabetes dataset to a pandas DataFrame
insta_dataset = pd.read_csv(r"C:\Users\USER\Downloads\train.csv")
# getting the statistical measures of the data
insta_dataset.describe()
insta_dataset['fake'].value_counts()

insta_dataset.groupby('fake').mean()
# separating the data and labels
X = insta_dataset.drop(columns='fake', axis=1)
Y = insta_dataset['fake']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
Y = insta_dataset['fake']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, stratify=Y, random_state=2)

# Instantiate Logistic Regression classifier
classifier = LogisticRegression()

# Training the Logistic Regression Classifier
classifier.fit(X_train, Y_train)

# Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data: ', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data: ', test_data_accuracy)

input_data = (0, 0, 1, 0, 0, 0, 0, 0, 0, 17, 44)

# Changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('The person the person Instagram id is not fake')
elif prediction[0] == 1:
    print('The person the person Instagram id is fake')

