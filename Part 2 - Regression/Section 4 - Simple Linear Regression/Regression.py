#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:07:38 2020

@author: salmaelshahawy
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 1/3, 
                                                    random_state = 0)

# fitting simple linear regression to the training set
regressor = LinearRegression()
print(regressor.fit(X_train, y_train))

# predicting the Test set result
y_pred = regressor.predict(X_test)

# visualizing the training results

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Salary vs. Experience (training set)')

plt.show()

# visualizing the Test results

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Salary vs. Experience (test set)')

plt.show()
