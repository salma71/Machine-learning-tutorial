#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:16:08 2020

@author: salmaelshahawy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# dataset preprocessing

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values # make sure that x is matrix
y = dataset.iloc[:, 2].values # y is a vector


# split the dataset

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fitting linear regression to the dataset
len_reg = LinearRegression()
len_reg.fit(X, y)


# polynomial regression

# transform x into x to the power  of n to draw the barabola curve
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

len_reg2 = LinearRegression()
len_reg2.fit(X_poly, y)

# visualize the linear regression
plt.scatter(X, y, color = "red")
plt.plot(X, len_reg.predict(X), color = "blue")
plt.title("Truth or bluff(Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()


# visualize the polynomial regression
plt.scatter(X, y, color = "red")
plt.plot(X, len_reg2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Truth or bluff(Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# predicting a new result with linear regression
len_reg.predict([[6.5]]) # predict only value, level6.5 years of experience

# predicting a new result sith polynomial regression 

print(len_reg2.predict(poly_reg.fit_transform([[6.5]])))














