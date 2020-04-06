#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:19:08 2020

@author: salmaelshahawy
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

regressor = DecisionTreeRegressor(criterion = 'mse', random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results (higher resolution) - none continous
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
