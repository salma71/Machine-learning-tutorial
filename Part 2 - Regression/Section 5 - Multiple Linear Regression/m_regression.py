#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:39:19 2020

@author: salmaelshahawy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values #IV
y = dataset.iloc[:, 4].values #DV


# dummy variables for categorical data
# avoiding the dummy variable trap, should take one away
# redunacy variable - remove california

X = pd.DataFrame(X)

X = pd.get_dummies(X, columns = [3], drop_first = True)

X = X.to_numpy()

# splitting the data 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# fitting multiple linear regression 

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting on the test set
y_pred = regressor.predict(X_test)

# building the optimal model using backword elimination 

X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# start backword elimination 
#X_opt = X[:, [0,1,2,3,4,5]]
"""
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype = float)
X_Modeled = backwardElimination(X_opt, SL)

print(X_Modeled)
"""
"""
def backwardElimination(x, SL):
    numVars = len(x[0]) # number of variables in my data 
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype = float)
X_Modeled = backwardElimination(X_opt, SL)

"""
X_opt = np.array(X[:, [0,1,2,3,4,5]], dtype = float)

model = sm.OLS(endog = y, exog = X_opt).fit()

print(model.summary())


"""
X_opt = X[:, [0,1,2,3,4]]

X_opt = np.array(X[:, [0,1,2,3,4]], dtype = float)

model = sm.OLS(endog = y, exog = X_opt).fit()

model.summary()


X_opt = X[:, [0,1,2,3]]

X_opt = np.array(X[:, [0,1,2,3]], dtype = float)

model = sm.OLS(endog = y, exog = X_opt).fit()

model.summary()

X_opt = X[:, [0,1,3]]

X_opt = np.array(X[:, [0,1,3]], dtype = float)

model = sm.OLS(endog = y, exog = X_opt).fit()

model.summary()

X_opt = X[:, [0,1]]

X_opt = np.array(X[:, [0,1]], dtype = float)

model = sm.OLS(endog = y, exog = X_opt).fit()

model.summary()
"""