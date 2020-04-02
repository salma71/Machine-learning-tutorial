# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# importing the dataset
df = pd.read_csv('Data.csv')
# creating matrix of features, matrix of independent variables
X = df.iloc[:, :-1].values # all the columns except the last one
# create in
y = df.iloc[:, 3].values

# handle the missing data

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer = imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])

# dealing with categorical variables

# encode bothe country and purchase into number

labelencoder_X =  LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# using dummy variables to overcome the order issue - so M/C don't count order attribute in the model
columntransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(columntransformer.fit_transform(X),dtype=np.float)

# do the same for the purchase variable

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting the daset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)
# feature scaling -> example, age difference and salary difference
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)




