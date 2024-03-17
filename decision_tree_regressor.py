#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

from ml_from_scratch.tree import DecisionTreeRegressor
from ml_from_scratch.metrics import mean_squared_error

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split



# LOAD DATA
X, y = load_diabetes(return_X_y = True)

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.3,
                                                    random_state = 42)


# CLASSIFY - A Very Fit Tree
# Create a decision tree classfier
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)



# Predict & calculate accuracy score test
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
print(f"MSE score Train : {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"MSE score Test  : {mean_squared_error(y_test, y_pred_test):.4f}")
print("")
print("")


# CLASSIFY - A Simple Tree
# Create a decision tree classfier
clf = DecisionTreeRegressor(max_depth=3)
clf.fit(X_train, y_train)


# Predict & calculate accuracy score test
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
print(f"MSE score Train : {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"MSE score Test  : {mean_squared_error(y_test, y_pred_test):.4f}")
print("")
print("")


