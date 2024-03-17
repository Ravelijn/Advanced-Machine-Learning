#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

from ml_from_scratch.tree import DecisionTreeClassifier
from ml_from_scratch.metrics import accuracy_score

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split




# LOAD DATA
iris = load_iris()
X = iris.data
y = np.where(iris.target==2,
             1,
             0)


# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.3,
                                                    stratify = y,
                                                    random_state = 42)



# CLASSIFY - A Very Fit Tree
# Create a decision tree classfier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# Predict & calculate accuracy score test
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
print(f"Accuracy score Train : {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Accuracy score Test  : {accuracy_score(y_test, y_pred_test):.4f}")
print("")
print("")


# CLASSIFY - A Simple Tree
# Create a decision tree classfier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)



# Predict & calculate accuracy score test
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
print(f"Accuracy score Train : {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Accuracy score Test  : {accuracy_score(y_test, y_pred_test):.4f}")
print("")
print("")




