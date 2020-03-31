import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import sys
import warnings
import os
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
data = pd.read_csv(r'C:\Users\Admin\AppData\Local\Programs\Python\Python35\ASLrec\Features\combined_csv4.csv')#Change the file path accordingly
X=data.drop('17',axis=1)

y=data['17']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=10) 
# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

# Make grid search classifier
clf_grid = GridSearchCV(SVC(), param_grid, verbose=1)

# Train the classifier
clf_grid.fit(X_train, y_train)

# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)

svm_model_linear = LinearSVC(multi_class='ovr', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
  
accuracy = svm_model_linear.score(X_test, y_test) 
  

cm = confusion_matrix(y_test, svm_predictions) 
print(cm)
print(accuracy)
