
# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('/home/hamza/Downloads/train.csv',header = None)
dataset_train_labels = pd.read_csv('/home/hamza/Downloads/trainLabels.csv',header = None)
dataset_test = pd.read_csv('/home/hamza/Downloads/test.csv',header = None)
X_train = dataset_train.iloc[:, [0, 39]].values
y_train = dataset_train_labels.iloc[:,:].values
X_test = dataset_test.iloc[:, [0,39]].values

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500,criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

f = open('result.csv','w')
f.write('ID,solution')
f.write('\n')
for i in range (0,9000):
    f.write('%d,%d' % (i+1,y_pred[i]))
    f.write('\n')
f.close()