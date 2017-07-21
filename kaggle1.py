
"""
Created on Thu Jul 20 11:40:15 2017

@author: hamza
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('/home/hamza/Downloads/train.csv',header = None)
dataset_train_labels = pd.read_csv('/home/hamza/Downloads/trainLabels.csv',header = None)
dataset_test = pd.read_csv('/home/hamza/Downloads/test.csv',header = None)
X_train = dataset_train.iloc[:, :].values
y_train = dataset_train_labels.iloc[:,:].values
X_test = dataset_test.iloc[:, :].values


##
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 12)
#X2D = pca.fit_transform(X_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators = 500,criterion = 'entropy')

from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB()

from sklearn.svm import SVC
classifier3 = SVC(kernel = 'rbf', random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
classifier4 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

from sklearn.ensemble import VotingClassifier
classifier = VotingClassifier(estimators=[('rf',classifier1),('nb',classifier2),
                                          ('svm',classifier3),('knn',classifier4)],voting='hard')

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


















