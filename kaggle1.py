
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
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X2D = pca.fit_transform(X_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

from sklearn.model_selection import cross_val_score
labels = y_train[:,0]

from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=5, min_samples_split=2,
  min_samples_leaf=3, max_features='auto',    bootstrap=False, oob_score=False, n_jobs=1, random_state=33,
  verbose=0)
#calculating individual model accuracy
accuracies0 = cross_val_score(estimator = classifier1, X = X_train , y = labels , cv = 10,n_jobs = -1)
accuracies = accuracies0.mean()

from sklearn.svm import SVC
classifier3 = SVC(C=7.9,gamma =0.01 ,kernel = 'rbf', random_state = 0)
classifier3.fit(X_train,y_train)
#calculating individual model accuracy
accuracies2 = cross_val_score(estimator = classifier3, X = X_train , y = labels , cv = 10,n_jobs = -1)
accuracies2 = accuracies2.mean()

#using voting classifier to improve model
from sklearn.ensemble import VotingClassifier
classifier = VotingClassifier(estimators=[('rf',classifier1),('svm',classifier3)],
                                          voting='hard', weights = [1,1])
classifier.fit(X_train, y_train)
### Predicting the Test set results
y_pred = classifier1.predict(X_test)



#Applying k-fold cross validation

accuracies4 = cross_val_score(estimator = classifier, X = X_train , y = labels , cv = 10,n_jobs = -1)
accuracies4 = accuracies4.mean()





# outputing results of test set to file 
f = open('result2.csv','w')
f.write('ID,solution')
f.write('\n')
for i in range (0,9000):
    f.write('%d,%d' % (i+1,y_pred[i]))
    f.write('\n')
f.close()


















