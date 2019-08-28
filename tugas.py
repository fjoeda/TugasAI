# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:46:37 2018

@author: fjoeda
"""

import pandas
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from pandas.plotting import scatter_matrix

animal_data = []
attribute = ["animal name","hair","feathers","eggs","milk","airborne","aquatic","predator",
"toothed","backbone","breathes","venomois","fins","legs","tail","domestic","catsize","type"]
        
dataframe = pandas.read_csv("zoo.data.txt", names = attribute)
print(dataframe.groupby('type').size())

x = np.array(dataframe.drop(['animal name','type'],1))
y = np.array(dataframe['type'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(x,y,test_size = 0.2)


clf = neighbors.KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train,Y_train)

accuracy = clf.score(X_test,Y_test)
print(accuracy)

cobaprediksi = np.array([0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1])
cobaprediksi = cobaprediksi.reshape(1,-1)

prediksi = clf.predict(cobaprediksi)

print(prediksi)


