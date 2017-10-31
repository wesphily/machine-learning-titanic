# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:35:44 2017

@author: ptillotson
"""
from feature_format import featureFormat, targetFeatureSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from time import time
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from csv_to_dict_train import train_list_function

#run data prep function. this also adds/modifies features
train_list = train_list_function()


#Features to consider and then split into train and test data
features_list = ["Survived", "Pclass", "Sex",  "Age", "family_size"]
data = featureFormat(train_list, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

#Apply pca to values
n_components = 270
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(features_train)
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)
print(pca.explained_variance_ratio_)


clf = RandomForestClassifier(min_samples_split=10, max_features="sqrt")
t0 = time()
clf.fit(features_train_pca, labels_train)
#print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test_pca)
#print "training time:", round(time()-t0, 3), "s"
accuracy = clf.score(features_test_pca, labels_test)
pred_recall = recall_score(labels_test, pred)
pred_precision = precision_score(labels_test, pred)
print "Recall", pred_recall
print "Precision", pred_precision
#print pred
print "Accuracy", accuracy


#plt.plot(features_test_pca, labels_test, 'ro')
#plt.show()