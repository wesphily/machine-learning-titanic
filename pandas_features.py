# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:07:48 2017

@author: ptillotson
"""
## converts training.csv to a dictionary of dictionarys. 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from time import time
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd
train_data = pd.read_csv("train.csv")
train_data["Survived"] = train_data["Survived"].astype("int")
train_data["Pclass"] = train_data["Pclass"].astype("int")
train_data["SibSp"] = train_data["SibSp"].astype("int")
train_data["Parch"] = train_data["Parch"].astype("int")
train_data["Embarked"] = train_data["Embarked"].astype("category")
#print(train_data.head())
    
#convert sex to number value
    #for key, value in train_list.iteritems():
     #   if value["Sex"] == "male":
      #      value["Sex"] = 1
       # else:
        #    value["Sex"] = 0

train_data['Sex'].replace(['male','female'],[0,1],inplace=True)
    
    
        
#fill in NaN age values
   # for key, value in train_list.iteritems():
    #    if value["Age"] == "":
     #       if value["SibSp"] > 1:
      #          value["Age"] = 9
       #     else:
        #        if value["SibSp"] == 1:
         #           if value["Parch"] > 0:
          #              value["Age"] == 9
           #         else:
            #            value["Age"] == 30

train_data.loc[(train_data.Age.isnull()) & (train_data["SibSp"] > 1), 'Age']=9
train_data.loc[(train_data.Age.isnull()) & (train_data["SibSp"] == 1) & (train_data["Parch"] > 0), 'Age']=9
train_data.loc[(train_data.Age.isnull()), 'Age']=30
train_data["Age"] = train_data["Age"].astype("int")



           
    
#Family Size Feature
   # family_size_list = []
    #family_size = []
    #family_size_counter = 0
    #for key, value in train_list.iteritems():
     #   family_size_list.append([int(value["SibSp"]), int(value["Parch"])])
    #for i in family_size_list:
     #   family_size.append(sum(i))
    #for key, value in train_list.iteritems():
     #   if "family_size" not in value:
      #      value.update({"family_size": family_size[family_size_counter]})
       #     family_size_counter += 1
    #return train_list

train_data['Family_Size']=0
train_data['Family_Size']=train_data['Parch']+train_data['SibSp']
    
train_data.drop(['Name', 'Ticket','Fare','Cabin', 'PassengerId'],axis=1,inplace=True)
train,test=cross_validation.train_test_split(train_data,test_size=0.3,random_state=42,stratify=train_data['Survived'])
features_train=train[train.columns[1:]]
labels_train=train[train.columns[:1]]
features_test=test[test.columns[1:]]
labels_test=test[test.columns[:1]]
features=train_data[train_data.columns[1:]]
labels=train_data['Survived']



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
print("Recall", pred_recall)
print("Precision", pred_precision)
#print pred
print("Accuracy", accuracy)    

