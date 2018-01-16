# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 13:24:15 2017

@author: ptillotson
"""

## converts training.csv to a dictionary of dictionarys. 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.datasets import make_classification
from time import time
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import RandomizedPCA
import pandas as pd
import numpy as np
import seaborn as sns
import re

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
#train_data["Survived"] = train_data["Survived"].astype("int")
#train_data["SibSp"] = train_data["SibSp"].astype("int")
#train_data["Parch"] = train_data["Parch"].astype("int")
train_data["Embarked"] = train_data["Embarked"].astype("category")
test_data["Embarked"] = train_data["Embarked"].astype("category")
IDtest = test_data["PassengerId"]
#print(train_data.head())

    


train_data['Sex'].replace(['male','female'],[0,1],inplace=True)
train_data["Sex"] = train_data["Sex"].astype("int")
train_data["Embarked"].replace(["S", "C", "Q"],[1,2,3],inplace=True)
train_data.loc[(train_data.Embarked.isnull()), "Embarked"]=1
train_data["Embarked"] = train_data["Embarked"].astype("int")

test_data['Sex'].replace(['male','female'],[0,1],inplace=True)
test_data["Sex"] = test_data["Sex"].astype("int")
test_data["Embarked"].replace(["S", "C", "Q"],[1,2,3],inplace=True)
test_data.loc[(test_data.Embarked.isnull()), "Embarked"]=1
test_data["Embarked"] = test_data["Embarked"].astype("int")
    


train_data.loc[(train_data.Age.isnull()) & (train_data["SibSp"] > 1), 'Age']=9
train_data.loc[(train_data.Age.isnull()) & (train_data["SibSp"] == 1) & (train_data["Parch"] > 0), 'Age']=9
train_data.loc[(train_data.Age.isnull()) & (train_data["SibSp"] <= 1) & (train_data["Parch"] == 2), 'Age']=9
train_data.loc[(train_data.Age.isnull()) & (train_data["SibSp"] <= 1) & (train_data["Parch"] == 1), 'Age']=30
train_data.loc[(train_data.Age.isnull()) & (train_data["SibSp"] <= 1) & (train_data["Parch"] == 0), 'Age']=30
#train_data["Age"] = train_data["Age"].astype("int")
####Top code age to 73 since that is the upper level of the Gaussian distribution
train_data.loc[(train_data["Age"]>=74), 'Age']=73

test_data.loc[(test_data.Age.isnull()) & (test_data["SibSp"] > 1), 'Age']=9
test_data.loc[(test_data.Age.isnull()) & (test_data["SibSp"] == 1) & (test_data["Parch"] > 0), 'Age']=9
test_data.loc[(test_data.Age.isnull()) & (test_data["SibSp"] <= 1) & (test_data["Parch"] == 2), 'Age']=9
test_data.loc[(test_data.Age.isnull()) & (test_data["SibSp"] <= 1) & (test_data["Parch"] == 1), 'Age']=30
test_data.loc[(test_data.Age.isnull()) & (test_data["SibSp"] <= 1) & (test_data["Parch"] == 0), 'Age']=30
test_data.loc[(test_data.Age.isnull()), 'Age']=30
#train_data["Age"] = train_data["Age"].astype("int")
####Top code age to 73 since that is the upper level of the Gaussian distribution
test_data.loc[(test_data["Age"]>=74), 'Age']=73

#Put Fare into quartile buckets since the distribution of values is skewed. 
train_data['Fare']=pd.qcut(train_data.Fare, q=8, labels=False)

test_data['Fare']=pd.qcut(test_data.Fare, q=8, labels=False)
test_data.loc[(test_data.Fare.isnull()), 'Fare']=0


##Rare title work such as Mr. and Mrs.
##Names have unique titles such as Mrs or Mr. Let's extract that and place it into a new column.
def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
##Add title column to train_data with title (i.e. Mr or Mrs)  
train_data['Title'] = train_data['Name'].apply(get_title)
##Convert Title to a number representing the Title
train_data.loc[(train_data["Title"]=="Mrs"), 'Title']=0
train_data.loc[(train_data["Title"]=="Mr"), 'Title']=1
train_data.loc[(train_data["Title"]=="Miss"), 'Title']=2
train_data.loc[(train_data["Title"]=="Master"), 'Title']=3
train_data.loc[(train_data["Title"]=="Other"), 'Title']=4

##Add title column to train_data with title (i.e. Mr or Mrs)  
test_data['Title'] = test_data['Name'].apply(get_title)
##Convert Title to a number representing the Title
test_data.loc[(test_data["Title"]=="Mrs"), 'Title']=0
test_data.loc[(test_data["Title"]=="Mr"), 'Title']=1
test_data.loc[(test_data["Title"]=="Miss"), 'Title']=2
test_data.loc[(test_data["Title"]=="Master"), 'Title']=3
test_data.loc[(test_data["Title"]=="Other"), 'Title']=4

train_data['Family_Size']=0
train_data['Family_Size']=train_data['Parch']+train_data['SibSp']
train_data.loc[(train_data['Family_Size']>7), 'Family_Size']=7

test_data['Family_Size']=0
test_data['Family_Size']=test_data['Parch']+test_data['SibSp']
test_data.loc[(test_data['Family_Size']>7), 'Family_Size']=7

test_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch', 'Embarked', 'Pclass'],axis=1,inplace=True)
train_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch', 'Embarked', 'Pclass'],axis=1,inplace=True)
print(train_data.head(1))
train,test=cross_validation.train_test_split(train_data,test_size=0.3,random_state=42,stratify=train_data['Survived'])
features_train=train[train.columns[1:]]
labels_train=train[train.columns[:1]]
features_test=test[test.columns[1:]]
labels_test=test[test.columns[:1]]
features=train_data[train_data.columns[1:]]
labels=train_data['Survived']

etc = ExtraTreesClassifier(bootstrap = False, criterion = 'gini', max_depth = None, min_samples_leaf = 3, min_samples_split = 3, n_estimators = 100)
#t0 = time()
etc.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#t0 = time()
etcpred = etc.predict(features_test)
#print "training time:", round(time()-t0, 3), "s"


##Random Forrest
rf = RandomForestClassifier(min_samples_split=10, max_features="sqrt")
#t0 = time()
rf.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#t0 = time()
rfpred = rf.predict(features_test)
#print "training time:", round(time()-t0, 3), "s"


##AdaBoost
abc = AdaBoostClassifier()
#t0 = time()
abc.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#t0 = time()
abcpred = abc.predict(features_test)
#print "training time:", round(time()-t0, 3), "s"


##GradientBoost
gbc = GradientBoostingClassifier()
#t0 = time()
gbc.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#t0 = time()
gbcpred = gbc.predict(features_test)
#print "training time:", round(time()-t0, 3), "s"



votingC = VotingClassifier(estimators=[('etc', etc),('rfc', rf),('gbc', gbc)], voting='soft', weights=[1,1,1])

votingC = votingC.fit(features_train,labels_train.values.ravel())
pred = votingC.predict(features_test)


accuracy = votingC.score(features_test, labels_test)
pred_recall = recall_score(labels_test, pred)
pred_precision = precision_score(labels_test, pred)
print("Recall", pred_recall)
print("Precision", pred_precision)
#print pred
print("Accuracy", accuracy)


#test_Survived = pd.Series(votingC.predict(test_data), name="Survived")
#results = pd.concat([IDtest,test_Survived],axis=1)
#results.to_csv("submit_to_kaggle.csv",index=False)







