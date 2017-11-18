# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:59:36 2017
@author: ptillotson
"""

## converts training.csv to a dictionary of dictionarys. 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.datasets import make_classification
from time import time
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import minmax_scale
import pandas as pd
import seaborn as sns

train_data = pd.read_csv("train.csv")
#train_data["Survived"] = train_data["Survived"].astype("int")
#train_data["SibSp"] = train_data["SibSp"].astype("int")
#train_data["Parch"] = train_data["Parch"].astype("int")
train_data["Embarked"] = train_data["Embarked"].astype("category")
#print(train_data.head())
    


train_data['Sex'].replace(['male','female'],[0,1],inplace=True)
train_data["Sex"] = train_data["Sex"].astype("int")
train_data["Embarked"].replace(["S", "C", "Q"],[1,2,3],inplace=True)
train_data.loc[(train_data.Embarked.isnull()), "Embarked"]=1
train_data["Embarked"] = train_data["Embarked"].astype("int")
    
    
        
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
train_data.loc[(train_data.Age.isnull()) & (train_data["SibSp"] <= 1) & (train_data["Parch"] == 2), 'Age']=9
train_data.loc[(train_data.Age.isnull()) & (train_data["SibSp"] <= 1) & (train_data["Parch"] == 1), 'Age']=30
train_data.loc[(train_data.Age.isnull()) & (train_data["SibSp"] <= 1) & (train_data["Parch"] == 0), 'Age']=30
#train_data["Age"] = train_data["Age"].astype("int")




train_data['Age_Category']=0
train_data.loc[(train_data["Age"] < 1), 'Age_Category']=1
train_data.loc[(train_data["Age"] >= 1) & (train_data["Age"] <= 12), 'Age_Category']=2
train_data.loc[(train_data["Age"] >= 13) & (train_data["Age"] <= 19), 'Age_Category']=3
train_data.loc[(train_data["Age"] >= 20) & (train_data["Age"] <= 29), 'Age_Category']=4
train_data.loc[(train_data["Age"] >= 30) & (train_data["Age"] <= 39), 'Age_Category']=5
train_data.loc[(train_data["Age"] >= 40) & (train_data["Age"] <= 59), 'Age_Category']=6
train_data.loc[(train_data["Age"] >= 60), 'Age_Category']=7

train_data["c_am_af"]=0
train_data.loc[(train_data["Age"] >= 18) & (train_data["Sex"]==0), "c_am_af"]=1
train_data.loc[(train_data["Age"] >= 18) & (train_data["Sex"]==1), "c_am_af"]=2

train_data["Age"] = minmax_scale(train_data[["Age"]])

#
#Make Fare Buckets
train_data.loc[train_data['Fare'] <= 7.91, 'Fare']=0
train_data.loc[(train_data['Fare'] > 7.91) & (train_data['Fare'] <= 14.454), 'Fare']=1
train_data.loc[(train_data['Fare'] > 14.454) & (train_data['Fare'] <= 31), 'Fare']=2
train_data.loc[train_data['Fare'] > 31, 'Fare']



train_data['Family_Size']=0
train_data['Family_Size']=train_data['Parch']+train_data['SibSp']

train_data["Cabin_or_Not"]=0
train_data.loc[(train_data.Cabin.notnull()), "Cabin_or_Not"]=1



    
train_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch'],axis=1,inplace=True)
print(train_data.head(1))

train,test=cross_validation.train_test_split(train_data,test_size=0.3,random_state=42,stratify=train_data['Survived'])
features_train=train[train.columns[1:]]
labels_train=train[train.columns[:1]]
features_test=test[test.columns[1:]]
labels_test=test[test.columns[:1]]
features=train_data[train_data.columns[1:]]
labels=train_data['Survived']


kfold = StratifiedKFold(n_splits=10)
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, features_train, labels_train.values.ravel(),  scoring = "accuracy", cv = kfold))
    
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
for i in cv_means:
    print(i)

#bar chart showing mean score of each algorythm    
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")