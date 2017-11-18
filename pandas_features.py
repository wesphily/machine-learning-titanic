# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:07:48 2017

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

#train_data["Age"] = minmax_scale(train_data[["Age"]])

#
#Make Fare Buckets
#train_data.loc[train_data['Fare'] <= 7.91, 'Fare']=0
#train_data.loc[(train_data['Fare'] > 7.91) & (train_data['Fare'] <= 14.454), 'Fare']=1
#train_data.loc[(train_data['Fare'] > 14.454) & (train_data['Fare'] <= 31), 'Fare']=2
#train_data.loc[train_data['Fare'] > 31, 'Fare']



train_data['Family_Size']=0
train_data['Family_Size']=train_data['Parch']+train_data['SibSp']

train_data["Cabin_or_Not"]=0
train_data.loc[(train_data.Cabin.notnull()), "Cabin_or_Not"]=1

## Explore Fare Data
#Number of passengers paying each fare price
#fig = train_data.Fare.hist(bins=50)
#fig.set_title('Fare Paid Distribution')
#fig.set_label('Fare')
#fig.set_ylabel('Number of Passengers')
#Fare by survived
#fig1 = train_data.boxplot(column='Fare', by='Survived')
#fig1.set_title('')
#fig1.set_label('Fare')
#fig1.set_ylabel('Survived')
train_data.Fare.describe()
#inter quantile range
Fare_IQR = train_data.Fare.quantile(.75) - train_data.Fare.quantile(.25)
#lower quantile. For extreme values multiply by 3 instead of 1.5
Fare_Lower_Fence = train_data.Fare.quantile(.25) - (Fare_IQR * 1.5)
#Uper quantile. For extreme values multiply by 3 instead of 1.5
Fare_Upper_Fence = train_data.Fare.quantile(.75) + (Fare_IQR * 1.5)
#How many passengers in different ranges. We chose to use IQR since the distribution is no equal like a temple or guassian.
#print('Total Number of Passengers: {}'.format(train_data.shape[0]))
#print('Total Number of Passengers Paid More Than 100: {}'.format(train_data[train_data.Fare>100].shape[0]))
#print('Total Number of Passengers Paid More Than 200: {}'.format(train_data[train_data.Fare>200].shape[0]))
#print('Total Number of Passengers Paid More Than 300: {}'.format(train_data[train_data.Fare>300].shape[0]))
#See extreme outliers to determine similarities or differences from the rest of the data. It appears they are all together.
#print(train_data[train_data.Fare>300])

##Explore Age Data
##show distribution of age of all passengers
#fig2 = train_data.Age.hist(bins=50)
#fig2.set_title('Age Distribution')
#fig2.set_label('Age')
#fig2.set_ylabel('Number of Passengers')
##show distribution of age by dead or alive
#fig3 = train_data.boxplot(column='Age', by='Survived')
#fig3.set_title('')
#fig3.set_label('Age')
#fig3.set_ylabel('Survived')
##describe age data
#print(train_data.Age.describe())
##Gaussian (Assumption of Normality)
##mean plus 3 multiplied by standard deviation
#Age_Gaussian_Upper_Boundary = train_data.Age.mean() + 3* train_data.Age.std()
#Age_Gaussian_Lower_Boundary = train_data.Age.mean() - 3* train_data.Age.std()
##inter quantile range
#Age_IQR = train_data.Age.quantile(.75) - train_data.Age.quantile(.25)
##lower quantile. For extreme values multiply by 3 instead of 1.5
#Age_Lower_Fence = train_data.Age.quantile(.25) - (Age_IQR * 1.5)
##Uper quantile. For extreme values multiply by 3 instead of 1.5
#Age_Upper_Fence = train_data.Age.quantile(.75) + (Age_IQR * 1.5)
##Different meassurements to represent the end of outliers. We went with Guassian because the distribution is somewhat even like a temple
#print('Total Number of Passengers: {}'.format(train_data.shape[0]))
#print('Passengers older than 73 (gaussian): {}'.format(train_data[train_data.Fare>100].shape[0]))
#print('Total Number of Passengers Paid More Than 200: {}'.format(train_data[train_data.Fare>200].shape[0]))
#print('Total Number of Passengers Paid More Than 300: {}'.format(train_data[train_data.Fare>300].shape[0]))

##Explore Unique Data Labels. Decision Trees suffer from too many labels so it is important to understand.
print('Number of categories in the variable Gender: {}'.format(len(train_data.Sex.unique())))
print('Number of categories in the variable Ticket: {}'.format(len(train_data.Ticket.unique())))
print('Number of categories in the variable Cabin: {}'.format(len(train_data.Cabin.unique())))
print('Number of categories in the Titanic: {}'.format(len(train_data)))

##Convert Cabin to fewer labels
train_data['Cabin_Mapped']=train_data['Cabin'].astype(str).str[0]
print(train_data.Cabin_Mapped.unique())
train_data.loc[(train_data['Cabin_Mapped']=="n"), 'Cabin_Mapped']=0
train_data.loc[(train_data['Cabin_Mapped']=="A"), 'Cabin_Mapped']=1
train_data.loc[(train_data['Cabin_Mapped']=="B"), 'Cabin_Mapped']=2
train_data.loc[(train_data['Cabin_Mapped']=="C"), 'Cabin_Mapped']=3
train_data.loc[(train_data['Cabin_Mapped']=="D"), 'Cabin_Mapped']=4
train_data.loc[(train_data['Cabin_Mapped']=="E"), 'Cabin_Mapped']=5
train_data.loc[(train_data['Cabin_Mapped']=="F"), 'Cabin_Mapped']=6
train_data.loc[(train_data['Cabin_Mapped']=="G"), 'Cabin_Mapped']=7
train_data.loc[(train_data['Cabin_Mapped']=="T"), 'Cabin_Mapped']=8

#print(train_data.Cabin_Mapped.unique())
#print(train_data.head())
##Histogram comparing cabin to fare
g = sns.factorplot(x="Survived",y="Fare",data=train,kind="bar", size = 300 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Far Vs Cabin")




































