# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:59:36 2017

@author: ptillotson
"""

## converts training.csv to a dictionary of dictionarys. 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from time import time
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd
train_data = pd.read_csv("train.csv")
train_data["Survived"] = train_data["Survived"].astype("int")
train_data["Pclass"] = train_data["Pclass"].astype("int")
train_data["SibSp"] = train_data["SibSp"].astype("int")
train_data["Parch"] = train_data["Parch"].astype("int")

#print(train_data.head())
    
#convert sex to number value
    #for key, value in train_list.iteritems():
     #   if value["Sex"] == "male":
      #      value["Sex"] = 1
       # else:
        #    value["Sex"] = 0

train_data['Sex'].replace(['male','female'],[0,1],inplace=True)
train_data["Sex"] = train_data["Sex"].astype("int")
train_data["Embarked"].replace(["S", "C", "Q"],[1,2,3],inplace=True)
train_data.loc[(train_data.Embarked.isnull()), "Embarked"]=1
train_data["Embarked"] = train_data["Embarked"].astype("int")
#print(train_data[train_data["Embarked"]==2])


    
    
        
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
train_data["Age"] = train_data["Age"].astype("int")

train_data['Age_Category']=0
train_data.loc[(train_data["Age"] < 1), 'Age_Category']=1
train_data.loc[(train_data["Age"] >= 1) & (train_data["Age"] < 5), 'Age_Category']=2
train_data.loc[(train_data["Age"] >= 5) & (train_data["Age"] < 10), 'Age_Category']=3
train_data.loc[(train_data["Age"] >= 10) & (train_data["Age"] < 15), 'Age_Category']=4
train_data.loc[(train_data["Age"] >= 15) & (train_data["Age"] < 20), 'Age_Category']=5
train_data.loc[(train_data["Age"] >= 20) & (train_data["Age"] < 25), 'Age_Category']=6
train_data.loc[(train_data["Age"] >= 25) & (train_data["Age"] < 30), 'Age_Category']=7
train_data.loc[(train_data["Age"] >= 30) & (train_data["Age"] < 35), 'Age_Category']=8
train_data.loc[(train_data["Age"] >= 35) & (train_data["Age"] < 40), 'Age_Category']=9
train_data.loc[(train_data["Age"] >= 40) & (train_data["Age"] < 45), 'Age_Category']=10
train_data.loc[(train_data["Age"] >= 45) & (train_data["Age"] < 50), 'Age_Category']=11
train_data.loc[(train_data["Age"] >= 50) & (train_data["Age"] < 55), 'Age_Category']=12
train_data.loc[(train_data["Age"] >= 55) & (train_data["Age"] < 60), 'Age_Category']=13
train_data.loc[(train_data["Age"] >= 60) & (train_data["Age"] < 65), 'Age_Category']=14
train_data.loc[(train_data["Age"] >= 65) & (train_data["Age"] < 70), 'Age_Category']=15
train_data.loc[(train_data["Age"] >= 70) & (train_data["Age"] < 75), 'Age_Category']=16
train_data.loc[(train_data["Age"] >= 75) & (train_data["Age"] < 80), 'Age_Category']=17
train_data.loc[(train_data["Age"] >= 80) & (train_data["Age"] < 85), 'Age_Category']=18
train_data.loc[(train_data["Age"] >= 85)]=19
#print(train_data[train_data["Age_Category"]==3])




           
    
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

train_data["Family_VS_Sex"]=0
train_data["Family_VS_Sex"]=train_data["Family_Size"]+train_data["Sex"]

#Bar Graph    
#f,ax=plt.subplots(1,2,figsize=(15,7.5))
#train_data[train_data['Survived']==0].Sex.plot.hist(ax=ax[0],bins=8,edgecolor='black',color='red')
#ax[0].set_title('Survived= 0')
#x1=list(range(1,10,1))
#ax[0].set_xticks(x1)
#train_data[train_data['Survived']==1].Sex.plot.hist(ax=ax[1],color='green',bins=8,edgecolor='black')
#ax[1].set_title('Survived=1')
#x2=list(range(1,10,1))
#ax[1].set_xticks(x2)
#plt.show()
    
#Violin Graph
#f,ax=plt.subplots(1,2,figsize=(18,8))
#sns.violinplot("Embarked","Pclass", hue="Survived", data=train_data,split=True,ax=ax[0])
#ax[0].set_title('Embarked and Pclass vs Survived')
#ax[0].set_yticks(range(0,4,1))
#sns.violinplot("Sex","Pclass", hue="Survived", data=train_data,split=True,ax=ax[1])
#ax[1].set_title('Sex and Pclass vs Survived')
#ax[1].set_yticks(range(0,5,1))
#plt.show()


#print(train_data[train_data["Embarked"]=="S" & train_data["Pclass"]==3])



print(train_data.sort_values("Embarked", na_position='last'))