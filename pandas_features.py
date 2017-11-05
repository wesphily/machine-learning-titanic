# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:07:48 2017

@author: ptillotson
"""
## converts training.csv to a dictionary of dictionarys. 
#def train_data_function():
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
train_data.loc[(train_data.Age.isnull()) & (train_data["SibSp"] == 1) & (train_data["Parch"] <= 0), 'Age']=30


print(train_data["Age"]==30)
           
    
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





    



    
    
    
    
    
    
        




    
