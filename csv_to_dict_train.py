# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 13:07:48 2017

@author: ptillotson
"""
## converts training.csv to a dictionary of dictionarys. 
def train_list_function():
    import csv
    reader = csv.DictReader(open('train.csv', 'rb'))
    train_list = {}
    dict_counter = 0
    for line in reader:
        train_list[dict_counter] = line
        dict_counter += 1
    
#convert sex to number value
    for key, value in train_list.iteritems():
        if value["Sex"] == "male":
            value["Sex"] = 1
        else:
            value["Sex"] = 0
        
#fill in NaN age values
    for key, value in train_list.iteritems():
        if value["Age"] == "":
            if value["SibSp"] > 1:
                value["Age"] = 9
            else:
                if value["SibSp"] == 1:
                    if value["Parch"] > 0:
                        value["Age"] == 9
                    else:
                        value["Age"] == 30
           
    
#Family Size Feature
    family_size_list = []
    family_size = []
    family_size_counter = 0
    for key, value in train_list.iteritems():
        family_size_list.append([int(value["SibSp"]), int(value["Parch"])])
    for i in family_size_list:
        family_size.append(sum(i))
    for key, value in train_list.iteritems():
        if "family_size" not in value:
            value.update({"family_size": family_size[family_size_counter]})
            family_size_counter += 1
    return train_list





    



    
    
    
    
    
    
        




    
