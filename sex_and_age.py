# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:42:25 2017

@author: ptillotson
"""
from collections import Counter

age_value = {}
sex_value = {}
age_and_sex_counter = 0


for key, value in train_list.iteritems():
    age_value[age_and_sex_counter] = value['Age']
    age_and_sex_counter += 1
    sex_value[age_and_sex_counter] = value['Sex']
    age_and_sex_counter += 1
      
        
print age_value


""" I have determined that creating a feature is hard. Need to be able to fill in "''" with NaN 
and then ignore that data.
"""