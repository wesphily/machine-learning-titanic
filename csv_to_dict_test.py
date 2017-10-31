# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:31:24 2017

@author: ptillotson
"""

## converts test.csv to a dictionary of dictionarys

import csv
reader = csv.DictReader(open('test.csv', 'rb'))
test_list = {}
dict_counter = 0
for line in reader:
    test_list[dict_counter] = line
    dict_counter += 1

#print test_list