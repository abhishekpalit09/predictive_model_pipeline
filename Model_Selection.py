# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:28:31 2020

@author: PalitAbhishek
"""

   
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

rfc = RandomForestClassifier()  

# Splitting 70/30 game
X_train,X_test,y_train,y_test = train_test_split(model_data,y, test_size =.3, stratify = y)

# Find the top 3 models
model_comparison(X_train ,y_train)