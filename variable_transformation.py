# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:43:05 2020

@author: PalitAbhishek
"""

import os
os. chdir("C:/Users/palitabhishek/Documents/Analysis/Python_Functions") 

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,RobustScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from pre_processing import *
from feature_engine import outlier_removers as outr
import random
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from pre_processing import Abhi_Trasformer

random.seed(9999)


rfc = RandomForestClassifier()  

df = pd.read_csv('C:/Users/palitabhishek/Documents/Analysis/German/german_credit_data.csv')

# np.random.shuffle(df)
# df, out = df[:80,:], df[80:,:]
# df = random.sample(df, int(.6*len(df)))

# df = df[1:1000]

Cat_Ordinal = ['Saving_accounts','Checking_account']
Cat_Onehot = ['Purpose','Sex']
Cont = ['Age','Job','Credit_amount','Duration']
Target = ['Target']

y = pd.DataFrame(df,columns = Target)

# Multi Colinearity Treatment
df_Cont = ReduceVIF().fit_transform(pd.DataFrame(df,columns = Cont))
Cont_Revised = df_Cont.columns

# Continous Variable Pipeline
Cont_pipeline = Pipeline(steps=[
    ('outlier',outr.Winsorizer(distribution='gaussian',tail='both', fold=1)),
    ('imputer',SimpleImputer(strategy='constant', fill_value=0)),
    # ('add_variables', NewVariablesAdder()),
    # ('outlier',Abhi_Trasformer(outlier_treatment)),
    ('rbst_scaler', RobustScaler())
    ])

# Categorical Variable Pipeline
Cat_Ordinal_pipeline = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='constant', fill_value=0)),
    ('Ordinal', OrdinalEncoder())])

Cat_Onehot_pipeline = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='constant', fill_value=0)),
    ('Onehot', OneHotEncoder())])

data_pipeline = ColumnTransformer([
    ('cont', Cont_pipeline, Cont_Revised),
    # ('cat_ordinal', Cat_Ordinal_pipeline, Cat_Ordinal),
    ('cat_onehot', Cat_Onehot_pipeline, Cat_Onehot),
    # ('feature_selection', SelectFromModel(rfc)),
    ])


# df_processed = data_pipeline.fit_transform(df)

df_processed = pd.DataFrame(data=data_pipeline.fit_transform(df), columns=get_column_names_from_ColumnTransformer(data_pipeline))

# names = get_column_names_from_ColumnTransformer(data_pipeline)

# data_pipeline.get_feature_names()

# Feat_Sel = Pipeline([
#   # ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
#   ('feature_selection', SelectFromModel(rfc)),
#   # ('classification', RandomForestClassifier())
# ])


var_imp_file = './var_imp_file.csv'

# Selecting top contributing variables for modelling 
model_data =Variable_Selector(Model =rfc,X = df_processed,y =y,var_imp_file = var_imp_file)


# model_data = pd.DataFrame(data=Feat_Sel.fit_transform(model_data, y), columns=Feat_Sel.Steps[1].get_feature_names)

        

    


