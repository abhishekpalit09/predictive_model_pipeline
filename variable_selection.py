# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 10:13:52 2020

@author: PalitAbhishek
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

sns.set(style='white', context='notebook', palette='deep')

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# random.seed(9999)

# Output Tables
os. chdir("C:/Users/palitabhishek/Documents/Analysis/German") 
Data_Dictionary = './summary.csv'

# Data Dictionary
df = pd.read_csv('C:/Users/palitabhishek/Documents/Analysis/German/german_credit_data.csv')
summary = df.describe(include ='all')
summary.to_csv(Data_Dictionary)

