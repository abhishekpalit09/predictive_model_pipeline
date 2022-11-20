# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 22:39:54 2020

@author: PalitAbhishek
"""

import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline


if __name__ == "__main__":
    df = pd.read_csv('C:/Users/palitabhishek/Documents/Analysis/santander/train.csv')
    X = df.drop("TARGET",axis=1).values
    y = df.TARGET.values
    
    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_jobs=-1)
    
    # classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    # # Grid Search is more time consuming than Random Seach
    # grid_param_grid = {
    #     "n_estimators" : [100,200,300,400],
    #     "max_depth" : [1,3,5,7],
    #     "criterion" : ["gini","entropy"],
    #     }
    # model = model_selection.GridSearchCV(
    #     estimator = classifier,
    #     param_grid = grid_param_grid,
    #     scoring = "accuracy",
    #     verbose = 10,
    #     n_jobs =-1,
    #     cv = 5
    #     )
    # model.fit(X,y)
    # print(model.best_score_)
    # print(model.best_estimator_.get_params())
    
    # # Random Search is more efficient
    # random_param_grid = {
    #     "n_estimators" : np.arange(100,1500,100),
    #     "max_depth" : np.arrange(1,20),
    #     "criterion" : ["gini","entropy"],
    #     }
    # model = model_selection.RandomizedSearchCV(
    #     estimator = classifier,
    #     param_distributions = random_param_grid,
    #     n_iter = 10,
    #     scoring = "accuracy",
    #     verbose = 10,
    #     n_jobs =-1,
    #     cv = 5
    #     )
    # model.fit(X,y)
    # print(model.best_score_)
    # print(model.best_estimator_.get_params())
    
    #Pipeline
    classifier = pipeline.Pipeline([("scaling",scl),("pca",pca),("rf",rf)])
    random_param_grid = {
        "pca__n_components" : np.arange(5,10),
        "rf__n_estimators" : np.arange(100,300,100),
        "rf__max_depth" : np.arange(1,5),
        "rf__criterion" : ["gini","entropy"],
        }
    model = model_selection.RandomizedSearchCV(
        estimator = classifier,
        param_distributions = random_param_grid,
        n_iter = 10,
        scoring = "accuracy",
        verbose = 10,
        n_jobs =-1,
        cv = 5
        )
    model.fit(X,y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
    
