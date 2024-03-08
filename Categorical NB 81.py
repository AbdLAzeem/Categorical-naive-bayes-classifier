# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:57:10 2020

@author: AbdelAzeem
"""

# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#----------------------------------------------------

#reading data

data = pd.read_csv('heart.csv')
data.describe()

#X Data
X = data.iloc[:,:-1]
#y Data
y = data.iloc[:,-1]
print('X Data is \n' , X.head())
print('X shape is ' , X.shape)


#---------Feature Selection = Logistic Regression 13=>7 -------------------

from sklearn.linear_model import  LogisticRegression

thismodel = LogisticRegression()


FeatureSelection = SelectFromModel(estimator = thismodel, max_features = None) # make sure that thismodel is well-defined
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension 
print('X Shape is ' , X.shape)
print('Selected Features are : ' , FeatureSelection.get_support())

#--------------------- Normalizing Data -------------------------------
#Normalizing Data

scaler = Normalizer(copy=True, norm='max') # you can change the norm to 'l1' or 'max' 
X = scaler.fit_transform(X)

#showing data
#print('X \n' , X[:10])
#print('y \n' , y[:10])


#------------ Splitting data ---33% Test  67% Training --------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


#   -------------- Categorical NB classifier 81 % --------

from sklearn.naive_bayes import CategoricalNB


#Applying BernoulliNB Model 

'''
sklearn.naive_bayes.CategoricalNB(alpha=1.0, fit_prior=True, class_prior=None)
'''

CategoricalNBModel = CategoricalNB()

CategoricalNBModel.fit(X_train, y_train)

#Calculating Details

#y_pred = BernoulliNBModel.predict(X_test)
y_pred = CategoricalNBModel.predict(X_test)

#                       = Score =

from sklearn.metrics import accuracy_score

#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))
AccScore = accuracy_score(y_test, y_pred, normalize=False)
print('Accuracy Score is : ', AccScore)



#            ###################### ((Grid Search)) ##############

from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import GridSearchCV
import pandas as pd

SelectedModel = CategoricalNB()

'''
sklearn.naive_bayes.CategoricalNB(alpha=1.0, fit_prior=True, class_prior=None)

'''


SelectedParameters = {'alpha':[0,1], 'fit_prior' : ('True','False') }

GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 10,return_train_score=True)

GridSearchModel.fit(X_train, y_train)

sorted(GridSearchModel.cv_results_.keys())

GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]

# Showing Results
print('All Results are :\n', GridSearchResults )
print('Best Score is :', GridSearchModel.best_score_)
print('Best Parameters are :', GridSearchModel.best_params_)
print('Best Estimator is :', GridSearchModel.best_estimator_)


#--------------------------- try 2 ----- 81 % ----

CategoricalNBModel = CategoricalNB(alpha=1, class_prior=None, fit_prior='True')

CategoricalNBModel.fit(X_train, y_train)

#Calculating Details
print('CategoricalNBModel Train Score is : ' , CategoricalNBModel.score(X_train, y_train))
print('CategoricalNBModel Test Score is : ' , CategoricalNBModel.score(X_test, y_test))
print('----------------------------------------------------')
