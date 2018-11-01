# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 19:19:12 2018
@author: abujr
"""
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn import model_selection

from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


os.chdir("E:\PhD-WPI\Fall-2018\DS504\Assignment_7")

#Load the data from CSV file
df_clasify = pd.read_csv('Python_Clean_PwP_HC.csv')
##################################################

#########################################################################################################
#Slide#12
# Plot the corelation for classsification Experiment.

# Compute the correlation matrix
corr = df_clasify.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})


#########################################################################################
# Bagging algorithms:
# Bagging meta-estimator
# Random forest   

df_clasify_1 = df_clasify.drop(['Unnamed: 0', 'education','employment'], axis=1)
df_clasify_1 = df_clasify[['professional.diagnosis','age', 'employment' ,'EQ.5D1','maritalStatus' ,'race' ,'GELTQ.1c' ,'GELTQ.1b' ,'education' , 'GELTQ.1a' ,'entropy.rate' ,'smoked' ,'gender' ,'cross.correlation','Gyro_entropy.rate', 'GELTQ.2' ,'minMaxDiff' ,'std' ,'wavelet.band'  ,'are.caretaker' ,'rms'  , 'energy.in._5.to.3','skewness','Gyro_Sway.X.Z','Gyro_Sway.Y.Z','wavelet.entropy','Gyro_std','peakFreq','Gyro_wavelet.band','Gyro_energy.in._5.to.3','spectralCentroid' ,'averageStepTime','Gyro_wavelet.entropy','Gyro_windowed.energy.in._5.to.3']]

#split dataset into train and test

train, test = train_test_split(df_clasify_1, test_size=0.1, random_state=0)

x_train=train.drop('professional.diagnosis',axis=1)
y_train=train['professional.diagnosis']

x_test=test.drop('professional.diagnosis',axis=1)
y_test=test['professional.diagnosis']

model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x_train, y_train)
model.score(x_test,y_test)

# Confusion Matrix
seed = 7
test_size = 0.1
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)

report = classification_report(Y_test, predicted)
print(report)


####################################################################
#RandomForest


model= RandomForestClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)

model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)

report = classification_report(Y_test, predicted)
print(report)
###########################################################################################
# Cross Validation
# Bagged Trees
array = df_clasify_1.values
X = array[:,1:30]
Y = array[:,0]

model1 = BaggingClassifier()
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)

scoring = 'accuracy'
results = model_selection.cross_val_score(model1, X, Y, cv=kfold, scoring=scoring)
print("Accuracy:") 
results
max(results)


scoring = 'roc_auc'
results = model_selection.cross_val_score(model1, X, Y, cv=kfold, scoring=scoring)
print("AUC:") 
max(results)



##########################################################
# Random Forest



predicted = model.predict(x_train)
matrix = confusion_matrix(y_test, predicted)
print(matrix)

