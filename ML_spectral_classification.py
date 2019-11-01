# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:51:19 2018

@author: ***
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import os
count = 0
intensity_peak = []
position_peak = []
FWHM_peak = []
def eachFile(filepath):
    
    pathDir = os.listdir(filepath)    
    for s in pathDir:
        newDir=os.path.join(filepath,s)    
        if os.path.isfile(newDir) :        
            if os.path.splitext(newDir)[1]==".txt": 
                Raman_Plot(newDir)                  
                pass
        else:
            eachFile(newDir)             

def Raman_Plot(filepath):
    global count, intensity_peak, position_peak, FWHM_peak 
    file_object = open(filepath, "r")
    Raman_original_data = file_object.readlines()
    Raman_original_data = Raman_original_data[1:]
    Row_data = len(Raman_original_data)
    Column_data = len(Raman_original_data[count].strip().split(' '))
    Raman_data = np.zeros((Row_data, Column_data), dtype=float)
    for n in range(0, Row_data-1):
        line = Raman_original_data[n].rstrip().split(' ')
        Raman_data[n] = line
    file_object.close()
    which_spec = int(3)
    while which_spec < Column_data: 
        plt_x = Raman_data[:, 0]
        plt_y = Raman_data[:, which_spec]
        which_spec_fit = which_spec - 4
        if which_spec_fit >= 0:
            intensity_peak[count][which_spec_fit] =  max(plt_y)
            position_peak[count][which_spec_fit] = float(plt_x[np.argwhere(
                    plt_y==max(plt_y))])
            FWHM_peak[count][which_spec_fit] = max(plt_x[np.argwhere(
                    plt_y>=max(plt_y)/2)]) - min(plt_x[np.argwhere(
                            plt_y>=max(plt_y)/2)])
        which_spec += 1
    count+=1
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_size=np.linspace(.1, 1.0, 5 )):
    if __name__ == '__main__':
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel('Training example')
        plt.ylabel('score')
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                                cv=cv, n_jobs=n_jobs,
                                                                train_sizes=train_size)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()#区域
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean,'o-',color="g",
                 label="Cross-validation score")
        plt.legend(loc="best")
        
        return plt     

Xdata = Xdata[1:]   

scaler = MinMaxScaler()
scaler.fit(Xdata)
scaler.data_max_
xdata_n=scaler.transform(Xdata)

xtrain, xtest, ytrain, ytest = train_test_split(xdata_n, ydata, test_size = 0.2)
clf = svm.SVC(C=10, kernel='rbf', gamma=0.1)#svm de Guassian core function
clf.fit(xtrain, ytrain_c)
train_score = clf.score(xtrain, ytrain_c)
test_score = clf.score(xtest, ytest_c)
print('train score: {0}; test score: {1}'.format(train_score, test_score))
ypred = clf.predict(xtest)
print("Test Set Predictions:\n {}".format(ypred))
'''Plotting learning curves'''
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
gammas = np.logspace(1, -3, 5)

for i in range(len(gammas)):
    estimator = svm.SVC(C=10, gamma=gammas[i])
        
    title = "Learning Curves(SVM, RBF kernel, $\gamma =$" + '%E' %gammas[i] + ")"
    plot_learning_curve(estimator, title, xtrain, ytrain, (0.25, 1.05), cv=cv, n_jobs=1)
plt.show()
gammas = np.logspace(2, -4, 7)
for i in range(len(gammas)):
    clf = svm.SVC(C=10, kernel='rbf', gamma=gammas[i])#svm de Guassian core function
    clf.fit(xtrain_c, ytrain_c)
    train_score = clf.score(xtrain_c, ytrain_c)
    test_score = clf.score(xtest_c, ytest_c)
    print('train score: {0}; test score: {1}'.format(train_score, test_score))
    










