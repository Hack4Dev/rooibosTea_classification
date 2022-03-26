import numpy as np
from sklearn.metrics import *
from sklearn import metrics
from scipy.spatial import distance

def measurments (data):
    cov = np.cov(data , rowvar=False)
    v1 = np.linalg.matrix_power(cov, -1)
    center = np.mean(data , axis=0)
    return cov, v1, center

def measurments1D (sample, data):
    #z scores of the points
    sd = np.std(data)
    center = np.mean(data)
    z = np.abs( (sample-center)/sd )
    return z



def nested (xtrain, ytrain, xtest, ytest, numFeat):
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain).flatten()
    xtest = np.array(xtest)
    ytest = np.array(ytest).flatten()

    
    xtrain_fer = xtrain[np.where(ytrain == 1)]
    xtrain_nf = xtrain[np.where(ytrain == 0)]
    testFer = []
    testNF = []
    
    if numFeat == 1:
        for test in xtest:
            testFer.append(measurments1D(test, xtrain_fer))
            testNF.append(measurments1D(test, xtrain_nf))
    
    else:    
        covFer, v1Fer, centerFer =  measurments(xtrain_fer)
        covNF, v1NF, centerNF =  measurments(xtrain_nf)
        
        for test in xtest:
            testFer.append(distance.mahalanobis(test, centerFer, v1Fer))
            testNF.append(distance.mahalanobis(test, centerNF, v1NF))
        
    yPred = []
    for i in range(len(xtest)):
        if testFer[i] >= testNF[i]:
            yPred.append(0)
        else:
            yPred.append(1)
                

    acc = accuracy_score(ytest, np.array(yPred))
    return acc
        
        

def get_accuracy_base(xtrain, ytrain, xtest, ytest, numFeat):
    accTot = nested(xtrain, ytrain, xtest, ytest, numFeat)
    
    jackTrainArr = []
    jackTestArr = []
            
    for i in range(len(xtrain)):
        x_train = np.delete(np.array(xtrain), i, 0)
        y_train = np.delete(np.array(ytrain), i, 0)
        
        scoreTrain = nested(x_train, y_train, xtest, ytest, numFeat)
        
        jackTrainArr.append(scoreTrain)
            
    for t in range (len(xtest)):
        x_test = np.delete(np.array(xtest), t, 0)
        y_test = np.delete(np.array(ytest), t, 0)
            
        scoreTest = nested(xtrain, ytrain, x_test, y_test, numFeat)
        
        jackTestArr.append(scoreTest)  
            
    return  accTot, jackTrainArr, jackTestArr