# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 21:21:12 2016

@author: rajpu
"""

import numpy as np

    

if __name__ == "__main__":
    print "Hello World"
    X = np.matrix
    Y = np.matrix
    print "letor"
    file = open("Querylevelnorm.txt")
    inputX = []
    inputY = []
    for line in file:
        inputY.append(np.double(line[0]))
        temp = line.split()[2:48]
        tempX = []
        for item in temp:
            tempX.append(np.double(item.split(":")[1]))
        
        inputX.append(tempX)
    X = np.matrix(inputX)
    Y = np.matrix(np.array(inputY))
    length = X.shape[0]
    trainX = X[0:(length*0.8)]
    trainY = Y[:,0:(length*0.8)]
    validX = X[(length*0.8):(length*0.9)]
    validY = Y[:,(length*0.8):(length*0.9)]
    testX = X[(length*0.9):]
    testY = Y[:,(length*0.9):]
    
    var = []
    for i in range(0,trainX.shape[1]):
        temp = np.var(trainX[:,i])
        if(temp == 0):
            var.append(0.000001)
        else:
            var.append(temp)

     
    sigmaInv = 0.1*np.linalg.inv(np.diag(np.array(var)))
	
    for m in range(4,5):
        randRows = np.random.randint(trainX.shape[0], size=m)        
        mu = trainX[randRows,:]
        phiTrain = np.ones(shape = (trainX.shape[0],m))
        
        for i in range(0,trainX.shape[0]):
            for j in range(1,m):
                XminusMu = np.subtract(trainX[i],mu[j-1])
                phiTrain[i][j]= np.exp(-0.5*(np.dot(np.dot(XminusMu,sigmaInv),(np.matrix.transpose(XminusMu)))))
                
        lamb = 0
        eye = np.eye(m, k=1)
        w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.matrix.transpose(phiTrain),phiTrain),eye)),np.matrix.transpose(phiTrain)),np.matrix.transpose(trainY))
        predic = np.dot(phiTrain,w)
        TminusP = (trainY - predic)
        
        """while(lamb < 0.1):
            eye = np.eye(m, k=1)
            lamb += 0.1
            w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.matrix.transpose(phiTrain),phiTrain),eye)),np.matrix.transpose(phiTrain)),np.matrix.transpose(trainY))
            #w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.matrix.transpose(phiTrain),phiTrain),eye)),np.matrix.transpose(phiTrain)),trainY)
            TminusP = np.subtract(trainY,np.dot(phiTrain,w))    
            Erms = np.dot(np.matrix.transpose(TminusP),TminusP)
            Erms = np.sqrt(Erms/trainY.shape[0])"""