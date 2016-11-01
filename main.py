# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:06:30 2016

@author: rajpu
"""
import numpy as np
import csv
import matplotlib.pyplot as plt

#def letor():
    
    

def synthetic():
    X = np.matrix
    Y = np.matrix
    inFile = open("input.csv", 'rU')
    inReader = csv.reader(inFile)
    inData = []
    for row in inReader:
        temp = []
        for value in row:
            temp.append(float(value))
        inData.append(temp)
    X = np.array(inData)
    
    outFile = open("output.csv", 'rU')
    outReader = csv.reader(outFile)
    outData = []
    for row in outReader:
        temp = []
        for value in row:
            temp.append(float(value))    
        outData.append(temp)
    Y = np.array(outData)
    length = len(X)
    trainLen = int(length*0.8)
    validLen = int((length - trainLen)/2)
    testLen = length - trainLen - validLen
    trainX = X[0:trainLen]
    trainY = Y[0:trainLen]
    validX = X[trainLen:trainLen + validLen]
    validY = Y[trainLen:trainLen + validLen]
    testX = X[trainLen + validLen:]
    testY = Y[trainLen + validLen:]
    
    var = []
    for i in range(0,trainX.shape[1]):
        temp = np.var(trainX[:,i])
        if(temp == 0):
            var.append(0.000001)
        else:
            var.append(temp)

     
    sigmaInv = 0.1*np.linalg.inv(np.diag(np.array(var)))
    
    trainMinValues = {}
    validMinValues = {}
    ErmsMinTrain = float("Inf")
    ErmsMinValid = float("Inf")
    for m in range(56,57):
        randRows = np.random.randint(trainX.shape[0], size=m)        
        mu = trainX[randRows,:]
        phiTrain = np.ones(shape = (trainX.shape[0],m))
        
        
        for i in range(0,trainX.shape[0]):
            for j in range(1,m):
                XminusMu = np.subtract(trainX[i],mu[j-1])
                phiTrain[i][j]= np.exp(-0.5*(np.dot(np.dot(XminusMu,sigmaInv),(np.transpose(XminusMu)))))
        
        phiValid = np.ones(shape = (validX.shape[0],m))
        
        for i in range(0,validX.shape[0]):
            for j in range(1,m):
                XminusMu = np.subtract(validX[i],mu[j-1])
                phiValid[i][j]= np.exp(-0.5*(np.dot(np.dot(XminusMu,sigmaInv),(np.transpose(XminusMu)))))
                
        lamb = 0.01
        
        while(lamb < 0.06):
            eye = np.eye(m, k = lamb)
            lamb += 0.01
            w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(phiTrain),phiTrain),eye)),np.transpose(phiTrain)),trainY)
            #w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.matrix.transpose(phiTrain),phiTrain),eye)),np.matrix.transpose(phiTrain)),trainY)
            
            phiW = np.dot(phiTrain,w)
            TminusP = trainY - phiW    
            Erms = np.dot(np.transpose(TminusP),TminusP)
            ErmsTrain = np.sqrt(Erms/trainLen)
            
            if(ErmsTrain < ErmsMinTrain):
                ErmsMinTrain = ErmsTrain
                trainMinValues['m'] = m
                trainMinValues['lamb'] = lamb
                trainMinValues['phiTrain'] = phiTrain
                trainMinValues['w'] = w
                trainMinValues['rms'] = ErmsTrain
    
            phiW = np.dot(phiValid,w)
            TminusP = validY - phiW    
            Erms = np.dot(np.transpose(TminusP),TminusP)
            ErmsValid = np.sqrt(Erms/validLen)
            
            if(ErmsValid < ErmsMinValid):
                ErmsMinValid = ErmsValid
                validMinValues['m'] = m
                validMinValues['lamb'] = lamb
                validMinValues['phiValid'] = phiValid
                validMinValues['w'] = w
                validMinValues['rms'] = ErmsValid
            
    #stochastic gradient descent
    eta = 0.03
    etaMin = 0
    ErmsShMin = float("Inf")
    while(eta < 1):
        wSh = np.ones((trainX.shape[1],1))
        for i in range(0,50):
            eyeSh = np.eye(trainX.shape[1], k = validMinValues['lamb'])
            lambWSh = np.transpose(np.dot(eyeSh,wSh))
            tempW = np.transpose(-eta*np.add((-1*np.dot(np.transpose(trainY-np.dot(trainX,wSh)),trainX)),lambWSh))
            wSh += (tempW/trainLen)
        
        phiWSh = np.dot(trainX,wSh)
        TminusPSh = trainY - phiWSh    
        ErmsSh = np.dot(np.transpose(TminusPSh),TminusPSh)
        ErmsSh = np.sqrt(ErmsSh/trainLen)
        if(ErmsSh < ErmsShMin):
            etaMin = eta
            ErmsShMin = ErmsSh
        eta = eta * 3
    
if __name__ == "__main__":
    print "Hello World"
    #letor()
    X = np.matrix
    Y = np.matrix
    print "letor"
    file = open("Querylevelnorm.txt")
    input = []
    for line in file:
        a=line.split()[0:48]
        a= a[0:1]+a[2:48]
    
        for i in range(0,47):
            if i==0:
                a[i]=float(a[i])
            else:
                a[i]=float(a[i].split(":")[1])
        input.append(a)
        
    data = np.array(input)
    X = data[:,1:47]
    Y = data[:,0:1]
    length = len(X)
    trainLen = int(length*0.8)
    validLen = int((length - trainLen)/2)
    testLen = length - trainLen - validLen
    trainX = X[0:trainLen]
    trainY = Y[0:trainLen]
    validX = X[trainLen:trainLen + validLen]
    validY = Y[trainLen:trainLen + validLen]
    testX = X[trainLen + validLen:]
    testY = Y[trainLen + validLen:]
    
    var = []
    for i in range(0,trainX.shape[1]):
        temp = np.var(trainX[:,i])
        if(temp == 0):
            var.append(0.000001)
        else:
            var.append(temp)

     
    sigmaInv = 0.1*np.linalg.inv(np.diag(np.array(var)))
    
    trainMinValues = {}
    validMinValues = {}
    ErmsMinTrain = float("Inf")
    ErmsMinValid = float("Inf")
    for m in range(4,5):
        randRows = np.random.randint(trainX.shape[0], size=m)        
        mu = trainX[randRows,:]
        phiTrain = np.ones(shape = (trainX.shape[0],m))
        
        
        for i in range(0,trainX.shape[0]):
            for j in range(1,m):
                XminusMu = np.subtract(trainX[i],mu[j-1])
                phiTrain[i][j]= np.exp(-0.5*(np.dot(np.dot(XminusMu,sigmaInv),(np.transpose(XminusMu)))))
        
        phiValid = np.ones(shape = (validX.shape[0],m))
        
        for i in range(0,validX.shape[0]):
            for j in range(1,m):
                XminusMu = np.subtract(validX[i],mu[j-1])
                phiValid[i][j]= np.exp(-0.5*(np.dot(np.dot(XminusMu,sigmaInv),(np.transpose(XminusMu)))))
                
        lamb = 0
        
        while(lamb < 0.1):
            eye = np.eye(m, k = lamb)
            lamb += 0.1
            w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(phiTrain),phiTrain),eye)),np.transpose(phiTrain)),trainY)
            #w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.matrix.transpose(phiTrain),phiTrain),eye)),np.matrix.transpose(phiTrain)),trainY)
            
            phiW = np.dot(phiTrain,w)
            TminusP = trainY - phiW    
            Erms = np.dot(np.transpose(TminusP),TminusP)
            ErmsTrain = np.sqrt(Erms/trainLen)
            
            if(ErmsTrain < ErmsMinTrain):
                ErmsMinTrain = ErmsTrain
                trainMinValues['m'] = m
                trainMinValues['lamb'] = lamb
                trainMinValues['phiTrain'] = phiTrain
                trainMinValues['w'] = w
                trainMinValues['rms'] = ErmsTrain
    
            phiW = np.dot(phiValid,w)
            TminusP = validY - phiW    
            Erms = np.dot(np.transpose(TminusP),TminusP)
            ErmsValid = np.sqrt(Erms/validLen)
            
            if(ErmsValid < ErmsMinValid):
                ErmsMinValid = ErmsValid
                validMinValues['m'] = m
                validMinValues['lamb'] = lamb
                validMinValues['phiValid'] = phiValid
                validMinValues['w'] = w
                validMinValues['rms'] = ErmsValid
                validMinValues['mu'] = mu
    

    #testing trained models
    phiTest = np.ones(shape = (testX.shape[0],validMinValues['m']))
    mu = validMinValues['mu']
    for i in range(0,testX.shape[0]):
        for j in range(1,validMinValues['m']):
            XminusMu = np.subtract(testX[i],mu[j-1])
            phiTest[i][j]= np.exp(-0.5*(np.dot(np.dot(XminusMu,sigmaInv),(np.transpose(XminusMu)))))
    
            
    predic = np.dot(phiTest,validMinValues['w'])
    graphX = list(range(testLen))
    TminusP = testY - predic
    Erms = np.dot(np.transpose(TminusP),TminusP)
    ErmsTest = np.sqrt(Erms/testLen)
    plt.figure(1)
    plt.plot(graphX,predic,'r--', graphX, testY, 'b--')
    plt.xlabel("data points")
    plt.ylabel("values")
    plt.show()
        
    #stochastic gradient descent
    eta = 0.03
    mSh = validMinValues['m']
    costValues = []
    wSh = np.ones((mSh,1))
    iterations = 100
    for i in range(0,iterations):
        eyeSh = np.eye(mSh, k = validMinValues['lamb'])
        lambWSh = np.transpose(np.dot(eyeSh,wSh))
        tempW = np.transpose(-eta*np.add((-1*np.dot(np.transpose(trainY-np.dot(phiTrain,wSh)),phiTrain)),lambWSh))
        wSh += (tempW/trainLen)
        cost1 = trainY - np.dot(phiTrain,wSh)
        cost2 = np.dot(np.transpose(cost1),cost1)
        costValues.append(cost2/trainLen)
    
    phiWSh = np.dot(phiTrain,wSh)
    TminusPSh = trainY - phiWSh    
    ErmsSh = np.dot(np.transpose(TminusPSh),TminusPSh)
    ErmsSh = np.sqrt(ErmsSh/trainLen)
    
    
    predic = np.dot(phiTest,wSh)
    graphX = list(range(testLen))
    
    plt.figure(2)
    plt.plot(graphX,predic,'r--', graphX, testY, 'b--')
    plt.xlabel("data points")
    plt.ylabel("values")
    plt.show()
    
    graphCostX = list(range(iterations))
    plt.figure(3)
    plt.scatter(graphCostX, costValues)
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.show()
    
    
    #synthetic()
    
        
    
    