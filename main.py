# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:06:30 2016

@author: rajpu
"""
import numpy as np
import csv
import matplotlib.pyplot as plt

def letor():
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
    
    randNums = np.random.choice(length, size = length, replace = False)
    trainX = X[randNums[0:trainLen]]
    trainY = Y[randNums[0:trainLen]]
    validX = X[randNums[trainLen:trainLen + validLen]]
    validY = Y[randNums[trainLen:trainLen + validLen]]
    testX = X[randNums[trainLen + validLen:]]
    testY = Y[randNums[trainLen + validLen:]]
    
    """trainX = X[0:trainLen]
    trainY = Y[0:trainLen]
    validX = X[trainLen:trainLen + validLen]
    validY = Y[trainLen:trainLen + validLen]
    testX = X[trainLen + validLen:]
    testY = Y[trainLen + validLen:]"""
    
    
    var = []
    for i in range(0,trainX.shape[1]):
        temp = np.var(trainX[:,i])
        if(temp == 0):
            var.append(0.000001)
        else:
            var.append(temp)

     
    sigmaInv = 0.1*np.linalg.inv(np.diag(np.array(var)))
    
    contourMatrix = np.zeros(shape=(50,6))

    
    trainMinValues = {}
    validMinValues = {}
    ErmsMinTrain = float("Inf")
    ErmsMinValid = float("Inf")
    
    for m in range(43,44):
        randRows = np.random.choice(trainX.shape[0], size=m, replace = False)
        #randRows = np.random.randint(trainX.shape[0], size=m)        
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
        
        while(lamb < 0.2):
            eye = np.eye(m, k = lamb)
            
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
            
            tempIndex = int(lamb*10)
            contourMatrix[m][tempIndex] = ErmsValid
            
            if(ErmsValid < ErmsMinValid):
                ErmsMinValid = ErmsValid
                validMinValues['m'] = m
                validMinValues['lamb'] = lamb
                validMinValues['phiValid'] = phiValid
                validMinValues['w'] = w
                validMinValues['rms'] = ErmsValid
                validMinValues['mu'] = mu
            lamb += 0.1
    
        
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
    plt.title("LeToR Closed Form Solution")
    plt.show()

    fig = plt.figure(2)    
    plt.imshow(contourMatrix, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()
    #stochastic gradient descent  #0.3
    eta = 1
    mSh = validMinValues['m']
    costValues = []
    wSh = np.ones((mSh,1))
    iterations = 50
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
    ErmsShTrain = np.sqrt(ErmsSh/trainLen)
    
    phiWSh = np.dot(phiValid,wSh)
    TminusPSh = validY - phiWSh    
    ErmsSh = np.dot(np.transpose(TminusPSh),TminusPSh)
    ErmsShValid = np.sqrt(ErmsSh/validLen)
    
    phiWSh = np.dot(phiTest,wSh)
    TminusPSh = testY - phiWSh    
    ErmsSh = np.dot(np.transpose(TminusPSh),TminusPSh)
    ErmsShTest = np.sqrt(ErmsSh/testLen)
    
    
    predic = np.dot(phiTest,wSh)
    graphX = list(range(testLen))
    
    plt.figure(3)
    plt.plot(graphX,predic,'r--', graphX, testY, 'b--')
    plt.xlabel("data points")
    plt.ylabel("values")
    plt.title("LeToR Stochastic Gradient Descent")
    plt.show()
    
    graphCostX = list(range(iterations))
    plt.figure(4)
    plt.scatter(graphCostX, costValues)
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.title("LeToR Cost Function for eta = 1")
    plt.show()
    
    
    

#def synthetic():
    
    
    
if __name__ == "__main__":
    print "Hello World"
    #letor()
    
    
    #synthetic()m = 56 eta = 0.03"""
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
    
    """randNums = np.random.choice(length, size = length, replace = False)
    trainX = X[randNums[0:trainLen]]
    trainY = Y[randNums[0:trainLen]]
    validX = X[randNums[trainLen:trainLen + validLen]]
    validY = Y[randNums[trainLen:trainLen + validLen]]
    testX = X[randNums[trainLen + validLen:]]
    testY = Y[randNums[trainLen + validLen:]]"""
    
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
    
    contourMatrix = np.zeros(shape=(50,6))

    
    trainMinValues = {}
    validMinValues = {}
    ErmsMinTrain = float("Inf")
    ErmsMinValid = float("Inf")
    
    for m in range(39,40):
        randRows = np.random.choice(trainX.shape[0], size=m, replace = False)
        #randRows = np.random.randint(trainX.shape[0], size=m)        
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
                
        lamb = 0.00
        
        while(lamb < 0.02):
            eye = np.eye(m, k = lamb)
            
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
            
            tempIndex = int(lamb*10)
            contourMatrix[m][tempIndex] = ErmsValid
            
            if(ErmsValid < ErmsMinValid):
                ErmsMinValid = ErmsValid
                validMinValues['m'] = m
                validMinValues['lamb'] = lamb
                validMinValues['phiValid'] = phiValid
                validMinValues['w'] = w
                validMinValues['rms'] = ErmsValid
                validMinValues['mu'] = mu
            lamb += 0.01
    
        
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
    plt.ylabel("Target and Predicted values")
    plt.title("Synthetic Closed Form Solution")
    plt.show()

    fig = plt.figure(2)    
    plt.imshow(contourMatrix, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()
    #stochastic gradient descent  #0.3
    eta = 0.01
    mSh = validMinValues['m']
    costValues = []
    wSh = np.ones((mSh,1))
    iterations = 50
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
    ErmsShTrain = np.sqrt(ErmsSh/trainLen)
    
    phiWSh = np.dot(phiValid,wSh)
    TminusPSh = validY - phiWSh    
    ErmsSh = np.dot(np.transpose(TminusPSh),TminusPSh)
    ErmsShValid = np.sqrt(ErmsSh/validLen)
    
    phiWSh = np.dot(phiTest,wSh)
    TminusPSh = testY - phiWSh    
    ErmsSh = np.dot(np.transpose(TminusPSh),TminusPSh)
    ErmsShTest = np.sqrt(ErmsSh/testLen)
    
    
    predic = np.dot(phiTest,wSh)
    graphX = list(range(testLen))
    
    plt.figure(3)
    plt.plot(graphX,predic,'r--', graphX, testY, 'b--')
    plt.xlabel("data points")
    plt.ylabel("Target and Predicted values")
    plt.title("Synthetic Stochastic Gradient Descent")
    plt.show()
    
    graphCostX = list(range(iterations))
    plt.figure(4)
    plt.scatter(graphCostX, costValues)
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.title("Synthetic Cost Function for eta = 0.003")
    plt.show()
        
    
    