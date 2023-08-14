#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist



#load data and preprocess data
(train_images, train_labels), (test_images,test_labels)=mnist.load_data()#include all numbers from 0 to 9 
index_train=np.where((train_labels ==2) | (train_labels ==6))#index of numbers 3 and 5 in training data
index_test=np.where((test_labels ==2) | (test_labels ==6))#index of numbers 3 and 5 in test data
train_images_26=train_images [index_train] 
train_images_26=train_images_26.reshape( (len(train_images_26), train_images_26[1].size)) 
#label of number 2: -1; label of number 6: +1 
train_labels_26=train_labels[index_train].astype('int') 
test_images_26=test_images[index_test] 
test_images_26=test_images_26.reshape((len(test_images_26),train_images_26[1].size)) 
test_labels_26=test_labels[index_test].astype('int')
#change labels from 2 and '6' to -1' and +1 
train_labels_26[np.where(train_labels_26==2)]= -1
train_labels_26[np.where(train_labels_26==6)]= 1
test_labels_26[np.where(test_labels_26==2)]= -1
test_labels_26[np.where(test_labels_26==6)]= 1
#print(test_labels_26)

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_images_26[i].reshape((28,28)),)
#     plt.xlabel('number'+ str(train_images_26[i]))
# plt.show()

#append dummy feature 1 to feature vectors, and then normalize 
train_images_26_w_dummy=np.insert(train_images_26,784,1,axis=1)/255
test_images_26_w_dummy=np.insert(test_images_26,784,1,axis=1)/255
#choose a subset of the enntire training dataset
train_images_26_w_dummy=train_images_26_w_dummy[range(1000)]
train_labels_26_w_dummy=train_labels_26[range(1000)]
#check the dimension, the feature vector of each sample shall be 785
print(train_images_26_w_dummy.shape)
print(test_images_26_w_dummy.shape)
print(train_labels_26_w_dummy.shape)


#implementing Perceptron Algorithm
aplha = 0.1
#creating W 1 vector
W=np.ones([785, 1])#785x1
testW = np.ones([785, 1])#785x1
iterations = 0
acc = 0
e= 0

# #training the train data
while(acc <=950):
    iterations = iterations + 1
    print("Iteration : " ,iterations)
    acc=0
    for i in range(0,999):
        X=train_images_26_w_dummy[i,:]
        #reshaping the X for dot product
        Xmatrix = np.reshape(X,(785,1))#785x1
        Y=train_labels_26_w_dummy[i]
        if(float(np.dot(W.T, Xmatrix)* Y) <= 0): #dot(1x785), 785x1
            #updating W when the wtranspose x xi x yi <0
            W = W + (aplha*Xmatrix*Y)
        else:
            acc = acc+1
    print("Perceptron Training Accuracy",acc)
    accrate = acc/10
    print(" Perceptron Traning Accuracy Percentage", accrate)
    testW = W

#testing the model on test data
test_acc = 0

for j in range(0,1989):
    X=test_images_26_w_dummy[j,:]
    Xmatrix = np.reshape(X,(785,1))#785x1
    Y=test_labels_26[j]
    if(float(np.dot(testW.T, Xmatrix)* Y) <= 0):#dot(1x785), 785x1
            continue
    else:
        test_acc = test_acc+1
print("----------------------------")
print("Perceptron Test Accuracy",test_acc)
test_accrate = (test_acc/1990)*100
print(" Perceptron Test Accuracy Rate", test_accrate)


######Findings for Perceptron Algorithm
#for stopping criteria at 80% , Training acc of 90.3 is attained at first iteration and Test acc of 95.72
#for stopping criteria at 95%, Traing acc of 96.2 is attained at second iteration and Test acc of 93.11

#implementing knn algorithm
difference = []
mind = 0
k1acc = 0
k1accrate = 0

for row in range(0,999):
    sourcepoint = train_images_26_w_dummy[row,:]
    for check in range(0,999):
            dist = np.sqrt(np.sum(np.square(np.subtract(sourcepoint,train_images_26_w_dummy[check,:]))))
            difference.append(dist)
    mindindex = difference.index(min(d for d in difference if d>0))
    predictedY = train_labels_26_w_dummy[mindindex]
    if(predictedY == train_labels_26_w_dummy[row]):
        k1acc = k1acc +1 
    difference = []
    k1accrate = k1acc/10
print("K(k=1) means on Training Data ",k1accrate)

testk1acc = 0
testk1accrate = 0

for row in range(0,1989):
    sourcepoint = test_images_26_w_dummy[row,:]
    for check in range(0,1989):
            dist = np.sqrt(np.sum(np.square(np.subtract(sourcepoint,test_images_26_w_dummy[check,:]))))
            difference.append(dist)
    mindindex = difference.index(min(d for d in difference if d>0))
    predictedY = test_labels_26[mindindex]
    if(predictedY == test_labels_26[row]):
        testk1acc = testk1acc +1
    difference = []
    testk1accrate = (k1acc/199)*10
print("K(k=1) means on Testing Data",testk1accrate)

##################for k = 3
k2acc = 0
k2accrate = 0
testk2acc = 0
testk2accrate = 0
for row in range(0,999):
    sourcepoint = train_images_26_w_dummy[row,:]
    for check in range(0,999):
            dist = np.sqrt(np.sum(np.square(np.subtract(sourcepoint,train_images_26_w_dummy[check,:]))))
            difference.append(dist)
    mindindex1 = difference.index(min(d for d in difference if d>0))
    difference[mindindex1] = 0
    mindindex2 =  difference.index(min(d for d in difference if d>0))
    difference[mindindex2] = 0
    mindindex3 =  difference.index(min(d for d in difference if d>0))
    dec = [train_labels_26_w_dummy[mindindex1], train_labels_26_w_dummy[mindindex2], train_labels_26_w_dummy[mindindex3]]
    predictedY = max(dec, key=dec.count)
    #print("Predicted Y", predictedY, "Real Y", train_labels_26_w_dummy[row])
    if(predictedY == train_labels_26_w_dummy[row]):
        k2acc = k2acc +1
    difference = []
    k2accrate = k2acc/10

print("K(k=3) means on Training Data",k2accrate)

for row in range(0,1989):
    sourcepoint = test_images_26_w_dummy[row,:]
    for check in range(0,1989):
            dist = np.sqrt(np.sum(np.square(np.subtract(sourcepoint,test_images_26_w_dummy[check,:]))))
            difference.append(dist)
    mindindex1 = difference.index(min(d for d in difference if d>0))
    difference[mindindex1] = 0
    mindindex2 =  difference.index(min(d for d in difference if d>0))
    difference[mindindex2] = 0
    mindindex3 =  difference.index(min(d for d in difference if d>0))
    dec = [test_labels_26[mindindex1], test_labels_26[mindindex2], test_labels_26[mindindex3]]
    predictedY = max(dec, key=dec.count)
    #print("Predicted Y", predictedY, "Real Y", train_labels_26_w_dummy[row])
    if(predictedY == test_labels_26[row]):
        testk2acc = testk2acc +1
    difference = []
    testk2accrate = (testk2acc/199)*10

print("K(k=3) means on Testing Data",testk2accrate)

######################for k = 5

k3acc = 0
k3accrate = 0
testk3acc = 0
testk3accrate = 0


for row in range(0,999):
    sourcepoint = train_images_26_w_dummy[row,:]
    for check in range(0,999):
            dist = np.sqrt(np.sum(np.square(np.subtract(sourcepoint,train_images_26_w_dummy[check,:]))))
            difference.append(dist)
    mindindex1 = difference.index(min(d for d in difference if d>0))
    difference[mindindex1] = 0
    mindindex2 =  difference.index(min(d for d in difference if d>0))
    difference[mindindex2] = 0
    mindindex3 =  difference.index(min(d for d in difference if d>0))
    difference[mindindex3] = 0
    mindindex4 =  difference.index(min(d for d in difference if d>0))
    difference[mindindex4] = 0
    mindindex5 =  difference.index(min(d for d in difference if d>0))
    dec = [train_labels_26_w_dummy[mindindex1], train_labels_26_w_dummy[mindindex2], train_labels_26_w_dummy[mindindex3]]
    predictedY = max(dec, key=dec.count)
    #print("Predicted Y", predictedY, "Real Y", train_labels_26_w_dummy[row])
    if(predictedY == train_labels_26_w_dummy[row]):
        k3acc = k3acc +1
    difference = []
    k3accrate = k3acc/10

print("K(k=5) means on Training Data",k3accrate)


for row in range(0,1989):
    sourcepoint = test_images_26_w_dummy[row,:]
    for check in range(0,1989):
            dist = np.sqrt(np.sum(np.square(np.subtract(sourcepoint,test_images_26_w_dummy[check,:]))))
            difference.append(dist)
    mindindex1 = difference.index(min(d for d in difference if d>0))
    difference[mindindex1] = 0
    mindindex2 =  difference.index(min(d for d in difference if d>0))
    difference[mindindex2] = 0
    mindindex3 =  difference.index(min(d for d in difference if d>0))
    difference[mindindex3] = 0
    mindindex4 =  difference.index(min(d for d in difference if d>0))
    difference[mindindex4] = 0
    mindindex5 =  difference.index(min(d for d in difference if d>0))
    dec = [test_labels_26[mindindex1], test_labels_26[mindindex2], test_labels_26[mindindex3],
            test_labels_26[mindindex4],test_labels_26[mindindex5]]
    predictedY = max(dec, key=dec.count)
    #print("Predicted Y", predictedY, "Real Y", train_labels_26_w_dummy[row])
    if(predictedY == test_labels_26[row]):
        testk3acc = testk3acc +1
    difference = []
    testk3accrate = (k3acc/199)*10
print("K(k=5) means on Testing Data",testk3accrate)


     


                  
                  


           

           
    






