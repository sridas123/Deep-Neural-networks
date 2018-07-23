"""
Created on Tue Mar 15 00:56:36 2016

@author: Srijita
"""

import mnist_loader
import network_grad
import matplotlib.pyplot as plt
import numpy as np
#import sys

path='D:/git/Neuralnet/Graphs/Gradient_first_1000/'
#sys.stdout = open(path+"out.txt", "w")

COLORS = ['#2A6EA6', '#FFCD33', '#FF7033','#FF00BF','#00FF00']
#'#B0649A','#9B80F2'
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network_grad.Network([784,30,30,30,30,10])

"""Below are the arguments of method SGD"""
"""training_data, epochs, mini_batch_size, eta,test_data=None"""

train_cost,test_accuracy,backprop_grad,weight_mag=net.SGD(training_data, 1, 1, 0.1, test_data=test_data)
"""Plotting the costs vs the epochs"""

fig = plt.figure()
ax = fig.add_subplot(111)
i=1

#print train_cost_final
#print test_accuracy_final
NUM_EPOCHS=len(backprop_grad)
NUM_INTERVAL=len(weight_mag)

barray=[]
warray=[]
bfarray=[]
wfarray=[]

bgrad={x: [] for x in range(0,len(backprop_grad[0]))}
wmag={x: [] for x in range(0,len(weight_mag[0]))}
#print bgrad
#print wmag

#for i in range(0,len(backprop_grad)):
#    print backprop_grad[i]

for i in range(0,len(backprop_grad)):
    for j in range(0,len(backprop_grad[0])):
        bgrad[j].append(backprop_grad[i][j])

for i in range(0,len(weight_mag)):
    for j in range(0,len(weight_mag[0])):
        wmag[j].append(weight_mag[i][j])
        
for i in range(0,len(backprop_grad[0])):
    bfarray.append(bgrad[i])
        
for i in range(0,len(weight_mag[0])):
    wfarray.append(wmag[i])
    
""" Code to plot the back propagated gradients"""  

fig = plt.figure()
ax = fig.add_subplot(111)

i=1

for grad, color in zip(bfarray,COLORS):
    ax.plot(np.arange(NUM_EPOCHS), grad, "o-",
            label="$layer$ = "+str(i),
            color=color)
    i=i+1  
    
ax.set_xlim([0, NUM_EPOCHS])
ax.set_xlabel('Update intervals')
ax.set_ylabel('Back_propagated gradients')
plt.legend(loc='upper right')
plt.savefig(path+'Back_propagated_gradient_all_first_1000.png', bbox_inches='tight')
plt.clf()
plt.cla()

