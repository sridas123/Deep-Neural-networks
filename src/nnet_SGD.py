"""
Created on Tue Mar 15 00:56:36 2016

@author: Srijita
"""
import mnist_loader
import network
import matplotlib.pyplot as plt
import numpy as np

NUM_EPOCHS=10
COLORS = ['#2A6EA6', '#FFCD33', '#FF7033']
path='D:/git/Neuralnet/Graphs/'
train_cost_final=[]
test_accuracy_final=[]

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784,30,10])
"""Below are the arguments of method SGD"""
"""training_data, epochs, mini_batch_size, eta,test_data=None"""

train_cost,test_accuracy=net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
train_cost_final.append(train_cost)
test_accuracy_final.append(test_accuracy)

net = network.Network([784,30,30,10])
train_cost,test_accuracy=net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
train_cost_final.append(train_cost)
test_accuracy_final.append(test_accuracy)


net = network.Network([784,30,30,30,10])
train_cost,test_accuracy=net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
train_cost_final.append(train_cost)
test_accuracy_final.append(test_accuracy)

"""Plotting the costs vs the epochs"""

fig = plt.figure()
ax = fig.add_subplot(111)
i=1

#print train_cost_final
#print test_accuracy_final

for train_cost, color in zip(train_cost_final, COLORS):
    ax.plot(np.arange(NUM_EPOCHS), train_cost, "o-",
            label="$layer$ = "+str(i),
            color=color)
    i=i+1        
ax.set_xlim([0, NUM_EPOCHS])
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Cost')
plt.legend(loc='upper right')
plt.savefig(path+'Epoch_cost_all_layers.png', bbox_inches='tight')
    
plt.clf()
plt.cla()

"""Plotting the accuracy vs the epochs"""

fig1 = plt.figure()
ax = fig1.add_subplot(111)
i=1
for accuracy, color in zip(test_accuracy_final, COLORS):
    ax.plot(np.arange(NUM_EPOCHS), accuracy, "o-",
            label="$layer$ = "+str(i),
            color=color)
    i=i+1        
ax.set_xlim([0, NUM_EPOCHS])
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
plt.legend(loc='upper right')
plt.savefig(path+'Epoch_accuracy_all_layers.png', bbox_inches='tight')
