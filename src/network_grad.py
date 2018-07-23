"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.backprop_grad=[]
        self.weight_mag=[]   
        self.update_weight_magnitude()   
        self.counter=0              

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        
        tcost=[]
        taccuracy=[]
        
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        if test_data: n_test = len(test_data)
        #print n_test
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}  Accuracy: {3}".format(
                    j, self.evaluate(test_data), n_test,
                    (float(self.evaluate(test_data)*100)/n_test))
                accuracy=float(self.evaluate(test_data)*100)/n_test
                cost= self.total_cost(training_data,False)  
                tcost.append(cost)
                taccuracy.append(accuracy)
                print "The length of the weight magnitudes are",len(self.backprop_grad),len(self.weight_mag)
            else:
                print "Epoch {0} complete".format(j)
        #for i in range(0,len(self.weight_mag)):
        #    if (self.backprop_grad[i][0] > self.backprop_grad[i][4]):
        #    print self.weight_mag[i]
            
        return  tcost,taccuracy,self.backprop_grad,self.weight_mag    

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.counter=self.counter+1
            
        bgrad_all_layers=[]   
        
        if self.counter<=1000:
           for i in range(0,len(nabla_w)):    
               magnitude=np.linalg.norm(nabla_w[i]) 
               bgrad_all_layers.append(magnitude)
            
           self.backprop_grad.append(bgrad_all_layers)
        
#        for i in range (0,len(self.backprop_grad)):
#            print self.backprop_grad[i]
        
        #print "The shape of nabla_b and nabla_w is",len(nabla_w)
        #print "The shape of self.bias and self.weights is",len(self.weights)
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
                            
        self.update_weight_magnitude()
        
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            #print activation
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        #print "activation last layer is",activations[-2]
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #print "magnitude of last layer",np.linalg.norm(nabla_w[-1]) 
        
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            #print "activation last layer is",activations[-l-1]
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            #print "magnitude of other layer",np.linalg.norm(nabla_w[-l])
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
        
    def total_cost(self, data,convert=False):
        
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            
            if convert: y = vectorized_result(y)
                
            cost += self.qcost(a, y)
        
        """This isthe average training cost"""
        return float(cost/len(data))
        
    def qcost(self,a, y):
     
        """Return the cost associated with an output ``a`` and desired output y."""
        return 0.5*np.linalg.norm(a-y)**2
        
    def update_weight_magnitude(self):
     
        """Update the weight magnitude array after mini batch update of size 10"""
        weight_all_layers=[]   
        
        for i in range(0,len(self.weights)):    
            magnitude=np.linalg.norm(self.weights[i]) 
            weight_all_layers.append(magnitude)
            
        self.weight_mag.append(weight_all_layers)

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere. This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
    
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
