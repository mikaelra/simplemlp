"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np

class mlp:
    def __init__(self, inputs, targets, nhidden):
        self.beta = 1
        self.eta = 0.1
        self.momentum = 0.0
        self.bias = -1
        self.inputs = inputs
        self.targets = targets
        self.nhidden = nhidden

        self.inputamount = len(self.inputs[0])
        self.outputamount = len(self.targets[0])
        # Need to construct the weight network
        # One weight from each node, pluss the bias node
        self.wlayer1 = np.ones((self.inputamount + 1, nhidden))
        self.wlayer2 = np.ones((nhidden + 1, self.outputamount))

        self.hiddennodes = np.zeros(self.nhidden)
        self.outputnodes = np.zeros(self.outputamount)



    # You should add your own methods as well!

    def earlystopping(self, inputs, targets, valid, validtargets):
        print('To be implemented')

    def train(self, inputs, targets, iterations=100):
        # for loop for x antall iterasjoner
        # og kanskje en for-loop for antall train targets

        currentinput = inputs[-1]
        currenttarget = targets[-1]
        delta_k, delta_j = self.backphase(self.forward(currentinput), currenttarget)

        # Change the weights in the second layer
        for j in range(self.nhidden):
            for k in range(self.outputamount):
                self.wlayer2[j][k] -= self.eta * delta_k[k] * self.hiddennodes[j]

        # Change the weights on the bias for the outputnodes
        for k in range(self.outputamount):
            self.wlayer2[-1][k] -= self.eta * delta_k[k] * self.bias


        # Change the weights in the first layer
        for i in range(self.inputamount):
            for j in range(self.nhidden):
                self.wlayer1[i][j] -= self.eta * delta_j[j] * currentinput[i]

        # Change the weights for the bias node
        for j in range(self.nhidden):
            self.wlayer1[-1][j] -= self.eta * delta_j[j] * self.bias

        print('--trained--')



    def backphase(self, outputs, targetoutputs):
        # Assumes all the data from the last forward is still stored
        # This should calculate the difference in the weights

        print('outputs:')
        print(outputs)
        print('target outputs:')
        print(targetoutputs)

        # Calculate the delta_k's
        dif = np.array(outputs - targetoutputs)
        der_out = np.zeros(self.outputamount)
        for i in range(self.outputamount):
            der_out[i] = self.relu_d(outputs[i])

        delta_k = dif * der_out

        # Calculate the delta_j's from the hidden layers
        delta_j = np.zeros(self.nhidden)

        for j in range(self.nhidden):
            for k in range(len(delta_k)):
                delta_j[j] += delta_k[k]*self.wlayer2[j][k]
                delta_j[j] *= self.sigmoid_function_d(self.hiddennodes[j])
                # Trenger også å oppdatere biasen

        return delta_k, delta_j


    def forward(self, inputs):
        # Forward works, a perfect neural net would give the right answer everytime

        # Set up calculations of layer 1
        for i in range(self.inputamount):
            for j in range(self.nhidden):
                # Calcluates the sum of inputs
                self.hiddennodes[j] += inputs[i]*self.wlayer1[i][j]
        # Need to add bias weight
        for i in range(self.nhidden):
            self.hiddennodes[i] += self.bias*self.wlayer1[-1][i]

        # Calculate the output from the hidden nodes
        for j in range(self.nhidden):
            self.hiddennodes[j] = self.sigmoid_function(self.hiddennodes[j])


        # Start on new layer
        for i in range(self.nhidden):
            for j in range(self.outputamount):
                self.outputnodes[j] += self.hiddennodes[i]*self.wlayer2[i][j]
        # Need to add bias weight
        for i in range(self.outputamount):
            self.outputnodes += self.bias*self.wlayer2[-1][i]
        # Calculate output to the last nodes, using relu
        for j in range(self.outputamount):
            self.outputnodes[j] = self.relu(self.outputnodes[j])

        return self.outputnodes



    def confusion(self, inputs, targets):
        print('To be implemented')

    def sigmoid_function(self, x):
        return 1./(1 + np.exp(x))

    def sigmoid_function_d(self, x):
        return self.sigmoid_function(x)*(1 - self.sigmoid_function(x))

    def relu(self, x):
        if x <= 0:
            return 0
        else:
            return x

    def relu_d(self, x):
        if x <= 0:
            return 0
        else:
            return 1

    def errorfunc(self, outputs, expectedoutputs):
        sum = 0
        for i in range(len(outputs)):
            sum+= (outputs[i] - expectedoutputs[i])**2
        return 1./2 * sum

    # The derivative of the bias would be 0, so define the biasfunction as x
    def biasfunc(self, x):
        return x
    # So it will always change if there is an error
    def biasfunc_d(self, x):
        return 1
