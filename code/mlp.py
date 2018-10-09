"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np

class mlp:
    def __init__(self, inputs, targets, nhidden):
        self.beta = 1
        self.eta = 0.5
        self.momentum = 0.0
        self.inputs = inputs
        self.targets = targets
        self.nhidden = nhidden

        self.inputamount = len(self.inputs[0])
        self.outputamount = len(self.targets[0])
        # Need to construct the weight network
        # One weight from each node, pluss the bias node
        self.wlayer1 = np.random.randn(self.inputamount + 1, nhidden)
        self.wlayer2 = np.random.randn(nhidden + 1, self.outputamount)

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

        for j in range(self.nhidden):
            for k in range(self.outputamount):
                self.wlayer2[j][k] -= self.eta * delta_k[k] * self.hiddennodes[j]
                # Mangler nok enda endring på bias node

        for i in range(self.inputamount):
            for j in range(self.nhidden):
                self.wlayer1[i][j] -= self.eta * delta_j[j] * currentinput[i]
                # Forandre bias node

        print('--trained--')



    def backphase(self, outputs, targetoutputs):
        # Assumes all the data from the last forward is still stored
        # This should caclulate the difference in the weights

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
        # Mangler delta_k til bias vekt
        # TRENGER VI BIAS-VEKT HER??

        # Calculate the delta_j's from the hidden layers
        delta_j = np.zeros(self.nhidden)

        for j in range(self.nhidden):
            for k in range(len(delta_k)):
                delta_j[j] += delta_k[k]*self.wlayer2[j][k]
                # Må gange me den deriverte av noe mhp j
                # Trenger også å oppdatere biasen

        return delta_k, delta_j


    def forward(self, inputs):
        # This shall calculate the outputs by running the MLP one time

        # Set up calculations of layer 1
        bias = -1
        for i in range(self.inputamount):
            for j in range(self.nhidden):
                # Calcluates the sum of inputs
                self.hiddennodes[j] += inputs[i]*self.wlayer1[i][j]
        # Need to add bias weight
        for i in range(self.nhidden):
            self.hiddennodes[i] += bias*self.wlayer1[-1][i]

        # Calculate the output from the hidden nodes
        for j in range(self.nhidden):
            self.hiddennodes[j] = self.sigmoid_function(self.hiddennodes[j])

        # Start on new layer
        for i in range(self.nhidden):
            for j in range(self.outputamount):
                self.outputnodes[j] += self.hiddennodes[i]*self.wlayer2[i][j]
        # Need to add bias weight
        for i in range(self.outputamount):
            self.outputnodes += bias*self.wlayer2[-1][i]
        # Calculate output to the last nodes, using relu
        for j in range(self.outputamount):
            self.outputnodes[j] = self.relu(self.outputnodes[j])

        return self.outputnodes



    def confusion(self, inputs, targets):
        print('To be implemented')

    def sigmoid_function(self, x):
        return 1./(1 + np.exp(x))

    def sigmoid_function_d(self, x):
        return sigmoid_function(x)*(1 - sigmoid_function(x))

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
