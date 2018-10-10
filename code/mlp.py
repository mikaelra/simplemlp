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
        self.inputs = inputs
        self.targets = targets
        self.nhidden = nhidden

        self.inputamount = len(self.inputs[0])
        self.outputamount = len(self.targets[0])
        # Need to construct the weight network
        # One weight from each node, pluss the bias node
        self.wlayer1 = np.random.uniform(low=-0.7, high=0.7, size=(self.inputamount , nhidden))
        self.wlayer2 = np.random.uniform(low=-0.7, high=0.7, size=(nhidden , self.outputamount))

        self.hiddennodes = np.zeros(self.nhidden)
        self.outputnodes = np.zeros(self.outputamount)



    # You should add your own methods as well!

    def earlystopping(self, inputs, targets, valid, validtargets):
        errorsum = 1e18
        #while(errorsum/len(valid) > 0.055):
        for i in range(0, 10):
            errorsum = 0
            self.train(inputs, targets, iterations=10)
            for i in range(len(valid)):
                errorsum += self.errorfunc(self.forward(valid[i]), validtargets[i])
            print('errorsummen er nå:')
            print(errorsum/len(valid))

    def train(self, inputs, targets, iterations=100):
        # This runs the algorithm trough al the training data b iterations
        for b in range(iterations):
            choice = np.random.choice(len(inputs))
            currentinput = inputs[choice]
            currenttarget = targets[choice]

            delta_k, delta_j = self.backphase(self.forward(currentinput), currenttarget)

            # Change the weights in the second layer
            for j in range(self.nhidden):
                for k in range(self.outputamount):
                    self.wlayer2[j][k] -= self.eta * delta_k[k] * self.hiddennodes[j]

            # Change the weights in the first layer
            for i in range(self.inputamount):
                for j in range(self.nhidden):
                    self.wlayer1[i][j] -= self.eta * delta_j[j] * currentinput[i]

    def backphase(self, outputs, targetoutputs):
        # Assumes all the data from the last forward is still stored
        # This should calculate the difference in the weights

        # Calculate the delta_k's

        #
        #
        #   SOMETHING IS WRONG WHEN CALCLATING THE DELTAS
        #   OR IN BACKPROPAGATION!
        #
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

        return delta_k, delta_j


    def forward(self, inputs):
        # Forward works, a perfect neural net would give the right answer everytime

        # Set up calculations of layer 1
        for i in range(self.inputamount):
            for j in range(self.nhidden):
                # Calcluates the sum of inputs
                self.hiddennodes[j] += inputs[i]*self.wlayer1[i][j]

        # Start on new layer
        for i in range(self.nhidden):
            for j in range(self.outputamount):
                self.outputnodes[j] += self.sigmoid_function(self.hiddennodes[i])*self.wlayer2[i][j]
        # Calculate output to the last nodes, using linear function
        for j in range(self.outputamount):
            self.outputnodes[j] = self.relu(self.outputnodes[j])

        """
        # Not sure if I should convert output to only ones and zeros
        # Outputs have now numbers on every row, change so that only one of ouputs is one
        index_max = np.argmax(self.outputnodes)
        outputsp = np.zeros(len(self.outputnodes))
        outputsp[index_max] = 1
        self.outputnodes = outputsp
        """

        return self.outputnodes


    # Confusion matrix produces confusion matrix and how a percentage vector
    # of how well our neural network works
    def confusion(self, inputs, targets):
        confmatrix = np.zeros((len(targets[0]),len(targets[0])))
        percentage_vector = np.zeros((len(targets[0])))
        for i in range(len(inputs)):
            # This adds the predicted value to the actual vector
            # A perfect neural network produces only values on the diagonal
            pred = self.forward(inputs[i])
            confmatrix[np.argmax(pred)][:] += targets[i][:]

            # Adds to percentage_vector
            actual = np.argmax(targets[i])
            if np.argmax(pred) == actual:
                # Adds one top the vector if it predicted correct
                percentage_vector[actual] += 1
        percentage_vector /= len(inputs)
        print('confusion matrix:')
        print(confmatrix)
        print('Percentage correct on each class:')
        print(percentage_vector)

    def sigmoid_function(self, x):
        return 1./(1 + np.exp(-x))

    def sigmoid_function_d(self, x):
        return self.sigmoid_function(x) * (1 - self.sigmoid_function(x))

    def relu(self, x):
        if x > 0:
            return x
        else:
            return 0
    def relu_d(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def linear(self, x):
        return x

    def linear_d(self, x):
        return 1

    def errorfunc(self, outputs, expectedoutputs):
        sum = 0
        if np.argmax(outputs) != np.argmax(expectedoutputs):
            sum += 1
        return 1./2 * sum

    # The derivative of the bias would be 0, so define the biasfunction as x
    def biasfunc(self, x):
        return x
    # So it will always change if there is an error
    def biasfunc_d(self, x):
        return 1
