"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np
import matplotlib.pyplot as plt

class mlp:
    def __init__(self, inputs, targets, nhidden):
        self.beta = 1
        self.eta = 0.1
        self.bias = 1
        self.momentum = 0.0
        self.inputs = inputs
        self.targets = targets
        self.nhidden = nhidden

        self.inputamount = len(self.inputs[0])
        self.outputamount = len(self.targets[0])
        # Need to construct the weight network
        # One weight from each node, pluss the bias node

        # Works greit with just ones as initial weights
        weighth = 1
        weightl = 0.9
        self.wlayer1 = np.random.uniform(low=-weightl, high=weighth, size=(self.inputamount + 1 , self.nhidden))
        self.wlayer2 = np.random.uniform(low=-weightl, high=weighth, size=(self.nhidden + 1 , self.outputamount))

        self.hiddennodes = np.zeros(self.nhidden)
        self.outputnodes = np.zeros(self.outputamount)

    def plotvaliderror(self, inputs, targets, valid, validtargets):
        error = 0
        errorlist = []
        epoch = []
        for i in range(0, 300):
            epoch.append(i)
            self.train(inputs, targets, iterations=10)
            error = 0
            for i in range(len(valid)):
                error += self.errorfunc(self.forward(valid[i]), validtargets[i])
            error /= len(valid)
            errorlist.append(error)
        plt.plot(epoch, errorlist)
        plt.show()



    def earlystopping(self, inputs, targets, valid, validtargets):
        # Implement plotting of the error fucntion to see when
        # it looks like we should stop the training

        # Make a new function for plotting the error

        error = 999
        epoch = 0
        while error > 0.3:
            epoch+=1
            self.train(inputs, targets, iterations=10)
            error = 0
            for i in range(len(valid)):
                error += self.errorfunc(self.forward(valid[i]), validtargets[i])
            error /= len(valid)
            # Add that value to a list, then plot
            # make a for loop for 100 epochs and plot
        print('Training done! %s epochs done' %epoch)


    def train(self, inputs, targets, iterations=100):
        # This runs the algorithm trough al the training data b iterations
        for b in range(iterations):
            choice = np.random.choice(len(inputs))
            #choice = 50
            currentinput = inputs[choice]
            currenttarget = targets[choice]

            delta_k, delta_j = self.backphase(self.forward(currentinput), currenttarget)
            """
            # TESTING
            print('-----------------')
            print('iterasjon nummer ' + str(b))
            print('predicted')
            print(self.forward(currentinput))
            print('expected')
            print(currenttarget)
            #print(delta_k)
            """

            # Change the weights in the second layer
            for j in range(self.nhidden):
                for k in range(self.outputamount):
                    self.wlayer2[j][k] -= self.eta * delta_k[k] * self.hiddennodes[j]
            # Change bias-weight
            for k in range(self.outputamount):
                self.wlayer2[-1][k] -= self.eta * delta_k[k] * self.bias

            # Change the weights in the first layer
            for i in range(self.inputamount):
                for j in range(self.nhidden):
                    self.wlayer1[i][j] -= self.eta * delta_j[j] * currentinput[i]
            # Change bias weight

            for j in range(self.nhidden):
                self.wlayer1[-1][j] -= self.eta * delta_j[j] * self.bias

            """
            #
            print('-')
            print('delta_k')
            print(delta_k)
            print('delta_j')
            print(delta_j)
            print('wlayer2[0]')
            print(self.wlayer2[0])
            print('wlayer1[0]')
            print(self.wlayer1[0])
            """


    def backphase(self, outputs, targetoutputs):
        # Assumes all the data from the last forward is still stored
        # This should calculate the difference in the weights

        # Calculate the delta_k's

        dif = np.zeros(len(outputs))
        for i in range(len(outputs)):
            dif[i] = outputs[i] - targetoutputs[i]
        #dif = np.array(outputs - targetoutputs)

        """
        # Using equation 4.14 from the book, our delta_k is just the difference
        der_out = np.zeros(self.outputamount)
        for i in range(self.outputamount):
            der_out[i] = self.linear_d(outputs[i])
        # Since the derivative is 1, I ignore this
        """
        delta_k = dif #* der_out

        # Calculate the delta_j's from the hidden layers
        delta_j = np.zeros(self.nhidden)

        for j in range(self.nhidden):
            for k in range(len(delta_k)):
                delta_j[j] += delta_k[k]*self.wlayer2[j][k]
            delta_j[j] *= self.sigmoid_function_d(self.hiddennodes[j])

        return delta_k, delta_j


    def forward(self, inputs):
        # Reset the nodes!!
        self.hiddennodes = np.zeros(self.nhidden)
        self.outputnodes = np.zeros(self.outputamount)

        # Forward works, a perfect neural net would give the right answer everytime

        # Set up calculations of layer 1
        for i in range(self.inputamount):
            for j in range(self.nhidden):
                # Calcluates the sum of inputs
                self.hiddennodes[j] += inputs[i]*self.wlayer1[i][j]
        for j in range(self.nhidden):
            self.hiddennodes[j] += self.bias*self.wlayer1[-1][j]

        # Start on new layer
        for i in range(self.nhidden):
            for j in range(self.outputamount):
                self.outputnodes[j] += self.sigmoid_function(self.hiddennodes[i])*self.wlayer2[i][j]
        for j in range(self.outputamount):
            self.outputnodes[j] += self.bias*self.wlayer2[-1][j]
        # Calculate output to the last nodes, using linear function
        for j in range(self.outputamount):
            self.outputnodes[j] = self.linear(self.outputnodes[j])

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

        # Calculates the percentage which are correct
        # Sums up the rows and then divides the correct score on it
        for i in range(len(targets[0])):
            sum = 0
            for j in range(len(targets[0])):
                sum += confmatrix[i][j]
            if sum == 0:
                percentage_vector[i] = 1
            else:
                percentage_vector[i] /= sum


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
        for i in range(len(outputs)):
            sum += (outputs[i] - expectedoutputs[i])**2
        return 1./2 * sum

    # The derivative of the bias would be 0, so define the biasfunction as x
    def biasfunc(self, x):
        return x
    # So it will always change if there is an error
    def biasfunc_d(self, x):
        return 1
