#!/usr/bin/env Python3
'''
    This file will read in data and start your mlp network.
    You can leave this file mostly untouched and do your
    mlp implementation in mlp.py.
'''
# Feel free to use numpy in your MLP if you like to.
import numpy as np
import mlp

filename = '../data/movements_day1-3.dat'

movements = np.loadtxt(filename,delimiter='\t')

# Subtract arithmetic mean for each sensor. We only care about how it varies:
movements[:,:40] = movements[:,:40] - movements[:,:40].mean(axis=0)

# Find maximum absolute value:
imax = np.concatenate(  ( movements.max(axis=0) * np.ones((1,41)) ,
                          np.abs( movements.min(axis=0) * np.ones((1,41)) ) ),
                          axis=0 ).max(axis=0)

# Divide by imax, values should now be between -1,1
movements[:,:40] = movements[:,:40]/imax[:40]

# Generate target vectors for all inputs 2 -> [0,1,0,0,0,0,0,0]
target = np.zeros((np.shape(movements)[0],8));
for x in range(1,9):
    indices = np.where(movements[:,40]==x)
    target[indices,x-1] = 1

# Randomly order the data
order = list(range(np.shape(movements)[0]))
np.random.shuffle(order)
movements = movements[order,:]
target = target[order,:]

# Split data into 3 sets

# Training updates the weights of the network and thus improves the network
train = movements[::2,0:40]
train_targets = target[::2]


# Validation checks how well the network is performing and when to stop
valid = movements[1::4,0:40]
valid_targets = target[1::4]

# Test data is used to evaluate how good the completely trained network is.
test = movements[3::4,0:40]
test_targets = target[3::4]

# Plots the error over epochs, tells us what to set the minimal error to in early stopping
"""
net = mlp.mlp(train, train_targets, 8)
net.plotvaliderror(train, train_targets, valid, valid_targets)

# Try networks with different number of hidden nodes:
for hidden in [6, 8, 12]:
    # Initialize the network:
    net = mlp.mlp(train, train_targets, hidden)
    net.earlystopping(train, train_targets, valid, valid_targets)

    # Check how well the network performed:
    print('%s hidden nodes:' %hidden)
    net.confusion(test, test_targets)
"""

"""
# Implementing k-cross validation
# Interpreting it as splitting the data in to k different pieces
# Then take out the same data each time as testing,
# but use different k-s as validation and training
"""

# Testk is the same testing
# Use testk as test for all k
testk = movements[400::,0:40]
testk_targets = target[400::]

restk = movements[:400, 0:40]
restk_targets = target[:400]

# Trenger å ta med tilhørende targets!!!




# Need to split restk into 10 different pieces
# Then use new piece as validation each time

# Will split up restk into 10 pieces
k = 10
# Now starts the k-fold
for i in range(k):
    # This just splits it up in even pieces
    # Doesn't necesserily work for other k
    testpieces = []
    testpieces_targets = []
    for j in range(k):
        testpieces.append(restk[j*40:j*40 + 40 ,0:40])
        testpieces_targets.append(restk_targets[j*40:j*40 + 40])

    kvalid = testpieces[i]
    testpieces.pop(i)
    kvalid_targets = testpieces_targets[i]
    testpieces_targets.pop(i)
    ktest = []
    ktest_targets = []
    for j in range(k-1):
        ktest.append(testpieces[j][0:40])
        ktest_targets.append(testpieces_targets[j])

    # Check that all the lists are correct















#
