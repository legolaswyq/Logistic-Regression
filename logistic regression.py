import numpy as np
import sys
import matplotlib.pyplot as plt
import util as ut
import scipy.optimize as op

# base on exam one score and exam two score
# predict a student could be enter an University
# data    exam1 exam2 (1 or 0)

filename = "ex2data1.txt"
data = np.loadtxt(filename, delimiter=",")

X = data[:, :2]
y = data[:, 2]

# visualize data
# ut.plot_data(X,y)

# add ones to X
# ones dimension need to match X
ones = np.ones([X.shape[0], 1])
X = np.hstack([ones, X])  # (100,3)

# init theta
theta = np.zeros(X.shape[1])  # (3,)


# helper function to use fmin_bfgs
# important note, theta must be a (n,) ndarray
# the cost function return a float number
def f(theta, X, y):
    cost, gradient = ut.cost_function(theta, X, y)
    return cost


def fprime(theta, X, y):
    cost, gradient = ut.cost_function(theta, X, y)
    return gradient


result = op.fmin_bfgs(f, x0=theta, args=(X, y), fprime=fprime, full_output=True)
print(f"the theta is {result[0]}")
print(f"the current cost is {result[1]}")

theta = result[0]

p = ut.predict(theta,X)
print(np.mean(p == y))

ut.plot_decision_boundary(theta,X,y)

sys.exit()
