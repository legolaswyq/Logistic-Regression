import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.optimize as op
import util as ut

# microchip test result  test1, test2
# base on the test result determine if the microchip is accepted or not

filename = "ex2data2.txt"
data = np.loadtxt(filename, delimiter=",")

X = data[:, :2]
y = data[:, 2]


# plot data set
# def plot_data(X, y):
#     # find pos points and neg points
#     pos_idx = np.where(y == 1)
#     pos_points = X[np.asarray(pos_idx[0])]
#
#     neg_idx = np.where(y == 0)
#     neg_points = X[np.asarray(neg_idx[0])]
#
#     plt.scatter(pos_points[:, 0], pos_points[:, 1], marker="+")
#     plt.scatter(neg_points[:, 0], neg_points[:, 1], marker=".")
#     plt.show()
#
#
# plot_data(X,y)


# feature mapping (x1,x2) return all polynomial combination of these two feature
def feature_mapping(x1, x2):
    degree = 6
    row = x1.shape[0]
    out = np.ones(row)

    for i in range(1, degree + 1):
        for k in range(i + 1):
            new_feature = np.power(x1, i - k) * np.power(x2, k)
            out = np.vstack([out, new_feature])

    return out.T


x1 = X[:, 0]
x2 = X[:, 1]
features = feature_mapping(x1, x2)


def sigmoid(x):
    return 1 / (1 + np.power(np.e, -x))


def cost_function(theta, X, y, lamb):
    h = sigmoid(np.dot(X, theta))
    m = len(y)
    cost = 1 / m * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + lamb / m * np.sum(np.power(theta[1:], 2))
    return cost


def gradient_function(theta, X, y, lamb):
    h = sigmoid(np.dot(X, theta))
    m = len(y)
    # not regularized the theta0
    gradient0 = (1 / m) * (h - y).T.dot(X[:, 0])
    gradient_other = (1 / m) * (h - y).T.dot(X[:, 1:]) + lamb / m * theta[1:]
    gradient = np.hstack([gradient0, gradient_other])
    return gradient


initial_theta = np.zeros(features.shape[1])
lamb = 1
cost = cost_function(initial_theta, features, y, lamb)
gradient = gradient_function(initial_theta, features, y, lamb)
#
#
optimized_theta = op.fmin_bfgs(cost_function, initial_theta, fprime=gradient_function, args=(features, y, lamb),
                               maxiter=1000)


# print(optimized_theta)


def plot_decision_boundary(theta, X, y):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.ndarray([len(u), len(v)])

    u1, v1 = np.meshgrid(u, v)
    for idx in range(len(u1)):
        z[:, idx] = np.dot(feature_mapping(u1[:, idx], v1[:, idx]), theta)

    # for the decision boundary, our cost function is sigmoid function
    # therefore when z > 0 is true
    # when z < 0 is false, so we need to plot a contour = 0
    plt.contour(u,v,z,0)
    plt.show()


ut.plot_data(X,y)
plot_decision_boundary(optimized_theta, X, y)


sys.exit()
