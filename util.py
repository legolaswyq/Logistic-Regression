import numpy as np
import matplotlib.pyplot as plt


def plot_data(X, y):
    # find pos points and neg points
    X = X[:,1:]
    pos_idx = np.where(y == 1)
    pos_points = X[np.asarray(pos_idx[0])]

    neg_idx = np.where(y == 0)
    neg_points = X[np.asarray(neg_idx[0])]

    print(pos_points,neg_points)

    plt.scatter(pos_points[:, 0], pos_points[:, 1], marker="+")
    plt.scatter(neg_points[:, 0], neg_points[:, 1], marker=".")


def sigmoid(x):
    return 1 / (1 + np.power(np.e, -x))


def cost_function(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    m = len(y)
    cost = 1 / m * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))
    gradient = (1 / m) * (h - y).T.dot(X)
    return [cost, gradient]


def predict(theta, X):
    x = np.dot(X, theta)
    return sigmoid(x) >= 0.5


def plot_decision_boundary(theta, X, y):
    # base on the predict hypothesis we could come out the decision boundary
    # 1/(1 + e^-(theta1 + theta2*x1 + theta3*x2)) = 1/2
    # 1 + e^-(theta1 + theta2*x1 + theta3*x2) = 2
    # e^-(theta1 + theta2*x1 + theta3*x2) = 1
    # log each side
    # -(theta1 + theta2*x1 + theta3*x2) = 0
    # x2 = -1/theta3 * (theta1 + theta2 * x1)
    # need two point the draw a line
    min_x1 = np.min(X[:, 2])
    max_x1 = np.max(X[:, 2])
    x1 = np.asarray([min_x1, max_x1])
    x2 = -1 / theta[2] * (theta[0] + theta[1] * x1)
    plt.plot(x1, x2)

    plot_data(X,y)

    plt.show()
