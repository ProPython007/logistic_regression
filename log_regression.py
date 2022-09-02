# Logistic Regression in Python

import numpy as np
from matplotlib import pyplot as plt

# Loading data:
data = np.genfromtxt('week_3.csv', delimiter=',')
x = data[:,0:2]
y = data[:,2]

# Data Scaling:
x = np.insert(x, 0, 1, axis=1)
range_x1 = max(x[:,1])-min(x[:,1])
range_x2 = max(x[:,2])-min(x[:,2])
x[:,1] = x[:,1]/range_x1
x[:,2] = x[:,2]/range_x2

# Data Plotting:
pos , neg = (y==1).reshape(100,1) , (y==0).reshape(100,1)
plt.scatter(x[pos[:,0],1], x[pos[:,0],2], color="orange", s=10)
plt.scatter(x[neg[:,0],1], x[neg[:,0],2], color="blue", s=10)

# Constants:
prev_costfunc_y = -999
istrained = 0
alpha = 0.1
m = len(y)

# Initializint the parameters:
theta = [0 for _ in range(3)]

# Sigmoid Function:
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Logistic Function:
def logistic_func(x):
    return sigmoid(np.sum(theta*x))

# Cost Function:
def cost_func():
    global istrained, prev_costfunc_y
    sum = 0

    for i in range(m):
        sum += y[i]*np.log(logistic_func(x[i])) + (1 - y[i])*np.log(1 - logistic_func(x[i]))
    jy = (-1/m)*sum

    if jy == prev_costfunc_y:
        istrained = 1
        return jy
    else:
        prev_costfunc_y = jy
        return jy

# Derivative cost_func() w.r.t (theta)j:
def d_cost_func(j):
    sum = 0
    for i in range(m):
        sum += (logistic_func(x[i]) - y[i])*x[i][j]

    return (1/m)*sum

# Gradient Descent Function:
def update():
    global theta
    temp = [0 for _ in range(len(theta))]

    for i in range(len(theta)):
        temp[i] = theta[i] - alpha*d_cost_func(i)

    for i in range(len(theta)):
        theta[i] = temp[i]

# Training:
for i in range(10**5):
    update()
    print("Pass:", i)
    print("Cost function value:", cost_func())
    print("theta = {}".format(theta))
    print()
    if istrained:
        print("Model fully trained at pass = {} with alpha = {}".format(i-1, alpha))
        break

# Decision Boundary:
def boundary(x):
    return -(theta[0] + theta[1]*x)/theta[2]

# Plotting the prediction:
x_value = np.array([np.min(x[:,1]),np.max(x[:,1])-0.1])
y_value = boundary(x_value)
plt.plot(x_value, y_value, label='Decision Boundary Plot', color='red')
plt.ylim(np.array([np.min(x[:,2])-0.1,np.max(x[:,2])+0.1]))
plt.legend()
plt.xlabel('x1 Data -->')
plt.ylabel('x2 Data -->')
plt.title('Logistic Regression in Python')
plt.tight_layout()
plt.show()
