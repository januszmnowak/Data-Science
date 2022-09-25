import numpy as np

"""Assume we have regression: 𝑓(𝐱) = 2*x1 - 3*x2 + 5
We need to find weights for x1 and x2 [2,3] and bias [5]
We use matrix X for stacked x1 and x2.
Using matrices we have regression: y = weights * X + bias"""

#input data creation
observations=1000
x1 = np.random.uniform(low=-10, high=10, size=(observations,1))
x2 = np.random.uniform(-10, 10, (observations,1))
noise = np.random.uniform(-1, 1, (observations,1))
y=2*x1-3*x2+5+noise
X = np.column_stack((x1,x2)) #two column matrix from x1 and x2

#random vector initialization
init_range=0.1
weights = np.random.uniform(low=-init_range, high=init_range, size=(2, 1)) #start vector with two values as weight for x1 and x2 estimate
bias = np.random.uniform(low=-init_range, high=init_range, size=1) #start vector(scalar) with one value as bias estimate

#gradient descent parameters
learn_rate=0.02
n_iter=1000

def gradient_weights(X,y,weights):
    residual=(np.dot(X,weights)+bias) - y
    residual_scaled = residual / len(X)
    return np.dot(X.T,residual_scaled)

def gradient_bias(X,y,bias):
    residual=(np.dot(X,weights)+bias) - y
    residual_scaled = residual / len(X)
    return np.sum(residual_scaled)

for i in range(n_iter):
    weights = weights - learn_rate * gradient_weights(X,y,weights)
    bias = bias - learn_rate * gradient_bias(X,y,bias)
    print (i, weights, bias)