#Batches

import numpy as np
inputs = [[1,2,3,2.5],
            [2.0,5.0,-1.0,2.0],
            [-1.5,2.7,3.3,-0.8]
          ]
weights =[[0.2,0.8,-0.5,1.0],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]
#need to transpose weights to match dimensions for dot product
output = np.dot(inputs ,np.array(weights).T ) + biases

print(output)

o/p
[[ 4.8    1.21   2.385]
 [ 8.9   -1.81   0.2  ]
 [ 1.41   1.051  0.026]]


#Batches ex-2

import numpy as np
inputs = [[1,2,3,2.5],
            [2.0,5.0,-1.0,2.0],
            [-1.5,2.7,3.3,-0.8]
          ]
weights =[[0.2,0.8,-0.5,1.0],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

weights2 =[[0.1,-0.14,0.5],
         [-0.5,0.12,-0.33],
         [-0.44,0.73,-0.13],]

biases2 = [-1,2,-0.5]

#need to transpose weights to match dimensions for dot product
layer1_outputs = np.dot(inputs ,np.array(weights).T ) + biases
layer2_outputs = np.dot(layer1_outputs ,np.array(weights2).T ) + biases2

print(layer2_outputs)


o/p

[[ 0.5031  -1.04185 -2.03875]
 [ 0.2434  -2.7332  -5.7633 ]
 [-0.99314  1.41254 -0.35655]]

#Batches ex-2

import numpy as np

np.random.seed(0)

X = [[1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
    def forward(self):
        pass


print(np.random.randn(4,3))
o/p
[[ 1.76405235  0.40015721  0.97873798]
 [ 2.2408932   1.86755799 -0.97727788]
 [ 0.95008842 -0.15135721 -0.10321885]
 [ 0.4105985   0.14404357  1.45427351]]

#Batches ex-2,modification

import numpy as np

np.random.seed(0)

X = [[1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)

o/p
[[ 0.10758131  1.03983522  0.24462411  0.31821498  0.18851053]
 [-0.08349796  0.70846411  0.00293357  0.44701525  0.36360538]
 [-0.50763245  0.55688422  0.07987797 -0.34889573  0.04553042]]

#Batches ex-2,modification

import numpy as np

np.random.seed(0)

X = [[1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)

o/p
[[ 0.148296   -0.08397602]
 [ 0.14100315 -0.01340469]
 [ 0.20124979 -0.07290616]]
