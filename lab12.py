# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:17:36 2019

@author: Sefa
"""


import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        k = 0
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
                k = k+1
        print("counter", k)
                
                

training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

#labels = np.array([1, 0, 0, 0])  # and
labels = np.array([0, 1, 1, 0])   # xor

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)

inputs = np.array([1, 1])
perceptron.predict(inputs) 
#=> 1

print(inputs)
print(perceptron.predict(inputs))

#inputs = np.array([0, 1])
inputs = np.array([1, 1])
perceptron.predict(inputs) 
#=> 0

print(inputs)
perceptron.predict(inputs) 

print(perceptron.weights)