#######################################################################################

# ML3: Intro To Logistic Regression

#######################################################################################

# Don't know about parameters, hypothesis, etc. 

# Binary Classification (2 possible outputs/choices only)

# f(x) = L/(1+e^(-k(x-x0))); L = cap value

# k = parameter 1; x0 = parameter 2; x & y are datapoints

# Error value: Closeness to 1

#######################################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sigmoid(z):
  return 1/(1+np.exp(-1*z))

def error(k, x0, x, y):
  
  n = len(y)
  hypY = sigmoid(k*x+x0)
  
  err = np.sum((y)*-1*np.log(hypY) - (1-y)*np.log(1-hypY))
  err = err * 1/n

  return err

#######################################################################################

df = pd.read_csv("hoodie.csv")

df.head()
x = df["x"]
y = df["y"]

#######################################################################################

k = 0
x0 = 0

EPOCHS = 5000
LEARNING_RATE = 0.007

for i in range(EPOCHS):
  
  prevError = error(k, x0, x, y)

  h = 0.004

  pDerK = (error(k+h, x0, x, y) - prevError)/h
  pDerX0 = (error(k, x0 + h, x, y) - prevError)/h

  k -= LEARNING_RATE * pDerK
  x0 -= LEARNING_RATE * pDerX0

print(error(k, x0, x, y))
plt.scatter(x, y)

#######################################################################################

plt.scatter(x,y)
plt.plot(x, sigmoid(k*x +x0))
print(error(k, x0, x, y))

#######################################################################################
