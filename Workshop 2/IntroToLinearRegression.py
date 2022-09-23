#####################################################################################################################

# ML2: Intro to Linear Regression

#####################################################################################################################

# Problem Formulation
# Supervised Learning Problem: Model learns from previous examples to predict future outcomes

# Linear Regression: Looking at a scatter of data and predicting a linear model

# Notation: x (input), y (output)
# Set 1: x0, y0 & so on;

# Continuous Output: Definite range of possible outputs
# Discrete Output: Random possible value of output

# y = mx + b; Goal is to find closest possible m and b so as to reduce the distance between the slope and datapoints

# Mean Squared Error: MSE = (1/n) sigma(i = 1 to n) (y'i - yi)^2
#                         = (1/n) sigma(i = 1 to n)  ((mx+b)-yi)^2

# n = number of samples 

# Gradient Descent: Minimize MSE(m,b)
# m = m - learning rate (step size) * gradient (direction of steepest descent)
# Gradient = (2/m) sigma(i = 1 to m) (((mx+b) - yI)*xI) 

# Gradient Descent in 3D: Both m and b require to be adjusted
# b = b - learning rate * (2/m) sigma(i = 1 to m) ((mx+b) - yI)

# Learning rate should be lower for less adjustment; 

#####################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv('Salary_Data.csv')

csv.head()

#####################################################################################################################

# This is uni-varied linear aggression (one input and one output)

plt.scatter(csv['YearsExperience'], csv['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs YOE')
plt.show()

#####################################################################################################################

# y = mx + b

m = 0
b = 0

x = csv['YearsExperience']
y = csv['Salary']

learning_rate = 0.01
epochs = 5000

n = float(x.shape[0])

error = []

for i in range(epochs):
  
  y_hat = m*x +b
  mse = (1/n)*np.sum((y-y_hat)**2)
  error.append(mse)

  gradient_m = (-2/n) * np.sum(x*(y-y_hat))
  gradient_b = (-2/n) * np.sum(y-y_hat)

  m = m - learning_rate * gradient_m
  b = b - learning_rate * gradient_b

  if i % 10 == 0:
    x_line = np.linspace(0,15,10)
    y_line = m*x_line + b

plt.plot(x_line,y_line)
plt.scatter(x,y)
plt.title('Epoch: '+str(i))
plt.show()

#####################################################################################################################
