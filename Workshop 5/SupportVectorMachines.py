#################################################################################################

# ML Workshop 5 - SVM: Support Vector Machines

#################################################################################################

# Linear SVM tries to find a linear model that separates 2 classes 
# Support vectors: closest set of vectors on either side
# Goal: Obtain large positive margin

#################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np

#################################################################################################

data = pd.read_csv('data.csv')
data.head()

#################################################################################################

data['diagnosis'] = data['diagnosis'].replace({'B' : 0, 'M' : 1})
data['diagnosis'].unique()
plt.scatter(data['texture_mean'], data['radius_mean'],c=data['diagnosis'])

#################################################################################################

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data['texture_mean'], data['radius_mean'], data['diagnosis'], c = data['diagnosis'])
plt.show()

#################################################################################################

x = data[['texture_mean','radius_mean']]
y = data['diagnosis']

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=.3,random_state=69)

model = SVC(kernel = 'linear')
model.fit(X_train,Y_train)

#################################################################################################

model.coef_

#################################################################################################

x1 = np.linspace(-10,30,100)
x2 = np.linspace(-10,30,100)

x1,x2 = np.meshgrid(x1,x2)
print(model.coef_[0][0])
print(model.coef_[0][1])

#################################################################################################

model.intercept_

#################################################################################################

equation = (-model.intercept_[0] - model.coef_[0][0]*x1 - model.coef_[0][1]*x2)/model.coef_[0][1]
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(data['texture_mean'], data['radius_mean'], data['diagnosis'], c = data['diagnosis'])
ax.plot_surface(x1,x2,equation)
ax.view_init(20,10)
plt.show()

#################################################################################################
