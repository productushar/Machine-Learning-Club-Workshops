# ML Workshop 4: K-Nearest Neighbour (KNNs) 

#########################################################################################

# Lazy learning: No data generalization and no descriminative function
# Most commonly used for recommendation systems
# Nearest neighbor -> Euclidean Distance -> sqrt((x2-x1)^2 - (y2-y1)^2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.model_selection import train_test_split

#########################################################################################

data = pd.read_csv("penguins.csv")
data = data.dropna()
data.head()

#########################################################################################

sns.scatterplot(x=data["bill_length_mm"],y = data["bill_depth_mm"],hue = data["species"])

#########################################################################################

X = data.drop(["species","island","sex","body_mass_g","flipper_length_mm"],axis=1)
Y = data["species"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.3,random_state=25)

#########################################################################################

def euclidean(point_1,point_2):
  distance = np.sqrt(np.sum((point_1-point_2)**2))
  return distance

euclidean(np.array([0,0]), np.array([1,1]))

#########################################################################################

def KNN(X_data, Y_labels, kvalue, X_points):
  
  predictions = []

  for X_test in X_points.to_numpy():
    distances = []

    for i in range(len(Y_train)):
      distances.append(euclidean(np.array(X_train.iloc[i]), X_test))

    distance_dataframe = pd.DataFrame(distances, columns=["Distance"], index=Y_train.index)

    neighbors = distance_dataframe.sort_values(by=["Distance"], axis = 0)[:kvalue]
    neighbor_classes = Y_train.loc[neighbors.index]

    predictions.append(mode(neighbor_classes).mode[0])

  return predictions

#########################################################################################

prediction = KNN(X_train, Y_train, kvalue = 7, X_points=X_test)

#########################################################################################

new_point = pd.DataFrame({"bill_length": [43], "bill_depth": [18]})
new_prediction = KNN(X_train, Y_train, 9, new_point)
print(new_prediction)

#########################################################################################

accuracy_values = []

for i in range(1,20):
  
  y_hat_test = KNN(X_train, Y_train, kvalue = i, X_points = X_test)
  correct = 0
  incorrect = 0
  
  for i in range(len(y_hat_test)):
    
    if(y_hat_test[i] == Y_test.iloc[i]):
      correct+=1
      
    else:
      incorrect+=1

  accuracy = float(correct) / float((incorrect+correct))
  accuracy_values.append(accuracy)

plt.xticks(np.arange(0,20,step = 1))
plt.plot(range(1,20), accuracy_values, color = "red", linestyle = "dashed", marker = 'x')

#########################################################################################
