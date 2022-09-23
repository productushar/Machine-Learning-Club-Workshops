############################################################################

# MLC Workshop 1: Intro to Python

############################################################################

# numpy arrays: memory efficient, really fast; actively used for ML

import numpy as np
a = np.array([0,1,2,3,4,5])

print(a)

for element in a:
  print(element)

b = np.array([[0,1,2,3,4,5],[6,7,8,9,10,11]])

print(b)

print(a.dtype)

b = b.astype('str')

for element in b[0]:
  print(type(element))
  
############################################################################

# Pandas library: super useful
import pandas as pd
csv = pd.read_csv('financial_data_sp500_companies_1.csv')
xlsx = pd.read_excel('financial_data_sp500_companies_final.xlsx')

# Allows to see first 5 rows
csv.head()
xlsx.head()

# Dropping all entries under Unnamed: 0
xlsx = xlsx.drop("Unnamed: 0", axis = 1)
csv = csv.drop("Unnamed: 0", axis = 1)

# (rows,columns)
csv.shape[1]

csv.info()

csv.shape

# To strip on either sides: lstrip() rstrip()

############################################################################

csv = csv.dropna()
csv.isna().sum()

############################################################################

#Data Visualization 

import matplotlib.pyplot as plt
plt.hist(csv['Gross Profit'])

############################################################################

plt.scatter(csv['Operating Income'],csv['Gross Profit'])
plt.xlabel('Operating Income')
plt.ylabel('Gross Profit')
plt.title('Gross Profit vs Operating Income')
plt.style.use('ggplot')
plt.show()

############################################################################

import seaborn as sns

correlations = csv.corr()
sns.heatmap(correlations, annot = True)

############################################################################
