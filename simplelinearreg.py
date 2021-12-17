import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


dataset = pd.read_csv('AmazonProductRating.csv')

# print sum of null cells in each column
# print(X.isnull().sum())
# print(Y.isnull().sum())

# Drop the rows with null values in manufacturer col
dataset.dropna(subset=['average_rating'], how='all', inplace=True)
dataset.dropna(subset=['manufacturer'], how='all', inplace=True)

X = dataset['manufacturer']
Y = dataset['average_rating']

# Encoding manufacturer values
le = preprocessing.LabelEncoder()
X = le.fit_transform(X.values)
# print(X)

# Encoding average_rating values
for index, value in Y.items():
    value = value[:-15]
    Y[index] = value
Y = pd.to_numeric(Y, errors='coerce')
print(Y.isnull().sum())
Y.fillna(Y.mean(), inplace=True)


X = np.expand_dims(X, axis=1)
Y = np.expand_dims(Y, axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, shuffle=True)


# Plotting
plt.scatter(X_train, Y_train)
plt.xlabel('manufacturer', fontsize=20)
plt.ylabel('avg_rating', fontsize=20)
plt.show()


lm = linear_model.LinearRegression()
lm.fit(X_train, Y_train)
prediction = lm.predict(X_test)
print('Co-efficient of linear regression', lm.coef_)
print('Intercept of linear regression model', lm.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y_test, prediction))

# Plotting
plt.scatter(X_test, prediction)
plt.xlabel('manufacturer', fontsize=20)
plt.ylabel('avg_rating', fontsize=20)
plt.show()
