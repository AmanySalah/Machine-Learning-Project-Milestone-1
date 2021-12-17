from sklearn import preprocessing
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None


dataset = pd.read_csv('AmazonProductRating.csv')
dataset.drop(['sellers', 'uniq_id', 'product_name',
              'amazon_category_and_sub_category', 'product_information'],inplace=True, axis=1)

# Drop the rows with null values in manufacturer, , ,  cols
# print(dataset.describe())
dataset.dropna(subset=['manufacturer', 'number_available_in_stock', 'number_of_reviews',
                       'number_of_answered_questions', 'average_rating'], inplace=True)
X = dataset.iloc[:, 0:7].copy()
Y = dataset['average_rating']

"""Encoding manufacturer values"""
col = 'manufacturer'
le = preprocessing.LabelEncoder()
X[col] = le.fit_transform(X[col].values)

"""convert data in "price" column to numeric values and filling the null values with the mean value"""
col = 'price'
X[col] = pd.to_numeric(X[col].values, errors='coerce')
X[col].fillna(X[col].mean(), inplace=True)
# print(X[col].isnull().sum())

"""convert data in "number_available_in_stock" column to numeric values"""
col = 'number_available_in_stock'
for index, value in X[col].items():
    value = value[:-3]
    X[col][index] = value

X[col] = pd.to_numeric(X[col], errors='coerce')
X[col].fillna(X[col].mean(), inplace=True)

"""convert data in "number_of_reviews" column to numeric values"""
col = "number_of_reviews"
X[col] = pd.to_numeric(X[col], errors='coerce')
X[col].fillna(X[col].mean(), inplace=True)
# print(X[col].isnull().sum())

"""convert data in "number_of_answered_questions" column to numeric values"""
col = "number_of_answered_questions"
X[col] = pd.to_numeric(X[col], errors='coerce')
X[col].fillna(X[col].mean(), inplace=True)
# print(X[col].isnull().sum())

"""convert data in "average_rating" column to numeric values"""
col = 'average_rating'
for index, value in X[col].items():
    value = value[:-15]
    X[col][index] = value
X[col] = pd.to_numeric(X[col], errors='coerce')
X[col].fillna(X[col].mean(), inplace=True)
Y = X['average_rating']
# print(X)

# Feature Selection
"""Get the correlation between the features"""
data_corr = X.corr()
print("correlation:->>>\n", data_corr)
top_feature = data_corr.index[abs(data_corr['average_rating']) > 0.02]
print("Top Features:->>> ", top_feature)
plt.subplots(figsize=(12, 8))
top_corr = X[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)  # removing 'average_rating'
X = X[top_feature]

lm = linear_model.LinearRegression()
lm.fit(X, Y)
prediction = lm.predict(X)
print('Co-efficient of linear regression = ', lm.coef_)
print('Intercept of linear regression model = ', lm.intercept_)
print('Mean Square Error = ', metrics.mean_squared_error(Y, prediction))
