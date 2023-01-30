# DEPENDENCIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px

# normalizing features
from sklearn.preprocessing import StandardScaler
# training/validation split
from sklearn.model_selection import train_test_split
# get performance metrics after model fitting
from sklearn.tree import DecisionTreeClassifier
# decision tree classifier
from sklearn.metrics import classification_report
# random forest classifier
from sklearn.ensemble import RandomForestClassifier
# adaboost classifier
from sklearn.ensemble import AdaBoostClassifier
# gradient boost classifier
from sklearn.ensemble import GradientBoostingClassifier

# xgboost classifier
import xgboost as xgb


# DATASET

# df = pd.read_csv("data/winequality-red_cleaned.csv")
df = pd.read_csv("data/winequality-white_cleaned.csv")


# DATA EXPLORATION

## number of rows and columns
# print("Rows, columns: " + str(df.shape))

## first five rows of the dataset
# print(df.head())

## missing values
# print(df.isna().sum())

## quality distribution
# fig = px.histogram(df,x='quality')
# fig.show()

## correlation matrix
# corr = df.corr()
# plt.subplots(figsize=(15,10))
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(240, 170, center="light", as_cmap=True))
# plt.show()


# DATA PREPROCESSING

## make binary quality classification
df['good'] = [1 if x >= 7 else 0 for x in df['quality']]
## separate feature and target variables
X = df.drop(['quality', 'good'], axis = 1)
y = df['good']
## check distribution
# print(df['good'].value_counts())
## print first 5 rows
# print(df.head())

## normalize feature
X_features = X
X = StandardScaler().fit_transform(X)

## 25/75 val/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)


# MODEL FITTING

## decision tree classifier
tree_model = DecisionTreeClassifier(random_state=42)
## use training dataset for fitting
tree_model.fit(X_train, y_train)
## run prediction based of the validation dataset
y_pred1 = tree_model.predict(X_test)
## get performance metrics
print("")
print(":: DecisionTreeClassifier ::")
print("")
print(classification_report(y_test, y_pred1))


## random forrest classifier
forrest_model = RandomForestClassifier(random_state=42)
## use training dataset for fitting
forrest_model.fit(X_train, y_train)
## run prediction based of the validation dataset
y_pred2 = forrest_model.predict(X_test)
## get performance metrics
print("")
print(":: RandomForestClassifier ::")
print("")
print(classification_report(y_test, y_pred2))


## adaboost classifier
adaboost_model = AdaBoostClassifier(random_state=42)
## use training dataset for fitting
adaboost_model.fit(X_train, y_train)
## run prediction based of the validation dataset
y_pred3 = adaboost_model.predict(X_test)
## get performance metrics
print("")
print(":: AdaBoostClassifier ::")
print("")
print(classification_report(y_test, y_pred3))


## gradient boost classifier
gradient_model = GradientBoostingClassifier(random_state=42)
## use training dataset for fitting
gradient_model.fit(X_train, y_train)
## run prediction based of the validation dataset
y_pred4 = gradient_model.predict(X_test)
## get performance metrics
print("")
print(":: GradientBoostingClassifier ::")
print("")
print(classification_report(y_test, y_pred4))


## xgboost classifier
xgboost_model = xgb.XGBClassifier(random_state=42)
## use training dataset for fitting
xgboost_model.fit(X_train, y_train)
## run prediction based of the validation dataset
y_pred5 = xgboost_model.predict(X_test)
## get performance metrics
print("")
print(":: XGBClassifier ::")
print("")
print(classification_report(y_test, y_pred5))


# FEATURE IMPORTANCE

## RandomForestClassifier
feat_importances_forrest = pd.Series(forrest_model.feature_importances_, index=X_features.columns)
feat_importances_forrest.nlargest(11).plot(kind='pie', figsize=(10,10), title="Feature Importance :: RandomForestClassifier")
plt.show()

## XGBClassifier
feat_importances_xbg = pd.Series(xgboost_model.feature_importances_, index=X_features.columns)
feat_importances_xbg.nlargest(11).plot(kind='pie', figsize=(10,10), title="Feature Importance :: XGBClassifier")
plt.show()


# get only good wines
df_good = df[df['good']==1]
print(":: Wines with good Quality ::")
print("")
print(df_good.describe())
# get only bad wines
df_bad = df[df['good']==0]
print("")
print(":: Wines with bad Quality ::")
print("")
print(df_bad.describe())