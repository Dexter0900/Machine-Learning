#Hill and Valley Prediction with Logistic Regression

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#Import CSV as DataFrame

df = pd.read_csv(r'https://github.com/YBIFoundation/Dataset/raw/main/Hill%20Valley%20Dataset.csv')

#Get the First Five Rows of DataFrame

df.head()

#Get Information of DataFrame

df.info()

#Get the Summary Statistics

df.describe()

#Get Column Names

print(df.columns)

#All columns name not printed

print(df.columns.tolist())

#Get Shape of DataFrame

print(df.shape)

#Get Unique Values (Class or Label) in y Variable

df['Class'].value_counts()

df.groupby('Class').mean()

#Define y (dependent or label or target variable) and X (independent or features or attribute variable)

y = df['Class']

y.shape

X = df.drop('Class', axis=1)

print(X.shape)

#Get Plot of First Two Rows

plt.plot(X.iloc[0,:])
plt.title('Valley');

plt.plot(X.iloc[1,:])
plt.title('Hill');

#Get X Variables Standardized

#Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn: they might behave badly if the individual features do not more or less look like standard normally distributed data; Gaussian with zero mean and unit variance. Next approach is go for MinMax Scaler

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)

X.shape

#Get Train Test Split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, stratify=y,random_state=2529)

print(X_train.shape, X_test.shape,y_train.shape,y_test.shape)

#Get Model Train

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

#Get Model Prediction

y_pred  = lr.predict(X_test)

y_pred.shape

print(y_pred)

#Get Probability of Each Predicted Class

lr.predict_proba(X_test)

#Get Model Evaluation

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

#Get Future Predictions

'''Lets select a random sample from existing dataset as new value

Steps to follow
1.   Extract a random row using sample function
2.   Separate X and y
3.   Standardize X
4.   Predict
'''


X_new = df.sample(1)

print(X_new)

X_new.shape

X_new = X_new.drop('Class',axis=1)

print(X_new)

X_new = ss.fit_transform(X_new)

y_pred_new = lr.predict(X_new)

print(y_pred_new)

lr.predict_proba(X_new)