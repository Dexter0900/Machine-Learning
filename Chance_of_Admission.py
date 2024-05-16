import pandas as pd

admission = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Admission%20Chance.csv')

admission.head()

admission.info()

admission.describe()

admission.columns

y = admission['Chance of Admit ']
X = admission.drop(['Serial No','Chance of Admit '], axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train,y_train)

model.intercept_

model.coef_

y_pred = model.predict(X_test)

print(y_pred)

from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(y_test,y_pred)

