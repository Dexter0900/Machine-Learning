import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv(r'https://github.com/YBIFoundation/Dataset/raw/main/Big%20Sales%20Data.csv')

df.head()

df.info()

df.columns

df.describe()


#Encoding

df['Item_Weight'].fillna(df.groupby(['Item_Type'])['Item_Weight'].transform('mean'),inplace=True)

df.info()

sns.pairplot(df)

df['Item_Fat_Content'].value_counts()

df.replace({'Item_Fat_Content':{'Low Fat':0,'Regular':1,'LF':0,'reg':1,'low fat':0}},inplace=True)

df[['Item_Type']].value_counts()

df.replace({'Item_Type':{'Fruits and Vegetables':0,'Snack Foods':0,'Household':1,
                         'Frozen Foods':0,'Dairy':0,'Baking Goods':0,
                         'Canned':0,'Health and Hygiene':1,
                         'Meat':0,'Soft Drinks':0,'Breads':0,'Hard Drinks':0,
                         'Others':2,'Starchy Foods':0,'Breakfast':0,'Seafood':0
                         }},inplace=True)

df[['Item_Type']].value_counts()

df[['Outlet_Identifier']].value_counts()

df.replace({'Outlet_Identifier':{'OUT027':0,'OUT013':1,'OUT035':2,'OUT046':3,'OUT049':4,
                                 'OUT045':5,'OUT018':6,'OUT017':7,'OUT010':8,'OUT019':9
                                 }},inplace=True)

df[['Outlet_Identifier']].value_counts()

df[['Outlet_Size']].value_counts()

df.replace({'Outlet_Size':{'Small':0,'Medium':1,'High':2}},inplace=True)

df[['Outlet_Size']].value_counts()

df[['Outlet_Location_Type']].value_counts()

df.replace({'Outlet_Location_Type':{'Tier 1':0,'Tier 2':1,'Tier 3':2}},inplace=True)

df[['Outlet_Location_Type']].value_counts()

df[['Outlet_Type']].value_counts()

df.replace({'Outlet_Type':{'Supermarket Type1':0,'Grocery Store':1,'Supermarket Type3':2,
                           'Supermarket Type2':3,}},inplace=True)

df[['Outlet_Type']].value_counts()

df.head()

df.info()

df.shape

#Define Features and Labels

y = df['Item_Outlet_Sales']

y.shape

X = df.drop(['Item_Outlet_Sales','Item_Identifier'],axis=1)

X.shape

print(X)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_std = df[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']]

X_std = sc.fit_transform(X_std)

print(X_std)

X[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']] = pd.DataFrame(X_std, columns = [['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']])

print(X)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.1, random_state=2529)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=2529)

rfr.fit(X_train,y_train)

y_pred = rfr.predict(X_test)

y_pred.shape

print(y_pred)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mean_squared_error(y_test,y_pred)

mean_absolute_error(y_test,y_pred)

r2_score(y_test,y_pred)

import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Price vs Predicted Price')
plt.show()