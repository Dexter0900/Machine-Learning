import pandas as pd

fish = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Fish.csv')

fish.head()

fish.info()

fish.describe()

fish.columns

y = fish['Weight']
X = fish.drop(['Category', 'Species', 'Weight'],axis = 1)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=2529)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error

mean_absolute_percentage_error(y_test,y_pred)

mean_absolute_error(y_test,y_pred)

