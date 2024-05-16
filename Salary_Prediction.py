# Import pandas.
import pandas as pd

# read Data and make a Data Frame.
house = pd.read_csv("https://github.com/YBIFoundation/Dataset/raw/main/Boston.csv")

house.head()

house.info()

house.describe()

house.columns

house.shape

# Declare Y and X.
y = house['MEDV']
X = house[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]



X.shape

y.shape

# import function to split the data for training and testing purpose from the sklearn library.
from sklearn.model_selection import train_test_split

# split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=2529)
# "random_state = 2529" helps to get the same data every time to maintain the consistency in aquracy.
print(X_train)

# select the model you want to use
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train model (fit model)
model.fit(X_train,y_train)

model.intercept_

model.coef_

# Prediction
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(y_test,y_pred)

