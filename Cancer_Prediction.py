import pandas as pd

cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv')

cancer.head()

cancer.info()

cancer.describe()

cancer.columns

y = cancer['diagnosis']

X = cancer.drop(['diagnosis','Unnamed: 32','id'],axis = 1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)

