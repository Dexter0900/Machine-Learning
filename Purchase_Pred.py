import pandas as pd

purchase = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Customer%20Purchase.csv')

purchase.head()

purchase.info()

purchase.describe()

purchase.columns

y = purchase['Purchased']

X = purchase.drop(['Purchased','Customer ID'],axis = 1)

# encoding categorical variable
X.replace({'Review':{'Poor':0,'Average':1,'Good':2}},inplace=True)
X.replace({'Education':{'School':0,'UG':1,'PG':2}},inplace=True)
X.replace({'Gender':{'Male':0,'Female':1}},inplace=True)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=2529)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

confusion_matrix(y_test,y_pred)

accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred))