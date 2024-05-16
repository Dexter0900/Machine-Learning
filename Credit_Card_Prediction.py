import pandas as pd

cred = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Credit%20Default.csv')

cred.head(10)

cred.info()

cred.describe()

cred.columns

y = cred['Default']
X = cred[['Income', 'Age', 'Loan', 'Loan to Income']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

from sklearn.linear_model import LogisticRegression
model  =  LogisticRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

confusion_matrix(y_test,y_pred)

accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred))