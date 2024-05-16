import pandas as pd

diab = pd.read_csv("https://github.com/YBIFoundation/Dataset/raw/main/Diabetes.csv")

diab.head()

diab.info()

diab.describe()

diab.columns

y = diab["diabetes"]
X = diab[['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi',
       'dpf', 'age']]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=2529)

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=500)

# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier()

# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()

model.fit(X_train,y_train,)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)

