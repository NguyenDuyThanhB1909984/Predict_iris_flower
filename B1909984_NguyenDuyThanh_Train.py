import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score


data = pd.read_csv("iris.csv")
X = data.iloc[:,0:4]
print(X)
y = data.variety
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state= 42)

model = tree.DecisionTreeClassifier(criterion="gini", max_depth=3)
model.fit(X_train, y_train)

pickle.dump(model, open("pima.pickle.dat", "wb"))
print("Saved model to: pima.pickle.dat")

loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
print("Loaded model from: pima.pickle.dat")

predictions = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
predict = loaded_model.predict([[5,4,2,0.5]])
print(predict[0])
