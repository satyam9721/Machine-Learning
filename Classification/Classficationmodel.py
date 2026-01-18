#start
#install dependincies then execute line by line cells in juypter, used `iris date set`.

Building a Classification Model for the Iris data set

## 1. Import libraries

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#execute the above cells fist then execute one by one.


## 2. Load the *iris* data set
iris = datasets.load_iris()

### 3.1. Input features

print(iris.feature_names)

o\p
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


### 3.2. Output features

print(iris.target_names)


print(iris.target)

o/p

[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]

## 4. Glimpse of the data

iris.data

iris.target

X = iris.data
Y = iris.target

X.shape

Y.shape


## 5. Build Classification Model using Random Forest


clf = RandomForestClassifier()

clf.fit(X, Y)

#RandomForestClassifier()


## 6. Feature Importance

print(clf.feature_importances_)

o/p

[0.0897064  0.03104855 0.46491081 0.41433423]


## 7. Make Prediction


X[0]

print(clf.predict([[5.1, 3.5, 1.4, 0.2]]))

print(clf.predict(X[[0]]))

print(clf.predict_proba(X[[0]]))

o/p

[[1. 0. 0.]]

clf.fit(iris.data, iris.target_names[iris.target])

## 8. Data split (80/20 ratio)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train.shape, Y_train.shape

X_test.shape, Y_test.shape


## 9. Rebuild the Random Forest Model

clf.fit(X_train, Y_train)

print(clf.predict([[5.1, 3.5, 1.4, 0.2]]))

o/p
[0]

print(clf.predict_proba([[5.1, 3.5, 1.4, 0.2]]))

o/p
[[1. 0. 0.]]

#predicted class table
print(clf.predict(X_test))

o/p:-
[0 0 0 0 2 1 2 0 2 0 2 2 1 0 2 1 2 2 1 0 1 1 0 0 1 0 0 1 0 2]

clf.fit(iris.data, iris.target_names[iris.target])

#Actual class labels
print(Y_test)

o/p

[0 0 0 0 2 1 2 0 2 0 2 2 1 0 2 1 1 2 1 0 1 1 0 0 1 0 0 1 0 1]

now compare the Xtest and Ytest then compare it , now we get the comparision and overview real values
which is Y_test and predcted value is X_test, so let's calculate the performance of models.


## 10. Model Performance
print(clf.score(X_test, Y_test))

o/p:-

0.9333333























