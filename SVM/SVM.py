#SVM using a dataset in Python using scikit-learn.


#Cell 1 — Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#📌 Cell 2 — Load Dataset

iris = datasets.load_iris()

X = iris.data      # features
y = iris.target    # labels

print(X.shape, y.shape)
o/p
(150, 4) (150,)


#📌 Cell 3 — Convert to DataFrame (Optional, for viewing)
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df.head()

#📌 Cell 4 — Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#📌 Cell 5 — Create SVM Model
svm_model = SVC(kernel='linear')   # you can try 'rbf', 'poly', 'sigmoid'
svm_model.fit(X_train, y_train)

#📌 Cell 6 — Prediction

y_pred = svm_model.predict(X_test)
print(y_pred)

[1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]

#📌 Cell 7 — Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
o/p
Accuracy: 1.0


print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
o/p
Confusion Matrix:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
o/p

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

#📌 Cell 8 — Test With New Sample

sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = svm_model.predict(sample)

print("Predicted class:", iris.target_names[prediction][0])

o/p
Predicted class: setosa










