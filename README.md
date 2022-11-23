# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Read the given dataset.
2.Fitting the dataset into the training set and test set.
3.Applying the feature scaling method.
4.Fitting the logistic regression into the training set.
5.Prediction of the test and result
6.Making the confusion matrix
6.Visualizing the training set results.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by : P Hariharan
RegisterNumber :  212220040038
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets = pd.read_csv("/content/Social_Network_Ads (1).csv")
X = datasets.iloc[:,[2,3]].values
Y = datasets.iloc[:,[4]].values

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X

X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.fit_transform(X_Test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_Train, Y_Train)

Y_Pred = classifier.predict(X_Test)
Y_Pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)
cm

from sklearn import metrics
accuracy = metrics.accuracy_score(Y_Test, Y_Pred)
accuracy

recall_sensitivity = metrics.recall_score(Y_Test, Y_Pred, pos_label = 1)
recall_specificity = metrics.recall_score(Y_Test, Y_Pred, pos_label = 0)
recall_sensitivity, recall_specificity

from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
X1,X2 = np.meshgrid(np.arange(start = X_Set[:,0].min()-1, stop = X_Set[:,0].max()+1, step = 0.01), 
                    np.arange(start = X_Set[:,1].min()-1, stop = X_Set[:,1].max()+1, step = 0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),
X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(), X2.max())
plt.ylim(X1.min(), X2.max())
for i,j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j,0],X_Set[Y_Set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.label('Estimated Salary')
plt.legend()
plt.show()
```

## Output:
### Prediction of Test Result

![x1](https://user-images.githubusercontent.com/95471278/174228270-242f5f9f-f68a-40ca-af09-48db4595a641.jpg)

### Confusion Matrix:

![x2](https://user-images.githubusercontent.com/95471278/174228279-04e0abd8-2dbd-4c5e-bde7-cdd719dd109a.jpg)

### Accuracy:

![x3](https://user-images.githubusercontent.com/95471278/174228287-4dbf79bf-026b-4ddb-a85b-55f624c5a092.jpg)

### Recalling Sensitivity and Specificity:
![x4](https://user-images.githubusercontent.com/95471278/174228299-ff61596f-0647-419f-9b7e-2d71e1eec182.jpg)


### Visulaizing Training set Result:
![x5](https://user-images.githubusercontent.com/95471278/174228311-8cd74836-db44-4b41-b8e5-4a154fdd9262.jpeg)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
