# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
~~~
Step 1

Import the required Python libraries such as NumPy, Matplotlib, and scikit-learn for data handling, visualization, and model creation.

Step 2

Load the dataset and separate it into independent variables (X) and dependent variable (y).

Step 3

Split the dataset into training data and testing data using the train-test split method.

Step 4

Create the Multivariate Linear Regression model and train it using the training dataset.

Step 5
~~~
Predict the output using the test data and evaluate the model performance using appropriate metrics and plots.
## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

housing = datasets.fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

print('Coefficients: ', reg.coef_)
print('Variance score: {}'.format(reg.score(X_test, y_test)))

plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(X_train),
            reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

plt.scatter(reg.predict(X_test),
            reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

plt.hlines(y=0, xmin=0, xmax=5, linewidth=2)
plt.legend(loc='upper right')
plt.title('Residual Errors')
plt.show()
```
## Output:

### Insert your output

<img width="1576" height="932" alt="mai10(1)" src="https://github.com/user-attachments/assets/63e2b559-58e2-45f7-9a3d-4c23292eb212" />
<img width="1315" height="801" alt="mai10(2)" src="https://github.com/user-attachments/assets/20ae3c74-1a5a-40d6-84eb-cfa89da420b5" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
