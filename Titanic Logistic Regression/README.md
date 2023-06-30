# Titanic Logistic Regression

Code copied from [Python Maratón's sklearn Logistic Regression Video](https://www.youtube.com/watch?v=VK6v9Ure8Lk). The scikit-learn and pandas packages made the logistic regression model and set up the data frame respectively.

## In-Depth Documentation: [Logistic Regression.ipynb](/Titanic%20Logistic%20Regression/Logistic%20Regression.ipynb)

### Setting up the machine learning model

```python
# Imports:
import pandas as pd  # For setting up the data frame
from sklearn.linear_model import LogisticRegression  # For the logistic regression model
from sklearn.model_selection import train_test_split  # For setting up the train-test modularization
from sklearn.metrics import accuracy_score  # For testing accuracy
import numpy as np  # For the coefficient data frame
import joblib  # Persistent model

# Constants:
DATASET_FILE_NAME = 'titanic.csv'

INDEP_VAR = ['Pclass', 'Sex', 'Age']
DEP_VAR = 'Survived'

CAT_VAR = ['Sex']

MODEL_NAME = 'titanic-model.joblib'
```

First, I imported the necessary packages for manipulating the data set, making and training the logistic regression, testing accuracy, and exporting the model.

The constants were used to categorize the variables in the data set and improve readability. 'Survived' was the dependent variable because it depended on the independent variables `'Pclass'`, `'Sex'`, and `'Age'`. `'Name'`, `'Siblings/Spouses Aboard'`, and `'Parents/Children Aboard'` did not affect the person's survival, so these variables were not included. `'Sex'` was a categorical variable because its values were non-numerical.

<br>

### Making and cleaning the data frame

```python
# Data frame:
X, y = pd.read_csv(DATASET_FILE_NAME, usecols=INDEP_VAR), pd.read_csv(DATASET_FILE_NAME)[DEP_VAR]
X = pd.get_dummies(X, columns=CAT_VAR, drop_first=True)
```

The goal of machine learning models is to use data from independent variables to predict the result of a dependent variable. By splitting up the original data frame into its dependent and independent variables, I could now use them to train the model. I accomplished this using `pandas`' `pd.read_csv()` function to extract data from the .csv file.

Sex was an important factor when predicting survival, so I included it as an independent variable. However, since it was a categorical variable (its values were non-numerical), the training algorithm could not interpret it. To circumvent this, I changed it into a dummy variable.

`pandas`' `pd.get_dummies(X, columns=CAT_VAR, drop_first=True)` function replaced the `'Sex'` column with `'Sex_male'`. This made all entries boolean values, where True represented male and False represented female. The `drop_first` parameter removed one of the dummy variables, since if all other dummy variables were false, then the removed one was true.

<br>

### Setting up the train-test modularization and logistic regression model

```python
# Testing and training sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# Logistic regression model:
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# log_reg = joblib.load(MODEL_NAME)
```

In order to train a model and test its accuracy, I had to allocate a portion of the data to training, and a portion to testing. Usually, 80% of the data is allocated to training, while the remaining 20% is allocated to testing.

I used `sklearn.model_selection`'s `train_test_split(X, y, train_size=0.8)` function to randomly split the data set into the desired ratio.

Then, I initialized the logistic regression with `sklearn.linear_model`'s `LogisticRegression()` constructor and trained it with the training data using the `log_reg.fit(X_train, y_train)` function.

*Note: The line that is commented out is used after the logistic regression has been trained and exported. Instead of having to train the model, `joblib`'s `joblib.load(MODEL_NAME)` function finds the exported model and uses it. If you are using this line, make sure to comment out the initialization and training of the logistic regression.*

<br>

### Predicting the test set and calculating the accuracy

```python
# Predictions:
predictions = log_reg.predict(X_test)

# Accuracy score:
score = accuracy_score(y_test, predictions) * 100
print("Accuracy:", score, '\b%')
```

Now that the logistic regression has been trained, I predicted the accuracy by testing it with the `log_reg.predict(X_test)` function. `sklearn.metrics`' `accuracy_score(y_test, predictions)` function compared the results and returned the accuracy. Since the accuracy ∈ [0, 1], multiplying it by 100 and concatenating a % sign turned it into a percentage.

<br>

### Exporting the model

```python
# Persistent model:
joblib.dump(log_reg, MODEL_NAME)
```

Finally, I exported the model with `joblib`'s `joblib.dump(log_reg, MODEL_NAME)` function. This allowed me to use the same model for future predictions without having to train a new one.
