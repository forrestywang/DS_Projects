# Music Decision Tree

Code copied from [Programming with Mosh's Python Machine Learning Tutorial](https://www.youtube.com/watch?v=7eh4d6sabA0).

<br>

## In-depth Documentation: [Decision Tree.ipynb](/Music%20Decision%20Tree/Decision%20Tree.ipynb)

### Setting up the decision tree

```python
# Imports:
import pandas as pd  # Reading the dataset
from sklearn.tree import DecisionTreeClassifier  # Decision tree machine learning algorithm
from sklearn.model_selection import train_test_split  # For making test sets
from sklearn.metrics import accuracy_score  # For testing accuracy
import joblib  # Persistent model

# Constants:
DATASET_FILE_NAME = 'music.csv'

INDEP_VAR = ['age', 'gender']
DEP_VAR = 'genre'

MODEL_NAME = 'music-recommender.joblib'
```

First, I imported the necessary packages for manipulating the data set, making and training the decision tree, testing accuracy, and exporting the model. The `scikit-learn` and `pandas` packages made the decision tree model and set up the data frame respectively.

The constants were used to categorize the variables in the data set and improve readability. `'genre'` was the dependent variable because it depended on the independent variables `'age'`, and `'gender'`.

<br>

### Making and cleaning the data frame

```python
# Data frame and input set:
X, y = pd.read_csv(DATASET_FILE_NAME, usecols=INDEP_VAR), pd.read_csv(DATASET_FILE_NAME)[DEP_VAR]
```

The goal of machine learning models is to use data from independent variables to predict the result of a dependent variable. By splitting up the original data frame into its dependent and independent variables, I could now use them to train the model. I accomplished this using `pandas`' `pd.read_csv()` function to extract data from the .csv file.

<br>

### Setting up the train-test modularization and decision tree model

```python
# Training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# Decision tree model:
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)

# dec_tree = joblib.load(MODEL_NAME)
```

To train a model and test its accuracy, I had to allocate a portion of the data to training, and a portion to testing. Usually, 80% of the data is allocated to training, while the remaining 20% is allocated to testing.

I used `sklearn.model_selection`'s `train_test_split(X, y, train_size=0.8)` function to randomly split the data set into the desired ratio.

Then, I initialized the decision tree with `sklearn.tree`'s `DecisionTreeClassifier()` constructor and trained it with the training data using the `dec_tree.fit(X_train, y_train)` function.

*Note: The line that is commented out is used after the decision tree has been trained and exported. Instead of having to train the model, `joblib`'s `joblib.load(MODEL_NAME)` function finds the exported model and uses it. If you are using this line, make sure to comment out the initialization and training of the decision tree.*

<br>

### Predicting the test set and calculating the accuracy

```python
# Predictions:
predictions = dec_tree.predict(X_test)

# Accuracy score:
score = accuracy_score(y_test, predictions) * 100
print("Accuracy:", score, '\b%')
```

Now that the decision tree has been trained, I predicted the accuracy by testing it with the `dec_tree.predict(X_test)` function. `sklearn.metrics`' `accuracy_score(y_test, predictions)` function compared the results and returned the accuracy. Since the accuracy $\in [0, 1]$, multiplying it by 100 and concatenating the % sign converted it into a percentage.

<br>

### Exporting the model

```python
# Persistent model:
joblib.dump(dec_tree, MODEL_NAME)
```

Finally, I exported the model with `joblib`'s `joblib.dump(dec_tree, MODEL_NAME)` function. This allowed me to use the same model for future predictions without having to train a new one.
