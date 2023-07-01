# Machine Learning Workshop

Submission for the [2023 McMaster & Co-operators Problem-Solving Workshop](https://math.mcmaster.ca/fourth-mcmaster-co-operators-problem-solving-workshop/). Although our team did not win, I am thankful that the workshop gave me an introduction to the world of data science and machine learning.

## In-Depth Documentation: [Logistic Regression.ipynb](/Machine%20Learning%20Workshop/Logistic%20Regression.ipynb)

### Setting up the machine learning model

```python
# Imports:
import pandas as pd  # For setting up the data frame
from sklearn.preprocessing import StandardScaler  # For scaling the numerical values
from sklearn.linear_model import LogisticRegression  # For the logistic regression model
from sklearn.model_selection import train_test_split  # For setting up the train-test modularization
from sklearn.metrics import accuracy_score  # For testing accuracy
import joblib  # Persistent model
import time  # For getting runtime

start_time = time.time()

# Constants:
DATASET_FILE_NAME = 'training-dataset.csv'

INDEP_VAR = ['Gender', 'policyHolderAge', 'hasCanadianDrivingLicense', 'territory', 'hasAutoInsurance', 'hadVehicleClaimInPast', 'homeInsurancePremium', 'isOwner', 'rentedVehicle', 'hasMortgage', 'nbWeeksInsured', 'vehicleStatus']
DEP_VAR = 'responseVariable'

CAT_VAR = ['Gender', 'territory', 'hadVehicleClaimInPast', 'vehicleStatus']
NUM_VAR = ['policyHolderAge', 'homeInsurancePremium', 'nbWeeksInsured']

MODEL_NAME = 'insurance-model.joblib'
```

First, I imported the necessary packages for manipulating the data set, making and training the logistic regression, testing accuracy, and exporting the model. The scikit-learn and pandas packages made the logistic regression model and set up the data frame respectively.

I started the timer here to see how long everything took to execute.

The constants were used to categorize the variables in the data set and improve readability. `'responseVariable'` was the dependent variable, while `'policyId'` was not included. `'Gender'`, `'territory'`, `'hadVehicleClaimInPast'`, and `'vehicleStatus'` were categorical variables because their values were either non-numerical, symbolic, or boolean. `'policyHolderAge'`, `'homeInsurancePremium'`, and `'nbWeeksInsured'` were numerical variables because their values were numerical.

<br>

### Making and cleaning the data frame

```python
# Data frame and input set:
X, y = pd.read_csv(DATASET_FILE_NAME, usecols=INDEP_VAR), pd.read_csv(DATASET_FILE_NAME)[DEP_VAR]

# Cleaning the data frame:
X = X.fillna(0)  # Replaces NA entries with 0 for hasMortgage
X = pd.get_dummies(X, columns=CAT_VAR, drop_first=True)  # Converts CAT_VAR into boolean variables

scaler = StandardScaler()
scaled_var = pd.DataFrame(scaler.fit_transform((X[NUM_VAR])), columns=NUM_VAR)  # Scaling the numerical variables
X = X.drop(columns=NUM_VAR)  # Dropping the old numerical variables
X = X.join(scaled_var)  # Joining the new numerical variables
```

The goal of machine learning models is to use data from independent variables to predict the result of a dependent variable. By splitting up the original data frame into its dependent and independent variables, I could now use them to train the model. I accomplished this using `pandas`' `pd.read_csv()` function to extract data from the .csv file.

`'hasMortgage'` had NA entries which could not be interpreted, so I changed them to 0 with the `X.fillna(0)` function.

Since categorical variables are non-numerical, the training algorithm could not interpret them. To circumvent this, I changed all categorical variables into dummy variables.

`pandas`' `pd.get_dummies(X, columns=CAT_VAR, drop_first=True)` function made all columns boolean for each possible value. For example, the `'Sex'` column was replaced with `'Sex_Male'`. This made all entries boolean values, where True represented male. The `drop_first` parameter removed one of the dummy variables, since if all other dummy variable values were false, then the removed one was true.

In order to scale the numerical variables, I initialized a scaler using the `StandardScaler()` constructor. Then, I scaled the numerical variables, converted them into a `pandas` data frame, and replaced the unscaled variables with the scaled ones.

<br>

### Setting up the train-test modularization and logistic regression model

```python
# Testing and training sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Logistic regression model:
log_reg = LogisticRegression(solver='lbfgs')  # Specific logistic regression algorithm
log_reg.max_iter = 10000
log_reg.fit(X_train, y_train)  # Parameters are supposed to be X, y
```

In order to train a model and test its accuracy, I had to allocate a portion of the data to training, and a portion to testing. Usually, 80% of the data is allocated to training, while the remaining 20% is allocated to testing.

I used `sklearn.model_selection`'s `train_test_split(X, y, train_size=0.8)` function to randomly split the data set into the desired ratio.

Then, I initialized the logistic regression with `sklearn.linear_model`'s `LogisticRegression(solver='lbfgs')` constructor with the [Limited-memory BFG Solver](https://en.wikipedia.org/wiki/Limited-memory_BFGS) and trained it with the training data using the `log_reg.fit(X_train, y_train)` function. The default number of iterations took too long, so I increased the limit using `log_reg.max_iter = 10000`.

*Note: Since there is an entire data frame for training the model, the only purpose of splitting the train test is to check the accuracy. When submitting the predictions, the fit function should look like `log_reg.fit(X, y)`.*

<br>

### Predicting the test set and calculating the accuracy

```python
# Predictions:
predictions = log_reg.predict(X_test)

# Accuracy score:
score = accuracy_score(y_test, predictions) * 100
print("Accuracy:", score, '\b%')
```

Now that the logistic regression has been trained, I predicted the accuracy by testing it with the `log_reg.predict(X_test)` function. `sklearn.metrics`' `accuracy_score(y_test, predictions)` function compared the results and returned the accuracy. Since the accuracy $\in [0, 1]$, multiplying it by 100 and concatenating a % sign turned it into a percentage.

<br>

### Exporting the model

```python
# Persistent model:
joblib.dump(log_reg, MODEL_NAME)

# Execution time:
end_time = time.time()
print("Execution time:", end_time - start_time, "seconds")
print()

# Number of 1s:
print(pd.DataFrame(log_reg.predict(X_test), columns=['predictedResponseVariable'])['predictedResponseVariable'].value_counts())
```

Finally, I exported the model with `joblib`'s `joblib.dump(log_reg, MODEL_NAME)` function. This allowed me to use the same model for future predictions without having to train a new one.

I stopped the timer here to calculate how long everything took to execute.

Since 'predictedResponseVariable' was boolean (1 represented buying car insurance), I printed a data frame that showed the distribution of zeros and ones.

## In-Depth Documentation: [Predictions.ipynb](/Machine%20Learning%20Workshop/Predictions.ipynb)

The documentation for every section is the same as the documentation above.

<br>

### Predicting and exporting the predictions

```python
# Loading the logistic regression model:
log_reg = joblib.load(MODEL_NAME)

# Predictions data frame:
pred_df = pd.DataFrame(log_reg.predict(score_df), columns=['predictedResponseVariable'])

# Policy ID data frame:
policyId_df = pd.read_csv(SCORE_DATASET_FILE_NAME, usecols=['policyId'])

# Exporting the submission data frame as a csv file:
policyId_df.join(pred_df).to_csv(SUBMISSION_FILE_NAME, index=False)

# Execution time:
end_time = time.time()
print("Execution time:", end_time - start_time, "seconds")

# Number of 1s:
print(pred_df['predictedResponseVariable'].value_counts())
```
In order to export the predictions, I concatenated the policy ID with the predictions data frame and exported it as a .csv file.
