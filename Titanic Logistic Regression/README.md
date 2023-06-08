# Titanic Logistic Regression

Code copied from [Python Maratón's sklearn Logistic Regression Video](https://www.youtube.com/watch?v=VK6v9Ure8Lk). The scikit-learn and pandas packages make the logistic regression model and set up the data frame respectively.

## In-Depth Documentation

### Setting up the machine learning model

First, we need to import the necessary packages for manipulating the data set, making and training the logistic regression, testing accuracy, and exporting the model.

The initialized constants are used to categorize the variables in the data set and improve readability.
# TALK ABOUT CAT VAR

<br>

### Making and cleaning the data frame

# TALK ABOUT CAT VAR AND READING THE CSV
Since sex is an important factor when predicting survival, it is one of the independent variables. Since the sex variable isn't a number (it's a categorical variable), the training algorithm will not be able to interpret it. Hence, we need to change it to a dummy variable.

`X = pd.get_dummies(X, columns=CAT_VAR, drop_first=True)` replaces the `'Sex'` column with `'Sex_male'`. Now, all entries are boolean, where True represents male and False represents female.

<br>

### Setting up the train-test modularization and logistic regression model

In order to train a model and test its accuracy, we must allocate a portion of the data to training, and a portion to testing. Usually, 80% of the data is allocated to training, while the remaining 20% is allocated to testing.

I used `sklearn.model_selection`'s `train_test_split(X, y, train_size=0.8)` method to randomly split the data set into the desired ratio.

Then, I initalized the logistic regression with `sklearn.linear_model`'s `LogisticRegression()` constructor and trained it with the training data using the `fit(X_train, y_train)` method.

*Note: The line that is commented out is used after the logistic regression has been trained and exported. Instead of having to train the model, `joblib`'s `load(MODEL_NAME)` method finds the exported model and uses it. If you are using this line, make sure to comment out the initialization and training of the logistic regression.*

<br>

### Predicting the test set and calculating the accuracy

Now that the logistic regression has been trained, we can predict the accuracy by testing it with the `predict(X_test)` method. `sklearn.metrics`' `accuracy_score(y_test, predictions)` method will compare the results and return the accuracy. Since the accuracy ∈ [0, 1], multiplying it by 100 and concatenating a % sign turns it into a percentage.

<br>

### Exporting the model

Finally, we can export the model with `joblib`'s `dump(log_reg, MODEL_NAME)` method. This allows us to use the same model for future predictions without having to train a new one.
