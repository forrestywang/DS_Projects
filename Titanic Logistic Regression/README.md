# Titanic Logistic Regression

Code copied from [Python Maratón's sklearn Logistic Regression Video](https://www.youtube.com/watch?v=VK6v9Ure8Lk). The scikit-learn and pandas packages made the logistic regression model and set up the data frame respectively.

## In-Depth Documentation: [Logistic Regression.ipynb](/Titanic%20Logistic%20Regression/Logistic%20Regression.ipynb)

### Setting up the machine learning model

First, I imported the necessary packages for manipulating the data set, making and training the logistic regression, testing accuracy, and exporting the model.

The constants were used to categorize the variables in the data set and improve readability. 'Survived' was the dependent variable because it depended on the independent variables 'Pclass', 'Sex', and 'Age'. 'Name', 'Siblings/Spouses Aboard', and 'Parents/Children Aboard' did not affect the person's survival, so these variables were not included. 'Sex' was a categorical variable because its values were non-numerical.

<br>

### Making and cleaning the data frame

The goal of machine learning models is to use data from independent variables to predict the result of a dependent variable. By splitting up the original data frame into its dependent and independent variables, I could now use them to train the model. I accomplished this using `pandas`' `read_csv()` function to extract data from the .csv file.

Sex was an important factor when predicting survival, so I included as an independent variable. However, since it was a categorical variable (its values were non-numerical), the training algorithm could not interpret it. To circumvent this, I changed it into a dummy variable.

`pandas`' `get_dummies(X, columns=CAT_VAR, drop_first=True)` function replaced the `'Sex'` column with `'Sex_male'`. This made all entries boolean values, where True represented male and False represented female. The `drop_first` parameter removed one of the dummy variables, since if all other dummy variables were false, then the removed one is true.

<br>

### Setting up the train-test modularization and logistic regression model

In order to train a model and test its accuracy, I had to allocate a portion of the data to training, and a portion to testing. Usually, 80% of the data is allocated to training, while the remaining 20% is allocated to testing.

I used `sklearn.model_selection`'s `train_test_split(X, y, train_size=0.8)` function to randomly split the data set into the desired ratio.

Then, I initalized the logistic regression with `sklearn.linear_model`'s `LogisticRegression()` constructor and trained it with the training data using the `fit(X_train, y_train)` function.

*Note: The line that is commented out is used after the logistic regression has been trained and exported. Instead of having to train the model, `joblib`'s `load(MODEL_NAME)` function finds the exported model and uses it. If you are using this line, make sure to comment out the initialization and training of the logistic regression.*

<br>

### Predicting the test set and calculating the accuracy

Now that the logistic regression has been trained, I can predict the accuracy by testing it with the `predict(X_test)` function. `sklearn.metrics`' `accuracy_score(y_test, predictions)` function will compare the results and return the accuracy. Since the accuracy ∈ [0, 1], multiplying it by 100 and concatenating a % sign turned it into a percentage.

<br>

### Exporting the model

Finally, I exported the model with `joblib`'s `dump(log_reg, MODEL_NAME)` function. This allowed me to use the same model for future predictions without having to train a new one.
