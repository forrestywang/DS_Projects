# Music Decision Tree

Code copied from [Programming with Mosh's Python Machine Learning Tutorial](https://www.youtube.com/watch?v=7eh4d6sabA0). The scikit-learn and pandas packages make the decision tree model and set up the data frame respectively.

## In-depth Documentation

### Setting up the machine learning model

First, I imported the necessary packages for manipulating the data set, making and training the decision tree, testing accuracy, and exporting the model.

The constants were used to categorize the variables in the data set and improve readability.

<br>

### Making and cleaning the data frame

The goal of machine learning models is to use data from independent variables to predict the result of a dependent variable. By splitting up the original data frame into its dependent and independent variables, I could now use them to train the model. I accomplished this using `pandas`' `read_csv()` method to extract data from the .csv file.

<br>

### Setting up the train-test modularization and decision tree model

In order to train a model and test its accuracy, I had to allocate a portion of the data to training, and a portion to testing. Usually, 80% of the data is allocated to training, while the remaining 20% is allocated to testing.

I used `sklearn.model_selection`'s `train_test_split(X, y, train_size=0.8)` method to randomly split the data set into the desired ratio.

Then, I initalized the decision tree with `sklearn.tree`'s `DecisionTreeClassifier()` constructor and trained it with the training data using the `fit(X_train, y_train)` method.

*Note: The line that is commented out is used after the decision tree has been trained and exported. Instead of having to train the model, `joblib`'s `load(MODEL_NAME)` method finds the exported model and uses it. If you are using this line, make sure to comment out the initialization and training of the decision tree.*

<br>

### Predicting the test set and calculating the accuracy

Now that the decision tree has been trained, I can predict the accuracy by testing it with the `predict(X_test)` method. `sklearn.metrics`' `accuracy_score(y_test, predictions)` method will compare the results and return the accuracy. Since the accuracy ∈ [0, 1], multiplying it by 100 and concatenating the % sign turns it into a percentage.

<br>

### Exporting the model

Finally, I exported the model with `joblib`'s `dump(dec_tree, MODEL_NAME)` method. This allowed me to use the same model for future predictions without having to train a new one.
