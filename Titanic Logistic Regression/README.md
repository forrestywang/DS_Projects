# Titanic Logistic Regression

Code copied from [Python Maratón's sklearn Logistic Regression Video](https://www.youtube.com/watch?v=VK6v9Ure8Lk). The scikit-learn and pandas packages make the logistic regression model and set up the data frame respectively.

## Dummy Variables

Since sex is an important factor when predicting survival, it is one of the independent variables. Since the sex variable isn't a number (it's a categorical variable), the training algorithm will not be able to interpret it. Hence, we need to change it to a dummy variable.

`X = pd.get_dummies(X, columns=CAT_VAR, drop_first=True)` replaces the 'Sex' column with 'Sex_male'. Now, all entries are boolean, where True represents male and False represents female.
