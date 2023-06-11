# Machine Learning Workshop

Submission for the [2023 McMaster & Co-operators Problem-Solving Workshop](https://math.mcmaster.ca/fourth-mcmaster-co-operators-problem-solving-workshop/). The scikit-learn and pandas packages made the logistic regression model and set up the data frame respectively.

## In-Depth Documentation: [Logistic Regression.ipynb](Machine%20Learning%20Workshop/Logistic%20Regression.ipynb)

### Setting up the machine learning model

First, I imported the necessary packages for manipulating the data set, making and training the logistic regression, testing accuracy, and exporting the model.

The constants were used to categorize the variables in the data set and improve readability. [WHICH VARIABLES WERE NOT INDEPENDENT NOR DEPENDENT, WHICH VARIABLES ARE CATEGORICAL AND NUMERICAL]

<br>

### Making and cleaning the data frame

The goal of machine learning models is to use data from independent variables to predict the result of a dependent variable. By splitting up the original data frame into its dependent and independent variables, I could now use them to train the model. I accomplished this using `pandas`' `read_csv()` function to extract data from the .csv file.

...
