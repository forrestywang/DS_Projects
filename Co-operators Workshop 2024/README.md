# Co-operators Workshop 2024

Submission for the [5<sup>th</sup> McMaster & Co-operators Problem-Solving Workshop](https://math.mcmaster.ca/fifth-annual-mcmaster-industrial-workshop-registration-open/).

## Files:

  `training-dataset.csv` - Same as Claims_Years_1_to_3.csv, but removed all entries with claim amounts >= 3216 (Q3 + 1.5 IQR)
  `scoring-dataset.csv` - Same as Submission_Data.csv
  `submission.csv` - Our submission using the Gamma GLM trained on all entries with claim amounts <= 3216 (Q3 + 1.5 IQR)

  `Gamma GLM.ipynb` - Cleans training-dataset.csv data frame to train and test the Gamma GLM, then exported to gamma-glm.joblib
  `Predictions.ipynb` - Cleans scoring-dataset.csv, then uses the exported model to make predictions

  `gamma-glm.joblib` - The exported Gamma GLM

<br>

## Submissions:

Forrest was responsible for `submission_gamma.csv` and `submission_gamma_q2.csv`. 

`submission_gamma.csv`'s model was trained on data with claim amounts <= 3216 (Q3 + 1.5 IQR), while `submission_gamma_q2.csv`'s model was trained on data with claim amounts <= Q2.
