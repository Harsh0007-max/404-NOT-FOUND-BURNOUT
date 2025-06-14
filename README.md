# 🏍 MotoGP Lap Time Prediction

This project uses machine learning (XGBoost) to predict MotoGP rider lap times based on various rider, team, bike, and session features. The model is trained on historical data and outputs predictions for test data, which are saved in a submission-ready CSV.

---

## 📁 Dataset Structure

- train.csv – Contains the training data with lap time labels.
- test.csv – Test data for which lap time predictions are required.
- sample_submission.csv – Format for the submission file.

---

## 🔧 Project Structure

| File                     | Description                                   |
|--------------------------|-----------------------------------------------|
| main.py                | Main training, evaluation, and prediction logic |
| train.csv              | Training dataset with features and labels     |
| test.csv               | Test dataset without labels                   |
| sample_submission.csv  | Sample format for submission output           |
| 404_not_found_output.csv | Final predictions in submission format        |

---

## 🚀 Features

- *Frequency Encoding* for categorical variables.
- *One-Hot Encoding* with column alignment for train/test.
- *Log Transformation* of target variable to reduce skew.
- *Early Stopping* during training to avoid overfitting.
- *Evaluation Metrics:* RMSE, MAE, and R² score.
- *Prediction Export* to a formatted CSV file.

---

## 🛠 Requirements

- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - xgboost==3.0.2
  - scikit-learn
  - matplotlib

You can install the required packages using:

```bash
pip install pandas numpy xgboost scikit-learn matplotlib
