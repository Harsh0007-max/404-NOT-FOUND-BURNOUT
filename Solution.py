import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def load_data():
    train = pd.read_csv('/content/sample_data/train.csv')
    test = pd.read_csv('/content/sample_data/test.csv')
    sample = pd.read_csv('/content/sample_data/sample_submission.csv')
    return train, test, sample


def smooth_frequency_encoding(train, test, cols, alpha=1):
    for col in cols:
        freq = train[col].value_counts()
        smoothed = (freq + alpha) / (len(train) + alpha * len(freq))
        train[col + '_freq'] = train[col].map(smoothed).fillna(0)
        test[col + '_freq'] = test[col].map(smoothed).fillna(0)
    return train, test


def preprocess(train, test, target, freq_cols):
    y = train[target]
    y_log = np.log1p(y)

    # Frequency Encoding
    train, test = smooth_frequency_encoding(train, test, freq_cols)

    # Drop columns
    drop_cols = [target, 'Unique ID'] + freq_cols
    X = train.drop(columns=drop_cols, errors='ignore')
    X_test = test.drop(columns=['Unique ID'] + freq_cols, errors='ignore')

    # One-hot encoding
    X = pd.get_dummies(X)
    X_test = pd.get_dummies(X_test)

    # Align test with train
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

    # Reduce memory
    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    return X, X_test, y_log


def train_model(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=9,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method='hist',
        verbosity=1,
        eval_metric='rmse',               # ✅ Must be in constructor
        early_stopping_rounds=50          # ✅ Must be in constructor
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    return model


def evaluate_model(model, X_val, y_val):
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    y_val_actual = np.expm1(y_val)

    rmse = mean_squared_error(y_val_actual, preds, squared=False)
    mae = mean_absolute_error(y_val_actual, preds)
    r2 = r2_score(y_val_actual, preds)

    print(f"\n📊 Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_val_actual, preds, alpha=0.4)
    plt.plot([y_val_actual.min(), y_val_actual.max()],
             [y_val_actual.min(), y_val_actual.max()], 'r--')
    plt.xlabel("Actual Lap Time")
    plt.ylabel("Predicted Lap Time")
    plt.title("Predicted vs Actual")
    plt.grid()
    plt.tight_layout()
    plt.show()


def save_submission(model, X_test, sample):
    test_preds_log = model.predict(X_test)
    test_preds = np.expm1(test_preds_log)

    submission = sample.copy()
    submission['Lap_Time_Seconds'] = test_preds
    submission.to_csv('404_not_found_output.csv', index=False)
    print(f"\n✅ Submission saved: {submission.shape[0]} predictions")
    print(submission.head(10))


def main():
    print("🔄 Loading data...")
    train, test, sample = load_data()

    target = 'Lap_Time_Seconds'
    freq_cols = ['rider_name', 'team_name', 'bike_name', 'shortname',
                 'circuit_name', 'rider', 'team', 'bike', 'Session',
                 'weather', 'track']

    print("🧼 Preprocessing data...")
    X, X_test, y_log = preprocess(train, test, target, freq_cols)

    print("🔀 Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_log, test_size=0.2, random_state=42)

    print("🚀 Training model...")
    model = train_model(X_train, y_train, X_val, y_val)

    print("📈 Evaluating model...")
    evaluate_model(model, X_val, y_val)

    print("📤 Generating submission...")
    save_submission(model, X_test, sample)


if _name_ == "_main_":
    main()
