import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


# Load Dataset

df = pd.read_csv(
    "dataset/water_potability.csv"
)

print("Dataset Loaded")


# Features and Target

X = df.drop(
    "Potability",
    axis=1
)

y = df["Potability"]


# Handle Missing Values

imputer = SimpleImputer(
    strategy="mean"
)

X = imputer.fit_transform(X)


# Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------
# Random Forest
# -------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_prediction = rf_model.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)


print("\nRandom Forest Accuracy:")
print(rf_accuracy)



# -------------------------
# XGBoost
# -------------------------

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)


xgb_model.fit(
    X_train,
    y_train
)


xgb_prediction = xgb_model.predict(
    X_test
)


xgb_accuracy = accuracy_score(
    y_test,
    xgb_prediction
)


print("\nXGBoost Accuracy:")
print(xgb_accuracy)



# -------------------------
# Select Best Model
# -------------------------

if xgb_accuracy > rf_accuracy:

    best_model = xgb_model

    model_name = "XGBoost"

else:

    best_model = rf_model

    model_name = "Random Forest"



print("\nBest Model:")
print(model_name)



# Save Model

os.makedirs(
    "trained_models",
    exist_ok=True
)


joblib.dump(
    best_model,
    "trained_models/best_water_quality_model.pkl"
)


joblib.dump(
    imputer,
    "trained_models/imputer.pkl"
)


print("\nModel saved successfully!")