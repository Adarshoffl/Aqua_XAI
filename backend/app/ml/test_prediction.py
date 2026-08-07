import joblib
import pandas as pd


MODEL_PATH = "backend/app/ml/models/"


# -------------------------
# Load Saved Files
# -------------------------

print("Loading saved model files...")

model = joblib.load(
    MODEL_PATH + "random_forest.pkl"
)

encoders = joblib.load(
    MODEL_PATH + "encoders.pkl"
)

target_encoder = joblib.load(
    MODEL_PATH + "target_encoder.pkl"
)

imputer = joblib.load(
    MODEL_PATH + "imputer.pkl"
)

feature_columns = joblib.load(
    MODEL_PATH + "feature_columns.pkl"
)


print("Models loaded successfully!")


# -------------------------
# Sample Water Data
# -------------------------

sample = {
    "Country": "Canada",
    "Waterbody Type": "River",

    "Ammonia (mg/l)": 0.05,
    "Biochemical Oxygen Demand (mg/l)": 2.0,
    "Dissolved Oxygen (mg/l)": 8.5,
    "Orthophosphate (mg/l)": 0.02,
    "pH (ph units)": 7.2,
    "Temperature (cel)": 15.0,
    "Nitrogen (mg/l)": 1.0,
    "Nitrate (mg/l)": 0.5,

    "Year": 2024,
    "Month": 6
}


data = pd.DataFrame(
    [sample]
)


# -------------------------
# Encode Categorical Data
# -------------------------

for column, encoder in encoders.items():

    data[column] = encoder.transform(
        data[column]
    )


# -------------------------
# Arrange Feature Order
# -------------------------

data = data[
    feature_columns
]


# -------------------------
# Handle Missing Values
# -------------------------

data = imputer.transform(data)

data = pd.DataFrame(
    data,
    columns=feature_columns
)


# -------------------------
# Prediction
# -------------------------

prediction = model.predict(
    data
)


probability = model.predict_proba(
    data
)


label = target_encoder.inverse_transform(
    prediction
)


confidence = probability.max() * 100


print("\nPrediction:")
print(label[0])

print("\nConfidence:")
print(f"{confidence:.2f}%")