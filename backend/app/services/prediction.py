import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models"
)


print("Loading ML Models from:", MODEL_PATH)


model = joblib.load(
    os.path.join(MODEL_PATH, "random_forest.pkl")
)

encoders = joblib.load(
    os.path.join(MODEL_PATH, "encoders.pkl")
)

target_encoder = joblib.load(
    os.path.join(MODEL_PATH, "target_encoder.pkl")
)

imputer = joblib.load(
    os.path.join(MODEL_PATH, "imputer.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_PATH, "feature_columns.pkl")
)


print("ML Model Loaded Successfully")


def predict_water_quality(data: dict):

    try:

        df = pd.DataFrame([data])


        print("Input Data:")
        print(df)


        # Encode categorical columns

        for column, encoder in encoders.items():

            df[column] = encoder.transform(
                df[column]
            )


        # Arrange columns exactly like training

        df = df[feature_columns]


        # Apply imputer

        df = imputer.transform(df)


        # Convert back to dataframe

        df = pd.DataFrame(
            df,
            columns=feature_columns
        )


        prediction = model.predict(df)


        probability = model.predict_proba(df)


        label = target_encoder.inverse_transform(
            prediction
        )


        confidence = float(
            probability.max() * 100
        )


        return {
            "water_quality": label[0],
            "confidence": round(confidence, 2)
        }


    except Exception as e:

        print("Prediction Error:", e)

        raise e