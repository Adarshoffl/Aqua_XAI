import os
import joblib

from train import train_models


MODEL_DIR = "backend/app/ml/models"


def save_models():

    print("Starting model training and saving...")


    (
        rf_model,
        xgb_model,
        encoders,
        target_encoder,
        imputer,
        feature_columns
    ) = train_models()


    # Create directory if not exists

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    # Save models

    joblib.dump(
        rf_model,
        f"{MODEL_DIR}/random_forest.pkl"
    )

    joblib.dump(
        xgb_model,
        f"{MODEL_DIR}/xgboost.pkl"
    )


    # Save preprocessing objects

    joblib.dump(
        encoders,
        f"{MODEL_DIR}/encoders.pkl"
    )


    joblib.dump(
        target_encoder,
        f"{MODEL_DIR}/target_encoder.pkl"
    )


    joblib.dump(
        imputer,
        f"{MODEL_DIR}/imputer.pkl"
    )


    joblib.dump(
        feature_columns,
        f"{MODEL_DIR}/feature_columns.pkl"
    )


    print("\nModels saved successfully!")

    print("\nSaved files:")

    for file in os.listdir(MODEL_DIR):
        print(file)



if __name__ == "__main__":

    save_models()