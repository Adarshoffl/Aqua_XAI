import os
import joblib
import pandas as pd
import numpy as np
import shap
import traceback

from app.services.treatment_engine import (
    get_treatment_recommendations
)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models"
)

print(
    "Loading ML Models from:",
    MODEL_PATH
)


# ============================================================
# LOAD SAVED MODELS
# ============================================================

model = joblib.load(
    os.path.join(
        MODEL_PATH,
        "random_forest.pkl"
    )
)

encoders = joblib.load(
    os.path.join(
        MODEL_PATH,
        "encoders.pkl"
    )
)

target_encoder = joblib.load(
    os.path.join(
        MODEL_PATH,
        "target_encoder.pkl"
    )
)

imputer = joblib.load(
    os.path.join(
        MODEL_PATH,
        "imputer.pkl"
    )
)

feature_columns = joblib.load(
    os.path.join(
        MODEL_PATH,
        "feature_columns.pkl"
    )
)


# ============================================================
# SHAP
# ============================================================

print(
    "Initializing SHAP TreeExplainer..."
)

explainer = shap.TreeExplainer(
    model
)

print(
    "ML Model and Explainer Loaded Successfully"
)


# ============================================================
# PREDICTION
# ============================================================

def predict_water_quality(data: dict):

    try:

        # ====================================================
        # 1. MAP INPUT
        # ====================================================

        mapped_data = {

            "Country":
                data["Country"],

            "Waterbody Type":
                data["Waterbody_Type"],

            "Ammonia (mg/l)":
                data["Ammonia"],

            "Biochemical Oxygen Demand (mg/l)":
                data["Biochemical_Oxygen_Demand"],

            "Dissolved Oxygen (mg/l)":
                data["Dissolved_Oxygen"],

            "Orthophosphate (mg/l)":
                data["Orthophosphate"],

            "pH (ph units)":
                data["pH"],

            "Temperature (cel)":
                data["Temperature"],

            "Nitrogen (mg/l)":
                data["Nitrogen"],

            "Nitrate (mg/l)":
                data["Nitrate"],

            "Year":
                data["Year"],

            "Month":
                data["Month"]
        }


        df = pd.DataFrame(
            [mapped_data]
        )


        # ====================================================
        # 2. ENCODE CATEGORICAL FEATURES
        # ====================================================

        for column, encoder in encoders.items():

            try:

                df[column] = encoder.transform(
                    df[column]
                )

            except ValueError:

                print(
                    f"Warning: unseen value in {column}. "
                    "Using first known category."
                )

                known_default = (
                    encoder.classes_[0]
                )

                df[column] = encoder.transform(
                    [known_default]
                )


        # ====================================================
        # 3. FEATURE ORDER + IMPUTATION
        # ====================================================

        df = df[
            feature_columns
        ]

        df_imputed = imputer.transform(
            df
        )

        df_processed = pd.DataFrame(
            df_imputed,
            columns=feature_columns
        )


        # ====================================================
        # 4. ML PREDICTION
        # ====================================================

        prediction = model.predict(
            df_processed
        )

        probability = model.predict_proba(
            df_processed
        )

        predicted_label = (
            target_encoder
            .inverse_transform(
                prediction
            )[0]
        )

        confidence = float(
            probability.max() * 100
        )


        # ====================================================
        # 5. SHAP
        # ====================================================

        shap_values = explainer.shap_values(
            df_processed
        )

        predicted_class_idx = int(
            prediction[0]
        )


        if isinstance(
            shap_values,
            list
        ):

            class_shap_values = (
                shap_values[
                    predicted_class_idx
                ][0]
            )

        elif (
            isinstance(
                shap_values,
                np.ndarray
            )
            and
            len(
                shap_values.shape
            ) == 3
        ):

            class_shap_values = (
                shap_values[
                    0,
                    :,
                    predicted_class_idx
                ]
            )

        else:

            shap_array = np.array(
                shap_values
            )

            if len(
                shap_array.shape
            ) == 3:

                class_shap_values = (
                    shap_array[
                        0,
                        :,
                        predicted_class_idx
                    ]
                )

            else:

                class_shap_values = (
                    shap_array[0]
                )


        # ====================================================
        # 6. BUILD SHAP EXPLANATION
        # ====================================================

        feature_impacts = []

        for i, feature_name in enumerate(
            feature_columns
        ):

            feature_impacts.append({

                "feature":
                    feature_name,

                "importance":
                    float(
                        class_shap_values[i]
                    )
            })


        feature_impacts.sort(
            key=lambda x:
                abs(
                    x["importance"]
                ),
            reverse=True
        )


        # ====================================================
        # 7. ACTUAL-VALUE TREATMENT ENGINE
        # ====================================================
        #
        # IMPORTANT:
        # SHAP does NOT determine treatment.
        #
        # Treatment is based on the actual input values.
        # ====================================================

        recommended_actions = (
            get_treatment_recommendations(
                data
            )
        )


        # ====================================================
        # 8. RETURN RESULT
        # ====================================================

        return {

            "water_quality":
                predicted_label,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "explanation":
                feature_impacts,

            "recommended_actions":
                recommended_actions,

            "ai_treatment_plan":
                None,

            "input_data":
                data
        }


    except Exception as err:

        print(
            "----- REAL ERROR TRACEBACK -----"
        )

        traceback.print_exc()

        raise ValueError(
            f"ML Processing Error: {str(err)}"
        )