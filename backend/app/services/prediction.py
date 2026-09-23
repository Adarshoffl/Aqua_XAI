import os
import joblib
import pandas as pd
import numpy as np
import shap
import traceback

# Database and Model Imports
from app.database.connection import SessionLocal
from app.models.water_models import Treatment
from app.chatbot.generator import generate_local_treatment

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "models")

print("Loading ML Models from:", MODEL_PATH)

model = joblib.load(os.path.join(MODEL_PATH, "random_forest.pkl"))
encoders = joblib.load(os.path.join(MODEL_PATH, "encoders.pkl"))
target_encoder = joblib.load(os.path.join(MODEL_PATH, "target_encoder.pkl"))
imputer = joblib.load(os.path.join(MODEL_PATH, "imputer.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_PATH, "feature_columns.pkl"))

print("Initializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)
print("ML Model and Explainer Loaded Successfully")

def predict_water_quality(data: dict):
    db = SessionLocal() # Open DB connection
    try:
        # 1. Map Pydantic input keys to training columns
        mapped_data = {
            "Country": data["Country"],
            "Waterbody Type": data["Waterbody_Type"],
            "Ammonia (mg/l)": data["Ammonia"],
            "Biochemical Oxygen Demand (mg/l)": data["Biochemical_Oxygen_Demand"],
            "Dissolved Oxygen (mg/l)": data["Dissolved_Oxygen"],
            "Orthophosphate (mg/l)": data["Orthophosphate"],
            "pH (ph units)": data["pH"],
            "Temperature (cel)": data["Temperature"],
            "Nitrogen (mg/l)": data["Nitrogen"],
            "Nitrate (mg/l)": data["Nitrate"],
            "Year": data["Year"],
            "Month": data["Month"]
        }
        
        df = pd.DataFrame([mapped_data])

       # 2. Safely encode categorical columns
        for column, encoder in encoders.items():
            try:
                df[column] = encoder.transform(df[column])
            except ValueError as e:
                print(f"⚠️ Warning: Unseen label in column '{column}'. Auto-correcting to prevent crash.")
                # If the user inputs an unknown label, fallback to the first known class in the encoder
                known_default = encoder.classes_[0]
                df[column] = encoder.transform([known_default])

        # 3. Arrange columns and apply imputer
        df = df[feature_columns]
        df_imputed = imputer.transform(df)
        df_processed = pd.DataFrame(df_imputed, columns=feature_columns)

        # 4. Prediction
        prediction = model.predict(df_processed)
        probability = model.predict_proba(df_processed)
        
        predicted_label = target_encoder.inverse_transform(prediction)[0]
        confidence = float(probability.max() * 100)

        # 5. Calculate SHAP Values
        shap_values = explainer.shap_values(df_processed)
        predicted_class_idx = int(prediction[0])
        
        if isinstance(shap_values, list):
            class_shap_values = shap_values[predicted_class_idx][0] 
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            class_shap_values = shap_values[0, :, predicted_class_idx]
        else:
            class_shap_values = np.array(shap_values)[0, :, predicted_class_idx] if len(np.array(shap_values).shape) == 3 else np.array(shap_values)[0]

        # 6. Map and Sort SHAP Features
        feature_impacts = []
        for i, feature_name in enumerate(feature_columns):
            feature_impacts.append({
                "feature": feature_name,
                "importance": float(class_shap_values[i])
            })
            
        feature_impacts.sort(key=lambda x: abs(x["importance"]), reverse=True)

        # 7. THE TREATMENT ENGINE (SQL)
        recommended_actions = []
        
        if predicted_label in ["Excellent", "Good"]:
            action_type = "Preventative Maintenance"
            target_condition = "Maintenance"
            target_features = [f for f in feature_impacts if f["importance"] > 0 and f["feature"] not in ["Country", "Year", "Month", "Waterbody Type"]]
        else:
            action_type = "Corrective Treatment"
            target_condition = "High" 
            target_features = [f for f in feature_impacts if f["importance"] < 0 and f["feature"] not in ["Country", "Year", "Month", "Waterbody Type"]]

        if target_features:
            top_feature_raw_name = target_features[0]["feature"]
            clean_feature_name = top_feature_raw_name.split(" (")[0]
            
            treatment_records = db.query(Treatment).filter(
                Treatment.parameter == clean_feature_name,
                Treatment.condition == target_condition
            ).all()

            for record in treatment_records:
                recommended_actions.append({
                    "action_type": action_type,
                    "parameter": record.parameter,
                    "treatment_name": record.treatment_name,
                    "description": record.description,
                    "working_principle": record.working_principle,
                    "advantages": record.advantages,
                    "limitations": record.limitations,
                    "maintenance": record.maintenance,
                    "estimated_cost": record.estimated_cost,
                    "precautions": record.precautions
                })

        # 8. THE GENERATIVE AI ENGINE (Local Phi-3)
        ai_generated_advice = None
        if target_features:
            print(f"🧠 Triggering Local Phi-3 AI for {target_condition} {clean_feature_name}...")
            try:
                ai_generated_advice = generate_local_treatment(
                    parameter=clean_feature_name, 
                    condition=target_condition
                )
            except Exception as e:
                print(f"AI Generation failed (Skipping): {e}")
                ai_generated_advice = "AI generation unavailable at this time."

        # 9. RETURN THE FULL PAYLOAD
        return {
            "water_quality": predicted_label,
            "confidence": round(confidence, 2),
            "explanation": feature_impacts,
            "recommended_actions": recommended_actions,
            "ai_treatment_plan": ai_generated_advice 
        }

    except Exception as err:
        print("--- REAL ERROR TRACEBACK ---")
        traceback.print_exc() 
        raise ValueError(f"ML Processing Error: {str(err)}")
    finally:
        db.close() # Always close the DB connection