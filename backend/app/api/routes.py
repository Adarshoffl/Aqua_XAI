from fastapi import APIRouter, HTTPException

from app.schemas import WaterQualityInput
from app.services.prediction import predict_water_quality
from app.services.who_rag_service import generate_grounded_answer


router = APIRouter()


@router.post("/analyze-water")
async def analyze_water(sample: WaterQualityInput):

    try:

        # ====================================================
        # 1. CONVERT INPUT TO DICTIONARY
        # ====================================================

        sample_data = sample.model_dump()

        # Remove question from water-quality data
        # because question is not an ML feature
        user_question = sample_data.pop(
            "question",
            None
        )


        # ====================================================
        # 2. RUN ML + SHAP + TREATMENT ENGINE
        # ====================================================

        ml_result = predict_water_quality(
            sample_data
        )


        # ====================================================
        # 3. PREPARE ML RESULT FOR GENERATIVE AI
        # ====================================================

        ml_prediction = {

            "label":
                ml_result["water_quality"],

            "confidence":
                ml_result["confidence"]
        }


        # ====================================================
        # 4. RUN RAG + PHI-3
        # ====================================================

        generative_ai = generate_grounded_answer(

            water_quality_data=
                sample_data,

            ml_prediction=
                ml_prediction,

            shap_results=
                ml_result["explanation"],

            treatment_recommendation=
                ml_result["recommended_actions"],

            user_question=
                user_question
        )


        # ====================================================
        # 5. FINAL RESPONSE
        # ====================================================

        return {

            "status": "success",

            "prediction": {

                "label":
                    ml_result["water_quality"],

                "confidence":
                    ml_result["confidence"]
            },

            "shap":
                ml_result["explanation"],

            "treatment":
                ml_result["recommended_actions"],

            "generative_ai": {

                "question":
                    user_question,

                "answer":
                    generative_ai["answer"],

                "who_sources":
                    generative_ai["sources"]
            }

        }


    except Exception as e:

        print(
            "ERROR in /analyze-water:"
        )

        print(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )