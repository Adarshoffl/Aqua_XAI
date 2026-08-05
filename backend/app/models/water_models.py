from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.database.connection import Base


# User Table
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String(100), nullable=False)

    email = Column(String(150), unique=True, nullable=False)

    password = Column(String(255), nullable=False)

    created_at = Column(
        DateTime,
        default=func.now()
    )


# Water Quality Data Table
class WaterQuality(Base):
    __tablename__ = "water_quality"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id")
    )

    ph = Column(Float)
    hardness = Column(Float)
    tds = Column(Float)
    chloramines = Column(Float)
    sulfate = Column(Float)
    conductivity = Column(Float)
    organic_carbon = Column(Float)
    trihalomethanes = Column(Float)
    turbidity = Column(Float)

    prediction = Column(String(50))

    confidence = Column(Float)

    created_at = Column(
        DateTime,
        default=func.now()
    )


# Treatment Knowledge Base
class Treatment(Base):
    __tablename__ = "treatments"

    id = Column(Integer, primary_key=True)

    parameter = Column(String(100))

    condition = Column(String(100))

    treatment_name = Column(String(150))

    description = Column(Text)

    working_principle = Column(Text)

    advantages = Column(Text)

    limitations = Column(Text)

    maintenance = Column(Text)

    estimated_cost = Column(String(100))

    precautions = Column(Text)


# SHAP Explanation Table
class SHAPResult(Base):
    __tablename__ = "shap_results"

    id = Column(Integer, primary_key=True)

    water_id = Column(
        Integer,
        ForeignKey("water_quality.id")
    )

    feature_name = Column(String(100))

    importance = Column(Float)


# AI Chat History
class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id")
    )

    question = Column(Text)

    answer = Column(Text)

    created_at = Column(
        DateTime,
        default=func.now()
    )