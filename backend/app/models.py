from sqlalchemy import Column, Integer, String, Float, DateTime
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="viewer", nullable=False)


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    zone = Column(String(100), nullable=False)
    violation_type = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    risk_impact = Column(Float, nullable=False)