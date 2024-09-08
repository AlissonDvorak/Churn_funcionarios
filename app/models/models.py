from pydantic import BaseModel
from typing import List

class Previsao(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: int
    average_montly_hours: float
    time_spend_company: int
    Work_accident: int
    promotion_last_5years: int
    Departments: str
    salary: str

class PrevisaoList(BaseModel):
    previsoes: List[Previsao]