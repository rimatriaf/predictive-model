from typing import List, Optional
from typing import Union
from pydantic import BaseModel
from datetime import datetime

class XYPoint(BaseModel):
    x: Union[str, int]
    y: int

class SeriesData(BaseModel):
    actual: List[XYPoint]
    prediction: List[XYPoint]

class PredictionLatest(BaseModel):
    actual: int
    actual_datetime: str             
    prediction: Optional[int]         
    prediction_datetime: str
    pressure_measurement: float
    displacement_measurement: float
    pressure_lower: float
    pressure_upper: float
    displacement_lower: float
    displacement_upper: float
    total_can_tested: Optional[str]
    rejection: float
    status: str

class TargetAndStandard(BaseModel):
    target_production_hour: Optional[str] = None
    target_value: Optional[float] = None
    target_hour: Optional[float] = None
    standard_hour: Optional[str] = None
    standard_percent: Optional[float] = None
    standard_percent_date: Optional[str] = None
    standard_over_limit: bool = False
    percentage_error: Optional[float] = None
    standard_hourly_target: Optional[float] = None

class FullPredictionResponse(BaseModel):
    series: SeriesData
    latest: PredictionLatest
    target_and_standard: TargetAndStandard
    prev_prediction: Optional[XYPoint] = None
    error_mae: Optional[float] = None
    production_start: Optional[str] = None
    production_stop: Optional[str] = None

class SavePredictionRequest(BaseModel):
    datetime_actual: str         
    reject_actual: int
    datetime_prediction: str    
    reject_prediction: int
    standard_percent: float