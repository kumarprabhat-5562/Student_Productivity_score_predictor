from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    study_hours_per_day: float = Field(..., ge=0, le=24)
    sleep_hours: float = Field(..., ge=0, le=24)
    phone_usage_hours: float = Field(..., ge=0, le=24)
    total_distraction_hours: float = Field(..., ge=0, le=24)
    attendance_percentage: float = Field(..., ge=0, le=100)