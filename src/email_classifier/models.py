from pydantic import BaseModel, Field
from typing import Literal

class EmailClassification(BaseModel):
    
    priority : Literal["low", "medium", "high", "urgent"]
    summary : str = Field(..., min_length=10, max_length=200)
    action_required : bool
    category : Literal["sales", "support", "billing", "spam", "other"]


