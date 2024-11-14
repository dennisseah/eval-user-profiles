from pydantic import BaseModel


class ComparisonResult(BaseModel):
    reasoning: str
    score: float
