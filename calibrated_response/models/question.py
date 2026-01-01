"""Question representations for forecasting tasks."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Type of forecasting question."""
    BINARY = "binary"
    NUMERIC = "numeric"
    DATE = "date"
    MULTIPLE_CHOICE = "multiple_choice"


class Question(BaseModel):
    """Base class for forecasting questions."""
    
    id: str = Field(..., description="Unique identifier for the question")
    title: str = Field(..., description="The question text")
    description: Optional[str] = Field(None, description="Additional context/description")
    question_type: QuestionType = Field(..., description="Type of question")
    
    # Resolution information
    resolution: Optional[Any] = Field(None, description="The resolved answer, if known")
    resolution_time: Optional[datetime] = Field(None, description="When the question was resolved")
    
    # Bounds for numeric/date questions
    lower_bound: Optional[float] = Field(None, description="Lower bound for numeric questions")
    upper_bound: Optional[float] = Field(None, description="Upper bound for numeric questions")
    
    # Metadata
    created_at: Optional[datetime] = Field(None, description="When the question was created")
    close_time: Optional[datetime] = Field(None, description="When forecasting closes")
    tags: list[str] = Field(default_factory=list, description="Question tags/categories")
    
    def is_resolved(self) -> bool:
        """Check if the question has been resolved."""
        return self.resolution is not None
    
    def get_domain(self) -> tuple[float, float] | None:
        """Get the domain bounds for numeric questions."""
        if self.question_type == QuestionType.NUMERIC:
            return (self.lower_bound or 0, self.upper_bound or float('inf'))
        return None


class MetaculusQuestion(Question):
    """Question loaded from Metaculus dataset."""
    
    metaculus_id: int = Field(..., description="Metaculus question ID")
    url: Optional[str] = Field(None, description="URL to the question page")
    
    # Prediction history
    prediction_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Historical community predictions"
    )
    community_prediction: Optional[float] = Field(
        None, 
        description="Final community prediction"
    )
    
    @classmethod
    def from_scraped_data(cls, data: dict[str, Any]) -> MetaculusQuestion:
        """Create a MetaculusQuestion from scraped JSON data."""
        q = data.get("question", data)
        hist = data.get("prediction_history", {})
        
        # Determine question type
        poss_type = (q.get("possibilities", {}).get("type", "") or "").lower()
        if poss_type == "binary":
            qtype = QuestionType.BINARY
        elif poss_type == "date":
            qtype = QuestionType.DATE
        else:
            qtype = QuestionType.NUMERIC
        
        # Extract bounds
        lower = None
        upper = None
        if "possibilities" in q:
            poss = q["possibilities"]
            lower = poss.get("scale", {}).get("min")
            upper = poss.get("scale", {}).get("max")
        
        # Parse timestamps
        def parse_time(s: str | None) -> datetime | None:
            if not s:
                return None
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except:
                return None
        
        # Extract prediction history timeseries
        history = []
        if isinstance(hist.get("prediction_timeseries"), list):
            history = hist["prediction_timeseries"]
        elif isinstance(hist.get("history"), list):
            history = hist["history"]
        
        return cls(
            id=str(q.get("id", "")),
            metaculus_id=q.get("id", 0),
            title=q.get("title", ""),
            description=q.get("description"),
            question_type=qtype,
            resolution=q.get("resolution"),
            resolution_time=parse_time(q.get("resolve_time")),
            lower_bound=lower,
            upper_bound=upper,
            created_at=parse_time(q.get("created_time")),
            close_time=parse_time(q.get("close_time")),
            url=f"https://www.metaculus.com/questions/{q.get('id')}",
            prediction_history=history,
            community_prediction=q.get("community_prediction"),
        )
