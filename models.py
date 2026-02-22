from pydantic import BaseModel
from typing import Optional, Literal

class Receipt(BaseModel):
    receiver: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    category: Literal["groceries", "restaurant", "transport", "utilities", "shopping", "personal_transaction", "healthcare", "entertainment", "other"]
    confidence: Optional[Literal["high", "medium", "low"]] = None