from datetime import datetime
from pydantic import BaseModel


class Event(BaseModel):
    """Jedan digitalni događaj (bankarska transakcija) koji ulazi u engine."""
    event_id: str
    user_id: str
    first_name: str = ""
    last_name: str = ""
    timestamp: datetime
    amount: float
    currency: str
    merchant_category: str
    country: str
    latitude: float
    longitude: float
    device_id: str
    channel: str  # npr. "online", "card_present", "atm"
    is_fraud: bool = False        # ground truth (skriveno od enginea)
    fraud_type: str | None = None  # koji obrazac, ako je prevara
