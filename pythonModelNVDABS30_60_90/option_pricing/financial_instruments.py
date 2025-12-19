from dataclasses import dataclass
from enum import Enum

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

@dataclass
class Asset:
    """Class representing the underlying asset."""
    ticker: str
    current_price: float
    volatility: float # Annualized volatility (sigma)
    risk_free_rate: float # Annualized risk-free rate (r)
    dividend_yield: float = 0.0 # (q)

@dataclass
class Option:
    """Class representing a European option."""
    asset: Asset
    strike_price: float # (K)
    time_to_maturity: float # In years (T)
    option_type: OptionType

    def __post_init__(self):
        if self.time_to_maturity <= 0:
            raise ValueError("Time to maturity must be positive.")
        if self.strike_price <= 0:
            raise ValueError("Strike price must be positive.")
