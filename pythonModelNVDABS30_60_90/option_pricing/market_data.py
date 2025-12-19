import yfinance as yf
from dataclasses import dataclass
from typing import Optional

@dataclass
class MarketData:
    current_price: float
    volatility: float
    risk_free_rate: float = 0.05 # Default assumption if not provided

class MarketDataFetcher:
    """
    Fetches market data from Yahoo Finance.
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate

    def fetch_data(self, ticker_symbol: str, days_ago: int = 0, volatility_window: int = 252) -> MarketData:
        """
        Retrieves current price and historical volatility for a given ticker.
        If days_ago > 0, retrieves data as of that many days ago.
        :param volatility_window: Number of trading days to use for volatility calculation.
        """
        from datetime import datetime, timedelta
        
        # Calculate the end date for our data fetching
        end_date = datetime.now() - timedelta(days=days_ago)
        # We need the start date to be far enough back for volatility calc
        # Adding a buffer to ensure we get enough trading days
        start_date = end_date - timedelta(days=int(volatility_window * 1.5) + 20) 
        
        ticker = yf.Ticker(ticker_symbol)
        
        # Current Price (or "Current" as of days_ago)
        if days_ago == 0:
            try:
                current_price = ticker.fast_info['last_price']
            except AttributeError:
                history = ticker.history(period='1d')
                if history.empty:
                    raise ValueError(f"Could not fetch data for {ticker_symbol}")
                current_price = history['Close'].iloc[-1]
        else:
            # Fetch history ending at our target date
            # We fetch a small window around the target date to ensure we get a price
            history = ticker.history(start=end_date - timedelta(days=5), end=end_date + timedelta(days=1))
            if history.empty:
                 raise ValueError(f"Could not fetch historical data for {ticker_symbol} around {end_date.date()}")
            current_price = history['Close'].iloc[-1]
            print(f"DEBUG: Using price from {history.index[-1].date()}: {current_price:.2f}")

        # Historical Volatility (Annualized)
        # We take the last 'volatility_window' trading days
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty or len(hist) < 2: 
             raise ValueError(f"Not enough historical data for {ticker_symbol} to calculate volatility before {end_date.date()}")
             
        # Use only the requested window size
        hist = hist.tail(volatility_window + 1)
        
        # Log returns
        import numpy as np
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        volatility = log_returns.std() * np.sqrt(252) # Annualize daily volatility

        return MarketData(
            current_price=current_price,
            volatility=volatility,
            risk_free_rate=self.risk_free_rate
        )
