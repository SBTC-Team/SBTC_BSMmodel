
import yfinance as yf
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

# Import our models
from option_pricing.financial_instruments import Asset, Option, OptionType
from option_pricing.models import BlackScholesAnalytical

def run_backtest(ticker_symbol="NVDA", days_back=30):
    print(f"\n--- Backtesting Greeks for {ticker_symbol} ---")
    print(f"Scenario: Bought an ATM Call Option {days_back} days ago.")
    
    # 1. Fetch Historical Data
    # Fetch enough data to calculate volatility before the start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Buffer for volatility calculation
    fetch_start = start_date - timedelta(days=60)
    
    print(f"Fetching data from {fetch_start.date()} to {end_date.date()}...")
    try:
        data = yf.download(ticker_symbol, start=fetch_start, end=end_date, progress=False)
        if data.empty:
            print("Error: No data fetched.")
            return
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Use 'Adj Close' if available, otherwise 'Close'
    # yfinance often returns a MultiIndex columns if not flattened, let's simple check
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']
        
    # Find closest trading day to our target start_date
    # We look for the index closest to start_date
    idx_start = prices.index.get_indexer([start_date], method='nearest')[0]
    real_start_date = prices.index[idx_start]
    
    # Get initial price (S0)
    S0 = float(prices.iloc[idx_start])
    
    # Get final price (ST) - The most recent data point
    ST = float(prices.iloc[-1])
    real_end_date = prices.index[-1]
    
    print(f"\n[Trade Setup]")
    print(f"Start Date: {real_start_date.date()} | Spot Price (S0): ${S0:.2f}")
    print(f"End Date:   {real_end_date.date()} | Spot Price (ST): ${ST:.2f}")
    
    price_change = ST - S0
    pct_change = (price_change / S0) * 100
    print(f"Move:       ${price_change:.2f} ({pct_change:.2f}%)")

    # 2. Calculate Inputs for the Model at T0 (Start Date)
    # Volatility: Calculate 20-day historical volatility ending at start_date
    # Get 20 days slice ending at idx_start
    window_data = prices.iloc[idx_start-20:idx_start]
    log_returns = np.log(window_data / window_data.shift(1)).dropna()
    volatility = float(log_returns.std() * np.sqrt(252)) # Annualize and cast to float
    
    risk_free_rate = 0.045 # Approx 4.5% risk free rate
    
    asset = Asset(
        ticker=ticker_symbol,
        current_price=S0,
        volatility=volatility,
        risk_free_rate=risk_free_rate
    )
    
    # ATM Call Option
    K = S0
    T_years = days_back / 365.0
    
    option = Option(
        asset=asset,
        strike_price=K,
        time_to_maturity=T_years,
        option_type=OptionType.CALL
    )
    
    # 3. Running the Model at T0
    premium = BlackScholesAnalytical.calculate_price(option)
    greeks = BlackScholesAnalytical.calculate_greeks(option)
    
    print(f"\n[Model Analysis at T-30]")
    print(f"Volatility (Hist): {volatility:.2%}")
    print(f"Theoretical Option Premium: ${premium:.2f}")
    print("-" * 30)
    print(f"Delta: {greeks['delta']:.4f}  (Exposure: ${greeks['delta']*S0:.2f})")
    print(f"Gamma: {greeks['gamma']:.4f}  (Convexity)")
    print(f"Theta: {greeks['theta']:.4f}  (Daily Decay)")
    print(f"Vega:  {greeks['vega']:.4f}")
    print("-" * 30)

    # 4. Actual Results at Expiry (T=0)
    # Intrinsic value at expiry
    intrinsic_value = max(ST - K, 0)
    actual_pnl = intrinsic_value - premium
    
    print(f"\n[Actual Results]")
    print(f"Option Value at Expiry: ${intrinsic_value:.2f}")
    print(f"Actual Net PnL:         ${actual_pnl:.2f}")

    # 5. Greek Attribution (Explaining the PnL)
    # We try to predict the PnL using the Greeks from T0
    # Note: Greeks are dynamic, so this is a linear/quadratic approximation of a dynamic process
    
    # Delta Contribution: Profit from direction
    delta_pnl = greeks['delta'] * price_change
    
    # Gamma Contribution: Profit from convexity (0.5 * Gamma * dS^2)
    gamma_pnl = 0.5 * greeks['gamma'] * (price_change ** 2)
    
    # Theta Contribution: Loss from time (Theta * days)
    # Note: Theta in our model is per year? No, we divided by 365 in models.py, so it's daily.
    # Let's verify models.py.
    # Yes: greeks['theta'] = theta / 365.0
    theta_pnl = greeks['theta'] * days_back
    
    predicted_pnl = delta_pnl + gamma_pnl + theta_pnl
    
    print(f"\n[Greek Attribution Analysis]")
    print(f"How did we get this PnL?")
    print(f"1. Delta (Direction):   ${delta_pnl:.2f}  (Impact of ${price_change:.2f} move)")
    print(f"2. Gamma (Curvature):   ${gamma_pnl:.2f}  (Benefit from volatility of price)")
    print(f"3. Theta (Time Decay):  ${theta_pnl:.2f}  (Cost of holding {days_back} days)")
    print(f"----------------------------------")
    print(f"Predicted Total PnL:    ${predicted_pnl:.2f}")
    print(f"Actual Total PnL:       ${actual_pnl:.2f}")
    
    diff = actual_pnl - predicted_pnl
    print(f"\nDifference (Unexplained): ${diff:.2f}")
    print("(Difference is due to changes in Volatility (Vega) and higher-order Greek changes during the 30 days.)")

if __name__ == "__main__":
    run_backtest()
