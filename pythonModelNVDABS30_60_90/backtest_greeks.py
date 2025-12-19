
import yfinance as yf
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

# Import our models
from option_pricing.financial_instruments import Asset, Option, OptionType
from option_pricing.models import BlackScholesAnalytical

def run_backtest(ticker_symbols=["JPM", "PLTR", "AMD", "RGTI", "NVDA", "TSLA", "LCID", "RIVN", "ACAD", "WVE", "QS",   "AAP", "MA", "JNJ", "NVO"], days_back=30):
    print(f"\n--- Backtesting Greeks for {len(ticker_symbols)} Tickers ---")
    print(f"Scenario: Bought an ATM Call Option {days_back} days ago.")
    
    results = []
    
    for ticker_symbol in ticker_symbols:
        print(f"\nProcessing {ticker_symbol}...")
        
        # 1. Fetch Historical Data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        fetch_start = start_date - timedelta(days=60)
        
        try:
            data = yf.download(ticker_symbol, start=fetch_start, end=end_date, progress=False)
            if data.empty:
                print(f"Error: No data fetched for {ticker_symbol}.")
                continue
        except Exception as e:
            print(f"Error fetching data for {ticker_symbol}: {e}")
            continue

        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            prices = data['Close']
            
        # Check if MultiIndex and flatten if needed
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
            
        idx_start = prices.index.get_indexer([start_date], method='nearest')[0]
        real_start_date = prices.index[idx_start]
        S0 = float(prices.iloc[idx_start])
        ST = float(prices.iloc[-1])
        real_end_date = prices.index[-1]
        
        price_change = ST - S0
        
        # 2. Calculate Inputs for the Model at T0
        window_data = prices.iloc[idx_start-20:idx_start]
        log_returns = np.log(window_data / window_data.shift(1)).dropna()
        volatility = float(log_returns.std() * np.sqrt(252))
        
        risk_free_rate = 0.045 # Approx 4.5% risk free rate
        
        asset = Asset(
            ticker=ticker_symbol,
            current_price=S0,
            volatility=volatility,
            risk_free_rate=risk_free_rate
        )
        
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
        
        # 4. Actual Results at Expiry (T=0)
        intrinsic_value = max(ST - K, 0)
        actual_pnl = intrinsic_value - premium
        
        # 5. Greek Attribution
        delta_pnl = greeks['delta'] * price_change
        gamma_pnl = 0.5 * greeks['gamma'] * (price_change ** 2)
        theta_pnl = greeks['theta'] * days_back
        predicted_pnl = delta_pnl + gamma_pnl + theta_pnl
        diff = actual_pnl - predicted_pnl
        
        # Store results
        res = {
            "Ticker": ticker_symbol,
            "S0": S0,
            "ST": ST,
            "Move %": (ST/S0 - 1) * 100,
            "Premium": premium,
            "Expiry Val": intrinsic_value,
            "Actual PnL": actual_pnl,
            "Predicted PnL": predicted_pnl,
            "Delta PnL": delta_pnl,
            "Gamma PnL": gamma_pnl,
            "Theta PnL": theta_pnl,
            "PnL_Diff": diff,
            "Diff (Unexp)": diff
        }
        results.append(res)

    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("BACKTEST GREEKS SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Save to CSV
        output_file = "backtest_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        return df
    else:
        print("No results to display.")
        return None

if __name__ == "__main__":
    run_backtest()
