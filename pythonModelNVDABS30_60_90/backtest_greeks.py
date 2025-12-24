
import yfinance as yf
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

# Import our models
from option_pricing.financial_instruments import Asset, Option, OptionType
from option_pricing.models import BlackScholesAnalytical

def run_backtest(ticker_symbols=["JPM", "PLTR", "AMD", "RGTI", "NVDA", "TSLA", "LCID", "RIVN", "ACAD", "WVE", "QS", "AAP", "MA", "JNJ", "NVO"]):
    print(f"\n--- Backtesting Greeks for Maturity Snapshots ---")
    
    results = []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for ticker_symbol in ticker_symbols:
        print(f"\nProcessing {ticker_symbol}...")
        
        try:
            # Fetch enough history
            fetch_start = today - timedelta(days=361 + 100)
            data = yf.download(ticker_symbol, start=fetch_start, end=today + timedelta(days=1), progress=False)
            
            if data.empty:
                print(f"Error: No data for {ticker_symbol}")
                continue
            
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            else:
                prices = data['Close']
            
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]

            maturities = [30, 60, 90]
            
            for maturity in maturities:
                offsets = range(361, maturity - 1, -maturity)
                
                for offset in offsets:
                    analysis_date_target = today - timedelta(days=offset)
                    expiry_date_target = analysis_date_target + timedelta(days=maturity)
                    
                    if expiry_date_target > prices.index[-1]:
                        continue # Skip if we don't have expiry data yet
                    
                    # T0 (Analysis Date)
                    idx0 = prices.index.get_indexer([analysis_date_target], method='nearest')[0]
                    t0 = prices.index[idx0]
                    S0 = float(prices.iloc[idx0])
                    
                    # T_expiry (Expiry Date)
                    idxT = prices.index.get_indexer([expiry_date_target], method='nearest')[0]
                    tT = prices.index[idxT]
                    ST = float(prices.iloc[idxT])
                    
                    # Volatility at T0 (using 30d window for backtest)
                    prices_before = prices.loc[:t0]
                    volatilities = []
                    for win in [30, 60, 90]:
                        log_returns = np.log(prices_before.tail(win+1) / prices_before.tail(win+1).shift(1)).dropna()
                        volatilities.append(log_returns.std() * np.sqrt(252))
                    
                    # Use 30d vol for the model Greeks
                    vol = volatilities[0]
                    if np.isnan(vol): continue

                    risk_free_rate = 0.045
                    asset = Asset(ticker=ticker_symbol, current_price=S0, volatility=vol, risk_free_rate=risk_free_rate)
                    option = Option(asset=asset, strike_price=S0, time_to_maturity=maturity/365.0, option_type=OptionType.CALL)
                    
                    premium = BlackScholesAnalytical.calculate_price(option)
                    greeks = BlackScholesAnalytical.calculate_greeks(option)
                    
                    # Actual PnL
                    intrinsic_value = max(ST - S0, 0)
                    actual_pnl = intrinsic_value - premium
                    
                    # Attribution
                    price_change = ST - S0
                    delta_pnl = greeks['delta'] * price_change
                    gamma_pnl = 0.5 * greeks['gamma'] * (price_change ** 2)
                    theta_pnl = greeks['theta'] * maturity
                    predicted_pnl = delta_pnl + gamma_pnl + theta_pnl
                    diff = actual_pnl - predicted_pnl
                    
                    res = {
                        "Ticker": ticker_symbol,
                        "Maturity": maturity,
                        "Analysis_Date": t0.strftime("%Y-%m-%d"),
                        "Expiry_Date": tT.strftime("%Y-%m-%d"),
                        "S0": S0,
                        "ST": ST,
                        "Premium": premium,
                        "Actual_PnL": actual_pnl,
                        "Delta_PnL": delta_pnl,
                        "Gamma_PnL": gamma_pnl,
                        "Theta_PnL": theta_pnl,
                        "PnL_Diff": diff
                    }
                    results.append(res)
                    
        except Exception as e:
            print(f"Error processing {ticker_symbol}: {e}")

    if results:
        df = pd.DataFrame(results)
        output_file = "backtest_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\nBACKTEST COMPLETE: {len(df)} records saved to {output_file}")
        return df
    else:
        print("No results to display.")
        return None

if __name__ == "__main__":
    run_backtest()
