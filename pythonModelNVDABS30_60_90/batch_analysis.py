import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from option_pricing.financial_instruments import Asset, Option, OptionType
from option_pricing.models import BlackScholesAnalytical, MonteCarloEngine
from option_pricing.visualization import Visualizer

def calculate_volatility(prices, window):
    """Calculates annualized volatility for a given window."""
    if len(prices) < window + 1:
        return np.nan
    log_returns = np.log(prices.tail(window + 1) / prices.tail(window + 1).shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)

def run_batch_analysis(tickers=["JPM", "PLTR", "AMD", "RGTI", "NVDA", "TSLA", "LCID", "RIVN", "ACAD", "WVE", "QS",   "AAP", "MA", "JNJ", "NVO"], start_date="2024-11-17", end_date="2024-12-17"):
    print(f"--- Running Batch Analysis for {len(tickers)} Tickers ---")
    print(f"Period: {start_date} to {end_date}")
    
    results = []
    
    # Convert dates to datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # We need extra data before the start_date for the 90-day volatility lookback
    fetch_start = start_dt - timedelta(days=150) # Buffer
    
    for ticker_symbol in tickers:
        print(f"\nProcessing {ticker_symbol}...")
        
        try:
            # Fetch history
            data = yf.download(ticker_symbol, start=fetch_start, end=end_dt + timedelta(days=1), progress=False)
            if data.empty:
                print(f"Error: No data for {ticker_symbol}")
                continue
            
            # Use 'Adj Close' if available, else 'Close'
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            else:
                prices = data['Close']
            
            # Ensure series logic
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]
            
            # Get only trading days within our target range
            date_range = prices.loc[start_date:end_date].index
            
            for current_day in date_range:
                # Current price on this day
                S0 = float(prices.loc[current_day])
                
                # Prices before this day for volatility calculation
                prices_before = prices.loc[:current_day]
                
                # Calculate vols for 30, 60, 90 days
                vol30 = calculate_volatility(prices_before, 30)
                vol60 = calculate_volatility(prices_before, 60)
                vol90 = calculate_volatility(prices_before, 90)
                
                # For each volatility window, calculate Greeks
                for vol_label, vol in [("30d", vol30), ("60d", vol60), ("90d", vol90)]:
                    if np.isnan(vol):
                        continue
                    
                    # Setup instrument for ATM Call, 30 DTE
                    risk_free_rate = 0.045
                    asset = Asset(
                        ticker=ticker_symbol,
                        current_price=S0,
                        volatility=vol,
                        risk_free_rate=risk_free_rate
                    )
                    
                    option = Option(
                        asset=asset,
                        strike_price=S0, # ATM
                        time_to_maturity=30/365.0, # 30 DTE
                        option_type=OptionType.CALL
                    )
                    
                    greeks = BlackScholesAnalytical.calculate_greeks(option)
                    price = BlackScholesAnalytical.calculate_price(option)
                    
                    res = {
                        "Date": current_day.strftime("%Y-%m-%d"),
                        "Ticker": ticker_symbol,
                        "Price": S0,
                        "Vol_Window": vol_label,
                        "Volatility": vol,
                        "BS_Price": price,
                        "Delta": greeks['delta'],
                        "Gamma": greeks['gamma'],
                        "Vega": greeks['vega'],
                        "Theta": greeks['theta']
                    }
                    results.append(res)
                    
            # Generate Summary and Visualizations for this Ticker
            ticker_data = [r for r in results if r['Ticker'] == ticker_symbol]
            if ticker_data:
                tdf = pd.DataFrame(ticker_data)
                print(f"\n--- Summary for {ticker_symbol} ---")
                print(f"Price Range: {tdf['Price'].min():.2f} - {tdf['Price'].max():.2f}")
                print(f"Avg Volatility (30d): {tdf[tdf['Vol_Window'] == '30d']['Volatility'].mean():.4f}")
                print(f"Avg Delta (30d): {tdf[tdf['Vol_Window'] == '30d']['Delta'].mean():.4f}")
                
                # 6. Visualization (based on the latest day in the batch)
                last_day_price = float(prices.loc[date_range[-1]])
                last_day_vol = calculate_volatility(prices.loc[:date_range[-1]], 30) # Use 30d vol for plots
                
                print(f"Generating Plots for {ticker_symbol} (using latest data)...")
                
                # Setup final instrument for plots
                risk_free_rate = 0.045
                final_asset = Asset(ticker=ticker_symbol, current_price=last_day_price, volatility=last_day_vol, risk_free_rate=risk_free_rate)
                final_option = Option(asset=final_asset, strike_price=last_day_price, time_to_maturity=30/365.0, option_type=OptionType.CALL)
                
                # Monte Carlo for paths/dist
                mc_engine = MonteCarloEngine(simulations=1000, time_steps=30)
                paths_df = mc_engine.simulate(final_option)
                
                # Greek Sensitivity Data
                spot_range = np.linspace(last_day_price * 0.8, last_day_price * 1.2, 50)
                greeks_data = {k: [] for k in ['delta', 'gamma', 'vega', 'theta', 'rho', 'gex']}
                for spot in spot_range:
                    temp_asset = Asset(ticker=ticker_symbol, current_price=spot, volatility=last_day_vol, risk_free_rate=risk_free_rate)
                    temp_option = Option(asset=temp_asset, strike_price=last_day_price, time_to_maturity=30/365.0, option_type=OptionType.CALL)
                    tmp_greeks = BlackScholesAnalytical.calculate_greeks(temp_option)
                    for key in greeks_data:
                        greeks_data[key].append(tmp_greeks[key])
                
                # Plotting
                Visualizer.plot_paths(paths_df, title=f"Monte Carlo Paths: {ticker_symbol}")
                Visualizer.plot_distribution(paths_df, strike_price=last_day_price)
                Visualizer.plot_greeks(spot_range, greeks_data, title=f"Griegos: {ticker_symbol} (Latest ATM)")
                Visualizer.plot_gex_profile(spot_range, np.array(greeks_data['gex']), last_day_price, title=f"Perfil GEX: {ticker_symbol}")

        except Exception as e:
            print(f"Error processing {ticker_symbol}: {e}")

    if results:
        df = pd.DataFrame(results)
        output_file = "batch_analysis_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n" + "="*80)
        print(f"BATCH ANALYSIS COMPLETE: {len(df)} records")
        print(f"Results saved to {output_file}")
        print("="*80)
        return df
    else:
        print("No results to display.")
        return None

if __name__ == "__main__":
    run_batch_analysis()
