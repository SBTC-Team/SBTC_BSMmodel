import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from option_pricing.financial_instruments import Asset, Option, OptionType
from option_pricing.models import BlackScholesAnalytical, MonteCarloEngine
from option_pricing.visualization import Visualizer
from matplotlib.backends.backend_pdf import PdfPages

def calculate_volatility(prices, window):
    """Calculates annualized volatility for a given window."""
    if len(prices) < window + 1:
        return np.nan
    log_returns = np.log(prices.tail(window + 1) / prices.tail(window + 1).shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)

def run_batch_analysis(tickers=["JPM", "PLTR", "AMD", "RGTI", "NVDA", "TSLA", "LCID", "RIVN", "ACAD", "WVE", "QS", "AAP", "MA", "JNJ", "NVO"]):
    print(f"--- Running Maturity-Specific Batch Analysis ---")
    
    results = []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    output_pdf = "batch_plots.pdf"
    with PdfPages(output_pdf) as pdf:
        for ticker_symbol in tickers:
            print(f"\nProcessing {ticker_symbol}...")
            
            try:
                # Fetch enough history: 361 days + 90 days vol lookback
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

                # We iterate through maturities: 30, 60, 90
                maturities = [30, 60, 90]
                
                for maturity in maturities:
                    # Offsets as requested: 
                    # 30d: 361, 331, 301, ..., 31
                    # 60d: 361, 301, 241, ..., 61
                    # 90d: 361, 271, 181, ..., 91
                    offsets = range(361, maturity - 1, -maturity)
                    
                    for offset in offsets:
                        analysis_date_target = today - timedelta(days=offset)
                        
                        # Find nearest trading day
                        idx = prices.index.get_indexer([analysis_date_target], method='nearest')[0]
                        current_day = prices.index[idx]
                        S0 = float(prices.iloc[idx])
                        
                        # Volatility calculation at Analysis Date
                        prices_before = prices.loc[:current_day]
                        vol30 = calculate_volatility(prices_before, 30)
                        vol60 = calculate_volatility(prices_before, 60)
                        vol90 = calculate_volatility(prices_before, 90)
                        
                        # Price options for each vol window
                        for vol_label, vol in [("30d", vol30), ("60d", vol60), ("90d", vol90)]:
                            if np.isnan(vol):
                                continue
                            
                            risk_free_rate = 0.045
                            asset = Asset(ticker=ticker_symbol, current_price=S0, volatility=vol, risk_free_rate=risk_free_rate)
                            option = Option(asset=asset, strike_price=S0, time_to_maturity=maturity/365.0, option_type=OptionType.CALL)
                            
                            price = BlackScholesAnalytical.calculate_price(option)
                            greeks = BlackScholesAnalytical.calculate_greeks(option)
                            
                            res = {
                                "Date": current_day.strftime("%Y-%m-%d"),
                                "Ticker": ticker_symbol,
                                "Maturity_Days": maturity,
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

                # Generate Visualizations for each Ticker (using latest snapshot for detailed plots)
                ticker_results = [r for r in results if r['Ticker'] == ticker_symbol]
                if ticker_results:
                    # Get the most recent snapshot for detailed plotting
                    latest_res = ticker_results[-1]
                    S_latest = latest_res['Price']
                    vol_latest = latest_res['Volatility']
                    mat_latest = latest_res['Maturity_Days']
                    
                    print(f"Generating Standard Plots for {ticker_symbol} (latest snapshot)...")
                    
                    # Setup instrument for plots
                    risk_free_rate = 0.045
                    plot_asset = Asset(ticker=ticker_symbol, current_price=S_latest, volatility=vol_latest, risk_free_rate=risk_free_rate)
                    plot_option = Option(asset=plot_asset, strike_price=S_latest, time_to_maturity=mat_latest/365.0, option_type=OptionType.CALL)
                    
                    # Monte Carlo for paths/dist
                    mc_engine = MonteCarloEngine(simulations=1000, time_steps=int(mat_latest))
                    paths_df = mc_engine.simulate(plot_option)
                    
                    # Greek Sensitivity Data
                    spot_range = np.linspace(S_latest * 0.8, S_latest * 1.2, 50)
                    greeks_data = {k: [] for k in ['delta', 'gamma', 'vega', 'theta', 'rho', 'gex']}
                    for spot in spot_range:
                        temp_asset = Asset(ticker=ticker_symbol, current_price=spot, volatility=vol_latest, risk_free_rate=risk_free_rate)
                        temp_option = Option(asset=temp_asset, strike_price=S_latest, time_to_maturity=mat_latest/365.0, option_type=OptionType.CALL)
                        tmp_greeks = BlackScholesAnalytical.calculate_greeks(temp_option)
                        for key in greeks_data:
                            greeks_data[key].append(tmp_greeks[key])
                    
                    # Plotting
                    Visualizer.plot_paths(paths_df, title=f"Monte Carlo Paths: {ticker_symbol} (T-{int(mat_latest)})", pdf=pdf)
                    Visualizer.plot_distribution(paths_df, strike_price=S_latest, pdf=pdf)
                    Visualizer.plot_greeks(spot_range, greeks_data, title=f"Greeks: {ticker_symbol} ({int(mat_latest)} DTE)", pdf=pdf)
                    Visualizer.plot_gex_profile(spot_range, np.array(greeks_data['gex']), S_latest, title=f"GEX Profile: {ticker_symbol}", pdf=pdf)
                
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
