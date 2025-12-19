import sys
import pandas as pd
import numpy as np
from option_pricing.market_data import MarketDataFetcher
from option_pricing.financial_instruments import Asset, Option, OptionType
from option_pricing.models import MonteCarloEngine, BlackScholesAnalytical
from option_pricing.visualization import Visualizer

def main():
    print("--- Black-Scholes Monte Carlo Simulation (Multi-Ticker & Visualization) ---")
    
    # User Inputs
    ticker_symbols = ["JPM", "PLTR", "AMD", "RGTI", "NVDA", "TSLA", "LCID", "RIVN", "ACAD", "WVE", "QS",   "AAP", "MA", "JNJ", "NVO"] # List of tickers to analyze
    strike_price = None # If None, will assume At-The-Money (current price)
    days_to_maturity = 30
    simulations = 5000
    
    results = []
    
    fetcher = MarketDataFetcher()
    engine = MonteCarloEngine(simulations=simulations, time_steps=int(days_to_maturity))
    
    for ticker in ticker_symbols:
        print(f"\n" + "="*60)
        print(f"PROCESSING: {ticker}")
        print("="*60)
        
        try:
            # 1. Fetch Market Data
            days_ago = 30
            market_data = fetcher.fetch_data(ticker, days_ago=days_ago)
            current_price = market_data.current_price
            
            # 2. Setup Instruments
            asset = Asset(
                ticker=ticker,
                current_price=current_price,
                volatility=market_data.volatility,
                risk_free_rate=market_data.risk_free_rate
            )
            
            s_price = strike_price if strike_price is not None else current_price
                
            option = Option(
                asset=asset,
                strike_price=s_price,
                time_to_maturity=days_to_maturity / 365.0,
                option_type=OptionType.CALL
            )
            
            # 3. Analytical Calculations
            bs_price = BlackScholesAnalytical.calculate_price(option)
            greeks = BlackScholesAnalytical.calculate_greeks(option)
            
            # 4. Monte Carlo Simulation
            print(f"Running {simulations} Monte Carlo simulations...")
            paths_df = engine.simulate(option)
            mc_price = engine.price_option(option, paths_df)
            
            # 5. Greek Sensitivity Analysis (for plots)
            print("Calculating Greek sensitivities over price range...")
            spot_range = np.linspace(current_price * 0.8, current_price * 1.2, 50)
            greeks_data = {k: [] for k in ['delta', 'gamma', 'vega', 'theta', 'rho', 'gex']}
            
            for spot in spot_range:
                temp_asset = Asset(ticker=ticker, current_price=spot, volatility=market_data.volatility, risk_free_rate=market_data.risk_free_rate)
                temp_option = Option(asset=temp_asset, strike_price=s_price, time_to_maturity=days_to_maturity/365.0, option_type=OptionType.CALL)
                tmp_greeks = BlackScholesAnalytical.calculate_greeks(temp_option)
                for key in greeks_data:
                    greeks_data[key].append(tmp_greeks[key])
            
            for k in greeks_data:
                greeks_data[k] = np.array(greeks_data[k])

            # 6. Visualization
            print("Generating Plots (Please check for popup windows)...")
            Visualizer.plot_paths(paths_df, title=f"Monte Carlo Paths: {ticker}", num_paths_to_plot=150)
            Visualizer.plot_distribution(paths_df, strike_price=s_price)
            Visualizer.plot_greeks(spot_range, greeks_data, title=f"Griegos: {ticker} (Strike={s_price:.2f})")
            Visualizer.plot_gex_profile(spot_range, greeks_data['gex'], current_price, title=f"Perfil GEX: {ticker}")
            
            # Collect high-level results
            res = {
                "Ticker": ticker,
                "Spot": current_price,
                "Strike": s_price,
                "Vol": market_data.volatility,
                "BS Price": bs_price,
                "MC Price": mc_price,
                "Delta": greeks['delta'],
                "Gamma": greeks['gamma'],
                "Vega": greeks['vega'],
                "Theta": greeks['theta'],
            }
            results.append(res)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final Summary Table
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*110)
        print("FINAL CONSOLIDATED RESULTS")
        print("="*110)
        print(df.to_string(index=False))
        print("="*110)
        
        # Save to CSV
        output_file = "main_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()
