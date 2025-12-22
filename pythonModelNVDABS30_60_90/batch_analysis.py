import yfinance as yf
import pandas as pd
import numpy as np
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

def run_batch_analysis(tickers=["JPM", "PLTR", "AMD", "RGTI", "NVDA", "TSLA", "LCID", "RIVN", "ACAD", "WVE", "QS",   "AAP", "MA", "JNJ", "NVO"], start_date="2024-11-17", end_date="2024-12-17"):
    print(f"--- Running Batch Analysis for {len(tickers)} Tickers ---")
    print(f"Period: {start_date} to {end_date}")
    
    results = []
    
    # Convert dates to datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # We need extra data before the start_date for the 90-day volatility lookback
    fetch_start = start_dt - timedelta(days=150) # Buffer
    
    output_pdf = "batch_plots.pdf"
    with PdfPages(output_pdf) as pdf:
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
                    
                    # 1. Comparison Plots (30d, 60d, 90d Vol vs Realized Value)
                    print(f"Generating Comparison Plots for {ticker_symbol}...")
                    
                    DTE = 30
                    comparison_rows = []
                    mc_sim_count = 200 # Faster MC for daily snapshots
                    
                    for current_day in date_range:
                        S0 = float(prices.loc[current_day])
                        expiry_date_target = current_day + timedelta(days=DTE)
                        future_prices = prices.loc[current_day:]
                        
                        real_val = np.nan
                        ST = np.nan
                        if len(future_prices) > 20: 
                            try:
                                idx_expiry = future_prices.index.get_indexer([expiry_date_target], method='nearest')[0]
                                ST = float(future_prices.iloc[idx_expiry])
                                real_val = max(ST - S0, 0)
                            except:
                                pass
                        
                        comp_row = {
                            "Date": current_day, 
                            "Price": S0, 
                            "Real_Value": S0 + real_val if not np.isnan(real_val) else np.nan,
                            "Actual_Future_Price": ST
                        }
                        
                        for win in ["30d", "60d", "90d"]:
                            # Get the BS price and Vol from our results
                            res_slice = tdf[(tdf['Date'] == current_day.strftime("%Y-%m-%d")) & (tdf['Vol_Window'] == win)]
                            if not res_slice.empty:
                                bs_p = res_slice['BS_Price'].values[0]
                                vol = res_slice['Volatility'].values[0]
                                comp_row[f"BS_{win}"] = S0 + bs_p
                                
                                # Quick MC price for comparison
                                try:
                                    mc_asset = Asset(ticker=ticker_symbol, current_price=S0, volatility=vol, risk_free_rate=0.045)
                                    mc_opt = Option(asset=mc_asset, strike_price=S0, time_to_maturity=DTE/365.0, option_type=OptionType.CALL)
                                    mc_engine_fast = MonteCarloEngine(simulations=mc_sim_count, time_steps=DTE)
                                    mc_p = mc_engine_fast.price_option(mc_opt)
                                    comp_row[f"MC_{win}"] = S0 + mc_p
                                except:
                                    comp_row[f"MC_{win}"] = np.nan
                            else:
                                comp_row[f"BS_{win}"] = np.nan
                                comp_row[f"MC_{win}"] = np.nan
                                
                        comparison_rows.append(comp_row)
                    
                    cdf = pd.DataFrame(comparison_rows)
                    
                    fig_comp, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
                    fig_comp.suptitle(f"Advanced Model Comparison: {ticker_symbol} (30 DTE)", fontsize=16)
                    
                    for i, win in enumerate(["30d", "60d", "90d"]):
                        ax = axes[i]
                        ax.plot(cdf['Date'], cdf['Price'], label='Current Spot Price ($)', color='gray', alpha=0.4, linestyle='--')
                        ax.plot(cdf['Date'], cdf[f"BS_{win}"], label=f'Model Break-even (BS - {win} Vol)', color='blue', linewidth=1.5)
                        ax.plot(cdf['Date'], cdf[f"MC_{win}"], label=f'Model Break-even (MC - {win} Vol)', color='cyan', linewidth=1.0, linestyle=':')
                        ax.plot(cdf['Date'], cdf['Real_Value'], label='Realized Intrinsic Value (Spot + IV at T+30)', color='green', linewidth=1.5)
                        ax.plot(cdf['Date'], cdf['Actual_Future_Price'], label='Actual Future Spot (S_T at T+30)', color='orange', linewidth=1.0, alpha=0.7)
                        
                        ax.set_title(f"Comparison using {win} Volatility Lookback", fontsize=12)
                        ax.legend(fontsize=8, loc='upper left', ncol=2)
                        ax.grid(True, alpha=0.2)
                        ax.set_ylabel("Price / Value ($ USD)")
                    
                    plt.xlabel("Analysis Date")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    pdf.savefig(fig_comp)
                    plt.close(fig_comp)

                    # 2. Visualization (based on the latest day in the batch)
                    last_day_price = float(prices.loc[date_range[-1]])
                    last_day_vol = calculate_volatility(prices.loc[:date_range[-1]], 30) # Use 30d vol for plots
                    
                    print(f"Generating Standard Plots for {ticker_symbol} (using latest data)...")
                    
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
                    Visualizer.plot_paths(paths_df, title=f"Monte Carlo Paths: {ticker_symbol}", pdf=pdf)
                    Visualizer.plot_distribution(paths_df, strike_price=last_day_price, pdf=pdf)
                    Visualizer.plot_greeks(spot_range, greeks_data, title=f"Greeks: {ticker_symbol} (Latest ATM)", pdf=pdf)
                    Visualizer.plot_gex_profile(spot_range, np.array(greeks_data['gex']), last_day_price, title=f"GEX Profile: {ticker_symbol}", pdf=pdf)
    
            except Exception as e:
                print(f"Error processing {ticker_symbol}: {e}")

    if results:
        df = pd.DataFrame(results)
        output_file = "batch_analysis_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n" + "="*80)
        print(f"BATCH ANALYSIS COMPLETE: {len(df)} records")
        print(f"Results saved to {output_file}")
        print(f"All plots saved to {output_pdf}")
        print("="*80)
        return df
    else:
        print("No results to display.")
        return None

if __name__ == "__main__":
    run_batch_analysis()
