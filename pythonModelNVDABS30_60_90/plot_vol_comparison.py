
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from option_pricing.financial_instruments import Asset, Option, OptionType
from option_pricing.models import BlackScholesAnalytical
from matplotlib.backends.backend_pdf import PdfPages

def calculate_volatility(prices, window):
    """Calculates annualized volatility for a given window."""
    if len(prices) < window + 1:
        return np.nan
    log_returns = np.log(prices.tail(window + 1) / prices.tail(window + 1).shift(1)).dropna()
    return log_returns.std() * np.sqrt(252)

def run_vol_comparison(tickers=["NVDA", "TSLA", "AMD", "JPM", "PLTR"], start_date="2024-10-01", end_date="2024-11-15"):
    print(f"--- Running Volatility Comparison Analysis for {len(tickers)} Tickers ---")
    
    # 30 DTE for our tests
    DTE = 30
    
    output_pdf = "vol_comparison_plots.pdf"
    results_all = []
    
    with PdfPages(output_pdf) as pdf:
        for ticker_symbol in tickers:
            print(f"\nProcessing {ticker_symbol}...")
            
            # Fetch extra data for volatility windows and looking ahead for real value
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            fetch_start = start_dt - timedelta(days=150)
            fetch_end = end_dt + timedelta(days=60) # Need future data for realized value
            
            try:
                data = yf.download(ticker_symbol, start=fetch_start, end=fetch_end, progress=False)
                if data.empty:
                    print(f"Error: No data for {ticker_symbol}")
                    continue
                
                prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                if isinstance(prices, pd.DataFrame):
                    prices = prices.iloc[:, 0]
                
                target_dates = prices.loc[start_date:end_date].index
                
                ticker_results = []
                
                for current_day in target_dates:
                    S0 = float(prices.loc[current_day])
                    prices_before = prices.loc[:current_day]
                    
                    # Real Value at Expiry (intrinsic value of ATM call)
                    # We look DTE (30 days) forward. 
                    # Note: We need the nearest trading day ~30 days later.
                    expiry_date_target = current_day + timedelta(days=DTE)
                    future_prices = prices.loc[current_day:]
                    
                    if len(future_prices) > 20: # Ensure we have enough data forward
                        # Find nearest date around DTE
                        idx_expiry = future_prices.index.get_indexer([expiry_date_target], method='nearest')[0]
                        ST = float(future_prices.iloc[idx_expiry])
                        real_val = max(ST - S0, 0)
                    else:
                        real_val = np.nan

                    # Calculate BS prices for 30, 60, 90 windows
                    row = {"Date": current_day, "Price": S0, "Real_Value": S0 + real_val}
                    
                    for window in [30, 60, 90]:
                        vol = calculate_volatility(prices_before, window)
                        if not np.isnan(vol):
                            asset = Asset(ticker=ticker_symbol, current_price=S0, volatility=vol, risk_free_rate=0.045)
                            option = Option(asset=asset, strike_price=S0, time_to_maturity=DTE/365.0, option_type=OptionType.CALL)
                            bs_price = BlackScholesAnalytical.calculate_price(option)
                            row[f"BS_{window}d"] = S0 + bs_price
                        else:
                            row[f"BS_{window}d"] = np.nan
                    
                    ticker_results.append(row)
                
                df_ticker = pd.DataFrame(ticker_results)
                
                # Plotting for this ticker (3 plots)
                fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
                fig.suptitle(f"Volatility Comparison: {ticker_symbol}", fontsize=16)
                
                for i, window in enumerate([30, 60, 90]):
                    ax = axes[i]
                    ax.plot(df_ticker['Date'], df_ticker['Price'], label='Spot Price', color='gray', alpha=0.5, linestyle='--')
                    ax.plot(df_ticker['Date'], df_ticker[f"BS_{window}d"], label=f'Price + BS({window}d Vol)', color='blue', linewidth=2)
                    ax.plot(df_ticker['Date'], df_ticker['Real_Value'], label='Realized Value (Price + Intrinsic)', color='green', linewidth=1)
                    
                    ax.set_title(f"Volatility window: {window}d")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_ylabel("Price / Value")

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)
                
                results_all.extend([{**r, "Ticker": ticker_symbol} for r in ticker_results])

            except Exception as e:
                print(f"Error processing {ticker_symbol}: {e}")

    if results_all:
        df_all = pd.DataFrame(results_all)
        df_all.to_csv("vol_comparison_results.csv", index=False)
        print(f"\nAnalysis complete. Plots saved to {output_pdf}")
        print("Detailed results saved to vol_comparison_results.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_vol_comparison()
