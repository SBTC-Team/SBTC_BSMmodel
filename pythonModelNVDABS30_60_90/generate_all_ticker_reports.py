import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def generate_reports():
    # 1. Load the batch analysis results
    csv_path = 'batch_analysis_results.csv'
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Run batch_analysis.py first.")
        return

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    tickers = df['Ticker'].unique()
    maturities = [30, 60, 90]
    output_pdf = 'all_tickers_analysis_report.pdf'
    
    print(f"Generating consolidated report for {len(tickers)} tickers...")
    
    with PdfPages(output_pdf) as pdf:
        for ticker in tickers:
            print(f"Processing {ticker}...")
            
            # Fetch market data for realization (buffer from 361 days ago to now)
            today = datetime.now()
            fetch_start = today - timedelta(days=361 + 120)
            try:
                market_data = yf.download(ticker, start=fetch_start, end=today + timedelta(days=1), progress=False)
                if market_data.empty:
                    print(f"No market data for {ticker}")
                    continue
                
                prices = market_data['Adj Close'] if 'Adj Close' in market_data.columns else market_data['Close']
                if isinstance(prices, pd.DataFrame): prices = prices.iloc[:, 0]
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                continue

            # Create a 3-subplot figure for this ticker
            fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=False)
            fig.suptitle(f"Comparative Model Performance: {ticker}", fontsize=20, fontweight='bold', y=0.95)
            
            for i, maturity in enumerate(maturities):
                ax = axes[i]
                
                # Filter for this ticker, maturity, and 30d vol window (standard baseline)
                t_df = df[(df['Ticker'] == ticker) & 
                          (df['Maturity_Days'] == maturity) & 
                          (df['Vol_Window'] == '30d')].copy()
                
                if t_df.empty:
                    ax.text(0.5, 0.5, f"No data for {maturity} DTE", ha='center', va='center')
                    continue

                t_df = t_df.sort_values('Date')
                
                # Calculate Realizations
                realized_values = []
                actual_future_prices = []
                
                for idx, row in t_df.iterrows():
                    analysis_date = row['Date']
                    s0 = row['Price']
                    expiry_date = analysis_date + timedelta(days=maturity)
                    
                    if expiry_date > prices.index[-1]:
                        realized_values.append(np.nan)
                        actual_future_prices.append(np.nan)
                    else:
                        try:
                            # Find nearest price at expiry
                            target_idx = prices.index.get_indexer([expiry_date], method='nearest')[0]
                            st = float(prices.iloc[target_idx])
                            rv = max(st - s0, 0)
                            realized_values.append(s0 + rv)
                            actual_future_prices.append(st)
                        except:
                            realized_values.append(np.nan)
                            actual_future_prices.append(np.nan)

                t_df['Realized_BE'] = realized_values
                t_df['Model_BE'] = t_df['Price'] + t_df['BS_Price']
                t_df['Future_Spot'] = actual_future_prices
                
                # Plotting this maturity
                ax.plot(t_df['Date'], t_df['Price'], 'k--', label='Spot at Analysis', alpha=0.3)
                ax.plot(t_df['Date'], t_df['Model_BE'], 'b-o', label='BS Model (Spot + BS Price)', markersize=4, linewidth=1)
                ax.plot(t_df['Date'], t_df['Realized_BE'], 'g-s', label='Realized (Spot + Intrinsic)', markersize=4, linewidth=1)
                ax.scatter(t_df['Date'], t_df['Future_Spot'], color='orange', label='Actual Future Spot (S_T)', alpha=0.6, zorder=5)
                
                ax.set_title(f"{maturity} DTE Snapshots", fontsize=14, loc='left')
                ax.set_ylabel("Price / Value ($ USD)", fontsize=10)
                ax.grid(True, linestyle=':', alpha=0.5)
                ax.legend(loc='best', fontsize=9, frameon=True)
                
                # Format x-axis
                ax.tick_params(axis='x', rotation=30)

            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            pdf.savefig()
            plt.close()

    print(f"\n================================================================================")
    print(f"REPORT GENERATED: {output_pdf}")
    print(f"================================================================================")

if __name__ == "__main__":
    generate_reports()
