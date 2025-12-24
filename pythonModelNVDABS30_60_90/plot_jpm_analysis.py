import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def generate_jpm_plot():
    # 1. Load the batch analysis results
    csv_path = 'batch_analysis_results.csv'
    df = pd.read_csv(csv_path)
    
    # 2. Filter for JPM, 30d vol window, and 30d maturity (as per standard request)
    jpm_df = df[(df['Ticker'] == 'JPM') & (df['Vol_Window'] == '30d') & (df['Maturity_Days'] == 30)].copy()
    jpm_df['Date'] = pd.to_datetime(jpm_df['Date'])
    jpm_df = jpm_df.sort_values('Date')
    
    if jpm_df.empty:
        print("No JPM 30d data found in CSV.")
        return

    # 3. Fetch market data for realization
    start_date = jpm_df['Date'].min()
    end_date = jpm_df['Date'].max()
    fetch_start = start_date - timedelta(days=5)
    fetch_end = datetime.now() # Fetch up to now for latest realizations
    
    print(f"Fetching JPM market data for realizations...")
    market_data = yf.download('JPM', start=fetch_start, end=fetch_end, progress=False)
    
    if market_data.empty:
        print("Failed to fetch market data.")
        return

    prices = market_data['Adj Close'] if 'Adj Close' in market_data.columns else market_data['Close']
    if isinstance(prices, pd.DataFrame): prices = prices.iloc[:, 0]

    # 4. Calculate Realizations
    realized_values = []
    actual_future_prices = []
    
    for idx, row in jpm_df.iterrows():
        analysis_date = row['Date']
        s0 = row['Price']
        maturity = row['Maturity_Days']
        expiry_date = analysis_date + timedelta(days=maturity)
        
        if expiry_date > prices.index[-1]:
            realized_values.append(np.nan)
            actual_future_prices.append(np.nan)
            continue

        try:
            target_idx = prices.index.get_indexer([expiry_date], method='nearest')[0]
            st = float(prices.iloc[target_idx])
            rv = max(st - s0, 0)
            realized_values.append(s0 + rv)
            actual_future_prices.append(st)
        except:
            realized_values.append(np.nan)
            actual_future_prices.append(np.nan)
            
    jpm_df['Realized_Break_Even'] = realized_values
    jpm_df['Model_Break_Even'] = jpm_df['Price'] + jpm_df['BS_Price']
    jpm_df['Actual_Future_Price'] = actual_future_prices

    # 5. Plotting
    output_pdf = 'jpm_analysis_report.pdf'
    with PdfPages(output_pdf) as pdf:
        plt.figure(figsize=(14, 8))
        
        plt.plot(jpm_df['Date'], jpm_df['Price'], 'k--', label='Spot at Analysis', alpha=0.3)
        plt.plot(jpm_df['Date'], jpm_df['Model_Break_Even'], 'b-o', label='BS Model (Spot + Premium)', markersize=4)
        plt.plot(jpm_df['Date'], jpm_df['Realized_Break_Even'], 'g-s', label='Realized (Spot + Intrinsic)', markersize=4)
        plt.scatter(jpm_df['Date'], jpm_df['Actual_Future_Price'], color='orange', label='Actual Future Spot (S_T)', alpha=0.6)
        
        plt.title('JPM Comparative Analysis: Model vs Real Market (Snapshots, 30 DTE)', fontsize=16)
        plt.xlabel('Date of Analysis', fontsize=12)
        plt.ylabel('Value ($ USD)', fontsize=12)
        plt.legend(frameon=True, shadow=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xticks(rotation=30)
        plt.tight_layout()
        
        pdf.savefig()
        plt.close()
        
    print(f"Plot saved to {output_pdf}")

if __name__ == "__main__":
    generate_jpm_plot()
