"""
Batch Price Analysis for Black-Scholes Model
Analyzes closing prices for different lookback periods (30, 60, 90 days)
for a specified date range.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import json

class BatchPriceAnalyzer:
    """
    Analyzes historical closing prices for multiple lookback periods.
    """
    
    def __init__(self, ticker: str, start_date: str, end_date: str):
        """
        Initialize the batch analyzer.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'JPM')
            start_date: Start date in format 'YYYY-MM-DD' (e.g., '2024-11-17')
            end_date: End date in format 'YYYY-MM-DD' (e.g., '2024-12-17')
        """
        self.ticker = ticker
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.lookback_periods = [30, 60, 90]  # Days
        
    def fetch_historical_data(self, lookback_days: int) -> pd.DataFrame:
        """
        Fetch historical data with specified lookback period.
        
        Args:
            lookback_days: Number of days to look back from start_date
            
        Returns:
            DataFrame with historical prices
        """
        # Calculate the data range needed
        data_start = self.start_date - timedelta(days=lookback_days + 10)  # Buffer for missing days
        data_end = self.end_date + timedelta(days=1)  # Include end date
        
        print(f"\nFetching {lookback_days}-day lookback data...")
        print(f"Data range: {data_start.date()} to {data_end.date()}")
        
        ticker_obj = yf.Ticker(self.ticker)
        hist = ticker_obj.history(start=data_start, end=data_end)
        
        if hist.empty:
            raise ValueError(f"No data available for {self.ticker}")
            
        return hist
    
    def calculate_metrics(self, prices: pd.Series) -> Dict:
        """
        Calculate statistical metrics for price series.
        
        Args:
            prices: Series of closing prices
            
        Returns:
            Dictionary with calculated metrics
        """
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        metrics = {
            'count': len(prices),
            'mean_price': float(prices.mean()),
            'std_price': float(prices.std()),
            'min_price': float(prices.min()),
            'max_price': float(prices.max()),
            'first_price': float(prices.iloc[0]),
            'last_price': float(prices.iloc[-1]),
            'price_change': float(prices.iloc[-1] - prices.iloc[0]),
            'price_change_pct': float((prices.iloc[-1] / prices.iloc[0] - 1) * 100),
            'volatility_daily': float(log_returns.std()),
            'volatility_annualized': float(log_returns.std() * np.sqrt(252)),
            'mean_return_daily': float(log_returns.mean()),
            'mean_return_annualized': float(log_returns.mean() * 252)
        }
        
        return metrics
    
    def process_lookback_period(self, lookback_days: int) -> Dict:
        """
        Process data for a specific lookback period.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with analysis results
        """
        # Fetch all historical data
        hist_data = self.fetch_historical_data(lookback_days)
        
        results = {
            'ticker': self.ticker,
            'lookback_days': lookback_days,
            'analysis_period': {
                'start': self.start_date.strftime('%Y-%m-%d'),
                'end': self.end_date.strftime('%Y-%m-%d')
            },
            'periods': []
        }
        
        # Filter data for the analysis period
        analysis_mask = (hist_data.index >= self.start_date) & (hist_data.index <= self.end_date)
        analysis_dates = hist_data[analysis_mask].index
        
        print(f"\nProcessing {len(analysis_dates)} trading days in analysis period...")
        
        for target_date in analysis_dates:
            # Get lookback window ending at target_date
            lookback_start = target_date - timedelta(days=lookback_days + 20)  # Buffer for weekends/holidays
            lookback_mask = (hist_data.index >= lookback_start) & (hist_data.index <= target_date)
            lookback_data = hist_data[lookback_mask]['Close']
            
            # Get exactly the last N trading days
            if len(lookback_data) > lookback_days:
                lookback_data = lookback_data.tail(lookback_days)
            
            if len(lookback_data) < lookback_days * 0.8:  # At least 80% of expected days
                print(f"Warning: Only {len(lookback_data)} days available for {target_date.date()}")
                continue
            
            # Calculate metrics
            metrics = self.calculate_metrics(lookback_data)
            
            period_result = {
                'target_date': target_date.strftime('%Y-%m-%d'),
                'actual_days': len(lookback_data),
                'lookback_start_date': lookback_data.index[0].strftime('%Y-%m-%d'),
                'metrics': metrics
            }
            
            results['periods'].append(period_result)
            
            # Print summary for this date
            print(f"  {target_date.date()}: Price={metrics['last_price']:.2f}, "
                  f"Vol={metrics['volatility_annualized']:.2%}, "
                  f"Days={len(lookback_data)}")
        
        return results
    
    def run_batch_analysis(self) -> Dict:
        """
        Run batch analysis for all lookback periods.
        
        Returns:
            Dictionary with complete analysis results
        """
        print("="*80)
        print(f"BATCH PRICE ANALYSIS FOR {self.ticker}")
        print(f"Analysis Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Lookback Periods: {self.lookback_periods}")
        print("="*80)
        
        all_results = {
            'ticker': self.ticker,
            'analysis_period': {
                'start': self.start_date.strftime('%Y-%m-%d'),
                'end': self.end_date.strftime('%Y-%m-%d')
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'lookback_analyses': []
        }
        
        for lookback_days in self.lookback_periods:
            print(f"\n{'='*80}")
            print(f"LOOKBACK PERIOD: {lookback_days} DAYS")
            print(f"{'='*80}")
            
            try:
                results = self.process_lookback_period(lookback_days)
                all_results['lookback_analyses'].append(results)
            except Exception as e:
                print(f"Error processing {lookback_days}-day lookback: {e}")
                continue
        
        return all_results
    
    def save_results(self, results: Dict, output_format: str = 'json'):
        """
        Save analysis results to file.
        
        Args:
            results: Analysis results dictionary
            output_format: Output format ('json' or 'csv')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_format == 'json':
            filename = f"{self.ticker}_batch_analysis_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {filename}")
            
        elif output_format == 'csv':
            # Flatten results for CSV
            rows = []
            for lookback_analysis in results['lookback_analyses']:
                lookback_days = lookback_analysis['lookback_days']
                for period in lookback_analysis['periods']:
                    row = {
                        'ticker': self.ticker,
                        'lookback_days': lookback_days,
                        'target_date': period['target_date'],
                        'actual_days': period['actual_days'],
                        'lookback_start_date': period['lookback_start_date'],
                        **period['metrics']
                    }
                    rows.append(row)
            
            df = pd.DataFrame(rows)
            filename = f"{self.ticker}_batch_analysis_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"\n✓ Results saved to: {filename}")
            
            return df
    
    def print_summary(self, results: Dict):
        """
        Print a summary of the analysis results.
        
        Args:
            results: Analysis results dictionary
        """
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        for lookback_analysis in results['lookback_analyses']:
            lookback_days = lookback_analysis['lookback_days']
            num_periods = len(lookback_analysis['periods'])
            
            if num_periods == 0:
                continue
            
            print(f"\n{lookback_days}-Day Lookback Period:")
            print(f"  Number of trading days analyzed: {num_periods}")
            
            # Calculate summary statistics
            all_prices = [p['metrics']['last_price'] for p in lookback_analysis['periods']]
            all_vols = [p['metrics']['volatility_annualized'] for p in lookback_analysis['periods']]
            
            print(f"  Price range: ${min(all_prices):.2f} - ${max(all_prices):.2f}")
            print(f"  Average price: ${np.mean(all_prices):.2f}")
            print(f"  Volatility range: {min(all_vols):.2%} - {max(all_vols):.2%}")
            print(f"  Average volatility: {np.mean(all_vols):.2%}")


def main():
    """
    Main execution function.
    """
    # Configuration
    TICKER = "JPM"  # JP Morgan Chase
    START_DATE = "2024-11-17"  # November 17, 2024
    END_DATE = "2024-12-17"    # December 17, 2024
    
    # Initialize analyzer
    analyzer = BatchPriceAnalyzer(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # Run analysis
    results = analyzer.run_batch_analysis()
    
    # Print summary
    analyzer.print_summary(results)
    
    # Save results
    df = analyzer.save_results(results, output_format='csv')
    analyzer.save_results(results, output_format='json')
    
    # Display sample of CSV data
    print("\n" + "="*80)
    print("SAMPLE DATA (First 10 rows):")
    print("="*80)
    print(df.head(10).to_string())
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
