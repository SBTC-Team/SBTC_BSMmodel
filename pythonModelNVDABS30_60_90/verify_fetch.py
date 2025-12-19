import sys
sys.path.append(".")
from option_pricing.market_data import MarketDataFetcher
from datetime import datetime, timedelta

def test_fetch():
    print("Testing MarketDataFetcher with days_ago=30...")
    try:
        fetcher = MarketDataFetcher()
        days_ago = 30
        data = fetcher.fetch_data("NVDA", days_ago=days_ago)
        print(f"Success!")
        print(f"Price: {data.current_price}")
        print(f"Volatility: {data.volatility}")
        
        expected_date = datetime.now() - timedelta(days=days_ago)
        print(f"Expected approx date: {expected_date.date()}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fetch()
