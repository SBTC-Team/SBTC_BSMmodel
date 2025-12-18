import sys
from option_pricing.market_data import MarketDataFetcher
from option_pricing.financial_instruments import Asset, Option, OptionType
from option_pricing.models import MonteCarloEngine, BlackScholesAnalytical
from option_pricing.visualization import Visualizer

def main():
    print("--- Black-Scholes Monte Carlo Simulation ---")
    
    # User Inputs (hardcoded for demonstration, could be arguments)
    ticker_symbol = "NVDA"
    strike_price = None # If None, will assume At-The-Money (current price)
    days_to_maturity = 30
    simulations = 5000
    
    # 1. Fetch Market Data
    days_ago = 30
    print(f"Fetching data for {ticker_symbol} from {days_ago} days ago...")
    try:
        fetcher = MarketDataFetcher()
        market_data = fetcher.fetch_data(ticker_symbol, days_ago=days_ago)
        print(f"Historical Price ({days_ago} days ago): {market_data.current_price:.2f}")
        print(f"Annualized Volatility: {market_data.volatility:.2%}")
        print(f"Risk Free Rate: {market_data.risk_free_rate:.2%}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # 2. Setup Instruments
    asset = Asset(
        ticker=ticker_symbol,
        current_price=market_data.current_price,
        volatility=market_data.volatility,
        risk_free_rate=market_data.risk_free_rate
    )
    
    if strike_price is None:
        strike_price = market_data.current_price
        
    option = Option(
        asset=asset,
        strike_price=strike_price,
        time_to_maturity=days_to_maturity / 365.0, # Convert days to years
        option_type=OptionType.CALL # Predicting a Call option
    )
    
    print(f"\nEvaluating {option.option_type.value.upper()} Option:")
    print(f"Strike Price: {option.strike_price:.2f}")
    print(f"Time to Maturity: {option.time_to_maturity:.4f} years")

    # 3. Method 1: Analytical Black-Scholes
    bs_price = BlackScholesAnalytical.calculate_price(option)
    print(f"\n[Analytical] Black-Scholes Price: {bs_price:.4f}")
    
    # Calculate and Display Greeks
    print("\n[Greeks & Risk Metrics]")
    greeks = BlackScholesAnalytical.calculate_greeks(option)
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Vega:  {greeks['vega']:.4f}")
    print(f"Theta: {greeks['theta']:.4f} (per day)")
    print(f"Rho:   {greeks['rho']:.4f}")
    print(f"GEX:   {greeks['gex']:.4f} (Est. Gamma Exposure for 1% move)")

    # 4. Method 2: Monte Carlo Simulation
    print(f"\n[Monte Carlo] Running {simulations} simulations...")
    engine = MonteCarloEngine(simulations=simulations, time_steps=int(days_to_maturity))
    paths_df = engine.simulate(option)
    
    mc_price = engine.price_option(option, paths_df)
    print(f"[Monte Carlo] Estimated Price:    {mc_price:.4f}")
    
    error = abs(mc_price - bs_price)
    print(f"Difference: {error:.4f} ({(error/bs_price)*100:.2f}%)")
    
    # 5. Visualization
    print("\nGenerating plots...")
    Visualizer.plot_paths(paths_df, title=f"Monte Carlo Paths for {ticker_symbol}", num_paths_to_plot=200)
    Visualizer.plot_distribution(paths_df, strike_price=strike_price)
    print("Done.")

if __name__ == "__main__":
    main()
