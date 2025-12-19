import numpy as np
import polars as pl
from scipy.stats import norm
from .financial_instruments import Option, OptionType

class MonteCarloEngine:
    """
    Monte Carlo Simulation Engine for European Options.
    Uses Geometric Brownian Motion (GBM).
    """

    def __init__(self, simulations: int = 10000, time_steps: int = 252, seed: int = None):
        """
        :param simulations: Number of paths to simulate.
        :param time_steps: Number of time steps per simulation (default 252 for daily/year).
        :param seed: Random seed for reproducibility.
        """
        self.simulations = simulations
        self.time_steps = time_steps
        if seed:
            np.random.seed(seed)

    def simulate(self, option: Option) -> pl.DataFrame:
        """ 
        Runs the Monte Carlo simulation and returns the paths.
        Returns a Polars DataFrame with simulation paths.
        """
        S0 = option.asset.current_price
        K = option.strike_price
        T = option.time_to_maturity
        r = option.asset.risk_free_rate
        sigma = option.asset.volatility
        
        dt = T / self.time_steps
        
        # Generating random noise (Interpreting "Gaussian distribution of Brownian noise")
        # Z ~ N(0, 1)
        Z = np.random.standard_normal((self.time_steps, self.simulations))
        
        # Geometric Brownian Motion Equation:
        # S_t = S_{t-1} * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
        
        # We can implement this vectorally
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        # Calculate returns at each step
        daily_returns = np.exp(drift + diffusion)
        
        # Prepend 1.0 for the start price scaling
        # shape: (time_steps + 1, simulations)
        price_paths = np.vstack([np.ones((1, self.simulations)), daily_returns])
        
        # Cumulative product to get price paths
        price_paths = S0 * np.cumprod(price_paths, axis=0)
        
        # Convert to Polars DataFrame for efficient handling
        # Since we have many columns (simulations), Polars is efficient, 
        # but wide format might be heavy for massive sims. 
        # For plotting efficiency, we might subset, but here we return all.
        # Column names: sim_0, sim_1, ...
        
        # Optimization: Polars is column-major. Creating thousands of columns can be slow if 'simulations' is huge.
        # However, for price path analysis (time series), this is standard.
        schema = {f"sim_{i}": pl.Float64 for i in range(min(self.simulations, 100))} # Naming just a few or all?
        # Creating dataframe directly from numpy array
        
        df = pl.DataFrame(price_paths, schema=[f"sim_{i}" for i in range(self.simulations)])
        
        # Add a time column
        time_axis = np.linspace(0, T, self.time_steps + 1)
        df_with_time = df.with_columns(pl.Series(name="time", values=time_axis))
        
        return df_with_time

    def price_option(self, option: Option, paths_df: pl.DataFrame) -> float:
        """
        Calculates the option price based on the final prices from the simulation.
        """
        # Get the final prices (last row of the simulation columns)
        # We assume 'time' is a column, so we exclude it.
        final_prices = paths_df.select(pl.all().exclude("time")).tail(1).to_numpy().flatten()
        
        if option.option_type == OptionType.CALL:
            payoffs = np.maximum(final_prices - option.strike_price, 0)
        else:
            payoffs = np.maximum(option.strike_price - final_prices, 0)
            
        # Discount the average payoff back to present value
        price = np.exp(-option.asset.risk_free_rate * option.time_to_maturity) * np.mean(payoffs)
        return price


class BlackScholesAnalytical:
    """
    Standard Black-Scholes Analytical Model for comparison/verification.
    """
    
    @staticmethod

    def calculate_greeks(option: Option) -> dict:
        """
        Calculates the Greeks (Delta, Gamma, Vega, Theta, Rho) and GEX for the option.
        """
        S = option.asset.current_price
        K = option.strike_price
        T = option.time_to_maturity
        r = option.asset.risk_free_rate
        sigma = option.asset.volatility
        q = option.asset.dividend_yield # Assuming we might want to use Dividend Yield (q) generally 0 for now as per Asset class default
        
        # d1 and d2 calculations
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Helper for N'(x) - Probability Density Function
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_d2 = norm.cdf(d2)
        cdf_neg_d2 = norm.cdf(-d2)

        greeks = {}
        
        # Delta
        if option.option_type == OptionType.CALL:
            delta = np.exp(-q * T) * cdf_d1
        else:
            delta = -np.exp(-q * T) * cdf_neg_d1
        greeks['delta'] = delta

        # Gamma (Same for Call and Put)
        gamma = (np.exp(-q * T) * pdf_d1) / (S * sigma * np.sqrt(T))
        greeks['gamma'] = gamma
        
        # Vega (Same for Call and Put)
        vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
        # Often expressed as change per 1% change in volatility, so we might divide by 100
        # But standard definition is per unit change. We will keep standard.
        greeks['vega'] = vega
        
        # Theta
        if option.option_type == OptionType.CALL:
            theta = (- (S * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * cdf_d2 
                     + q * S * np.exp(-q * T) * cdf_d1)
        else:
            theta = (- (S * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * cdf_neg_d2 
                     - q * S * np.exp(-q * T) * cdf_neg_d1)
        # Theta is often expressed as "per day" decay
        greeks['theta'] = theta / 365.0 

        # Rho
        if option.option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * cdf_d2
        else:
            rho = -K * T * np.exp(-r * T) * cdf_neg_d2
        greeks['rho'] = rho
        
        # GEX (Gamma Exposure) Proxy
        # Standard GEX approximation: Gamma * Spot^2 * 0.01 (representing exposure for 1% move)
        # Or often just Gamma * Spot * Spot. 
        # We will use Gamma * Spot^2 / 100 to represent dollar gamma for a 1% move.
        greeks['gex'] = gamma * (S ** 2) * 0.01

        return greeks

    @staticmethod
    def calculate_price(option: Option) -> float:
        S = option.asset.current_price
        K = option.strike_price
        T = option.time_to_maturity
        r = option.asset.risk_free_rate
        sigma = option.asset.volatility
        q = option.asset.dividend_yield
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option.option_type == OptionType.CALL:
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            
        return price
