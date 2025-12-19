import polars as pl
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """
    Handles plotting of simulation results.
    """
    
    @staticmethod
    def plot_paths(df: pl.DataFrame, title: str = "Simulación de precios con caminos de Monte Carlo NVDA", num_paths_to_plot: int = 100, pdf=None):
        """
        Plots the simulated price paths.
        """
        # Extract time
        time = df["time"].to_numpy()
        
        # Select a subset of simulation columns to avoid overcrowding the plot
        # Get simulation column names (excluding 'time')
        sim_cols = [col for col in df.columns if col.startswith("sim_")]
        
        # Limit to num_paths_to_plot
        cols_to_plot = sim_cols[:num_paths_to_plot]
        
        # Extract data for plotting
        data = df.select(cols_to_plot).to_numpy()
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(time, data, lw=1, alpha=0.6)
        plt.title(title)
        plt.xlabel("Tiempo (Años)")
        plt.ylabel("Precio del activo (USD)")
        plt.grid(True, alpha=0.3)
        
        if pdf:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_distribution(df: pl.DataFrame, strike_price: float = None, pdf=None):
        """
        Plots the histogram of the final prices.
        """
        # Get final prices
        final_prices = df.select(pl.all().exclude("time")).tail(1).to_numpy().flatten()
        
        fig = plt.figure(figsize=(10, 6))
        plt.hist(final_prices, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        if strike_price:
            plt.axvline(strike_price, color='r', linestyle='--', label=f"Strike Price ({strike_price})")
            plt.legend()
            
        plt.title("Distribución de los precios finales de los activos financieros")
        plt.xlabel("Precio (USD)")
        plt.ylabel("Densidad de probabilidad")
        plt.grid(True, alpha=0.3)
        
        if pdf:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_greeks(spot_prices: np.ndarray, greeks_dict: dict, title: str = "Sensibilidad de los Griegos ante cambios en el precio del activo", pdf=None):
        """
        Plots Delta, Gamma, Vega, Theta, and Rho against spot price.
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Flatten axes for easy iteration
        axes = axes.flatten()
        
        # Greek keys to plot
        greek_keys = ['delta', 'gamma', 'vega', 'theta', 'rho']
        
        for i, greek in enumerate(greek_keys):
            ax = axes[i]
            ax.plot(spot_prices, greeks_dict[greek], lw=2)
            ax.set_title(greek.capitalize())
            ax.set_xlabel("Spot Price")
            ax.set_ylabel(greek.capitalize())
            ax.grid(True, alpha=0.3)
            
        # Hide the last empty subplot if we have 5 greeks and 6 slots
        if len(greek_keys) < len(axes):
            for i in range(len(greek_keys), len(axes)):
                axes[i].axis('off')
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if pdf:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_gex_profile(spot_prices: np.ndarray, gex_values: np.ndarray, current_price: float, title: str = "Perfil de Gamma Exposure (GEX)", pdf=None):
        """
        Plots the Gamma Exposure (GEX) against spot price.
        """
        fig = plt.figure(figsize=(10, 6))
        plt.plot(spot_prices, gex_values, color='purple', lw=2, label='GEX')
        
        # Highlight current price
        plt.axvline(x=current_price, color='black', linestyle='--', label=f'Precio Actual: {current_price:.2f}')
        
        # Fill areas
        plt.fill_between(spot_prices, gex_values, where=(gex_values > 0), color='green', alpha=0.3, label='Positive GEX (Long Gamma)')
        plt.fill_between(spot_prices, gex_values, where=(gex_values < 0), color='red', alpha=0.3, label='Negative GEX (Short Gamma)')
        
        plt.title(title)
        plt.xlabel("Precio del Activo (Spot)")
        plt.ylabel("GEX ($ Gamma Exposure)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if pdf:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()
