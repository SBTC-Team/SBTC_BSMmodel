import numpy as np
import sys
import matplotlib.pyplot as plt

# Ensure local modules can be imported
sys.path.append(".")

from option_pricing.market_data import MarketDataFetcher
from option_pricing.financial_instruments import Asset, Option, OptionType
from option_pricing.models import BlackScholesAnalytical
from option_pricing.visualization import Visualizer

def main():
    print("--- Visualización de Griegos y GEX (Gamma Exposure) ---")
    
    ticker = "NVDA"
    print(f"Obteniendo datos para {ticker}...")
    
    try:
        fetcher = MarketDataFetcher()
        market_data = fetcher.fetch_data(ticker)
        current_price = market_data.current_price
        print(f"Precio Actual: {current_price:.2f}")
    except Exception as e:
        print(f"Error obteniendo datos: {e}")
        # Fallback for testing if api fails or offline
        current_price = 100.0
        market_data = type('obj', (object,), {'current_price': 100.0, 'volatility': 0.4, 'risk_free_rate': 0.05})
        print(f"Usando datos de prueba: Precio={current_price}")

    # Configuration
    strike_price = current_price # ATM
    days_to_maturity = 30
    asset = Asset(
        ticker=ticker,
        current_price=current_price,
        volatility=market_data.volatility,
        risk_free_rate=market_data.risk_free_rate
    )
    
    # Create an Option object (Call)
    option = Option(
        asset=asset,
        strike_price=strike_price,
        time_to_maturity=days_to_maturity / 365.0,
        option_type=OptionType.CALL 
    )

    # 1. Generate Spot Price Range (e.g., +/- 20%)
    spot_range = np.linspace(current_price * 0.8, current_price * 1.2, 100)
    
    # 2. Calculate Greeks for each spot price
    greeks_data = {
        'delta': [],
        'gamma': [],
        'vega': [],
        'theta': [],
        'rho': [],
        'gex': []
    }
    
    for spot in spot_range:
        # Create a temporary asset with the modified spot price
        temp_asset = Asset(
            ticker=ticker,
            current_price=spot,
            volatility=market_data.volatility,
            risk_free_rate=market_data.risk_free_rate
        )
        temp_option = Option(
            asset=temp_asset,
            strike_price=strike_price,
            time_to_maturity=days_to_maturity / 365.0,
            option_type=OptionType.CALL
        )
        
        greeks = BlackScholesAnalytical.calculate_greeks(temp_option)
        for key in greeks_data:
            greeks_data[key].append(greeks[key])

    # Convert lists to numpy arrays
    for key in greeks_data:
        greeks_data[key] = np.array(greeks_data[key])
        
    # 3. Visualization
    print("\nGenerando gráficos de los Griegos...")
    Visualizer.plot_greeks(spot_range, greeks_data, title=f"Griegos de una Opción CALL de {ticker} (Strike={strike_price:.2f})")
    
    print("\nGenerando perfil de GEX...")
    # For GEX profile, we typically look at Market Makers perspective.
    # If Market Makers are Short Gamma (Simulation: Client buys Call, MM sells Call -> MM is Short Call)
    # Short Call has Negative Gamma.
    # If Market Makers are Long Gamma (Simulation: Client sells Call, MM buys Call -> MM is Long Call)
    # Long Call has Positive Gamma.
    
    # Let's plot the GEX for the option we created (Long Call Perspective)
    # Note: GEX is technically open interest weighted, but here we show the per-contract GEX curve.
    Visualizer.plot_gex_profile(spot_range, greeks_data['gex'], current_price, title=f"Perfil de Gamma Exposure (GEX) - Long Call {ticker}")

    # 4. Explanation of GEX
    print_gex_explanation()

def print_gex_explanation():
    explanation = """
    ================================================================================
    EXPLICACIÓN DE GEX (GAMMA EXPOSURE)
    ================================================================================
    
    ¿Qué es GEX?
    GEX mide la exposición de los Creadores de Mercado (Market Makers) a la Gamma.
    Gamma es la tasa de cambio de Delta con respecto al precio del activo subyacente.
    Básicamente, cuánto tienen que ajustar su cobertura (hedging) los Market Makers
    cuando el precio se mueve.

    Interpretación:
    --------------------------------------------------------------------------------
    1. GEX Positivo (Long Gamma):
       - Ocurre cuando los Market Makers tienen posiciones largas en opciones (compraron opciones).
       - Para mantener su Delta Neutral, deben:
         * VENDER cuando el precio SUBE.
         * COMPRAR cuando el precio BAJA.
       - EFECTO: Esto reduce la volatilidad del mercado. Actúa como un amortiguador.
    
    2. GEX Negativo (Short Gamma):
       - Ocurre cuando los Market Makers han vendido opciones (están cortos en opciones).
       - Para mantener su Delta Neutral, deben:
         * COMPRAR cuando el precio SUBE (persiguen el precio).
         * VENDER cuando el precio BAJA.
       - EFECTO: Esto exacerba la volatilidad. Los movimientos se vuelven más bruscos.
       - "Gamma Sqeeze": Un aumento rápido de precios obliga a comprar más, subiendo más el precio.

    En el gráfico:
    - Las áreas VERDES indican zonas donde la Gamma es positiva (Estabilidad).
    - Las áreas ROJAS indican zonas donde la Gamma sería negativa (Volatilidad), 
      dependiendo de si se está Largo o Corto en la opción.
      
    Nota: El gráfico mostrado es para una posición LARGA en una Call. 
    Si fueras el vendedor (Short Call), la curva se invertiría.
    ================================================================================
    """
    print(explanation)

if __name__ == "__main__":
    main()
