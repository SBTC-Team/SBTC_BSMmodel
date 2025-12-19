# Black-Scholes Monte Carlo Simulator

This project implements a Monte Carlo simulation model to estimate the price of European Options. It adheres to Object-Oriented Programming (OOP) principles and is designed for modularity and scalability.

## Features

*   **Real-time Market Data**: Fetches live stock prices and calculates historical volatility using `yfinance`.
*   **Monte Carlo Engine**: Simulates thousands of price paths using Geometric Brownian Motion (GBM).
*   **High Performance**: Utilizes **Polars** for efficient data handling of large simulation datasets.
*   **Visualization**: Generates plots for simulation paths and final price distribution using `matplotlib`.
*   **Validation**: Compares Monte Carlo results with the analytical Black-Scholes formula.

## Architecture

The project is structured into the `option_pricing` package:

*   `market_data.py`: `MarketDataFetcher` handles API interactions with Yahoo Finance.
*   `financial_instruments.py`: Defines `Asset` and `Option` data classes.
*   `models.py`: Contains `MonteCarloEngine` (Logic) and `BlackScholesAnalytical` (Validation).
*   `visualization.py`: `Visualizer` class for plotting results.

## Prerequisites

*   Python 3.8+
*   `yfinance`
*   `polars`
*   `matplotlib`
*   `numpy`
*   `scipy`

## Usage

1.  Install dependencies:
    ```bash
    pip install yfinance polars matplotlib numpy scipy
    ```

2.  Run the main script:
    ```bash
    python main.py
    ```

The script will:
1.  Fetch the latest data for AAPL (default).
2.  Calculate the option price using the Black-Scholes formula.
3.  Run 5,000 simulations to estimate the price.
4.  Show the price difference.
5.  Display two plots: Price Paths and Distribution of Final Prices.

## Scalability

The use of the `MonteCarloEngine` class allows for easy extension to other stochastic processes (e.g., Heston model) or other option types (e.g., Asian options) by inheriting and overriding the `simulate` method.
