
import sys
import unittest
import numpy as np
from option_pricing.financial_instruments import Asset, Option, OptionType
from option_pricing.models import BlackScholesAnalytical

class TestGreeks(unittest.TestCase):
    def test_call_greeks(self):
        # Standard test case
        # S=100, K=100, T=1, r=0.05, sigma=0.2, q=0
        # Expected values can be approximately known
        asset = Asset("TEST", current_price=100.0, volatility=0.2, risk_free_rate=0.05, dividend_yield=0.0)
        option = Option(asset, strike_price=100.0, time_to_maturity=1.0, option_type=OptionType.CALL)
        
        greeks = BlackScholesAnalytical.calculate_greeks(option)
        
        # Delta for ATM call is roughly 0.6 (N(d1))
        # d1 = (ln(1) + (0.05 + 0.02) * 1) / 0.2 = 0.07 / 0.2 = 0.35
        # N(0.35) approx 0.6368
        self.assertAlmostEqual(greeks['delta'], 0.6368, places=2)
        
        # Gamma should be positive
        self.assertGreater(greeks['gamma'], 0)
        
        # Vega should be positive
        self.assertGreater(greeks['vega'], 0)
        
        # Theta for call usually negative
        self.assertLess(greeks['theta'], 0)
        
        # Rho for call positive
        self.assertGreater(greeks['rho'], 0)
        
        # GEX should be positive
        self.assertGreater(greeks['gex'], 0)
        
        print("\nCalculated Call Greeks:")
        for k, v in greeks.items():
            print(f"{k}: {v:.4f}")

    def test_put_greeks(self):
        asset = Asset("TEST", current_price=100.0, volatility=0.2, risk_free_rate=0.05, dividend_yield=0.0)
        option = Option(asset, strike_price=100.0, time_to_maturity=1.0, option_type=OptionType.PUT)
        
        greeks = BlackScholesAnalytical.calculate_greeks(option)
        
        # Delta for put is negative
        self.assertLess(greeks['delta'], 0)
        # Put Delta approx Call Delta - 1 => 0.6368 - 1 = -0.3632
        self.assertAlmostEqual(greeks['delta'], -0.3632, places=2)
        
        # Gamma same as call
        self.assertGreater(greeks['gamma'], 0)
        
        print("\nCalculated Put Greeks:")
        for k, v in greeks.items():
            print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    unittest.main()
