"""
Strategy: Volatility Breakout
Author: ChatGPT
Description:
    Captures directional moves after periods of low volatility.
    Uses ATR-based breakout levels to generate LONG and SHORT signals.
    Compatible with the interactive backtester (app.py) and batch processing.
    
Usage:
    - Place this file in your `strategies/` folder.
    - Import and select in the backtester GUI.
    - Parameters can be adjusted via the strategy panel.
"""

import pandas as pd
import numpy as np

class VolatilityBreakout:
    def __init__(self, data, atr_period=30, k=2, stop_atr_mult=1.5, take_atr_mult=3):
        """
        Parameters:
            data : pd.DataFrame
                OHLCV data with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
            atr_period : int
                Lookback period for ATR calculation
            k : float
                Multiplier for breakout bands
            stop_atr_mult : float
                Stop loss in ATR multiples
            take_atr_mult : float
                Take profit in ATR multiples
        """
        self.data = data.copy()
        self.atr_period = atr_period
        self.k = k
        self.stop_atr_mult = stop_atr_mult
        self.take_atr_mult = take_atr_mult
        self.signals = pd.Series(index=self.data.index, dtype=str)
        self.calculate_atr()
        self.generate_signals()
    
    def calculate_atr(self):
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.data['ATR'] = tr.rolling(self.atr_period, min_periods=1).mean()
    
    def generate_signals(self):
        upper_band = self.data['Close'].shift(1) + self.k * self.data['ATR']
        lower_band = self.data['Close'].shift(1) - self.k * self.data['ATR']

        long_condition = self.data['Close'] > upper_band
        short_condition = self.data['Close'] < lower_band

        self.signals[long_condition] = 'LONG'
        self.signals[short_condition] = 'SHORT'
        self.signals.fillna('HOLD', inplace=True)
    
    def get_signals(self):
        return self.signals

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/US30_1min.csv", parse_dates=['Datetime'], index_col='Datetime')
    strategy = VolatilityBreakout(df)
    signals = strategy.get_signals()
    print(signals.tail(20))
