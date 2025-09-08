# strategy_1.py
class Strategy:
    def init(self, data):
        """
        Called once to initialize the strategy.
        data: pandas DataFrame with historical data
        """
        self.data = data

    def on_bar(self, bar):
        """
        Called for each new bar.
        bar: pandas Series for current bar
        Returns: signals dictionary {'buy': True/False, 'sell': True/False, ...}
        """
        # Example placeholder logic
        signal = {"buy": False, "sell": False}
        return signal
