import os
import importlib.util

STRATEGY_DIR = 'strategies'

def load_strategies():
    strategies = {}
    if not os.path.exists(STRATEGY_DIR):
        os.makedirs(STRATEGY_DIR)
    for file in os.listdir(STRATEGY_DIR):
        if file.endswith('.py'):
            path = os.path.join(STRATEGY_DIR, file)
            spec = importlib.util.spec_from_file_location(file[:-3], path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, 'generate_signal'):
                strategies[file[:-3]] = mod.generate_signal
    return strategies

strategies = load_strategies()
strategy_names = list(strategies.keys())
strategy_selected = st.selectbox("Select Strategy", strategy_names)
strategy_func = strategies[strategy_selected]
