"""Utility functions for training scripts."""

import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators focused on pure price action.
    
    Args:
        df: DataFrame with at least 'close' column (optionally 'high', 'low')
        
    Returns:
        DataFrame with technical indicators:
        - rsi_14: Relative Strength Index (14 periods)
        - sma_7: Simple Moving Average (7 bars)
        - sma_21: Simple Moving Average (21 bars)
        - atr_14: Average True Range (14 periods) - volatility
    """
    close = df["close"].astype(float)
    
    # RSI(14)
    try:
        import talib  # type: ignore
        rsi = pd.Series(talib.RSI(close.values, timeperiod=14), index=df.index)
    except Exception:
        # Fallback calculation without TA-Lib
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14.0, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14.0, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

    # Simple Moving Averages
    sma7 = close.rolling(window=7, min_periods=1).mean()
    sma21 = close.rolling(window=21, min_periods=1).mean()
    
    # ATR (Average True Range) - volatility indicator
    try:
        import talib  # type: ignore
        # Try to use high/low if available
        if "high" in df.columns and "low" in df.columns:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            atr = pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=14), index=df.index)
        else:
            # Fallback: use close-to-close range (less accurate but works)
            atr = close.diff().abs().rolling(window=14, min_periods=1).mean()
    except Exception:
        # Fallback without TA-Lib: simple close-to-close ATR
        atr = close.diff().abs().rolling(window=14, min_periods=1).mean()

    # Combine all indicators (pure price action only)
    tech = pd.DataFrame({
        "rsi_14": rsi,
        "sma_7": sma7,
        "sma_21": sma21,
        "atr_14": atr,
    })
    
    # Fill NaN values
    tech = tech.bfill().ffill().fillna(0.0)
    return tech

