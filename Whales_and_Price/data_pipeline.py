"""
data_pipeline.py - Reusable Data Fetching and Feature Engineering

Shared module used by both Train.py and app.py for data operations.
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys with fallback to environment variables
# In Docker, dotenv might not work, so we also check os.environ directly
DUNE_API_KEY = os.getenv("DUNE_WHALES_API") or os.environ.get("DUNE_WHALES_API")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY") or os.environ.get("COINGECKO_API_KEY")
QUERY_ID = "6184996"


def fetch_dune_data(query_id: str = QUERY_ID, api_key: str = None) -> pd.DataFrame:
    """
    Fetch latest whale data from Dune Analytics
    
    Returns:
        DataFrame with whale activity data (excludes today)
    """
    if api_key is None:
        api_key = DUNE_API_KEY
    
    if not api_key:
        raise ValueError("❌ DUNE_WHALES_API key not found")
    
    print("\n🔄 Fetching whale data from Dune...")
    
    headers = {"x-dune-api-key": api_key}
    
    # Execute query
    execute_url = f"https://api.dune.com/api/v1/query/{query_id}/execute"
    execute_response = requests.post(execute_url, headers=headers)
    execute_data = execute_response.json()
    
    execution_id = execute_data.get("execution_id")
    if not execution_id:
        raise ValueError(f"❌ No execution_id: {execute_data}")
    
    print(f"   Execution ID: {execution_id}")
    
    # Poll for completion
    status_url = f"https://api.dune.com/api/v1/execution/{execution_id}/status"
    results_url = f"https://api.dune.com/api/v1/execution/{execution_id}/results"
    
    max_attempts = 30
    for attempt in range(max_attempts):
        status_response = requests.get(status_url, headers=headers).json()
        state = status_response.get("state")
        
        if state == "QUERY_STATE_COMPLETED":
            break
        elif state == "QUERY_STATE_FAILED":
            raise RuntimeError(f"❌ Query failed: {status_response}")
        
        print(f"   Waiting... (attempt {attempt + 1}/{max_attempts})")
        time.sleep(10)
    
    # Fetch results
    results = requests.get(results_url, headers=headers).json()
    df = pd.DataFrame(results["result"]["rows"])
    df['block_date'] = pd.to_datetime(df['block_date']).dt.date
    
    # Exclude today (incomplete data)
    today = datetime.now().date()
    df = df[df['block_date'] < today]
    
    print(f"   ✅ Retrieved {len(df)} rows")
    print(f"   Date range: {df['block_date'].min()} → {df['block_date'].max()}")
    
    return df


def fetch_coingecko_price(coin_id: str, from_date: str, to_date: str, 
                          api_key: str = None) -> pd.DataFrame:
    """
    Fetch daily prices from CoinGecko
    
    Args:
        coin_id: 'ethereum' or 'bitcoin'
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        api_key: CoinGecko API key
    
    Returns:
        DataFrame with date and price columns
    """
    if api_key is None:
        api_key = COINGECKO_API_KEY
    
    if not api_key:
        raise ValueError("❌ COINGECKO_API_KEY not found")
    
    print(f"\n📈 Fetching {coin_id.upper()} prices...")
    
    from_ts = int(pd.Timestamp(from_date).timestamp())
    to_ts = int(pd.Timestamp(to_date).timestamp())
    
    url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    headers = {'accept': 'application/json', 'x-cg-pro-api-key': api_key}
    params = {'vs_currency': 'usd', 'from': from_ts, 'to': to_ts}
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    
    prices = data['prices']
    df = pd.DataFrame({
        'timestamp': [p[0] for p in prices],
        'price': [p[1] for p in prices]
    })
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    df = df.groupby('date', as_index=False).agg({'price': 'last'})
    
    print(f"   ✅ {len(df)} days | ${df['price'].min():.0f} - ${df['price'].max():.0f}")
    
    return df


def add_price_features(df: pd.DataFrame, price_col: str, prefix: str) -> pd.DataFrame:
    """
    Add price-based features (returns, MA, volatility, RSI)
    
    Args:
        df: DataFrame with price data
        price_col: Name of price column
        prefix: Feature prefix (e.g., 'eth', 'btc')
    
    Returns:
        DataFrame with added features
    """
    df = df.sort_values('block_date').reset_index(drop=True)
    
    # Returns
    df[f'{prefix}_daily_return'] = df[price_col].pct_change()
    df[f'{prefix}_log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Moving averages
    df[f'{prefix}_ma7'] = df[price_col].rolling(7, min_periods=1).mean()
    df[f'{prefix}_ma30'] = df[price_col].rolling(30, min_periods=1).mean()
    
    # Momentum
    df[f'{prefix}_vs_ma7'] = df[price_col] / df[f'{prefix}_ma7']
    df[f'{prefix}_vs_ma30'] = df[price_col] / df[f'{prefix}_ma30']
    
    # Volatility
    df[f'{prefix}_vol7'] = df[f'{prefix}_daily_return'].rolling(7, min_periods=1).std()
    df[f'{prefix}_vol30'] = df[f'{prefix}_daily_return'].rolling(30, min_periods=1).std()
    
    # Returns
    df[f'{prefix}_ret7d'] = df[price_col].pct_change(7)
    df[f'{prefix}_ret30d'] = df[price_col].pct_change(30)
    
    # RSI
    returns = df[f'{prefix}_daily_return']
    gains = returns.where(returns > 0, 0).rolling(14, min_periods=1).mean()
    losses = -returns.where(returns < 0, 0).rolling(14, min_periods=1).mean()
    rs = gains / (losses + 1e-10)
    df[f'{prefix}_rsi'] = 100 - (100 / (1 + rs))
    
    # Lags
    for lag in [1, 3, 7]:
        df[f'{prefix}_ret_lag{lag}'] = df[f'{prefix}_daily_return'].shift(lag)
    
    return df


def add_correlation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ETH-BTC correlation features
    
    Args:
        df: DataFrame with eth_price, btc_price, and return columns
    
    Returns:
        DataFrame with correlation features
    """
    df['eth_btc_ratio'] = df['eth_price'] / df['btc_price']
    df['eth_btc_ratio_ma7'] = df['eth_btc_ratio'].rolling(7, min_periods=1).mean()
    df['eth_btc_corr_30d'] = df['eth_daily_return'].rolling(30, min_periods=20).corr(
        df['btc_daily_return']
    )
    df['eth_outperformance'] = df['eth_daily_return'] - df['btc_daily_return']
    
    return df


def fetch_and_prepare_data(save_file: str = 'whale_prices_ml_ready.csv') -> pd.DataFrame:
    """
    Complete pipeline: Fetch data from APIs and engineer features
    
    This is the main function to get fresh data and prepare it for predictions.
    
    Args:
        save_file: Path to save the prepared data
    
    Returns:
        DataFrame with all features ready for model
    """
    print("\n" + "="*70)
    print(" DATA REFRESH PIPELINE ".center(70))
    print("="*70)
    
    # 1. Fetch whale data
    df_whales = fetch_dune_data()
    
    # 2. Determine date range (add buffer for moving averages)
    min_date = pd.to_datetime(df_whales['block_date'].min()) - timedelta(days=100)
    max_date = pd.to_datetime(df_whales['block_date'].max())
    
    # 3. Fetch ETH prices
    df_eth = fetch_coingecko_price(
        'ethereum',
        min_date.strftime('%Y-%m-%d'),
        max_date.strftime('%Y-%m-%d')
    )
    df_eth = df_eth.rename(columns={'price': 'eth_price'})
    
    time.sleep(0.5)  # Rate limiting
    
    # 4. Fetch BTC prices
    df_btc = fetch_coingecko_price(
        'bitcoin',
        min_date.strftime('%Y-%m-%d'),
        max_date.strftime('%Y-%m-%d')
    )
    df_btc = df_btc.rename(columns={'price': 'btc_price'})
    
    # 5. Merge data
    print("\n🔗 Merging data...")
    df_merged = pd.merge(df_whales, df_eth, left_on='block_date', right_on='date', how='inner')
    df_merged = df_merged.drop('date', axis=1)
    
    df_merged = pd.merge(df_merged, df_btc, left_on='block_date', right_on='date', how='inner')
    df_merged = df_merged.drop('date', axis=1)
    
    print(f"   ✅ Merged: {len(df_merged)} rows")
    
    # 6. Engineer features
    print("\n⚙️  Engineering features...")
    df_merged = add_price_features(df_merged, 'eth_price', 'eth')
    df_merged = add_price_features(df_merged, 'btc_price', 'btc')
    df_merged = add_correlation_features(df_merged)
    
    print(f"   ✅ Total features: {len(df_merged.columns)}")
    
    # 7. Save to file
    if save_file:
        df_merged.to_csv(save_file, index=False)
        print(f"\n💾 Saved: {save_file}")
    
    return df_merged


if __name__ == "__main__":
    # Test the pipeline
    df = fetch_and_prepare_data()
    print(f"\n✅ Pipeline complete!")
    print(f"   Shape: {df.shape}")
    print(f"   Latest date: {df['block_date'].max()}")