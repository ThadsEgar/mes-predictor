"""
Preprocessing script to convert Databento MES futures data to FinRL format
"""
import pandas as pd
import os
from datetime import datetime


def prepare_data(
    input_file='datasets/glbx-mdp3-20190414-20251003.ohlcv-1m copy.csv',
    output_file='datasets/mes_finrl_ready.csv',
    sample_rows=None  # Set to a number like 10000 for testing
):
    """
    Convert Databento CSV to FinRL format

    Args:
        input_file: Path to input CSV
        output_file: Path to output CSV
        sample_rows: If set, only process first N rows (for testing)
    """
    print(f"Starting data preparation...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    # Read the data
    if sample_rows:
        print(f"Reading first {sample_rows} rows for testing...")
        df = pd.read_csv(input_file, nrows=sample_rows)
    else:
        print("Reading full dataset (this may take a while)...")
        df = pd.read_csv(input_file)

    print(f"Loaded {len(df):,} rows")
    print(f"Original columns: {list(df.columns)}")

    # Rename columns to FinRL format
    df = df.rename(columns={
        'ts_event': 'timestamp',
        'symbol': 'tic'
    })

    # Keep only needed columns for FinRL
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'tic']]

    # Convert timestamp to datetime
    print("Converting timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Remove timezone info (FinRL works better with timezone-naive timestamps)
    if hasattr(df['timestamp'].dt, 'tz_localize'):
        try:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        except TypeError:
            # Already timezone-naive
            pass

    # Sort by timestamp
    print("Sorting by timestamp...")
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Print some stats
    print("\n=== Data Summary ===")
    print(f"Total rows: {len(df):,}")
    try:
        unique_tics = df['tic'].unique()
        print(f"Unique symbols: {unique_tics[:10]}{'...' if len(unique_tics) > 10 else ''}")
    except Exception:
        pass
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # Preview
    print("\n=== First 5 rows ===")
    print(df.head())

    print("\n=== Last 5 rows ===")
    print(df.tail())

    # Save the cleaned data
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)

    file_size = os.path.getsize(output_file) / (1024**2)  # Size in MB
    print(f"\u2713 Saved successfully! File size: {file_size:.2f} MB")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare MES data for FinRL')
    parser.add_argument('--input', '-i',
                       default='datasets/glbx-mdp3-20190414-20251003.ohlcv-1m copy.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', '-o',
                       default='datasets/mes_finrl_ready.csv',
                       help='Output CSV file path')
    parser.add_argument('--sample', '-s', type=int,
                       help='Process only first N rows (for testing)')

    args = parser.parse_args()

    # Run the preprocessing
    _ = prepare_data(
        input_file=args.input,
        output_file=args.output,
        sample_rows=args.sample
    )

    print("\n\u2713 Done! You can now use the cleaned data with FinRL.")
