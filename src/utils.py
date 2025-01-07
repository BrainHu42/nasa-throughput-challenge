import re
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict
from config import *
import os


def _load_features(airport: str, feature_name: str, split: str) -> pd.DataFrame:
    """Load feature data for a given airport and feature type from cache or CSV files."""
    
    feature_name = feature_name.lower()
    cache_file = f'{cache_directory}/{airport}_{feature_name}_{split}.parquet'

    if os.path.exists(cache_file):
        data_df = pd.read_parquet(cache_file, engine='pyarrow')
    else:
        file_list = find_csv_files(airport, feature_name + '_data_set', split=split)
        data_df = merge_csv_files(file_list, feature_name)
        
        # Optionally cache the full dataset for better read speeds
        # data_df.to_parquet(cache_file, engine='pyarrow')

    return data_df


def find_csv_files(airport=None, dataset=None, start_date=None, end_date=None, split='train'):
    """Find CSV files based on the given parameters, skipping files with a range of dates in name"""
    if split == 'train':
        root = str(data_directory)
    else:
        root = f'{data_directory}/test_data'
    
    matching_files = []

    # Convert dates to strings for comparison
    start_date_str = start_date.strftime("%Y-%m-%d") if start_date else None 
    end_date_str = end_date.strftime("%Y-%m-%d") if end_date else None

    base_path = f"{root}/{airport}"
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)

                # Skip files with date range format
                if re.search(r'\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}', file):
                    continue
                    
                file_parts = file.split('_')

                # Check if the file matches the airport criteria
                if airport and not file.startswith(airport):
                    continue
                
                # Extract date from filename
                if len(file_parts) >= 3:
                    file_date = file_parts[1]

                    # Check if the file is within the date range
                    if start_date_str and file_date < start_date_str:
                        continue
                    if end_date_str and file_date > end_date_str:
                        continue

                # Check if the file matches the dataset criteria
                if dataset and dataset.lower() not in file.lower():
                    continue

                matching_files.append(file_path)

    return matching_files


import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def read_csv_with_kwargs(file, kwargs):
    """Helper function to read csv file with kwargs robust timestamp parsing"""
    df = pd.read_csv(file, **kwargs)
    
    # Convert any parse_dates columns that aren't datetime
    for col in kwargs.get('parse_dates', []):
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f'Converting datetime for {col} in {file}')
            
            # Store original values before any conversion attempts
            original_values = df[col].copy()
            
            # First try automatic parsing with coerce to handle mixed formats
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Handle any unparsed values
            nat_mask = df[col].isna()
            if nat_mask.any():
                print(f"\nParsing {nat_mask.sum()} datetime values in {col}")
                
                # Try common datetime formats in order of specificity
                formats = [
                    '%Y-%m-%d %H:%M:%S',     # Standard datetime
                    '%Y-%m-%d',              # Just date
                ]
                
                for fmt in formats:
                    if df[col].isna().any():
                        try:
                            # Only attempt to parse still-null values
                            null_mask = df[col].isna()
                            parsed_values = pd.to_datetime(
                                original_values[null_mask],
                                format=fmt,
                                errors='coerce'
                            )
                            df.loc[null_mask, col] = parsed_values
                        except (ValueError, TypeError):
                            continue
    
    return df


def merge_csv_files(file_list, feature_name, n_workers=8):
    """Merge multiple CSV files into a single DataFrame using parallel processing."""

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_file = {
            executor.submit(read_csv_with_kwargs, file, read_csv_kwargs[feature_name]): file 
            for file in file_list
        }

        dataframes = []
        first_df = None
        for future in tqdm(as_completed(future_to_file), total=len(file_list), desc="Merging CSV files"):
            df = future.result()
            if df is not None:
                # Initialize first_df with correct dtypes
                if first_df is None:
                    first_df = pd.DataFrame(columns=df.columns)
                    date_columns = read_csv_kwargs[feature_name].get('parse_dates', [])
                    for col in date_columns:
                        if col in first_df.columns:
                            first_df[col] = pd.Series(dtype='datetime64[ns]')
                
                dataframes.append(df)

    if dataframes:
        # Concatenate with the properly initialized empty DataFrame
        dataframes.insert(0, first_df)
        merged_df = pd.concat(dataframes, copy=False)
        return merged_df
    else:
        print("No valid DataFrames to merge.")
        return None


def parse_submission_format():
    """Parse the competition submission format file and structure data for predictions."""
    df = pd.read_csv(submission_format_path)

    # Split the ID column
    df[['airport', 'date', 'time', 'minutes']] = df['ID'].str.split('_', expand=True)

    # Convert date and time to datetime
    df['interval_start'] = pd.to_datetime('20' + df['date'] + df['time'], format='%Y%m%d%H%M')
    df['interval_start'] += pd.to_timedelta(df['minutes'].astype(int) - 15, unit='m')

    # Rename 'Value' column to 'arrivals'
    df = df.rename(columns={'Value': 'arrivals'})

    # Select and order the desired columns
    result_df = df[['ID', 'airport', 'interval_start', 'arrivals']].sort_values(['airport', 'interval_start'])
    result_df = result_df.reset_index(drop=True)

    return result_df


def filter_labels_split(df, split='test'):
    """
    Checks if the provided DataFrame contains only data from the specified split
    (Training or Testing) based on a cyclical 24-day Training / 8-day Testing pattern.
    Removes rows not belonging to the specified split and prints a message if all rows
    are within the split.

    Args:
        df: pandas DataFrame with an 'interval_start' column (datetime64[ns] dtype).
        split: 'train' or 'test', indicating which split to check.

    Returns:
        df: The DataFrame with only the rows in the specified split
    """
    
    if split not in ['train', 'test']:
        raise ValueError("Invalid 'split' argument. Must be 'train' or 'test'.")

    start_date = pd.to_datetime('2022-09-01')

    df['days_since_start'] = (df['interval_start'] - start_date).dt.days
    if split == 'train':
        mask = (df['days_since_start'] % 32) < 24
    else:
        mask = (df['days_since_start'] % 32) >= 24

    df.drop(columns=['days_since_start'], inplace=True)
    filtered_df = df[mask].copy()

    return filtered_df


def verify_and_align_submission(predictions):
    """Verify and align prediction IDs into proper order and format."""
    
    # Read the CSV files
    submission_format = pd.read_csv(submission_format_path)

    # Check if the number of rows is the same
    if len(submission_format) != len(predictions):
        print("ERROR", "Number of rows doesn't match")
        return None

    # Check if all IDs in submission_format exist in predictions
    missing_in_predictions = set(submission_format['ID']) - set(predictions['ID'])
    if missing_in_predictions:
        print("ERROR", "Not all IDs from submission_format exist in predictions")
        print("Missing IDs:", list(missing_in_predictions)[:10])
        return None

    # Check if all IDs in predictions exist in submission_format
    extra_in_predictions = set(predictions['ID']) - set(submission_format['ID'])
    if extra_in_predictions:
        print("ERROR", "Predictions contains IDs not present in submission_format")
        print("Extra IDs:", list(extra_in_predictions)[:10])
        return None

    # If IDs don't match order, rearrange predictions
    if not (submission_format['ID'] == predictions['ID']).all():
        predictions = predictions.set_index('ID').loc[submission_format['ID']].reset_index()
        print("WARNING", "Predictions were rearranged to match submission_format order")
        return predictions

    # If we've made it this far, everything checks out and is in the correct order
    print("SUCCESS", "Verification successful, no rearrangement needed")
    return predictions
