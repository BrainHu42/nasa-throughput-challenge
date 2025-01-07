# %%
import re
import pandas as pd
from datetime import datetime
from typing import Dict, List


def parse_taf_file(filename: str) -> pd.DataFrame:
    # Pre-compile regex patterns
    date_pattern = re.compile(r'(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2})')
    taf_pattern = re.compile(r'(?:AMD\s+)?([A-Z]{4})\s+(\d{6})Z')
    valid_period_pattern = re.compile(r'(\d{4}/\d{4})')
    wind_pattern = re.compile(r'(\d{2,3})(?:G(\d{2,3}))?KT')
    vis_pattern = re.compile(r'\s(\d{4})\s')
    
    severe_conditions = {
        'TS': 'Thunderstorm',
        'TSRA': 'Thunderstorm with rain',
        '+TSRA': 'Heavy thunderstorm with rain', 
        'SQ': 'Squall',
        'GR': 'Hail',
        'DS': 'Dust storm',
        'SS': 'Sandstorm',
        '+SHRA': 'Heavy rain showers',
        'G': 'Wind gusts'
    }
    
    wind_threshold = 25
    est_size = 2000
    data: Dict[str, List] = {
        'airport': [None] * est_size,
        'timestamp': [None] * est_size,
        'condition': [None] * est_size, 
        'raw_text': [None] * est_size,
        'valid_period': [None] * est_size,
    }
    
    idx = 0
    
    try:
        with open(filename, 'r', encoding="Windows-1252") as file:
            airport = ""
            taf_time = ""
            report_date = None
            
            for line in file:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse header date and time
                date_match = date_pattern.match(line)
                if date_match:
                    date_str, time_str = date_match.groups()
                    report_date = datetime.strptime(date_str, '%Y/%m/%d')
                    continue
                    
                # Handle TAF lines, including amendments
                if line.startswith('TAF') or 'AMD' in line:
                    match = taf_pattern.search(line)
                    if match:
                        airport = match.group(1)
                        taf_time = match.group(2)
                        continue
                
                if not (airport and report_date):
                    continue
                    
                valid_period = valid_period_pattern.search(line)
                valid_period = valid_period.group(1) if valid_period else None
                
                # Create timestamp using report date and TAF time
                try:
                    ts = pd.Timestamp(
                        year=report_date.year,
                        month=report_date.month,
                        day=int(taf_time[:2]),
                        hour=int(taf_time[2:4]),
                        minute=int(taf_time[4:])
                    )
                except (ValueError, TypeError):
                    continue
                
                # Check conditions
                for condition, description in severe_conditions.items():
                    if condition in line:
                        if idx >= est_size:
                            for key in data:
                                data[key].extend([None] * est_size)
                            est_size *= 2
                            
                        data['airport'][idx] = airport
                        data['timestamp'][idx] = ts
                        data['condition'][idx] = description
                        data['raw_text'][idx] = line
                        data['valid_period'][idx] = valid_period
                        idx += 1
                
                # Check wind conditions
                wind_match = wind_pattern.search(line)
                if wind_match:
                    wind_speed = int(wind_match.group(1))
                    gust_speed = int(wind_match.group(2)) if wind_match.group(2) else None
                    
                    if wind_speed >= wind_threshold or (gust_speed and gust_speed >= wind_threshold):
                        if idx >= est_size:
                            for key in data:
                                data[key].extend([None] * est_size)
                            est_size *= 2
                            
                        data['airport'][idx] = airport
                        data['timestamp'][idx] = ts
                        data['condition'][idx] = f'Strong winds {wind_speed}KT' + \
                            (f' gusting to {gust_speed}KT' if gust_speed else '')
                        data['raw_text'][idx] = line
                        data['valid_period'][idx] = valid_period
                        idx += 1
                        
                # Check visibility
                vis_match = vis_pattern.search(line)
                if vis_match and int(vis_match.group(1)) < 3000:
                    if idx >= est_size:
                        for key in data:
                            data[key].extend([None] * est_size)
                        est_size *= 2
                        
                    data['airport'][idx] = airport
                    data['timestamp'][idx] = ts
                    data['condition'][idx] = f'Low visibility {vis_match.group(1)}m'
                    data['raw_text'][idx] = line
                    data['valid_period'][idx] = valid_period
                    idx += 1
    
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return pd.DataFrame()

    # Trim lists and create DataFrame
    for key in data:
        data[key] = data[key][:idx]
        
    df = pd.DataFrame(data)
    return df.sort_values(['airport', 'timestamp']).reset_index(drop=True)


import os
import concurrent.futures
from tqdm import tqdm

def parse_and_filter(file_path: str, airports: list) -> pd.DataFrame:
    """
    Parse TAF file and filter for specific airports before returning
    """
    df = parse_taf_file(file_path)
    if not df.empty:
        # Filter for specified airports only
        df = df[df['airport'].isin(airports)]
    return df

def process_directory(directory_path: str, airports: list) -> pd.DataFrame:
    """
    Process all TAF text files in a directory in parallel and combine them into a single DataFrame.
    Only keeps data for specified airports to reduce memory usage.
    """
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        
        if not files:
            print(f"No TAF files found in {directory_path}")
            return pd.DataFrame()
            
        dataframes = []
        
        # Process files in parallel with progress bar and immediate filtering
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for file in files:
                file_path = os.path.join(directory_path, file)
                futures.append(
                    executor.submit(parse_and_filter, file_path, airports)
                )
                
            # Show progress bar while processing
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures),
                             desc="Processing TAF files"):
                df = future.result()
                if not df.empty:
                    dataframes.append(df)
                    
        if not dataframes:
            print(f"No valid data found for specified airports in {directory_path}")
            return pd.DataFrame()
            
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Sort and clean up combined dataframe
        return combined_df.sort_values(['airport', 'timestamp']).reset_index(drop=True)
        
    except Exception as e:
        print(f"Error processing directory {directory_path}: {str(e)}")
        return pd.DataFrame()

# List of airports to keep
airports = ['KATL', 'KCLT', 'KDEN', 'KDFW', 'KJFK', 'KMEM', 'KMIA', 'KORD', 'KPHX', 'KSEA']

# Process entire directory
directory = "/home/brianhu/workspace/TAF_train/"
combined_df = process_directory(directory, airports)
    
    
# %%
import pandas as pd
from datetime import datetime, timedelta
from config import *

split = 'test'

for airport in airports:
    test_df = pd.read_pickle(f'{train_directory}/{airport}_{split}_features_{VERSION}_h0.pkl.zip')
    severe_weather_df = pd.read_csv(f'{data_directory}/norm_{split}_severe_weather.csv')

    # Convert timestamp to datetime
    severe_weather_df['timestamp'] = pd.to_datetime(severe_weather_df['timestamp'])
    # Filter for TSRA conditions
    severe_weather_df = severe_weather_df[severe_weather_df['raw_text'].str.contains('TSRA', regex=False, na=False)]
    severe_weather_df.dropna(subset=['valid_period'], inplace=True)
    
    # Parse valid_period into start and end timestamps
    def parse_valid_period(row):
        if pd.isna(row['valid_period']):
            return pd.NA, pd.NA
        
        period = row['valid_period']
        base_date = row['timestamp']
        
        # Start with first day of the month
        month_start = base_date.replace(day=1, hour=0, minute=0, second=0)
        
        # Split period into start and end components
        start_str, end_str = period.split('/')
        
        # Extract days and hours
        start_day = int(start_str[:2])
        start_hour = int(start_str[2:])
        end_day = int(end_str[:2])
        end_hour = int(end_str[2:])
        
        # Handle special case where hour is 24
        if end_hour == 24:
            end_hour = 0
            end_day += 1
        
        # Calculate start and end times using timedelta
        start_time = month_start + pd.Timedelta(days=start_day-1, hours=start_hour)
        end_time = month_start + pd.Timedelta(days=end_day-1, hours=end_hour)
        
        # If end_time is before start_time, add one month to end_time
        if end_time < start_time:
            end_time = end_time + pd.DateOffset(months=1)
            
        return start_time, end_time

    # Add start and end times to severe_weather_df
    severe_weather_df[['period_start', 'period_end']] = severe_weather_df.apply(parse_valid_period, axis=1, result_type='expand')
    
    # Initialize weather_condition column as False
    test_df['weather_condition'] = False
    
    # Update weather_condition based on valid periods
    for _, weather_event in severe_weather_df.iterrows():
        mask = (test_df['interval_start'] >= weather_event['period_start']) & \
               (test_df['interval_start'] <= weather_event['period_end'])
        test_df.loc[mask, 'weather_condition'] = True

    test_df.to_pickle(f'{train_directory}/{airport}_{split}_features_{VERSION}_h0.pkl.zip')
# %%