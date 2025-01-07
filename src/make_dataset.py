import numpy as np
import pandas as pd
import re

from utils import _load_features, filter_labels_split, parse_submission_format
from config import *


def merge_lamp_features(df, lamp, train_hour=0):
    """
    Process and merge LAMP weather forecast features with input dataframe.
    
    Maps categorical values, filters forecasts for specific hours, and calculates 
    3-hour maximum values for weather metrics like temperature, wind, visibility etc.
    """
    
    print("PROCESSING LAMP FEATURES")
    
    # Initial preprocessing
    lamp['timestamp'] = pd.to_datetime(lamp['timestamp'])
    lamp['forecast_timestamp'] = pd.to_datetime(lamp['forecast_timestamp'])
    lamp = lamp[lamp['timestamp'].dt.hour % 4 == train_hour].copy()

    # Map categorical values
    lamp['precip'] = lamp['precip'].map({True: 1, False: 0})
    lamp['lightning_prob'] = lamp['lightning_prob'].map({'N': 0, 'L': 1, 'M': 2, 'H': 3})
    lamp['cloud'] = lamp['cloud'].map({'CL': 0, 'FW': 1, 'SC': 2, 'BK': 3, 'OV': 4})
    
    forecast_cols = ['temperature', 'wind_speed', 'wind_gust', 'cloud_ceiling', 
                    'visibility', 'lightning_prob', 'precip', 'cloud']

    max_values = []
    
    for interval_start in df['interval_start'].unique():
        interval_end = interval_start + pd.Timedelta(hours=3)
        
        valid_forecasts = lamp[(lamp['timestamp'] <= interval_start) & (lamp['timestamp'] >= interval_start - pd.Timedelta(hours=8))]
        if valid_forecasts.empty:
            continue

        window_mask = ((valid_forecasts['forecast_timestamp'] >= interval_start - pd.Timedelta(hours=1)) & 
                      (valid_forecasts['forecast_timestamp'] <= interval_end))
        window_data = valid_forecasts[window_mask]
        if window_data.empty:
            continue
        
        # Calculate max values for this interval
        max_vals = window_data[forecast_cols].max()
        max_vals['interval_start'] = interval_start
        max_values.append(max_vals)
    
    if not max_values:
        return df
    max_values_df = pd.DataFrame(max_values)
    max_values_df = max_values_df.rename(
        columns={col: f'max_{col}_3h' for col in forecast_cols}
    )

    return df.merge(max_values_df, on='interval_start', how='left')


def merge_flight_count_features(full_df, mfs) -> pd.DataFrame:
    """
    Add flight count features to full_df based on 15-minute intervals from mfs data.
    """
    print("PROCESSING FLIGHT COUNT FEATURES")
    
    # Convert timestamps to datetime if needed
    timestamp_cols = ['arrival_runway_actual_time', 'arrival_stand_actual_time',
                     'departure_runway_actual_time', 'departure_stand_actual_time']
    
    for col in timestamp_cols:
        if not pd.api.types.is_datetime64_any_dtype(mfs[col]):
            mfs[col] = pd.to_datetime(mfs[col])

    def get_4hour_group(timestamp):
        return ((timestamp.hour - 0) // 4) * 4 + timestamp.dayofyear * 24 + timestamp.year * 24 * 365

    # Add 4hour group to mfs based on arrival_runway_actual_time
    mfs['non_nan_count'] = mfs.notna().sum(axis=1)
    mfs = mfs.sort_values('non_nan_count', ascending=False).drop_duplicates(subset='gufi', keep='first')
    mfs = mfs.drop('non_nan_count', axis=1).dropna(subset=['arrival_runway_actual_time'])
    
    mfs['isarrival'] = mfs['isarrival'].astype(bool)
    mfs['isdeparture'] = mfs['isdeparture'].astype(bool)
    mfs['4hour_group'] = mfs['arrival_runway_actual_time'].apply(get_4hour_group)
    
    # Filter for first hour only
    mfs = mfs[mfs['arrival_runway_actual_time'].dt.hour % 4 == 0]
    
    full_df['4hour_group'] = full_df['interval_start'].apply(get_4hour_group)

    def get_group_metrics(group_df):
        # 2. Calculate taxi times
        arrival_taxi_mask = (group_df['isarrival'] & group_df['arrival_runway_actual_time'].notna() & 
                           group_df['arrival_stand_actual_time'].notna())
        arrival_taxi_time = np.nan
        if arrival_taxi_mask.any():
            arrival_taxi = (group_df[arrival_taxi_mask]['arrival_stand_actual_time'] - 
                          group_df[arrival_taxi_mask]['arrival_runway_actual_time'])
            arrival_taxi_time = np.maximum(arrival_taxi.dt.total_seconds(), 0).mean()

        departure_taxi_mask = (group_df['isdeparture'] & group_df['departure_runway_actual_time'].notna() & 
                             group_df['departure_stand_actual_time'].notna())
        departure_taxi_time = np.nan
        if departure_taxi_mask.any():
            departure_taxi = (group_df[departure_taxi_mask]['departure_runway_actual_time'] - 
                            group_df[departure_taxi_mask]['departure_stand_actual_time'])
            departure_taxi_time = np.maximum(departure_taxi.dt.total_seconds(), 0).mean()

        # 3. Calculate flight times
        flight_time_mask = (group_df['departure_runway_actual_time'].notna() & 
                          group_df['arrival_runway_actual_time'].notna())
        flight_time = np.nan
        if flight_time_mask.any():
            flight_times = (group_df[flight_time_mask]['arrival_runway_actual_time'] - 
                          group_df[flight_time_mask]['departure_runway_actual_time'])
            flight_time = np.maximum(flight_times.dt.total_seconds(), 0).mean()

        return pd.Series([arrival_taxi_time, departure_taxi_time, flight_time],
                        index=['avg_arrival_taxi_time', 
                              'avg_departure_taxi_time', 'avg_flight_time'])

    # Calculate metrics for each 4hour group
    group_metrics = (mfs.groupby('4hour_group')
                    .apply(get_group_metrics)
                    .reset_index())

    full_df = full_df.merge(group_metrics, on='4hour_group', how='left').drop(columns=['4hour_group'])
    return full_df


def add_runway_counts(full_df, configs_df):
    # Count runways for each config row
    configs_df['num_arrival_runways'] = configs_df['arrival_runways'].str.count(',') + 1
    configs_df['num_departure_runways'] = configs_df['departure_runways'].str.count(',') + 1
    
    # For each interval, find the latest config before it
    result = pd.merge_asof(
        full_df.sort_values('interval_start'),
        configs_df[['start_time', 'num_arrival_runways', 'num_departure_runways']].sort_values('start_time'),
        left_on='interval_start',
        right_on='start_time',
        direction='backward'
    ).drop('start_time', axis=1)  # Drop the start_time column
    
    # Fill any missing values with the mode (most common value)
    result['num_arrival_runways'] = result['num_arrival_runways'].fillna(configs_df['num_arrival_runways'].mode()[0])
    result['num_departure_runways'] = result['num_departure_runways'].fillna(configs_df['num_departure_runways'].mode()[0])
    
    return result


def merge_arrival_features(df, tfm, tbfm, runways, etd, train_hour=0):
    print("EXTRACTING ARRIVAL FEATURES")
    
    df['interval_end'] = df['interval_start'] + pd.Timedelta(minutes=15)
    
    # Only convert if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(tfm['timestamp']):
        print("WARNING: Reparsing tfm timestamp")
        tfm['timestamp'] = pd.to_datetime(tfm['timestamp'])
    if not pd.api.types.is_datetime64_any_dtype(tbfm['timestamp']):
        print("WARNING: Reparsing tbfm timestamp")
        tbfm['timestamp'] = pd.to_datetime(tbfm['timestamp'])
    
    arrivals = runways[runways['arrival_runway_actual_time'].notnull()]
    arrivals.drop_duplicates(subset=['gufi'], keep='last', inplace=True)
    if not pd.api.types.is_datetime64_any_dtype(arrivals['arrival_runway_actual_time']):
        print("WARNING: Reparsing arrival timestamp")
        arrivals['arrival_runway_actual_time'] = pd.to_datetime(arrivals['arrival_runway_actual_time'])

    print("CONVERTED TIMESTAMPS TO DATETIME")
    
    def get_4hour_group(timestamp):
        return ((timestamp.hour - train_hour) // 4) * 4 + timestamp.dayofyear * 24 + timestamp.year * 24 * 365
    
    # Add 4-hour group column to both dataframes
    df['4hour_group'] = df['interval_start'].apply(get_4hour_group)
    
    arrivals = arrivals[arrivals['arrival_runway_actual_time'].dt.hour % 4 == train_hour]
    arrivals['interval_start'] = arrivals['arrival_runway_actual_time'].dt.floor('15min')
    arrivals['4hour_group'] = arrivals['interval_start'].apply(get_4hour_group)    

    # Group arrivals by 4hour_group
    arrivals_grouped = arrivals.groupby('4hour_group')
    
    def get_interval_metrics(x):
        # Create complete range of 15min intervals for this 4hour group
           
        start_time = x.min()
        end_time = start_time + pd.Timedelta(hours=1)
        full_intervals = pd.date_range(start=start_time, end=end_time, freq='15min')[:-1]
        
        # Get counts with 0s for missing intervals
        counts = x.value_counts()
        counts = counts.reindex(full_intervals, fill_value=0).sort_index()
        
        if counts.empty:
            print("Counts series is empty")
            return pd.Series({
                'arrivals_mean_abs_diff': 0,
                'arrivals_mean': 0,
                'arrivals_mode': 0
            }, name='metrics')
        
        return pd.Series({
            'arrivals_mean_abs_diff': abs(counts.diff()).mean(),
            'arrivals_mean': counts.mean(),
            'arrivals_mode': counts.mode()[0]
        }, name='metrics')
        
    # Calculate metrics in one operation
    arrivals_metrics = (arrivals_grouped['interval_start']
                       .apply(get_interval_metrics)
                       .unstack()
                       .reset_index())
    
    # Merge back to the original dataframe
    df = df.merge(arrivals_metrics, on='4hour_group', how='left')
    
    print("DONE extracting true arrival stats for first hour of each 4 hour group", 
          df.columns, flush=True)

    # Filter tfm to only include rows within the first hour of each 4-hour group
    tfm = tfm[tfm['timestamp'].dt.hour % 4 == train_hour].copy()
    tfm['4hour_group'] = tfm['timestamp'].apply(get_4hour_group)
    tfm['interval_start'] = tfm['arrival_runway_estimated_time'].dt.floor('15min')
    tfm.sort_values(['timestamp'], inplace=True)

    # Calculate estimate changes within 15min intervals
    estimate_changes = []
    for (group, interval_start), group_data in tfm.groupby(['4hour_group', 'interval_start']):
        # Calculate differences for all flights at once using groupby + diff
        diffs = abs(group_data.groupby('gufi')['arrival_runway_estimated_time']
                   .diff()
                   .dt.total_seconds())
        avg_change = diffs.mean() if len(diffs) > 0 else 0
        
        estimate_changes.append({
            '4hour_group': group,
            'interval_start': interval_start,
            'avg_estimate_change': avg_change
        })

    estimate_changes_df = pd.DataFrame(estimate_changes)
    df = df.merge(estimate_changes_df[['4hour_group', 'interval_start', 'avg_estimate_change']], 
                 on=['4hour_group', 'interval_start'], 
                 how='left')

    # Continue with existing logic for tfm_grouped calculations
    tfm_grouped = {}
    avg_diffs = []
    var_diffs = []
    for group, tfm_group in tfm.groupby('4hour_group'):
        tfm_group.drop_duplicates('gufi', keep='last', inplace=True)
        if group in arrivals_grouped.groups:
            arrival_data = arrivals_grouped.get_group(group).merge(tfm_group, on='gufi', how='left')
            arrival_data['time_diff'] = (arrival_data['arrival_runway_actual_time'] - arrival_data['arrival_runway_estimated_time']).dt.total_seconds()
            avg_diff = arrival_data['time_diff'].mean()
            var_diff = arrival_data['time_diff'].var()
        else:
            avg_diff = 0
            var_diff = np.nan
        
        avg_diffs.append({'4hour_group': group, 'avg_arrival_error': avg_diff})
        var_diffs.append({'4hour_group': group, 'var_arrival_error': var_diff})
        tfm_grouped[group] = tfm_group

    # Create dataframes from the lists of dictionaries
    avg_diffs_df = pd.DataFrame(avg_diffs)
    var_diffs_df = pd.DataFrame(var_diffs)

    # Merge avg_diff and var_diff into the main df
    df = df.merge(avg_diffs_df, on='4hour_group', how='left')
    df = df.merge(var_diffs_df, on='4hour_group', how='left')


    # Filter tbfm to only include rows within the first hour of each 4-hour group
    tbfm = tbfm[tbfm['timestamp'].dt.hour % 4 == train_hour].copy()
    # Rename sta column and convert to datetime
    tbfm = tbfm.rename(columns={'arrival_runway_sta': 'arrival_runway_estimated_time'})
    
    if not pd.api.types.is_datetime64_any_dtype(tbfm['arrival_runway_estimated_time']):
        print('WARNING: Reparsing tbfm arrival_runway_sta timestamp')
        tbfm['arrival_runway_estimated_time'] = pd.to_datetime(tbfm['arrival_runway_estimated_time'])
    
    tbfm['4hour_group'] = tbfm['timestamp'].apply(get_4hour_group)
    tbfm.sort_values('timestamp', inplace=True)
    
    # Filter tbfm to only include rows within the first hour of each 4-hour group
    tbfm = tbfm[tbfm['timestamp'].dt.hour % 4 == train_hour].copy()
    # Rename sta column and convert to datetime
    tbfm = tbfm.rename(columns={'arrival_runway_sta': 'arrival_runway_estimated_time'})
    
    if not pd.api.types.is_datetime64_any_dtype(tbfm['arrival_runway_estimated_time']):
        print('WARNING: Reparsing tbfm arrival_runway_sta timestamp')
        tbfm['arrival_runway_estimated_time'] = pd.to_datetime(tbfm['arrival_runway_estimated_time'])
    
    tbfm['4hour_group'] = tbfm['timestamp'].apply(get_4hour_group)
    tbfm['interval_start'] = tbfm['arrival_runway_estimated_time'].dt.floor('15min')
    tbfm.sort_values('timestamp', inplace=True)

    # Calculate estimate changes within 15min intervals for tbfm
    tbfm_estimate_changes = []
    for (group, interval_start), group_data in tbfm.groupby(['4hour_group', 'interval_start']):
        # Calculate differences for all flights at once using groupby + diff
        diffs = abs(group_data.groupby('gufi')['arrival_runway_estimated_time']
                   .diff()
                   .dt.total_seconds())
        avg_change = diffs.mean() if len(diffs) > 0 else 0
        
        tbfm_estimate_changes.append({
            '4hour_group': group,
            'interval_start': interval_start,
            'tbfm_avg_estimate_change': avg_change
        })

    tbfm_estimate_changes_df = pd.DataFrame(tbfm_estimate_changes)
    df = df.merge(tbfm_estimate_changes_df[['4hour_group', 'interval_start', 'tbfm_avg_estimate_change']], 
                 on=['4hour_group', 'interval_start'], 
                 how='left')
    
    # Calculate mean & variance of difference between estimated and actual arrival times for first hour.
    tbfm_grouped = {}
    tbfm_avg_diffs = []
    tbfm_var_diffs = []
    for group, tbfm_group in tbfm.groupby('4hour_group'):
        tbfm_group.drop_duplicates('gufi', keep='last', inplace=True)
        if group in arrivals_grouped.groups:
            arrival_data = arrivals_grouped.get_group(group).merge(tbfm_group, on='gufi', how='left')
            arrival_data['time_diff'] = (arrival_data['arrival_runway_actual_time'] - arrival_data['arrival_runway_estimated_time']).dt.total_seconds()
            avg_diff = arrival_data['time_diff'].mean()
            var_diff = arrival_data['time_diff'].var()
        else:
            avg_diff = 0
            var_diff = np.nan
        
        tbfm_avg_diffs.append({'4hour_group': group, 'tbfm_avg_arrival_error': avg_diff})
        tbfm_var_diffs.append({'4hour_group': group, 'tbfm_var_arrival_error': var_diff})
        tbfm_group['interval_start'] = tbfm_group['arrival_runway_estimated_time'].dt.floor('15min')
        tbfm_grouped[group] = tbfm_group

    # Create dataframes from the lists of dictionaries
    tbfm_avg_diffs_df = pd.DataFrame(tbfm_avg_diffs)
    tbfm_var_diffs_df = pd.DataFrame(tbfm_var_diffs)

    # Merge tbfm avg_diff and var_diff into the main df
    df = df.merge(tbfm_avg_diffs_df, on='4hour_group', how='left')
    df = df.merge(tbfm_var_diffs_df, on='4hour_group', how='left')
    
    print("DONE extracting tfm and tbfm arrival count error for first hour of 4 hour group", df.columns)
    
    # Calculate average and variance of time between consecutive arrivals
    avg_var_time_between_arrivals = []
    for group, arrival_group in arrivals_grouped:
        arrival_group = arrival_group.sort_values('arrival_runway_actual_time')
        time_diffs = arrival_group['arrival_runway_actual_time'].diff()
        time_diffs_seconds = time_diffs.dt.total_seconds()
        
        avg_time = time_diffs_seconds.mean()
        var_time = time_diffs_seconds.var()
        
        avg_var_time_between_arrivals.append({
            '4hour_group': group, 
            'avg_time_between_arrivals': avg_time,
            'var_time_between_arrivals': var_time
        })

    # Create dataframe from the list of dictionaries
    avg_var_time_between_arrivals_df = pd.DataFrame(avg_var_time_between_arrivals)

    # Merge avg_var_time_between_arrivals into the main df
    df = df.merge(avg_var_time_between_arrivals_df, on='4hour_group', how='left')
    
    # Calculate average and variance of time between estimated times of consecutive arrivals for TBFM
    avg_var_time_between_tbfm_estimated_arrivals = []
    for group, tbfm_group in tbfm_grouped.items():
        tbfm_group = tbfm_group.sort_values('arrival_runway_estimated_time')
        
        # Group by interval_start
        grouped = tbfm_group.groupby('interval_start')
        interval_groups = dict(list(grouped))
        
        # Get sorted list of all interval starts
        all_intervals = sorted(interval_groups.keys())
        
        for i, current_interval in enumerate(all_intervals):
            # Get current interval and next 3 intervals if they exist
            combined_groups = []
            combined_groups.append(interval_groups[current_interval])  # Always include current interval
            
            # Safely add next intervals if they exist
            for offset in range(1, 2):  # Next 2 intervals
                if i + offset < len(all_intervals):
                    next_interval = all_intervals[i + offset]
                    if next_interval < current_interval + pd.Timedelta(hours=1):  # Ensure within 1 hour
                        combined_groups.append(interval_groups[next_interval])
            
            # Only proceed if we have data to analyze
            if combined_groups:
                # Concatenate all relevant intervals
                combined_df = pd.concat(combined_groups)
                combined_df = combined_df.sort_values('arrival_runway_estimated_time')
                
                # Calculate time differences
                time_diffs = combined_df['arrival_runway_estimated_time'].diff()
                time_diffs_seconds = time_diffs.dt.total_seconds()
                
                # Only calculate statistics if we have valid time differences
                if len(time_diffs_seconds.dropna()) > 0:
                    avg_time = time_diffs_seconds.mean()
                    var_time = time_diffs_seconds.var()
                else:
                    avg_time = np.nan
                    var_time = np.nan
                
                avg_var_time_between_tbfm_estimated_arrivals.append({
                    '4hour_group': group,
                    'interval_start': current_interval,
                    'tbfm_avg_time_between_estimated_arrivals': avg_time,
                    'tbfm_var_time_between_estimated_arrivals': var_time
                })

    # Create dataframe from the list of dictionaries
    avg_var_time_between_tbfm_estimated_arrivals_df = pd.DataFrame(avg_var_time_between_tbfm_estimated_arrivals)
    df = df.merge(avg_var_time_between_tbfm_estimated_arrivals_df, on=['4hour_group', 'interval_start'], how='left')

    
    # Calculate average and variance of time between estimated times of consecutive arrivals
    avg_var_time_between_estimated_arrivals = []
    for group, tfm_group in tfm_grouped.items():
        tfm_group = tfm_group.sort_values('arrival_runway_estimated_time')
        
        # Group by interval_start
        grouped = tfm_group.groupby('interval_start')
        interval_groups = dict(list(grouped))
        
        # Get sorted list of all interval starts
        all_intervals = sorted(interval_groups.keys())
        
        for i, current_interval in enumerate(all_intervals):
            # Get current interval and next 3 intervals if they exist
            combined_groups = []
            combined_groups.append(interval_groups[current_interval])  # Always include current interval
            
            # Safely add next intervals if they exist
            for offset in range(1, 2):  # Next 2 intervals
                if i + offset < len(all_intervals):
                    next_interval = all_intervals[i + offset]
                    if next_interval < current_interval + pd.Timedelta(hours=1):  # Ensure within 1 hour
                        combined_groups.append(interval_groups[next_interval])
            
            # Only proceed if we have data to analyze
            if combined_groups:
                # Concatenate all relevant intervals
                combined_df = pd.concat(combined_groups)
                combined_df = combined_df.sort_values('arrival_runway_estimated_time')
                
                # Calculate time differences
                time_diffs = combined_df['arrival_runway_estimated_time'].diff()
                time_diffs_seconds = time_diffs.dt.total_seconds()
                
                # Only calculate statistics if we have valid time differences
                if len(time_diffs_seconds.dropna()) > 0:
                    avg_time = time_diffs_seconds.mean()
                    var_time = time_diffs_seconds.var()
                else:
                    avg_time = np.nan
                    var_time = np.nan
                
                avg_var_time_between_estimated_arrivals.append({
                    '4hour_group': group,
                    'interval_start': current_interval,
                    'avg_time_between_estimated_arrivals': avg_time,
                    'var_time_between_estimated_arrivals': var_time
                })

    # Create dataframe from the list of dictionaries
    avg_var_time_between_estimated_arrivals_df = pd.DataFrame(avg_var_time_between_estimated_arrivals)
    df = df.merge(avg_var_time_between_estimated_arrivals_df, on=['4hour_group', 'interval_start'], how='left')

    print("DONE calculating avg and var of difference between consecutive tfm and tbfm estimates", df.columns)
    
    departures = runways[runways['departure_runway_actual_time'].notnull()]
    departures.drop_duplicates(subset=['gufi'], keep='last', inplace=True)
    
    if not pd.api.types.is_datetime64_any_dtype(departures['departure_runway_actual_time']):
        print('WARNING: Reparsing departures timestamp')
        departures['departure_runway_actual_time'] = pd.to_datetime(departures['departure_runway_actual_time'])
    
    departures = departures[departures['departure_runway_actual_time'].dt.hour % 4 == train_hour]
    departures['interval_start'] = departures['departure_runway_actual_time'].dt.floor('15min')
    departures['4hour_group'] = departures['interval_start'].apply(get_4hour_group)    

    # Filter etd to only include rows within the first hour of each 4-hour group
    etd = etd[etd['timestamp'].dt.hour % 4 == train_hour].copy()
    etd['4hour_group'] = etd['timestamp'].apply(get_4hour_group)
    etd.sort_values('timestamp', inplace=True)

    # Create helper function to count departures for each interval
    def count_etd_departures(start, end, group):
        departures_df = pd.DataFrame()
        
        if (group - 4) in etd_grouped:
            departures_df = pd.concat([departures_df, etd_grouped[group - 4]])
        if group in etd_grouped:
            departures_df = pd.concat([departures_df, etd_grouped[group]])
        
        if departures_df.empty:
            return 0
        # Drop duplicates keeping last occurrence (most recent group)
        departures_df = departures_df.drop_duplicates(subset=['gufi'], keep='last')
        
        return ((departures_df['departure_runway_estimated_time'] >= start) & 
                (departures_df['departure_runway_estimated_time'] < end)).sum()


    # Group ETD data by 4hour_group for easier lookup
    etd_grouped = {}
    for group, etd_group in etd.groupby('4hour_group'):
        etd_group['interval_start'] = etd_group['departure_runway_estimated_time'].dt.floor('15min')
        etd_group = etd_group.sort_values('departure_runway_estimated_time')
        etd_group.drop_duplicates('gufi', keep='last', inplace=True)
        etd_grouped[group] = etd_group
        
    # Apply the helper function to each row
    df['etd_departures'] = df.apply(lambda row: count_etd_departures(
        row['interval_start'],
        row['interval_end'],
        row['4hour_group']
    ), axis=1)

    print("DONE extracting ETD features", df.columns)

    def normalize_runway(name):
        match = re.match(r'(\d{1,2})([LCR])?', name)
        if match:
            num, pos = match.groups()
            num = int(num)
            opposite = (num + 18) % 36 or 36
            return tuple(sorted([f"{num:02d}{pos or ''}", f"{opposite:02d}{pos or ''}"]))
        return (name,)  # Return as a single-element tuple for non-matching names

    # New section: Add columns for each unique runway
    unique_arrival_runways = arrivals['arrival_runway_actual'].unique()
    unique_departure_runways = departures['departure_runway_actual'].unique()
    
    # Group runways that represent the same physical runway
    arrival_runway_groups = {}
    departure_runway_groups = {}
    for runway in unique_arrival_runways:
        key = normalize_runway(runway)
        if key not in arrival_runway_groups:
            arrival_runway_groups[key] = []
        arrival_runway_groups[key].append(runway)
    for runway in unique_departure_runways:
        key = normalize_runway(runway)
        if key not in departure_runway_groups:
            departure_runway_groups[key] = []
        departure_runway_groups[key].append(runway)

    # Calculate throughput for each physical runway
    arrival_runway_throughputs = []
    departure_runway_throughputs = []
    for physical_runway, runways in arrival_runway_groups.items():
        throughput = (
            arrivals[arrivals['arrival_runway_actual'].isin(runways)]
            .groupby('4hour_group')
            .size()
            .reset_index(name='throughput')
        )
        arrival_runway_throughputs.append(throughput)
    for physical_runway, runways in departure_runway_groups.items():
        throughput = (
            departures[departures['departure_runway_actual'].isin(runways)]
            .groupby('4hour_group')
            .size()
            .reset_index(name='throughput')
        )
        departure_runway_throughputs.append(throughput)

    # Combine throughputs and calculate average
    if arrival_runway_throughputs:
        combined_arrival_throughput = pd.concat(arrival_runway_throughputs).groupby('4hour_group')['throughput'].mean().reset_index(name='avg_arrival_runway_throughput')
        df = df.merge(combined_arrival_throughput, on='4hour_group', how='left')
        df['avg_arrival_runway_throughput'].fillna(0, inplace=True)
    else:
        df['avg_arrival_runway_throughput'] = 0

    if departure_runway_throughputs:
        combined_departure_throughput = pd.concat(departure_runway_throughputs).groupby('4hour_group')['throughput'].mean().reset_index(name='avg_departure_runway_throughput')
        df = df.merge(combined_departure_throughput, on='4hour_group', how='left')
        df['avg_departure_runway_throughput'].fillna(0, inplace=True)
    else:
        df['avg_departure_runway_throughput'] = 0

    print("DONE extracting runway throughput features", df.columns)
    
    
    # Create helper functions to count arrivals for each interval
    def count_tfm_arrivals(start, end, group):
        arrivals_df = pd.DataFrame()

        if (group - 4) in tfm_grouped:
            arrivals_df = pd.concat([arrivals_df, tfm_grouped[group - 4]])
        if group in tfm_grouped:
            arrivals_df = pd.concat([arrivals_df, tfm_grouped[group]])

        if arrivals_df.empty:
            return 0, 0

        arrivals_df = arrivals_df.drop_duplicates(subset=['gufi'], keep='last')
        
        # Count current interval
        current = ((arrivals_df['arrival_runway_estimated_time'] >= start) & 
                  (arrivals_df['arrival_runway_estimated_time'] < end)).sum()

        # Count next interval
        next_end = end + pd.Timedelta(minutes=15)
        next_interval = ((arrivals_df['arrival_runway_estimated_time'] >= end) & 
                        (arrivals_df['arrival_runway_estimated_time'] < next_end)).sum()
        
        return current, next_interval - current

    def count_tbfm_arrivals(start, end, group):
        arrivals_df = pd.DataFrame()
        
        if (group - 4) in tbfm_grouped:
            arrivals_df = pd.concat([arrivals_df, tbfm_grouped[group - 4]])
        if group in tbfm_grouped:
            arrivals_df = pd.concat([arrivals_df, tbfm_grouped[group]])
            
        if arrivals_df.empty:
            return 0, 0
            
        arrivals_df = arrivals_df.drop_duplicates(subset=['gufi'], keep='last')
        
        # Count current interval
        current = ((arrivals_df['arrival_runway_estimated_time'] >= start) & 
                  (arrivals_df['arrival_runway_estimated_time'] < end)).sum()
                  
        # Count next interval
        next_end = end + pd.Timedelta(minutes=15)
        next_interval = ((arrivals_df['arrival_runway_estimated_time'] >= end) & 
                        (arrivals_df['arrival_runway_estimated_time'] < next_end)).sum()
        
        return current, next_interval - current

    # Apply the helper functions to each row
    df[['tfm_arrivals', 'tfm_arrivals_next']] = df.apply(lambda row: pd.Series(count_tfm_arrivals(
        row['interval_start'], 
        row['interval_end'], 
        row['4hour_group']
    )), axis=1)

    df[['tbfm_arrivals', 'tbfm_arrivals_next']] = df.apply(lambda row: pd.Series(count_tbfm_arrivals(
        row['interval_start'], 
        row['interval_end'], 
        row['4hour_group']
    )), axis=1)

    
    # Calculate the average difference between TBFM and TFM estimates for each 15-minute interval
    print("Calculating TBFM-TFM estimate differences")

    tbfm_tfm_diffs = []
    combined_arrivals = []
    for group in tfm_grouped:
        # Get TFM data for this group
        tfm_data = tfm_grouped[group]
        tfm_data['interval_start'] = tfm_data['arrival_runway_estimated_time'].dt.floor('15min')
        
        if group in tbfm_grouped:
            # Get TBFM data for this group
            tbfm_data = tbfm_grouped[group]
            
            # Merge TFM and TBFM data on GUFI with outer join to keep all flights
            merged = tfm_data.merge(
                tbfm_data[['gufi', 'arrival_runway_estimated_time']], 
                on='gufi',
                suffixes=('_tfm', '_tbfm'), 
                how='outer'
            )
            
            # Calculate time difference for flights with both estimates
            merged['estimate_diff'] = (
                merged['arrival_runway_estimated_time_tbfm'] - 
                merged['arrival_runway_estimated_time_tfm']
            ).dt.total_seconds()

            # Use TBFM time when available, otherwise TFM time
            merged['final_time'] = np.where(
                merged['arrival_runway_estimated_time_tbfm'].notna(),
                merged['arrival_runway_estimated_time_tbfm'],
                merged['arrival_runway_estimated_time_tfm']
            )

            # Group by 15-minute intervals
            interval_stats = merged.groupby('interval_start').agg({
                'estimate_diff': 'mean',
                'final_time': 'count'
            }).reset_index()
            
            interval_stats['4hour_group'] = group
            tbfm_tfm_diffs.append(interval_stats)
        else:
            # For TFM only data, count arrivals
            interval_stats = tfm_data.groupby('interval_start').agg({
                'arrival_runway_estimated_time': 'count'
            }).reset_index()
            interval_stats = interval_stats.rename(columns={
                'arrival_runway_estimated_time': 'final_time'
            })
            interval_stats['4hour_group'] = group
            interval_stats['estimate_diff'] = np.nan
            tbfm_tfm_diffs.append(interval_stats)

    if tbfm_tfm_diffs:
        # Combine all differences
        tbfm_tfm_diffs_df = pd.concat(tbfm_tfm_diffs)
        # Merge with main dataframe
        df = df.merge(
            tbfm_tfm_diffs_df.rename(columns={
                'estimate_diff': 'tbfm_tfm_time_diff',
                'final_time': 'combined_arrivals'
            }),
            on=['4hour_group', 'interval_start'],
            how='left'
        )
    else:
        df['tbfm_tfm_time_diff'] = np.nan
        df['combined_arrivals'] = np.nan

    print("DONE calculating TBFM-TFM differences")
    
    # Remove temporary columns
    df = df.drop(columns=['interval_end', '4hour_group'])
    
    return df


# Features are extracted such that future data is never used
def extract_features(data_dict, airport, split, train_hour, save=True):
    print('Loading training label file')
    if split == 'train':
        labels = pd.read_csv(data_directory / f"train_labels_{airport}.csv", parse_dates=["interval_start"])
    else:
        labels = parse_submission_format()
        labels = labels[labels['airport'] == airport]
        labels.drop(columns=['airport'], inplace=True)

    labels = filter_labels_split(labels, split)
    labels['interval_start'] = pd.to_datetime(labels['interval_start'])
    
    tfm_track = data_dict[airport]['tfm_track']
    runways = data_dict[airport]['runways']
    lamp = data_dict[airport]['lamp']
    tbfm = data_dict[airport]['tbfm']
    etd = data_dict[airport]['etd']
    mfs = data_dict[airport]['mfs']
    first_position = data_dict[airport]['first_position']
    configs = data_dict[airport]['configs']
    
    # Remove rows with missing estimated times
    tbfm = tbfm.dropna(subset=['arrival_runway_sta'])
    first_position.drop_duplicates(subset=['gufi'], inplace=True)
    
    full_df = merge_arrival_features(labels, tfm_track, tbfm, runways, etd)
    
    # Sort full_df by interval_start
    full_df = full_df.sort_values('interval_start')
    
    full_df['tfm_arrivals_diff'] = full_df['tfm_arrivals'].diff(periods=1)
    full_df['tfm_arrivals_diff_2'] = full_df['tfm_arrivals'].diff(periods=2)
    
    full_df['tbfm_arrivals_diff'] = full_df['tbfm_arrivals'].diff(periods=1)
    full_df['tbfm_arrivals_diff_2'] = full_df['tbfm_arrivals'].diff(periods=2)
    
    full_df['tfm_tbfm_diff'] = full_df['tfm_arrivals'] - full_df['tbfm_arrivals']
    
    full_df = merge_lamp_features(full_df, lamp, train_hour)
    
    full_df = merge_flight_count_features(full_df, mfs)
    
    
    print("PROCESSING FIRST POSITION DATA")
    arrivals = runways[runways['arrival_runway_actual_time'].notnull()]
    arrivals.drop_duplicates(subset=['gufi'], keep='last', inplace=True)
    arrivals = arrivals[arrivals['arrival_runway_actual_time'].dt.hour % 4 == train_hour]
    
    # Merge first_position with arrivals on gufi
    first_position_merged = first_position.merge(
        arrivals[['gufi', 'arrival_runway_actual_time']], 
        on='gufi', 
        how='left'
    )
    
    def calc_first_position_data(interval_start):
        interval_end = interval_start + pd.Timedelta(minutes=15)
        interval_data = first_position_merged[
            (first_position_merged['time_first_tracked'] >= interval_start) & 
            (first_position_merged['time_first_tracked'] < interval_end)
        ]
        
        if len(interval_data) == 0:
            return pd.Series({
                'first_position_count': 0,
                'avg_time_to_arrival': 0
            })
            
        count = len(interval_data)
        
        # Use boolean indexing for arrival calculations
        arrival_data = interval_data[interval_data['arrival_runway_actual_time'].notna()]
        if len(arrival_data):
            time_diff = (arrival_data['arrival_runway_actual_time'] - 
                        arrival_data['time_first_tracked'])
            # Ensure time differences are not negative
            time_diff_seconds = time_diff.dt.total_seconds()
            time_to_arrival = np.maximum(time_diff_seconds, 0).mean()
        else:
            time_to_arrival = 0
            
        return pd.Series({
            'first_position_count': count,
            'avg_time_to_arrival': time_to_arrival
        })

    metrics = full_df['interval_start'].apply(calc_first_position_data)
    full_df[['first_position_count', 'avg_time_to_arrival']] = metrics.values
    
    full_df = add_runway_counts(full_df, configs)
    
    print(full_df.columns, flush=True)

    # Save extracted features as zipped pickle file
    if save:
        print('Saving extracted features')
        full_df.to_pickle(train_directory / f"{airport}_{split}_raw_{VERSION}_h{train_hour}.pkl.zip")
    
    return full_df


def encode_features(features, airport, split, train_hour, save=False):
    features['min_cos'] = np.cos(features['interval_start'].dt.minute*(2.0*np.pi/60))
    features['min_sin'] = np.sin(features['interval_start'].dt.minute*(2.0*np.pi/60))
    
    features['hour_cos'] = np.cos(features['interval_start'].dt.hour*(2.0*np.pi/24))
    features['hour_sin'] = np.sin(features['interval_start'].dt.hour*(2.0*np.pi/24))
    features['dow_cos'] = np.cos(features['interval_start'].dt.day_of_week*(2.0*np.pi/7))
    features['dow_sin'] = np.sin(features['interval_start'].dt.day_of_week*(2.0*np.pi/7))
    features['day_cos'] = np.cos(features['interval_start'].dt.day*(2.0*np.pi/31))
    features['day_sin'] = np.sin(features['interval_start'].dt.day*(2.0*np.pi/31))
    features['month_cos'] = np.cos(features['interval_start'].dt.month*(2.0*np.pi/12))
    features['month_sin'] = np.sin(features['interval_start'].dt.month*(2.0*np.pi/12))
    
    features['first_hour'] = (features['interval_start'].dt.hour // 4) * 4
    features['intervals_since_first'] = ((features['interval_start'].dt.hour - features['first_hour']) * 4 + 
                                       features['interval_start'].dt.minute // 15)

    # Calculate diff from 4 hours ago (16 fifteen-minute intervals)
    features['arrivals_diff'] = features['arrivals_mean'] - features['arrivals_mean'].shift(16)

    # Save encoded features as zipped pickle file
    if save:
        features.to_pickle(train_directory / f'{airport}_{split}_features_{VERSION}_h{train_hour}.pkl.zip')

    return features


import multiprocessing as mp
from functools import partial

def make_dataset(airport, split, save=True):
    print(f'Extracting features for {airport}')
    train_directory.mkdir(parents=True, exist_ok=True)
    print('Loading feature files')
    data_dict = defaultdict(dict)
    
    for feature_name in feature_names:
        print(f"Loading dataset:", feature_name)
        data_dict[airport][feature_name] = _load_features(airport, feature_name, split)
    
    for train_hour in range(1 if split == 'train' else 1):
        features = extract_features(data_dict, airport, split, train_hour)
        data_df = encode_features(features, airport, split, train_hour, save=save)

    # Clean up memory
    del data_dict
    
    return data_df

if "__main__" == __name__:
    # Process airports in parallel
    pool = mp.Pool(processes=2)
    make_dataset_partial = partial(make_dataset, split='train')
    pool.map(make_dataset_partial, airports)
    
    pool.close()
    pool.join()