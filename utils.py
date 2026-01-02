import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd 

# Functions to read in and process real SDOT data
def get_daily_occupancy_df(occupancy_df_path, target_date_str, parking_key_list, plot=True):
    """ 
    Function to filter the occupancy data to only the target date and selected parking lots/spaces, then
    compute occupancy across all selected parking lots/spaces. 

    Args:
        occupancy_df_path: path to the occupancy data CSV file
        target_date_str: the target date to filter the data to (format: '01/30/2025')
        parking_key_list: list of SDOT 'SourceElementKey' for the parking areas to include in the analysis
        plot: whether to plot the calculated daily occupancy data

    Returns:
        local_occupancy_df: DataFrame with 'OccupancyDateTime' and 'occupancy_pct' columns

    """

    occupancy_df = pd.read_csv(occupancy_df_path)

    # get the size in GB of the dataframe
    print(f"Original size: {occupancy_df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

    # Convert 'OccupancyDateTime' col to datetime
    occupancy_df['OccupancyDateTime'] = pd.to_datetime(occupancy_df['OccupancyDateTime'])
    # Get the date from the datetime column
    occupancy_df['date'] = occupancy_df['OccupancyDateTime'].dt.date
    # Convert target date to date object
    target_date = pd.to_datetime(target_date_str).date()  
    # Filter the dataframe to only the target date
    occupancy_df = occupancy_df[occupancy_df['date'] == target_date]
    # Drop the date column
    occupancy_df.drop(columns=['date'], inplace=True)

    # get the size in GB of the dataframe
    print(f"Size when filtered to only {target_date} date: {occupancy_df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

    # Select rows where 'SourceElementKey' is in key_list
    local_occupancy_df = occupancy_df[occupancy_df['SourceElementKey'].isin(parking_key_list)]

    # Groupby time and sum the PaidOccupancy and ParkingSpaceCount columns at each time
    local_occupancy_df = local_occupancy_df.groupby('OccupancyDateTime')[['PaidOccupancy', 'ParkingSpaceCount']].sum().reset_index()

    # Compute occupancy pct
    local_occupancy_df['occupancy_pct'] = local_occupancy_df['PaidOccupancy'] / local_occupancy_df['ParkingSpaceCount'] * 100

    num_parking_spaces = local_occupancy_df['ParkingSpaceCount'].unique()[0]
    print(f"Number of parking spaces in {parking_key_list} parking areas:\n\t{num_parking_spaces}")

    if plot:
        # Plot paid occupancy over time
        plt.plot(local_occupancy_df['OccupancyDateTime'], local_occupancy_df['occupancy_pct'])
        # Format x-axis to show only hours and minutes
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        plt.title(f"SDOT paid parking occupancy over time at {num_parking_spaces} spaces\n in {parking_key_list} lots on {target_date_str}")
        plt.ylabel("Occupancy (%)")
        plt.xticks(rotation=45)
        plt.show()
        plt.close()

    return local_occupancy_df


def get_daily_transactions_df(transactions_df_path, target_date_str, parking_key_list, plot=True):
    """
    Function to filter the transaction data to only the target date and selected parking lots/spaces.

    Args:
        transactions_df_path: path to the transaction data CSV file
        target_date_str: the target date to filter the data to (format: '01/30/2025')
        parking_key_list: list of SDOT 'Element key' for the parking areas to include in the analysis
        plot: whether to plot the transactions

    Returns:
        local_transactions_df: DataFrame with 'Transaction DateTime' column where rows represent transactions
    """

    # Read in transaction data 
    transactions_df = pd.read_csv(transactions_df_path)

    # get the size in GB of the dataframe
    print(f"Original size: {transactions_df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

    # Convert to datetime
    transactions_df['Transaction DateTime'] = pd.to_datetime(transactions_df['Transaction DateTime'])
    # Get the date from the datetime column
    transactions_df['date'] = transactions_df['Transaction DateTime'].dt.date
    # Convert target date to date object
    target_date = pd.to_datetime(target_date_str).date()  
    # Filter the dataframe for the target date
    transactions_df = transactions_df[transactions_df['date'] == target_date]
    # Drop the date column
    transactions_df.drop(columns=['date'], inplace=True)

    # get the size in GB of the dataframe
    print(f"Size when filtered to only {target_date} date: {transactions_df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

    # Select rows where 'Element key' is in key_list
    local_transactions_df = transactions_df[transactions_df['Element key'].isin(parking_key_list)]

    # Sort by Transaction DateTime
    local_transactions_df = local_transactions_df.sort_values(by='Transaction DateTime')

    if plot: 
        # Plot dots to show when transactions occurred 
        plt.scatter(local_transactions_df['Transaction DateTime'], np.zeros(len(local_transactions_df)), s=10)
        # Format x-axis to show only hours and minutes
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        plt.xticks(rotation=45)
        plt.yticks([])
        plt.title(f"SDOT paid parking transactions over time across\n {parking_key_list} lots on {target_date_str}")
        plt.show()
        plt.close()

    return local_transactions_df


def get_observed_probabilities_from_data(local_occupancy_df, local_transactions_df):
    """
    Given a list of true occupancy data and data with a series of transaction times, returns the 
    observed trajectory of occupancies, where an occupancy is assumed to be unchanged until a new one is observed,
    and transactions are assumed to indicate observations. 

    Args:
        local_occupancy_df: DataFrame with 'OccupancyDateTime' (datetime format) and occupancy rates in 'occupancy_pct' column.
        local_transactions_df: DataFrame with 'Transaction DateTime' (datetime format), where each row represents a transaction.

    Returns:
        A pandas DataFrame with 'OccupancyDateTime', 'occupancy_pct', and 'observed_occupancy_pct' columns.
    """

    # Sort both DataFrames by datetime:
    local_occupancy_df = local_occupancy_df.sort_values('OccupancyDateTime')
    local_transactions_df = local_transactions_df.sort_values('Transaction DateTime')

    # Initialize observed probabilities Series:
    observed_probabilities = pd.Series(index=local_occupancy_df['OccupancyDateTime'], dtype='float64')

    # Set initial probability
    # Note that we allow a 'freebie' observation by knowing the initial probability
    # (This is reasonable since initial occupancy is likely to be reliably low, since it's early in the morning)
    observed_probabilities.iloc[0] = local_occupancy_df.iloc[0]['occupancy_pct'] 
    last_observed_prob = observed_probabilities.iloc[0]

    transaction_index = 0
    for _, row in local_occupancy_df.iterrows():
        current_time = row['OccupancyDateTime']

        # If current_time is now the time of a transaction, update the observed occupancy and move through transactions until you get to a future time. Otherwise, keep the observed occupancy the same.
        while transaction_index < len(local_transactions_df) and local_transactions_df.iloc[transaction_index]['Transaction DateTime'] <= current_time:
            # Get the occupancy_pct value at the time of the transaction
            last_observed_prob = local_occupancy_df.loc[local_occupancy_df['OccupancyDateTime'] == local_transactions_df.iloc[transaction_index]['Transaction DateTime']]['occupancy_pct'].values[0]
            # increment the transaction index
            transaction_index += 1

        observed_probabilities.loc[current_time] = last_observed_prob

    # Create the result DataFrame:
    result_df = pd.DataFrame({
        'OccupancyDateTime': observed_probabilities.index,
        'occupancy_pct': local_occupancy_df['occupancy_pct'],
        'observed_occupancy_pct': observed_probabilities.values,
    })
    return result_df


def get_daily_connected_transactions_and_observed_occupancy(pct_connected_users, local_occupancy_df, local_transactions_df, random_seed, verbose=True):
    """
    Function to simulate connected user transactions and the resulting observed occupancy probabilities for a given date and parking areas.
    Assumes only connected users can observe true occupancy and that observations remain constant until updated with a new one.

    Args:
        pct_connected_users: the percentage of users who are connected and can observe true occupancy (e.g., 50=50% connected user rate)
        local_occupancy_df: DataFrame with 'OccupancyDateTime' (datetime format) and occupancy rates in 'occupancy_pct' column.
        local_transactions_df: DataFrame with 'Transaction DateTime' (datetime format), where each row represents a transaction.
        random_seed: random seed for reproducibility
        verbose: whether to print updates

    Returns:
        observed_occupancy_df: DataFrame with 'OccupancyDateTime', 'occupancy_pct', and 'observed_occupancy_pct' columns.
        connected_user_transactions_df: DataFrame with 'Transaction DateTime' and 'observed_occupancy_pct' columns, where each row represents a transaction by a connected user.
    """

    # Get number of rows in df
    num_rows_old = len(local_transactions_df)
    # Drop local_transactions_df rows prior to 8am
    local_transactions_df = local_transactions_df[local_transactions_df['Transaction DateTime'] >= '2025-01-30 08:00:00']
    if verbose:
        print(f"Dropped {num_rows_old - len(local_transactions_df)} row(s) from local_transactions_df prior to 8am")

    # Get transactions from randomly-selected connected users
    pct = pct_connected_users / 100
    n_rows_to_keep = int(len(local_transactions_df) * pct)
    connected_user_transactions_df = local_transactions_df.sample(n=n_rows_to_keep, random_state=random_seed)

    if verbose:
        print(f"{len(connected_user_transactions_df)} connected user transactions from {pct_connected_users}% of users, compared to {len(local_transactions_df)} transactions from all users")

    observed_occupancy_df = get_observed_probabilities_from_data(local_occupancy_df, connected_user_transactions_df)

    # Create new column in connected_user_transactions_df for 'observed_occupancy_pct' and merge in data from test_df based on the 'Transaction DateTime' column
    connected_user_transactions_df['observed_occupancy_pct'] = connected_user_transactions_df['Transaction DateTime'].map(observed_occupancy_df.set_index('OccupancyDateTime')['observed_occupancy_pct'])

    return observed_occupancy_df, connected_user_transactions_df


def check_time_validity(timestamp, parking_probs_dict):
    """
    Check if the given timestamp is within the time range of the parking probabilities in the provided dictionary.
    (Since we can't know probabilities for which we don't have data)
    
    Args:
        timestamp (pd.Timestamp): The timestamp to check for validity.
        parking_probs_dict (dict): A dictionary where keys are lot names and values are DataFrames with 'timestamp' column.

    Raises:
        ValueError: If the timestamp is outside the time range for any lot.
    """
    for lot_name, prob_df in parking_probs_dict.items():
        if timestamp < prob_df['timestamp'].min() or timestamp > prob_df['timestamp'].max():
            raise ValueError(f"Timestamp {timestamp} is outside the time range for {lot_name}: {prob_df['timestamp'].min()} - {prob_df['timestamp'].max()}")
