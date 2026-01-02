import pandas as pd
import numpy as np
from datetime import timedelta

class ParkingEnv:
    """
    A Markov Decision Process (MDP) environment for a parking simulation.

    This class models the problem of finding a parking spot as an infinite-horizon
    stochastic shortest path problem. The agent's goal is to choose a sequence
    of parking lots to visit to minimize the total time (driving + waiting + walking).

    Attributes:
        travel_times (pd.DataFrame): A matrix of travel times between all locations.
        parking_probs (dict): A dictionary mapping lot names to DataFrames of their
                              time-dependent parking probabilities.
        t_wait (int): The time in minutes incurred when waiting at a lot.
        num_lots (int): The number of available parking lots.
        lot_names (list): A list of names for the parking lots (e.g., ['lot_1', ...]).
    """

    def __init__(self, travel_times_df: pd.DataFrame, parking_probs_dict: dict, t_wait: int, seed: int = None):
        """
        Initializes the parking environment.

        Args:
            travel_times_df (pd.DataFrame): A DataFrame with locations as both index
                and columns, containing pairwise travel times in minutes.
                Required locations: 'origin', 'destination', and 'lot_i'.
            parking_probs_dict (dict): A dictionary where keys are lot names (e.g., 'lot_1')
                and values are DataFrames. Each DataFrame must have a 'timestamp'
                (datetime) and 'probability' (float) column.
            t_wait (int): The time in minutes to wait for another chance at the same lot.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.travel_times = travel_times_df
        self.parking_probs = parking_probs_dict
        self.t_wait = t_wait
        self.seed = seed

        # Set the random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Automatically determine the number of lots from the travel times matrix
        self.lot_names = [col for col in travel_times_df.columns if 'lot' in col]
        self.num_lots = len(self.lot_names)

        # Mapping from location index to name for easier lookup
        self._loc_idx_to_name = {0: 'origin'}
        for i, name in enumerate(self.lot_names, 1):
            self._loc_idx_to_name[i] = name

        # State variables to be managed by the simulation
        self.current_state = None
        self.current_time = None

    def reset(self, departure_time: str) -> tuple:
        """
        Resets the environment to an initial state for a new simulation.

        Args:
            departure_time (str): The starting time of the simulation in a
                                  pandas-compatible format (e.g., "2025-01-25 08:00:00").

        Returns:
            tuple: A tuple containing the initial state `(location_index, status)`
                   and an empty info dictionary, conforming to the gymnasium API.
        """
        self.current_state = (0, 'u')  # Start at origin (0), unparked ('u')
        self.current_time = pd.to_datetime(departure_time)
        # print(f"Simulation started at: {self.current_time}, departing from Origin.")
        return self.current_state, {}

    def step(self, action: int) -> tuple:
        """
        Executes one time step in the environment.

        Args:
            action (int): The index of the lot to attempt to park in (from 1 to N).

        Returns:
            tuple: A tuple containing:
                - next_state (tuple): The state after the action.
                - reward (float): The negative time elapsed for this step.
                - terminated (bool): True if the agent has parked, False otherwise.
                - truncated (bool): Always False for this problem.
                - info (dict): A dictionary with auxiliary diagnostic information.
        """
        if self.current_state[1] == 'o':
            # Cannot take actions from a terminal (parked) state
            return self.current_state, 0, True, False, {}

        current_loc_idx, _ = self.current_state
        current_loc_name = self._loc_idx_to_name[current_loc_idx]
        target_lot_name = f'lot_{action}'

        # 1. Determine time cost and update simulation time
        if action == current_loc_idx:
            # Case: Waiting at the current lot
            time_cost = self.t_wait
        else:
            # Case: Driving to a new lot
            time_cost = self.travel_times.loc[current_loc_name, target_lot_name]
        
        self.current_time += timedelta(minutes=float(time_cost))

        # 2. Get the probability of successful parking at the target lot
        prob_df = self.parking_probs[target_lot_name]
        # Use merge_asof for efficient time-based lookup
        # It finds the last known probability at or before the current time
        prob_lookup = pd.merge_asof(
            pd.DataFrame({'timestamp': [self.current_time]}),
            prob_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        p_success = prob_lookup['probability'].iloc[0]

        # 3. Simulate parking attempt
        if np.random.rand() < p_success:
            # SUCCESS: Agent parked
            next_state = (action, 'o')
            terminated = True
            t_walk = self.travel_times.loc[target_lot_name, 'destination']
            reward = -time_cost - t_walk
            self.current_time += timedelta(minutes=float(t_walk)) # Add walk time to current time
        else:
            # FAILURE: Agent remains unparked
            next_state = (action, 'u')
            terminated = False
            reward = -time_cost

        # Update internal state
        self.current_state = next_state
        info = {'current_time': self.current_time.strftime('%Y-%m-%d %H:%M:%S')}

        return next_state, reward, terminated, False, info