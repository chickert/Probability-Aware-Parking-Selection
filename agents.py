import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Tuple, Set

from utils import check_time_validity

def naive_patient_policy(travel_times: pd.DataFrame, lot_names: list) -> int:
    """
    A policy that finds the lot closest to the destination and waits there.
    This policy does not need the current state, as it always
    chooses the lot with the shortest walk time to the destination.

    Args:
        travel_times (pd.DataFrame): The matrix of travel times.
        lot_names (list): A list of the names of the parking lots.

    Returns:
        int: The action (lot index) to take.
    """
    # Find the lot with the shortest walk time to the destination
    walk_times = travel_times.loc[lot_names, 'destination']
    closest_lot_name = walk_times.idxmin()
    # Extract the index from the name (e.g., 'lot_1' -> 1)
    best_lot_idx = int(closest_lot_name.split('_')[1])

    return best_lot_idx

def naive_impatient_policy(state: Tuple[int, str], travel_times: pd.DataFrame, lot_names: list, loc_idx_to_name: dict, visited_lots: Set[str]) -> int:
    """
    A policy that goes to the lot closest to the destination, but if parking fails, it drives
    to the unvisited lot closest from its current position.
    Once all lots have been visited, it resets the search cycle.

    Args:
        state (Tuple[int, str]): The current state (location_idx, status).
        travel_times (pd.DataFrame): The matrix of travel times.
        lot_names (list): A list of the names of the parking lots.
        loc_idx_to_name (dict): Mapping from location index to name.
        visited_lots (Set[str]): A set of lot names that have been visited.

    Returns:
        int: The action (lot index) to take.
    """
    current_loc_idx, _ = state

    if current_loc_idx == 0:
        # If at the origin, go to the lot with the shortest walk time.
        walk_times = travel_times.loc[lot_names, 'destination']
        closest_lot_name = walk_times.idxmin()
        return int(closest_lot_name.split('_')[1])
    else:
        # If at a lot, find the nearest other lot to drive to.
        current_loc_name = loc_idx_to_name[current_loc_idx]
        
        # Determine the set of lots that have not yet been visited.
        all_lots_set = set(lot_names)
        unvisited_lots = list(all_lots_set - visited_lots)

        # Get drive times from the current lot to all other lots
        drive_times_from_current = travel_times.loc[current_loc_name, lot_names]
        
        if unvisited_lots:
            # If there are unvisited lots, choose the closest one from that set.
            target_lot_pool = drive_times_from_current[unvisited_lots]
        else:
            # If all lots have been visited, reset by considering all lots (except the current one, since impatient) again.
            # The simulation loop should clear the visited_lots set after this.
            target_lot_pool = drive_times_from_current.drop(current_loc_name)

        # Find the name of the closest lot from the chosen pool
        next_closest_lot_name = target_lot_pool.idxmin()
        return int(next_closest_lot_name.split('_')[1])

def probability_aware_policy(
    state: Tuple[int, str],
    current_time: pd.Timestamp,
    travel_times: pd.DataFrame,
    observed_parking_probs_dict: Dict[str, pd.DataFrame],
    t_wait: int,
    lot_names: list,
    loc_idx_to_name: dict,
    lookahead_type: str,
    use_oracle: bool = False 
) -> int:
    """
    A policy that chooses the action (drive or wait) that minimizes the one-step, two-step, or three-step expected cost
    of attempting to park at a lot, given the current state and observed parking probabilities.

    The expected cost is a function of travel/wait time, walk time, and the
    observed probability of successfully parking at the lot(s) in question. See below for
    details on the lookahead types.

    We do not assume the agent can know future parking probabilities,
    so we only use the current time to look up parking probabilities.

    Args:
        state (Tuple[int, str]): The current state (location_idx, status).
        current_time (pd.Timestamp): The current simulation time.
        travel_times (pd.DataFrame): The matrix of travel times.
        observed_parking_probs_dict (Dict): The agent's (possibly noisy or outdated) belief about
                                      parking probabilities.
        t_wait (int): The time cost for waiting at a lot.
        lot_names (list): A list of the names of the parking lots.
        loc_idx_to_name (dict): Mapping from location index to name.
        lookahead_type (str): The type of lookahead to apply ('one_step', 'two_step', or 'three_step').
            "one_step" only considers the odds of parking at the target lot
            "two_step" considers the odds of parking at the target lot and the next-best lot
                        (which may be the same lot, if waiting is optimal)
            "three_step" considers the odds of parking at the target lot, the next-best lot, 
                        and the third-best lot
        use_oracle (bool): If True, uses the true parking probabilities instead of the observed ones. Default is False.

    Returns:
        int: The action (lot index) to take.
    """
    current_loc_idx, _ = state
    current_loc_name = loc_idx_to_name[current_loc_idx]

    min_expected_cost = float('inf')
    best_action = -1

    # Check if the current_time is within the time range of the parking probabilities
    # (Since we can't know probabilities for which we don't have data)
    check_time_validity(current_time, observed_parking_probs_dict)

    # Evaluate all possible actions (go to any lot, or wait at current lot)
    for target_lot_name in lot_names:
        action_idx = int(target_lot_name.split('_')[1])

        # 1. Determine time cost to take the action (drive or wait)
        if current_loc_idx == 0: # Must drive from origin
            time_to_act = travel_times.loc[current_loc_name, target_lot_name]
        elif action_idx == current_loc_idx: # Wait at the current lot
            time_to_act = t_wait
        else: # Drive to a different lot
            time_to_act = travel_times.loc[current_loc_name, target_lot_name]
        
        # 2. Look up the observed probability of parking at the target lot
        # NOTE: We use current time instead of projecting arrival time since we do not 
        #       assume the policy can know future parking probabilities.
        observed_prob_df = observed_parking_probs_dict[target_lot_name]
        observed_prob_lookup = pd.merge_asof(
            pd.DataFrame({'timestamp': [current_time]}),
            observed_prob_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        p_success = observed_prob_lookup['observed_probability'].iloc[0] if not use_oracle else observed_prob_lookup['probability'].iloc[0]
        p_success = max(p_success, 1e-9)  # Avoid division by zero

        # 3. Get walk time from the target lot
        t_walk = travel_times.loc[target_lot_name, 'destination']

        # 4. Calculate expected cost of the target lot (action) based on the desired depth of lookahead
        if lookahead_type == 'one_step':
            # One-step optimism: only consider the immediate action, risk-adjusted
            expected_cost = (float(time_to_act) / p_success) + float(t_walk)

        elif lookahead_type == 'two_step':
            # Two-step optimism: consider the second-choice lot.
            # Calculate the expected cost of failure to park at target_lot_name, 
            # using the risk-adjusted expected cost of the best next action.
            expected_cost_of_failure = 0
            if p_success < 1.0:
                # If we fail, we'll be at `target_lot_name`. What's the best move from there?
                min_next_action_cost = float('inf')
                
                # Consider all possible next actions from the `target_lot_name`
                for next_lot_name in lot_names:
                    # Option A: Wait at the target lot
                    if next_lot_name == target_lot_name:
                        time_to_act_2 = t_wait
                        t_walk_2 = t_walk
                    # Option B: Drive from target lot to a different lot
                    else:
                        # Cost = drive time + walk time from the new lot
                        time_to_act_2 = travel_times.loc[target_lot_name, next_lot_name]
                        t_walk_2 = travel_times.loc[next_lot_name, 'destination']

                    # Again, we do not assume we can know future probabilities, 
                    # so we use the current time to look up the parking probability.
                    observed_prob_df_2 = observed_parking_probs_dict[next_lot_name]
                    observed_prob_lookup_2 = pd.merge_asof(
                        pd.DataFrame({'timestamp': [current_time]}),
                        observed_prob_df_2.sort_values('timestamp'),
                        on='timestamp',
                        direction='backward'
                    )
                    p_success_2 = observed_prob_lookup_2['observed_probability'].iloc[0] if not use_oracle else observed_prob_lookup_2['probability'].iloc[0]
                    p_success_2 = max(p_success_2, 1e-9) # Avoid division by zero

                    cost = (float(time_to_act_2) / p_success_2) + float(t_walk_2)

                    # If the cost is better, update the minimum cost
                    if cost < min_next_action_cost:
                        min_next_action_cost = cost
                
                expected_cost_of_failure = min_next_action_cost

            # Calculate the total expected cost for the initial action
            expected_cost = (
                float(time_to_act) +
                (p_success * float(t_walk)) +
                ((1 - p_success) * float(expected_cost_of_failure))
            )

        elif lookahead_type == 'three_step':
            # This helper calculates the best risk-adjusted cost for a single move from a given lot.
            def get_best_next_move_cost(from_lot_name):
                min_cost = float('inf')
                for next_lot_name in lot_names:
                    if next_lot_name == from_lot_name: # Wait
                        time_to_act_next = t_wait
                        t_walk_next = travel_times.loc[from_lot_name, 'destination']
                    else: # Drive
                        time_to_act_next = travel_times.loc[from_lot_name, next_lot_name]
                        t_walk_next = travel_times.loc[next_lot_name, 'destination']

                    observed_prob_df_next = observed_parking_probs_dict[next_lot_name]
                    observed_prob_lookup_next = pd.merge_asof(pd.DataFrame({'timestamp': [current_time]}), observed_prob_df_next.sort_values('timestamp'), on='timestamp', direction='backward')
                    p_success_next = observed_prob_lookup_next['observed_probability'].iloc[0] if not use_oracle else observed_prob_lookup_next['probability'].iloc[0]
                    p_success_next = max(p_success_next, 1e-9) # Avoid division by zero

                    cost = (float(time_to_act_next) / p_success_next) + float(t_walk_next)
                    min_cost = min(min_cost, cost)
                return min_cost

            # Calculate expected cost from the first failure point (at target_lot_name)
            expected_cost_from_lot1 = 0
            if p_success < 1.0:
                min_cost_from_lot1 = float('inf')
                # Consider all possible second moves
                for lot_2_name in lot_names:
                    if lot_2_name == target_lot_name: # Wait
                        time_to_act2 = t_wait
                        t_walk_2 = t_walk
                    else: # Drive
                        time_to_act2 = travel_times.loc[target_lot_name, lot_2_name]
                        t_walk_2 = travel_times.loc[lot_2_name, 'destination']

                    observed_prob_df_2 = observed_parking_probs_dict[lot_2_name]
                    observed_prob_lookup_2 = pd.merge_asof(pd.DataFrame({'timestamp': [current_time]}), observed_prob_df_2.sort_values('timestamp'), on='timestamp', direction='backward')
                    p_success_2 = observed_prob_lookup_2['observed_probability'].iloc[0] if not use_oracle else observed_prob_lookup_2['probability'].iloc[0]
                    p_success_2 = max(p_success_2, 1e-9) # Avoid division by zero

                    # Cost of failure for the second action is the risk-adjusted cost of the best third action.
                    cost_of_best_third_move = 0
                    if p_success_2 < 1.0:
                        cost_of_best_third_move = get_best_next_move_cost(lot_2_name)

                    # TODO: Check this
                    # Total expected cost for this particular second action
                    cost_2 = float(time_to_act2) + (p_success_2 * float(t_walk_2)) + ((1 - p_success_2) * float(cost_of_best_third_move))
                    min_cost_from_lot1 = min(min_cost_from_lot1, cost_2)
                
                expected_cost_from_lot1 = min_cost_from_lot1

            # Final expected cost for the initial action
            expected_cost = float(time_to_act) + (p_success * float(t_walk)) + ((1 - p_success) * expected_cost_from_lot1)

        else:
            raise ValueError("Invalid lookahead_type. Choose 'one_step', 'two_step', or 'three_step'.")

        if expected_cost < min_expected_cost:
            min_expected_cost = expected_cost
            best_action = action_idx

    return best_action
