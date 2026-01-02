import pandas as pd
import numpy as np
from datetime import timedelta

from env import ParkingEnv
from agents import naive_patient_policy, naive_impatient_policy, probability_aware_policy
from utils import check_time_validity

##################################################################
# Vars that may change

destination_mapping = {
    'PikePlaceMarket': {
        'pikeplace': 'lot_1',
        'secondave': 'lot_2',
        'alaskanway': 'lot_3'
    },
    'BotanicalGarden': {
        'garden': 'lot_1',
        'marina': 'lot_2',
        'ballardyard': 'lot_3'
    }
}
agents_to_use = ['naive_patient', 'naive_impatient', 'probability_aware-one_step', 'probability_aware-one_step-oracle', 'probability_aware-two_step', 'probability_aware-two_step-oracle', 'probability_aware-three_step', 'probability_aware-three_step-oracle']
pct_connected_users_list = [10, 50]
data_seed_list = [1, 2, 3, 4, 5]
day_types = ['weekday', 'weekend']  

t_wait = 5  # Time in minutes to wait for another chance at the same lot

verbose_sim = False  # Set to True for detailed output
##################################################################


##################################################################
# Vars that probably won't change much

sim_data_dir = 'sim_data'
parking_probabilities_dir = "parking_probs"
output_path = f'sim_results/sim_results.csv'

increment_minutes = 60

# Max number of minutes parking search can take before we end the sim (models giving up the search at current lots)
# This is to prevent infinite loops in the naive patient policy or handle case when all lots are full. 
max_search_limit = 60

seed = 1 # Seed for reproducibility
np.random.seed(seed)
##################################################################


# Initialize for storing final results for all agents, pct_connected_users, and destinations
final_results_list = [] 

for destination in destination_mapping.keys():
    # --- Load in Pairwise Travel Times DataFrame ---
    # Values are travel times in minutes. t(i, j) can be different from t(j, i).
    # The diagonal is 0. Times from lots to 'destination' are walk times.
    assert destination in ['PikePlaceMarket', 'BotanicalGarden'], f"Unknown destination: {destination}"
    travel_times_fname = 'travel_times_PPM.csv' if destination == 'PikePlaceMarket' else 'travel_times_BG.csv'
    travel_times_df = pd.read_csv(f"{sim_data_dir}/{travel_times_fname}", index_col=0)

    for day_type in day_types:

        target_date_str = "01-30-2025" if day_type == 'weekday' else "02-01-2025"
        if destination == 'PikePlaceMarket':
            # For Pike Place
            start_time = f"{target_date_str} 08:00:00"
            end_time = f"{target_date_str} 18:00:00"
        elif destination == 'BotanicalGarden':
            # For Botanical Garden (since parking data doesn't go as late)
            start_time = f"{target_date_str} 08:00:00"
            end_time = f"{target_date_str} 16:00:00"
        else:
            raise ValueError(f"Unknown destination: {destination}")
        departure_times = pd.date_range(start_time, end_time, freq=f'{increment_minutes}min').strftime("%Y-%m-%d %H:%M:%S").tolist()
        

        for pct_connected_users in pct_connected_users_list:

            for agent in agents_to_use:

                # Initialize results storage
                total_time_results = {}

                if agent == 'naive_impatient':
                    # Initialize an empty set to track visited lots
                    visited_lots = set()

                if agent.startswith('probability_aware'):
                    # Extract the lookahead type from the agent name
                    lookahead_type = agent.split('-')[-2] if 'oracle' in agent else agent.split('-')[-1]
                    # Determine if the agent uses the oracle
                    use_oracle = 'oracle' in agent

                for data_seed in data_seed_list:

                    # --- Load in Dynamic Parking Probability DataFrames ---
                    true_and_observed_parking_probs_dict = {}
                    for lot_name in destination_mapping[destination].keys():
                        # Construct path to the true and observed parking probabilities for the current
                        # date, lot, pct_connected_users, and data_seed.
                        lot_probs_fname = f"{sim_data_dir}/{parking_probabilities_dir}/{target_date_str}/lot-{lot_name}_pctcon-{pct_connected_users}_seed-{data_seed}.csv"
                        lot_probs_df = pd.read_csv(lot_probs_fname)
                        
                        # Convert the 'timestamp' column to datetime
                        lot_probs_df['timestamp'] = pd.to_datetime(lot_probs_df['timestamp'])

                        # Store in the dict for the env
                        true_and_observed_parking_probs_dict[destination_mapping[destination][lot_name]] = lot_probs_df

                    # Initialize the environment for the given location, day, and true/observed parking probabilities.
                    env = ParkingEnv(
                        travel_times_df=travel_times_df,
                        parking_probs_dict=true_and_observed_parking_probs_dict,
                        t_wait=t_wait,
                    )

                    # Initialize results storage for this data seed -- we'll store a list with the results from each departure time
                    total_time_results[f"data_seed_{str(data_seed)}"] = []

                    # Assess multiple departure times
                    for departure_time in departure_times:

                        # Check if the departure time is valid with respect to the parking probability data
                        check_time_validity(pd.to_datetime(departure_time), true_and_observed_parking_probs_dict)

                        if verbose_sim:
                            print(f"Simulating for destination '{destination}', day '{day_type}', pct_connected_users={pct_connected_users}%, agent='{agent}', data_seed={data_seed}, departure_time={departure_time}...")

                        # 2. Run a single simulation (episode)
                        state, _ = env.reset(departure_time=departure_time)
                        terminated = False
                        total_reward_this_sim = 0

                        while not terminated:

                            # Get the name of the current location.
                            current_loc_name = env._loc_idx_to_name[state[0]]

                            # Get the action from the desired agent
                            if agent == 'naive_patient':
                                action = naive_patient_policy(
                                    travel_times=env.travel_times,
                                    lot_names=env.lot_names
                                )
                            elif agent == 'naive_impatient':
                                # Add the current location to the set if it's a lot.
                                # This ensures we track which lots the agent has been to.
                                if 'lot' in current_loc_name:
                                    visited_lots.add(current_loc_name)
                                    if verbose_sim:
                                        print(f"  (Visited lots so far: {visited_lots})")

                                # Check if all lots have been visited. If so, clear the set.
                                # This allows the policy to restart its search cycle.
                                if len(visited_lots) == env.num_lots:
                                    if verbose_sim:
                                        print("  -> All lots have been visited. Resetting impatient policy search cycle.")
                                    visited_lots.clear()

                                action = naive_impatient_policy(
                                    state=state,
                                    travel_times=env.travel_times,
                                    lot_names=env.lot_names,
                                    loc_idx_to_name=env._loc_idx_to_name,
                                    visited_lots=visited_lots
                                )

                            elif agent.startswith('probability_aware'):
                                action = probability_aware_policy(
                                    state=state,
                                    current_time=env.current_time,
                                    travel_times=env.travel_times,
                                    observed_parking_probs_dict=true_and_observed_parking_probs_dict,
                                    t_wait=env.t_wait,
                                    lot_names=env.lot_names,
                                    loc_idx_to_name=env._loc_idx_to_name,
                                    lookahead_type=lookahead_type,  # 'one_step' or 'two_step' or 'three_step' for progressively deeper lookahead
                                    use_oracle=use_oracle
                                )
                            else:
                                raise ValueError(f"Unknown agent type: {agent}")

                            
                            if verbose_sim:
                                print(f"\nCurrently at '{current_loc_name}', taking action: Go to Lot {action}")

                            # Take a step in the environment
                            next_state, reward, terminated, _, info = env.step(action)
                            
                            # Update total reward
                            total_reward_this_sim += reward

                            # Add a timeout condition to prevent infinite loops in naive patient policy 
                            # (otherwise will wait forever if best lot is full at day's end in data)
                            if env.current_time >= pd.to_datetime(departure_time) + timedelta(minutes=max_search_limit):
                                if verbose_sim:
                                    print(f"  -> Simulation timed out after reaching end time {env.current_time}. Ending simulation.")
                                total_time_results[f"data_seed_{str(data_seed)}"].append(-total_reward_this_sim)
                                break
                            
                            if verbose_sim:
                                status = "SUCCESSFULLY PARKED" if terminated else "FAILED to park"
                                print(f"  -> Outcome at {info['current_time']}: {status}")
                                print(f"  -> Time cost for this step: {-reward:.1f} minutes")
                                print(f"  -> New state: (Location {next_state[0]}, Status '{next_state[1]}')")
                                
                            # Update the current state
                            state = next_state

                        total_time_results[f"data_seed_{str(data_seed)}"].append(-total_reward_this_sim)
                        if verbose_sim:
                            # Final summary at the end of the current simulation
                            print("\n-------------------------------------------")
                            print("SIMULATION COMPLETE")
                            print(f"Final State: Parked at Lot {state[0]}")
                            print(f"Arrival Time at Destination: {env.current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"Total Time from Departure: {-total_reward_this_sim:.1f} minutes")
                            print("-------------------------------------------")

                ## Calculate and print the mean and standard deviation.
                # Flatten the results across all data seeds for overall statistics from this agent on this day/place/pct_connected_users
                all_values = np.array([val for sublist in total_time_results.values() for val in sublist])
                mean = all_values.mean()
                std_dev = all_values.std()

                # Print results for the agent
                print(f"Results for agent '{agent}':")
                # print(f"\tTotal simulations results: {[float(result) for result in total_time_results]}")
                print(f"\tMean time to park: {mean:.1f} minutes, std dev: {std_dev:.1f} minutes")

                # --- Store the result in a dictionary and append to the list ---
                result_data = {
                    'destination': destination,
                    'day': day_type,
                    'pct_connected_user': pct_connected_users,
                    'agent': agent,
                    'mean_time': mean,
                    'std_dev_time': std_dev,
                    'raw_results': total_time_results
                }
                final_results_list.append(result_data)

# Save results
final_results_df = pd.DataFrame(final_results_list)
final_results_df.to_csv(output_path, index=False, float_format='%.2f')

print(f"✅ Results successfully saved to {output_path}")