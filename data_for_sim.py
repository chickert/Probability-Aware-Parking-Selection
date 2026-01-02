import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use('seaborn-v0_8-darkgrid')

from utils import (
    get_daily_occupancy_df,
    get_daily_transactions_df,
    get_daily_connected_transactions_and_observed_occupancy,
)

##################################################################
# Vars that may change
target_date_str = "01/30/2025"
pct_connected_users_list = [10]
random_seed_list = [5]

destination = "PikePlaceMarket" 
alaskan_way_parking_key_list = [31493, 31498, 54006, 54010, 76621]
pike_place_parking_key_list = [12869, 58629, 80198, 80878]
second_ave_parking_key_list = [11398, 13301, 13302, 46254, 70745, 70746]
lots_near_destination = {
    "Alaskan Way": alaskan_way_parking_key_list,
    "Pike Place": pike_place_parking_key_list,
    "Second Ave": second_ave_parking_key_list,
}

# destination = "BotanicalGarden" 
# garden_parking_key_list = [94732]
# marina_parking_key_list = [41725]
# ballard_yard_parking_key_list = [41726]
# lots_near_destination = {
#     "garden": garden_parking_key_list,
#     "marina": marina_parking_key_list,
#     "ballardyard": ballard_yard_parking_key_list,
# }

plot_probs = True
##################################################################

##################################################################
# Vars that probably won't change much
date_dir = target_date_str.replace("/", "-")
occupancy_source_data_path = "seattle_data/Paid_Parking_Occupancy__Last_30_Days__20250203.csv"
transactions_source_data_path = "seattle_data/Paid_Parking_Transaction_Data_20250203.csv"
sim_data_dir = "sim_data"  # Directory to which we save the generated data for the sim
parking_probabilities_dir = (
    "parking_probs"  # Directory to which we save the parking probabilities
)
color_list = [
    "steelblue",
    "lightseagreen",
    "lightcoral",
]
fig_size = (8, 4) 
##################################################################

# Make output dir if it doesn't exist
output_data_dir = f"{sim_data_dir}/{parking_probabilities_dir}/{date_dir}"
os.makedirs(output_data_dir, exist_ok=True)
# Make plots dir if it doesn't exist
os.makedirs(f"{output_data_dir}/plots", exist_ok=True)


for pct_connected_users in pct_connected_users_list:

    for random_seed in random_seed_list:

        if plot_probs:
            plt.figure(figsize=fig_size)

        for i, (lot_name, lot_key_list) in enumerate(lots_near_destination.items()):

            # Get the occupancy data for the target date and location
            local_occupancy_df = get_daily_occupancy_df(
                occupancy_source_data_path, target_date_str, lot_key_list, plot=False
            )
            # Get the transactions data for the target date and location
            local_transactions_df = get_daily_transactions_df(
                transactions_source_data_path, target_date_str, lot_key_list, plot=False
            )

            # Create a filename for the parking probabilities that includes the date, lot name, and random seed
            parking_probs_output_path = f"{sim_data_dir}/{parking_probabilities_dir}/{date_dir}/lot-{lot_name}_pctcon-{pct_connected_users}_seed-{random_seed}.csv"

            # Simulate *connected user* transactions as a fraction of all transactions
            # and the associated occupancy probabilities that would be observed by those users.
            observed_occupancy_df, connected_user_transactions_df = (
                get_daily_connected_transactions_and_observed_occupancy(
                    pct_connected_users,
                    local_occupancy_df,
                    local_transactions_df,
                    random_seed,
                    verbose=False,
                )
            )

            ## Format the dataframe approriately for the simulation
            # Convert occupancy to probability and observed occupancy to observed probability
            observed_occupancy_df["probability"] = 1 - (
                observed_occupancy_df["occupancy_pct"] / 100
            )
            observed_occupancy_df["observed_probability"] = 1 - (
                observed_occupancy_df["observed_occupancy_pct"] / 100
            )

            # Cap both probabilities at 0 and 1
            observed_occupancy_df["probability"] = observed_occupancy_df[
                "probability"
            ].clip(0, 1)
            observed_occupancy_df["observed_probability"] = observed_occupancy_df[
                "observed_probability"
            ].clip(0, 1)

            # Rename OccupancyDateTime to 'timestamp'
            observed_occupancy_df.rename(
                columns={"OccupancyDateTime": "timestamp"}, inplace=True
            )

            # Save the df
            observed_occupancy_df.to_csv(parking_probs_output_path, index=False)

            # Optionally plot the observed occupancy probabilities
            # (Currently only plotting the final seed and pct_connected_users)
            if plot_probs:
                # Plot the final seed and pct_connected_users for each lot
                plt.plot(observed_occupancy_df['timestamp'], observed_occupancy_df['probability'] * 100, color=color_list[i], label=f'{lot_name}')
                plt.plot(observed_occupancy_df['timestamp'], observed_occupancy_df['observed_probability'] * 100, color=color_list[i], linestyle='--', alpha=0.6)
                # Also plot when transactions occur as dots and put them at the height of the observed probability
                connected_user_transactions_df['observed_probability'] = (100 - connected_user_transactions_df['observed_occupancy_pct']).clip(0, 100)
                plt.scatter(connected_user_transactions_df['Transaction DateTime'], connected_user_transactions_df['observed_probability'], color=color_list[i], s=25)

        # Add formatting to the plot
        if plot_probs:
            plot_path = f"{sim_data_dir}/{parking_probabilities_dir}/{date_dir}/plots/dest-{destination}_pctcon-{pct_connected_users}_seed-{random_seed}.pdf"
            print(plot_path)

            def format_time(x, pos=None):
                time_str = mdates.num2date(x).strftime('%I %p').lower()
                return time_str.lstrip('0').replace(" ", "")  # Remove leading zeros
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_time))

            # Set to every-other tick
            ticks = plt.gca().get_xticks()
            plt.xticks(ticks[::2])

            plt.ylim(-3, 103)
            plt.yticks()

            plt.xlabel("Time")
            plt.ylabel("Parking probability (%)")

            destination = "Pike Place Market" if destination == "PikePlaceMarket" else "Botanical Garden"

            plt.title(f"Observed parking probabilities for {destination} at {len(lots_near_destination)} nearby parking lots\non {target_date_str} with {pct_connected_users}% adoption rate")

            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
