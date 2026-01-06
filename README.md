# Probability-Aware-Parking-Selection
Code repo for the paper "Probability-Aware Parking Selection" in IEEE Transactions on Intelligent Transportation Systems by Cameron Hickert, Sirui Li, Zhengbing He, and Cathy Wu. Available at https://arxiv.org/abs/2601.00521. Citation to come.

### Project Files
```tree
.
├── figs
├── seattle_data
├── sim_data
│   ├── parking_probs
│   ├── travel_times_BG.csv
│   ├── travel_times_PPM.csv
│   └── travel_times_test.csv
├── sim_results
├── agents.py
├── data_for_sim.py
├── env.py
├── environment.yml
├── process_results.py
├── sim.py
├── stoch_obs.ipynb
└── utils.py
```

### File Descriptions

To run the stochastic observation experiments, simply run the `stoch_obs.ipynb` notebook. Note that for the empirical portion of the notebook, you will need to download SDOT data for your desired analysis and include it in the `seattle_data` dir. See more detail below. 

To run the Markov Decision Process (MDP) simulations, the basic flow is `data_for_sim.py` --> `sim.py` --> `process_results.py`. 

* Source code for paper
    * Stochastic observation experiments
        * `stoch_obs.ipynb`: Code for computing, analyzing, and displaying stochastic errors of parking probability observations for random walks and SDOT data.
            * Saves output plots to `figs/` directory. 
        * `utils.py`: Utility functions used across the project.
    * Comparative parking experiments in MDP simulations
        * `env.py`: Script defining the simulation environment.
        * `agents.py`: Python script defining agents (parking strategies) for the simulation.
        * `data_for_sim.py`: Script to extract SDOT parking & transaction data (uncomment the desired location or build your own), then construct true and observed parking probabilities at the desired parking lots (or combinations of parking spaces that comprise effective parking lots).
            * Saves outputs to `sim_data/parking_probs`
        * `sim.py`: Main simulation script.
            * Saves outputs to `sim_results/` directory.
        * `process_results.py`: Script to process the simulation results.
* Data
    * `seattle_data/Paid_Parking_Occupancy_Last_30_Days_20250203.csv`: Seattle (SDOT) paid parking occupancy data used to compute dynamic lot-specific parking probabilities. See below for how to download it. 
        * Note that simulation lots may be comprised of multiple SDOT pay station parking spots in a similar location.
    * `seattle_data/Paid_Parking_Transaction_Data_20250203.csv`: Seattle paid parking transaction data.
        * Used to sample connected user observations for purpose of determining observed parking probabilities. See below for how to download it.  
    * `sim_data/parking_probs`: Directory with day-specific true and observed parking probabilities for the simulation, extracted from the SDOT files via `data_for_sim.py`.
        * Also includes plots of true and observed probabilities at each lot for each Markov Decision Process (MDP). 
    * `sim_data/travel_times_*.csv`: Travel time data (driving and walking) extracted from Google Maps for the respective  setting (MDP denoted by * in the .csv file name).
* Outputs
    * `figs`: Stores plots from stochastic observation experiments.
    * `sim_results/sim_results.csv`: All MDP simulation results.
    * `sim_results/formatted_results.csv`: Formatted file w/ polished simulation results. 
    * `sim_results/results_table.tex`: Formatted results converted to a LaTeX-friendly format.

    
### Seattle Data
Data from Seattle Dept. of Transportation for the stochastic observation and MDP simulations can be found at: 
* Occupancy data: https://data.seattle.gov/Transportation/Paid-Parking-Occupancy-Last-30-Days-/rke9-rsvs/about_data
* Transaction data: https://data.seattle.gov/Transportation/Paid-Parking-Transaction-Data/gg89-k5p6/about_data

Note that data used in the paper itself were accessed from these sources on Feb. 3rd, 2025. For full reproducibility, refer to the dates in the paper. 
