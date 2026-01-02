import pandas as pd
import numpy as np
import os

def calculate_improvement(current_mean, baseline_mean):
    """Calculates the percentage improvement of current_mean over baseline_mean."""
    if pd.isna(baseline_mean) or pd.isna(current_mean) or baseline_mean == 0:
        return np.nan
    # Formula: ((Old - New) / Old) * 100
    return ((baseline_mean - current_mean) / baseline_mean) * 100

def process_simulation_data(input_path):
    """Loads, processes, and pivots the simulation data."""
    df = pd.read_csv(input_path)
    
    # 1. Pre-computation and formatting
    df['mean_std_str'] = df.apply(
        lambda row: f"{row['mean_time']:.1f} ± {row['std_dev_time']:.1f}", axis=1
    )
    
    # 2. Calculate relative improvements within each experimental group
    def apply_relative_calcs(group):
        # Find baseline means for the group
        patient_mean = group.loc[group['agent'] == 'naive_patient', 'mean_time'].values[0]
        impatient_mean = group.loc[group['agent'] == 'naive_impatient', 'mean_time'].values[0]
        
        # Get the mean for the corresponding oracle agent, if it exists
        def get_oracle_mean(agent_name):
            if 'oracle' not in agent_name:
                oracle_name = f"{agent_name}-oracle"
                oracle_row = group.loc[group['agent'] == oracle_name, 'mean_time']
                return oracle_row.values[0] if not oracle_row.empty else np.nan
            return np.nan

        # Compute relative improvements
        group['rel_patient'] = group['mean_time'].apply(
            lambda x: calculate_improvement(x, patient_mean)
        )
        group['rel_impatient'] = group['mean_time'].apply(
            lambda x: calculate_improvement(x, impatient_mean)
        )
        group['rel_oracle'] = group.apply(
            lambda row: calculate_improvement(row['mean_time'], get_oracle_mean(row['agent'])), 
            axis=1
        )

        # Exclude self-comparisons and patient comparison to impatient (since weaker baseline)
        group.loc[group['agent'] == 'naive_patient', 'rel_patient'] = np.nan
        group.loc[group['agent'] == 'naive_patient', 'rel_impatient'] = np.nan
        group.loc[group['agent'] == 'naive_impatient', 'rel_impatient'] = np.nan

        return group

    df_processed = df.groupby(['destination', 'day', 'pct_connected_user']).apply(apply_relative_calcs).reset_index(drop=True)

    # 3. Pivot the table to the desired wide format
    pivot_df = df_processed.pivot_table(
        index=['destination', 'day', 'agent'],
        columns='pct_connected_user',
        values=['mean_std_str', 'rel_patient', 'rel_impatient', 'rel_oracle'],
        aggfunc='first' # Use first since values are already unique per group
    )

    # Swap and sort the column levels to group by percentage
    pivot_df.columns = pivot_df.columns.swaplevel(0, 1)
    
    # pivot_df = pivot_df.sort_index(axis=1, level=0)
    # 5. Define the final, specific column order and apply it
    pct_levels = sorted(df['pct_connected_user'].unique())
    value_order = ['mean_std_str', 'rel_patient', 'rel_impatient', 'rel_oracle']
    
    # Create the desired MultiIndex from the two ordered lists
    final_col_order = pd.MultiIndex.from_product(
        [pct_levels, value_order],
        names=['pct_connected_user', None]
    )

    # Filter for columns that actually exist in the pivot table
    final_col_order_existing = [col for col in final_col_order if col in pivot_df.columns]

    # Reindex the DataFrame to match this final, custom-sorted order
    pivot_df = pivot_df[final_col_order_existing]

    # Clean up and reorder for final output
    # Define the desired order of agents for sorting
    agent_order = [
        'naive_patient', 'naive_impatient', 
        'probability_aware-one_step', 'probability_aware-one_step-oracle',
        'probability_aware-two_step', 'probability_aware-two_step-oracle',
        'probability_aware-three_step', 'probability_aware-three_step-oracle',
    ]

    pivot_df = pivot_df.reset_index()
    pivot_df['agent'] = pd.Categorical(pivot_df['agent'], categories=agent_order, ordered=True)
    pivot_df = pivot_df.sort_values(by=['destination', 'day', 'agent']).set_index(['destination', 'day', 'agent'])
    
    return pivot_df

def generate_outputs(final_df, csv_path, latex_path):
    """Saves the final DataFrame to CSV and generates LaTeX code."""
    # --- Save to CSV ---
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_df = final_df.copy()
    csv_df.columns = [f'{val[0]}_{val[1]}%' for val in csv_df.columns]
    csv_df.to_csv(csv_path)
    print(f"✅ Formatted table saved to: {csv_path}")

    # --- Generate LaTeX Code ---
    latex_df = final_df.copy()

    # Drop oracle rows, since those are not needed in the LaTeX table
    latex_df = latex_df[~latex_df.index.get_level_values('agent').str.contains('oracle')]

    # 1. Clean up agent names in the index for display
    def format_agent_name(name):
        return name.replace('_', ' ').replace('-', ' ').replace('probability aware', 'PA').title()

    # CORRECTED PART: Use the rename method, which is designed for this task.
    latex_df = latex_df.rename(index=format_agent_name, level='agent')

    # 2. Clean up column level names for the LaTeX header
    latex_df.columns.names = ['\\% Connected Users', 'Metric']

    # 3. Define formatters for LaTeX columns based on the 'Metric' level of the MultiIndex
    formatters = {}
    for col_tuple in latex_df.columns:
        # col_tuple is like (10, 'mean_std_str')
        metric_name = col_tuple[1]
        if metric_name.startswith('rel'):
            # Format relative improvements with one decimal place and a % sign
            formatters[col_tuple] = lambda x: f"{x:.1f}\\%" if pd.notna(x) else "---"
        elif metric_name == 'mean_std_str':
            # Keep the string as is, or use '---' for missing data
            formatters[col_tuple] = lambda x: x if pd.notna(x) else "---"

    # 4. Generate the LaTeX string using the styler
    latex_string = latex_df.style.format(formatters, na_rep='---', escape="latex").to_latex(
        hrules=True,
        multicol_align="c",
        multirow_align="t",
        caption="Comparison of Probability-Aware Selection Policies with Patient and Impatient Baselines Across Temporal and Spatial Settings.",
        label="tab:agent_performance",
    )

    # 5. Final adjustments for better presentation
    latex_string = latex_string.replace("\\begin{table}", "\\begin{table*}")
    latex_string = latex_string.replace("\\end{table}", "\\end{table*}")
    latex_string = latex_string.replace("\\begin{tabular}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}")
    latex_string = latex_string.replace("\\end{tabular}", "\\end{tabular*}")
    latex_string = latex_string.replace('mean_std_str', 'Mean ± Std Dev')
    latex_string = latex_string.replace('rel_patient', '\\% vs Patient')
    latex_string = latex_string.replace('rel_impatient', '\\% vs Impatient')
    latex_string = latex_string.replace('rel_oracle', '\\% vs Oracle')
    latex_string = latex_string.replace("{llllrrrlrrrlrrr}", 'llllrrrlrrr}')
    latex_string = latex_string.replace("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}{llllrrrlrrr}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lllrrrrrrrr}")
    latex_string = latex_string.replace(" &  & \\% Connected Users & \\multicolumn{4}{c}{10} & \\multicolumn{4}{c}{50}", " &  & & \\multicolumn{4}{c}{10\\% Adoption Rate} & \\multicolumn{4}{c}{50\\% Adoption Rate}")
    latex_string = latex_string.replace(" &  & Metric & Mean ± Std Dev & \\% vs Patient & \\% vs Impatient & \\% vs Oracle & Mean ± Std Dev & \\% vs Patient & \\% vs Impatient & \\% vs Oracle", "\\cmidrule(lr){4-7} \\cmidrule(lr){8-11} Destination & Day & Policy & \\makecell{Mean Time\\\\$\\pm$ Std. (\\textdownarrow)} & \\makecell{Gain vs.\\\\BL-Pat. (\\textuparrow)}  & \\makecell{Gain vs.\\\\BL-Imp. (\\textuparrow)} & \\makecell{Perf. vs.\\\\Oracle (\\textuparrow)} & \\makecell{Mean Time\\\\$\\pm$ Std. (\\textdownarrow)} & \\makecell{Gain vs.\\\\BL-Pat. (\\textuparrow)}  & \\makecell{Gain vs.\\\\BL-Imp. (\\textuparrow)} & \\makecell{Perf. vs.\\\\Oracle (\\textuparrow)}")
    filtered_lines = [line for line in latex_string.splitlines() if "destination & day & agent" not in line]
    latex_string = "\n".join(filtered_lines)


    with open(latex_path, 'w') as f:
        f.write(latex_string)
    print(f"✅ LaTeX code saved to: {latex_path}")
    print("\n--- LaTeX Code for Overleaf ---")
    print(latex_string)


if __name__ == "__main__":
    # --- Configuration ---
    INPUT_CSV = 'sim_results/sim_results.csv'
    OUTPUT_DIR = 'sim_results'
    FORMATTED_CSV = os.path.join(OUTPUT_DIR, 'formatted_results.csv')
    LATEX_CODE_FILE = os.path.join(OUTPUT_DIR, 'results_table.tex')
    
    # --- Execution ---
    final_table = process_simulation_data(INPUT_CSV)
    generate_outputs(final_table, FORMATTED_CSV, LATEX_CODE_FILE)