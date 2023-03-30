import os
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation
from sklearn.utils import resample
from multiprocessing import Pool

data_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
space = 'MNI'

# define constants
n_iterations = 100
n_trials_per_operation = 30
n_subjects = 19
operations = ['maintain', 'replace', 'suppress']

# create an empty dataframe to store the bootstrap results
bootstrap_results = pd.DataFrame()

def run_mediation(op, i):
    filename = os.path.join(data_dir, 'iteration_data', f'bootstrap_{space}_study_{op}_iteration_{i}_dataframe.csv')
    data = pd.read_csv(filename)

    # split the data into exogenous and endogenous variables
    exog_vars = ['operation_evi', 'scene_evi']
    endog_var = 'memory'
    exog_data = data[exog_vars].copy()
    endog_data = data[endog_var]

    # specify the mediator variable and create a mediation object
    mediator_var = 'scene_evi'
    med_formula = f"{mediator_var} ~ {' + '.join(exog_vars)}"
    med_model = sm.OLS.from_formula(med_formula, data=exog_data)

    # specify the models for the treatment, mediator, and outcome
    treatment_formula = f"{endog_var} ~ operation_evi"
    outcome_formula = f"{endog_var} ~ operation_evi + {mediator_var}"
    exp_model = sm.OLS.from_formula(treatment_formula, data=data)
    out_model = sm.OLS.from_formula(outcome_formula, data=data)

    med = Mediation(out_model, med_model, 'operation_evi', mediator=mediator_var)

    # run the mediation analysis
    med_results = med.fit()

    # summarize results
    summary_table = med_results.summary()
    results_dict = {
        'iteration': i,
        'operation': op,
        'ACME_control': summary_table.loc['ACME (control)', 'Estimate'],
        'ACME_treated': summary_table.loc['ACME (treated)', 'Estimate'],
        'ADE_control': summary_table.loc['ADE (control)', 'Estimate'],
        'ADE_treated': summary_table.loc['ADE (treated)', 'Estimate'],
        'Total_effect': summary_table.loc['Total effect', 'Estimate'],
        'Prop_med_control': summary_table.loc['Prop. mediated (control)', 'Estimate'],
        'Prop_med_treated': summary_table.loc['Prop. mediated (treated)', 'Estimate'],
        'Prop_med_avg': summary_table.loc['Prop. mediated (average)', 'Estimate']
    }
    results_df = pd.DataFrame(results_dict, index=[0])

    return results_df

# loop through each iteration
start_time = time.time()

with Pool(processes=32) as pool:
    for i in range(n_iterations):
        iteration_start_time = time.time()
        print(f"Starting iteration {i + 1}...")
        
        results_list = []
        for op in operations:
            results_list.append(pool.apply_async(run_mediation, (op, i)))

        for result in results_list:
            bootstrap_results = bootstrap_results.append(result.get())

        iteration_end_time = time.time()
        print(f"Iteration {i + 1} complete. Time taken: {iteration_end_time - iteration_start_time:.2f}")

# group the bootstrap results by operation and calculate the average effect sizes
grouped_results = bootstrap_results.groupby('operation').mean()

# print the results
print(grouped_results)
