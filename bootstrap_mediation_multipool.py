import os
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation
from sklearn.utils import resample
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

data_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
space = 'MNI'

# define constants
n_iterations = 10000
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
    med_formula = f"{mediator_var} ~ {exog_vars[0]}"
    med_model = sm.Logit.from_formula(med_formula, data=exog_data)

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

def run_mediation_wrapper(args):
    return run_mediation(*args)

with Pool(processes=cpu_count()) as pool:
    results_iter = pool.imap_unordered(run_mediation_wrapper, [(op, i) for i in range(n_iterations) for op in operations])
    for result in tqdm(results_iter, total=n_iterations * len(operations)):
        bootstrap_results = bootstrap_results.append(result, ignore_index=True)


bootstrap_results.to_csv(os.path.join(data_dir,'MNI_bootstrap_mediation_results.csv'))

# define the significance level
alpha = 0.05

# loop through each operation
for op in operations:
    print(f"\nOperation: {op}")
    print("------------------------------")
    
    # get the relevant data for this operation
    op_data = bootstrap_results.loc[bootstrap_results['operation'] == op, :]
    
    # loop through each mediation effect
    for effect in ['ACME', 'ADE', 'Prop_med']:
        effect_key = f"{effect}_treated"
        
        # calculate the mean and standard deviation of the effect
        effect_mean = op_data[effect_key].mean()
        effect_std = op_data[effect_key].std()
        
        # calculate the lower and upper bounds of the confidence interval
        lower_bound = effect_mean - 1.96 * effect_std
        upper_bound = effect_mean + 1.96 * effect_std
        
        # determine if the effect is significant
        if lower_bound <= 0 and upper_bound >= 0:
            print(f"{effect} is not significant for {op}")
        else:
            if effect == 'Prop_med':
                if lower_bound > 0:
                    print(f"{effect} is significant for {op} (positive mediation: {lower_bound:.2f} - {upper_bound:.2f})")
                elif upper_bound < 0:
                    print(f"{effect} is significant for {op} (negative mediation: {lower_bound:.2f} - {upper_bound:.2f})")
            else:
                if effect_mean > 0:
                    print(f"{effect} is significant for {op} (positive mediation: {lower_bound:.2f} - {upper_bound:.2f})")
                elif effect_mean < 0:
                    print(f"{effect} is significant for {op} (negative mediation: {lower_bound:.2f} - {upper_bound:.2f})")
