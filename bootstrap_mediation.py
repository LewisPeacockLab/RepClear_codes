#code for mediation analysis

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from statsmodels.stats.mediation import Mediation
from sklearn.utils import resample

data_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
space = 'MNI'

# load the dataset
n_iterations = 10000
n_trials_per_operation = 30
n_subjects = 19

space='MNI'
# create an empty dataframe to store the bootstrap results
bootstrap_results = pd.DataFrame()

# loop through each iteration
for i in range(n_iterations):
    # read the csv file for the current iteration

    operations = ['maintain', 'replace', 'suppress']

    for op in operations:

        filename = os.path.join(data_dir,'iteration_data',f'bootstrap_{space}_study_{op}_iteration_{i}_dataframe.csv')
        data = pd.read_csv(filename)

        # split the data into exogenous and endogenous variables
        exog_vars = ['operation_evi', 'scene_evi']
        endog_var = 'memory'
        exog_data = data[exog_vars].copy()
        endog_data = data[endog_var]

        # specify the mediator variable and create a mediation object
        mediator_var = 'scene_evi'
        med_formula = f"{mediator_var} ~ {' + '.join(exog_vars)}"
        # med = Mediation(exog_data, endog_var, mediator_var, med_formula)

        # specify the models for the treatment, mediator, and outcome
        treatment_formula = f"{endog_var} ~ operation_evi"
        mediator_formula = f"{mediator_var} ~ operation_evi"
        outcome_formula = f"{endog_var} ~ operation_evi + {mediator_var}"

        # specify the mediator model
        med_model = sm.OLS.from_formula(mediator_formula, data=exog_data)
        # specify the exposure model
        exp_model = sm.OLS.from_formula(treatment_formula, data=data)
        # specify the outcome model
        out_model = sm.OLS.from_formula(outcome_formula, data=data)

        med = Mediation(out_model, med_model, 'operation_evi', mediator=mediator_var)


        med.treatment(treatment_formula)
        med.mediator(mediator_formula)
        med.outcome(outcome_formula)

        # run the mediation analysis
        med_results = med.fit()

        summary_table = med_results.summary()

        results_dict = {}
        results_dict['iteration'] = i
        results_dict['operation'] = op
        results_dict['ACME_control'] = summary_table.loc['ACME (control)', 'Estimate']
        results_dict['ACME_treated'] = summary_table.loc['ACME (treated)', 'Estimate']
        results_dict['ADE_control'] = summary_table.loc['ADE (control)', 'Estimate']
        results_dict['ADE_treated'] = summary_table.loc['ADE (treated)', 'Estimate']
        results_dict['Total_effect'] = summary_table.loc['Total effect', 'Estimate']
        results_dict['Prop_med_control'] = summary_table.loc['Prop. mediated (control)', 'Estimate']
        results_dict['Prop_med_treated'] = summary_table.loc['Prop. mediated (treated)', 'Estimate']
        results_dict['Prop_med_avg'] = summary_table.loc['Prop. mediated (average)', 'Estimate']

        # store the results in the bootstrap results dataframe

# group the bootstrap results by operation and calculate the average effect sizes
grouped_results = bootstrap_results.groupby('operation').mean()

# print the results
print(grouped_results)