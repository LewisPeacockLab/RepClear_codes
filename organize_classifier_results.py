#extract and organize classifier results
import warnings
import sys
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scipy
import seaborn as sns
cmap = sns.color_palette("crest", as_cmap=True)
import fnmatch
import pickle
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp


data_dir='/Users/zb3663/Desktop/School_Files/Repclear_files/manuscript/preremoval_classifier_results'

subIDs = ['002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','020','023','024','025','026']

group_aucs=[]
group_scores=[]

def calccohensd(array1, mean2):
    mean1=np.array(array1).mean()
    std=np.array(array1).std()

    cohens_d=(mean1-mean2)/std
    return cohens_d

for subID in subIDs:

    in_file=os.path.join(data_dir,f'sub-{subID}_task-preremoval_space-T1w_VVS_lrxval.npz') #get numpy file name

    temp_dict=np.load(in_file) #load in numpy dict

    group_aucs.append(temp_dict['auc_scores'].mean()) #take the mean of the x-valdiation AUCs and append to group level table for stats

    group_scores.append(temp_dict['scores'].mean())

#one-sample t-test against chance values: AUC=0.5, Scores=0.33

auc_t_stat, auc_p_value = ttest_1samp(group_aucs,0.5)
print('##### Reporting stats for AUC #####')
print(f'Mean = {np.array(group_aucs).mean()}')
print(f'SEM = {scipy.stats.sem(group_aucs)}')
print(f'Degrees of freedom = {(len(group_aucs)-1)}')
print(f'T-stat = {auc_t_stat}')
print(f'p-value = {auc_p_value}')
print(f'cohens d =  {calccohensd(group_aucs,0.5)}')

score_t_stat, score_p_value = ttest_1samp(group_scores,0.33)
print('##### Reporting stats for accuracy #####')
print(f'Mean = {np.array(group_scores).mean()}')
print(f'SEM = {scipy.stats.sem(group_scores)}')
print(f'Degrees of freedom = {(len(group_scores)-1)}')
print(f'T-stat = {score_t_stat}')
print(f'p-value = {score_p_value}')
print(f'cohens d =  {calccohensd(group_scores,0.33)}')

#######
# collect results for the operation classifier #

group_aucs=[]
group_scores=[]

data_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
for subID in subIDs:

    in_file=os.path.join(data_dir,f'sub-{subID}',f'sub-{subID}_T1w_study_operation_auc.csv') #get dataframe file name

    temp_dict=pd.read_csv(in_file) #load in dataframe

    group_aucs.append(temp_dict['AUC'].mean()) #take the mean of the x-valdiation AUCs and append to group level table for stats

    #group_scores.append(temp_dict['scores'].mean())


#one-sample t-test against chance values: AUC=0.5, Scores=0.33

auc_t_stat, auc_p_value = ttest_1samp(group_aucs,0.5)
print('##### Reporting stats for AUC #####')
print(f'Mean = {np.array(group_aucs).mean()}')
print(f'SEM = {scipy.stats.sem(group_aucs)}')
print(f'Degrees of freedom = {(len(group_aucs)-1)}')
print(f'T-stat = {auc_t_stat}')
print(f'p-value = {auc_p_value}')
print(f'cohens d =  {calccohensd(group_aucs,0.5)}')
