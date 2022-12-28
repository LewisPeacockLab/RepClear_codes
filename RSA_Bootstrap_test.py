# bootstrap stats for RSA analysis:
#Imports
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
import nibabel as nib
import scipy as scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import fnmatch
import pandas as pd
import pickle
import re
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import PredefinedSplit, cross_validate, cross_val_predict, GridSearchCV, LeaveOneGroupOut
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest, SelectFpr
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
from nilearn.image import clean_img
from nilearn.signal import clean
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from statannot import add_stat_annotation
from statsmodels.stats.anova import AnovaRM
from scipy.stats import bootstrap
from scipy.stats import ttest_rel


subs=['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','20','23','24','25','26']
brain_flag='MNI'

n_iterations=1000
n_trials=10

container_path='/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'
param_dir =  '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs'

group_item_remember_pre_post=pd.DataFrame()
group_item_forgot_pre_post=pd.DataFrame()

group_by_sub_item_remember_pre_post=pd.DataFrame()
group_by_sub_item_forgot_pre_post=pd.DataFrame()

# Python program to get average of a list
def Average(lst):
    return sum(lst) / len(lst)

for subID in subs:
    data_path=os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag)

    #pull the ones sorted by memory:
    item_r_pre_post=os.path.join(data_path,'itemweighted_pre_post_remember_fidelity.csv')
    group_item_remember_pre_post=group_item_remember_pre_post.append(pd.read_csv(item_r_pre_post,usecols=[1,2,3]),ignore_index=True) #add column 4 if we want pre-exposed

    item_f_pre_post=os.path.join(data_path,'itemweighted_pre_post_forgot_fidelity.csv')
    group_item_forgot_pre_post=group_item_forgot_pre_post.append(pd.read_csv(item_f_pre_post,usecols=[1,2,3]),ignore_index=True)

    if subID in ['06','07','14','23']: #these subjects have an operation without forgetting, thus this messes up my subject level assesment and I am leaving them out for now
        continue

    group_by_sub_item_remember_pre_post=group_by_sub_item_remember_pre_post.append(pd.read_csv(item_r_pre_post,usecols=[1,2,3]).mean(),ignore_index=True)

    group_by_sub_item_forgot_pre_post=group_by_sub_item_forgot_pre_post.append(pd.read_csv(item_f_pre_post,usecols=[1,2,3]).mean(),ignore_index=True) 


###Calculate bootstrap t-test###
bootstrap_remember=[]
bootstrap_forgot=[]

for i in range(n_iterations):
    iteration_supersub_r=[]
    iteration_supersub_f=[]
    for sub in range(len(subs)):
        xboot_sub_r=random.sample(list(group_item_remember_pre_post['suppress'].dropna()),n_trials)
        iteration_supersub_r=iteration_supersub_r+xboot_sub_r

        xboot_sub_f=random.sample(list(group_item_forgot_pre_post['suppress'].dropna()),n_trials)
        iteration_supersub_f=iteration_supersub_f+xboot_sub_f

    bootstrap_remember=bootstrap_remember+[Average(iteration_supersub_r)]
    bootstrap_forgot=bootstrap_forgot+[Average(iteration_supersub_f)]


#now test to see if the mean of remember is larger than the mean of forget:
if Average(bootstrap_remember)>Average(bootstrap_forgot):
    diff=np.array(bootstrap_remember)-np.array(bootstrap_forgot) #calcuate the difference of all iterations

    precent_boot=(len(np.where(diff>0)[0])/1000)*100 #calculate the % of iterations where the difference is greater than 0

print(precent_boot)
####

####Calculate two-way ANOVA###

melt_remember=group_by_sub_item_remember_pre_post.melt()
melt_forgot=group_by_sub_item_forgot_pre_post.melt()

melt_remember.rename(columns={'variable':'Operation','value':'Fidelity'},inplace=True)
melt_forgot.rename(columns={'variable':'Operation','value':'Fidelity'},inplace=True)

melt_remember['Memory']='Remember'
melt_forgot['Memory']='Forgot'

melt_remember['Subject']=np.tile((range(len(subs)-4)),3)+1 #repeats 1-18 3 times, for the sub list
melt_forgot['Subject']=np.tile((range(len(subs)-4)),3)+1

combined_df=pd.concat([melt_remember,melt_forgot])

print(AnovaRM(data=combined_df,depvar='Fidelity',subject='Subject', within=['Operation','Memory']).fit())
