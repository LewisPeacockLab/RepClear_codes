#code to get operation variation across all trials, along with updated bootstrapping 
import warnings
import sys
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
from nilearn.image import clean_img
from nilearn.signal import clean
import scipy as scipy
import scipy.io as sio
import seaborn as sns
cmap = sns.color_palette("crest", as_cmap=True)
import fnmatch
import pickle
import time
from random import choices #random sampling with replacement
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, PredefinedSplit  #train_test_split, PredefinedSplit, cross_validate, cross_val_predict, 
from sklearn.feature_selection import SelectFpr, f_classif  #VarianceThreshold, SelectKBest, 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from joblib import Parallel, delayed
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_1samp, f_oneway, ttest_ind, ttest_rel


# global consts
subIDs = ['002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','020','023','024','025','026']
temp_subIDs=['002','003','004','005','008','009','010','011','012','013','014','015','016','017','018','020','024','025','026']


phases = ['rest', 'preremoval', 'study', 'postremoval']
runs = np.arange(6) + 1  
spaces = {'T1w': 'T1w', 
            'MNI': 'MNI152NLin2009cAsym'}
descs = ['brain_mask', 'preproc_bold']
ROIs = ['wholebrain']
shift_sizes_TR = [5]

save=1
shift_size_TR = shift_sizes_TR[0]


stim_labels = {0: "Rest",
                1: "Scenes",
                2: "Faces"}

operation_labels = {0: "Rest",
                    1: "Maintain",
                    2: "Replace",
                    3: "Suppress"}

workspace = 'scratch'
data_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
param_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/params/'

def organize_evidence(subID,space,task,save=True):
    ROIs = ['wholebrain']

    print("\n *** loading evidence values from subject dataframe ***")

    sub_dir = os.path.join(data_dir, f"sub-{subID}")

    in_fname_template = f"sub-{subID}_{space}_{task}_operation_evidence.csv"   

    sub_df=pd.read_csv(os.path.join(sub_dir,in_fname_template))  
    sub_df.drop(columns=sub_df.columns[0], axis=1, inplace=True) #now drop the extra index column

    sub_images,sub_index=np.unique(sub_df['image_id'], return_index=True) #this searches through the dataframe to find each occurance of the image_id. This allows me to find the start of each trial, and linked to the image_ID #
    #category_df contains the 0's for the shift, while operation evidence already has removed that  

    #now to sort the trials, we need to figure out what the operation performed is:
    sub_condition_list=sub_df['operation'][sub_index].values.astype(int) #so using the above indices, we will now grab what the operation is on each image

    counter=0

    maintain_trials_mean={}
    replace_trials_mean={}
    suppress_trials_mean={}          

    if task=='study': x=9
    if task=='postremoval': x=5

    for i in sub_condition_list:
        if i==0: #this is the first rest period (because we had to shift of hemodynamics. So this "0" condition is nothing)
            print('i==0')
            counter+=1                         
            continue
        elif i==1:
            temp_image=sub_images[counter]
            maintain_trials_mean[temp_image]=sub_df[['maintain_evi']][sub_index[counter]:sub_index[counter]+4].values.mean()
          
            counter+=1

        elif i==2:
            temp_image=sub_images[counter]     
            replace_trials_mean[temp_image]=sub_df[['replace_evi']][sub_index[counter]:sub_index[counter]+4].values.mean()
            
            counter+=1

        elif i==3:
            temp_image=sub_images[counter]       
            suppress_trials_mean[temp_image]=sub_df[['suppress_evi']][sub_index[counter]:sub_index[counter]+4].values.mean()

            counter+=1

    #need to grab the total trials of the conditions, will need to then pull memory evidence of these trials and append to the dataframe:
    all_maintain=pd.DataFrame(data=maintain_trials_mean.values(),index=maintain_trials_mean.keys(),columns=['evidence'])
    all_maintain['operation']='Maintain'

    all_replace=pd.DataFrame(data=replace_trials_mean.values(),index=replace_trials_mean.keys(),columns=['evidence'])
    all_replace['operation']='Replace'

    all_suppress=pd.DataFrame(data=suppress_trials_mean.values(),index=suppress_trials_mean.keys(),columns=['evidence'])
    all_suppress['operation']='Suppress'

    all_operations=pd.concat([all_maintain,all_replace,all_suppress]) #combine all these into 1 dataframe to get the subject level z-score            

    #z-score within subject
    all_operations['evidence']=stats.zscore(all_operations['evidence'])
    all_operations.reset_index(inplace=True)

    #calculate the subject level variance for the operations
    operation_var=pd.DataFrame(columns=['Maintain','Replace','Suppress'],index=[0])
    operation_var['Maintain']=np.var(all_operations[all_operations['operation']=='Maintain']['evidence'])
    operation_var['Replace']=np.var(all_operations[all_operations['operation']=='Replace']['evidence'])
    operation_var['Suppress']=np.var(all_operations[all_operations['operation']=='Suppress']['evidence'])

    # save for future use
    if save: 
        sub_dir = os.path.join(data_dir, f"sub-{subID}")
        out_fname_template = f"sub-{subID}_{space}_{task}_operation_zscore_evidence.csv"  
        print(f"\n Saving the sorted z-scored operation evidence dataframe for {subID} - phase: {task} - as {out_fname_template}")
        all_operations.to_csv(os.path.join(sub_dir,out_fname_template))   
    return all_operations, operation_var

#now combine all the subject operation dfs into 1 to plot:
combined_df=pd.DataFrame()
combined_var=pd.DataFrame()

for subID in subIDs:
    temp_df, temp_subject_var=organize_evidence(subID,'T1w','study',save=True)
    combined_df=pd.concat([combined_df,temp_df])
    combined_var=pd.concat([combined_var,temp_subject_var])

del temp_df

combined_var.reset_index(inplace=True,drop=True)

plt.style.use('seaborn-paper')

ax=sns.violinplot(data=combined_df,x='operation',y='evidence',palette=['green','blue','red'])
ax.set(xlabel='Operation',ylabel='Z-Scored Evidence')
ax.set_title('Operation engagement variability across trials', loc='center', wrap=True)
ax.axhline(0,color='k',linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(data_dir,'figs','All_operation_trials_zscore.svg'))
plt.savefig(os.path.join(data_dir,'figs','All_operation_trials_zscore.png'))
plt.clf()

plt.style.use('seaborn-paper')

var_plotting_df=combined_var.melt()
var_plotting_df['sub']=np.tile(subIDs,3)

ax=sns.violinplot(data=var_plotting_df,x='variable',y='value',palette=['green','blue','red'],inner='quartile')
sns.swarmplot(data=var_plotting_df,x='variable',y='value',color= "white")
# sns.pointplot(data=var_plotting_df,x='variable',y='value',hue='sub')
ax.set(xlabel='Operation',ylabel='Variance')
ax.set_title('Operation engagement variability', loc='center', wrap=True)
ax.axhline(0,color='k',linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(data_dir,'figs','All_operation_variability_zscore.svg'))
plt.savefig(os.path.join(data_dir,'figs','All_operation_variability_zscore.png'))
plt.clf()

#pull out values for stats:
temp_maintain=combined_df[combined_df['operation']=='Maintain']['evidence']
temp_replace=combined_df[combined_df['operation']=='Replace']['evidence']
temp_suppress=combined_df[combined_df['operation']=='Suppress']['evidence']

F_stat, P_value = f_oneway(temp_maintain,temp_replace,temp_suppress) #compare all groups

F_stat_maintain_replace, P_value_maintain_replace = f_oneway(temp_maintain,temp_replace) #compare maintain vs. replace
F_stat_maintain_suppress, P_value_maintain_suppress = f_oneway(temp_maintain,temp_suppress) #compare maintain vs. suppress
F_stat_replace_suppress, P_value_replace_suppress = f_oneway(temp_replace,temp_suppress) 



### Bootstrap ###
iterations=10000


results={}

for i in range(iterations):
    bootstrap_m_var=[]
    bootstrap_r_var=[]
    bootstrap_s_var=[]
    for j in range(len(subIDs)):
        bootstrap_maintain=temp_maintain.sample(30,replace=True)
        bootstrap_replace=temp_replace.sample(30,replace=True)
        bootstrap_suppress=temp_suppress.sample(30,replace=True)

        temp_m_var=np.var(bootstrap_maintain)
        temp_r_var=np.var(bootstrap_replace)
        temp_s_var=np.var(bootstrap_suppress)

        bootstrap_m_var.append(temp_m_var)
        bootstrap_r_var.append(temp_r_var)
        bootstrap_s_var.append(temp_s_var)

    results[i]={'Maintain vs. Replace':ttest_ind(bootstrap_m_var,bootstrap_r_var),'Maintain vs. Suppress':ttest_ind(bootstrap_m_var,bootstrap_s_var),'Replace vs. Suppress': ttest_ind(bootstrap_r_var,bootstrap_s_var)}
    del bootstrap_m_var,bootstrap_r_var,bootstrap_s_var

#now test difference between groups
ttest_ind(bootstrap_m_var,bootstrap_r_var)
ttest_ind(bootstrap_m_var,bootstrap_s_var)
ttest_ind(bootstrap_r_var,bootstrap_s_var)

plot_bootstrap_var=pd.DataFrame(columns=['Variance','Operation'])

temp_df=pd.DataFrame()
temp_df['Variance']=bootstrap_m_var
temp_df['Operation']='Maintain'

plot_bootstrap_var=pd.concat([plot_bootstrap_var,temp_df])
del temp_df

temp_df=pd.DataFrame()
temp_df['Variance']=bootstrap_r_var
temp_df['Operation']='Replace'

plot_bootstrap_var=pd.concat([plot_bootstrap_var,temp_df])
del temp_df

temp_df=pd.DataFrame()
temp_df['Variance']=bootstrap_s_var
temp_df['Operation']='Suppress'

plot_bootstrap_var=pd.concat([plot_bootstrap_var,temp_df])
del temp_df

plt.style.use('seaborn-paper')

ax=sns.violinplot(data=plot_bootstrap_var,x='Operation',y='Variance',palette=['green','blue','red'])
ax.set(xlabel='Operation',ylabel='Variance')
ax.set_title('Bootstrapped Variance of Operation Engagement', loc='center', wrap=True)
plt.tight_layout()
plt.savefig(os.path.join(data_dir,'figs','All_operation_trials_bootstrap_variance.svg'))
plt.savefig(os.path.join(data_dir,'figs','All_operation_trials_bootstrap_variance.png'))
plt.clf()
