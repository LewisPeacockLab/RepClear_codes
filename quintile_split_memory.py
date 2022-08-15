#code to visualize the evidence of categories during operation times

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
import fnmatch
import pandas as pd
import pickle
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

subs=['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','20','23','24','25','26'] #"all overall clean subjects"

#subs_subset=['02','03','04','05','06','07','08','09','11','12','14','17','23','24','25','26'] #"subjects with no notable issues in data"

TR_shift=5
brain_flag='T1w'

#masks=['wholebrain','vtc'] #wholebrain/vtc
mask_flag='vtc'

param_dir =  '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'

memory_df=pd.read_csv(os.path.join(param_dir,f'{brain_flag}_evidence_for_memory.csv')) #new table - of evidence during removal
#memory_df=pd.read_csv(os.path.join(param_dir,'T1w_evidence_for_stimuli.csv')) #new table - of evidence during encoding


group_condition_df=pd.DataFrame()
group_overall_df=pd.DataFrame()

memory_df['subject id']=np.repeat(subs, 90)

for sub in subs:

    sub_df=memory_df[memory_df['subject id']==sub]
    sub_df['quintiles']=pd.qcut(sub_df['category_evi'],5,labels=False)
    sub_condition=sub_df.groupby(['condition','quintiles'])['memory'].mean().values
    sub_overall=sub_df.groupby(['quintiles'])['memory'].mean().values

    try:
        temp_df=pd.DataFrame()
        temp_df['condition']=['maintain','maintain','maintain','maintain','maintain','replace','replace','replace','replace','replace','suppress','suppress','suppress','suppress','suppress']
        temp_df['quintiles']=[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
        temp_df['memory']=sub_condition
        temp_df['subject']=sub

        temp_df2=pd.DataFrame()
        temp_df2['quintiles']=[1,2,3,4,5]
        temp_df2['memory']=sub_overall
        temp_df2['subject']=sub

        group_condition_df=group_condition_df.append(temp_df)
        group_overall_df=group_overall_df.append(temp_df2)
    except:
        print(f'sub-0{sub} is having issues appending (likely due to missing quintiles)... skipping for now')

sns.violinplot(data=group_condition_df,x='condition',y='memory',hue='quintiles',palette='flare')
plt.tight_layout()
plt.savefig(os.path.join(param_dir,f'{brain_flag}_group_by_condition_memory_removal_quintiles.png'))
plt.clf()

sns.violinplot(data=group_overall_df,x='quintiles',y='memory',palette='flare')
plt.tight_layout()
plt.savefig(os.path.join(param_dir,f'{brain_flag}_group_overall_memory_removal_quintiles.png'))
plt.clf()