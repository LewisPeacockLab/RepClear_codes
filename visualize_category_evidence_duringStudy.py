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

subs=['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','20','23','24','25','26']
TR_shift=5
brain_flag='MNI'

#masks=['wholebrain','vtc'] #wholebrain/vtc
mask_flag='vtc'

    sub_num=subs[num]

    print('Running sub-0%s...' %sub_num)
    #define the subject
    sub = ('sub-0%s' % sub_num)
    container_path='/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'

    os.chdir(os.path.join(container_path,sub,'func'))

    evi_df=pd.read_csv("train_local(1256)_test_study_category_evidence.csv", delimiter=",",header=None)
    category_labels=pd.read_csv("Operation_labels.csv", delimiter=",",header=None)

    evi_df.insert(0,"operation",category_labels)

    evi_df.columns = ['operation','rest','faces','fruit']

    evi_df['operation'][(np.where(evi_df['operation']==0))[0]]='rest'

    evi_df['operation'][(np.where(evi_df['operation']==1))[0]]='maintain'

    evi_df['operation'][(np.where(evi_df['operation']==2))[0]]='replace'

    evi_df['operation'][(np.where(evi_df['operation']==3))[0]]='suppress'

