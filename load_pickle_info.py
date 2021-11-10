import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
import nibabel as nib
import scipy as scipy
import scipy.io as sio
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fnmatch
import pandas as pd
import pickle


subs=['02','03','04']

TR_array=[4,5,6]
brain_flag='T1w'

masks=['wholebrain','vtc'] #wholebrain/vtc

mask_flag=masks[0]

phase='study' #localizer

for i in [0,1,2]:
    for TR_shift in TR_array:
        sub=('sub-0%s' % subs[i])


        container_path='/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'

        bold_path=os.path.join(container_path,sub)
        os.chdir(bold_path)

        if phase=='localizer':
            sub_dict=pickle.load(open("%s-preremoval_%s_%s_%sTR lag_data.pkl" % (sub,brain_flag,mask_flag,TR_shift),'rb'))

            random_acc=sub_dict["Shuffled Data"]
            raw_classif=sub_dict["L2 Raw Scores (only On)"]
            predicts=sub_dict["L2 Predictions (only On)"]
            trues=sub_dict["L2 True (only On)"]
        elif phase=='study':
            sub_dict=pickle.load(open("%s-studyoperation_%s_%s_%sTR lag_data.pkl" % (sub,brain_flag,mask_flag,TR_shift),'rb'))

            random_acc=0.333
            raw_classif=sub_dict["L2 Raw Scores (No Rest)"]
            predicts=sub_dict["L2 Predictions (No Rest)"]
            trues=sub_dict["L2 True (No Rest)"]

        print('-----------------------------')
        print('Running %s...' % sub)
        print('TR lag = %s  |  Brain-Space = %s  |  Mask = %s' % (TR_shift,brain_flag,mask_flag))
        print('-----------------------------')
        print('Random accuracy = %s' % random_acc)
        print('Localizer X-Validation scores = %s ' % raw_classif)
        print('-----------------------------')
        if phase=='localizer':
            print('Classifier Predictions = %s , %s' % (predicts[0],predicts[1]))
        elif phase=='study':
            print('Classifier Predictions = %s , %s' % (predicts[0],predicts[1],predicts[2]))
        print('-----------------------------')
        print('Classifier correct labels = %s' % np.asarray(trues).flatten())
        print('-----------------------------')