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

mask_flag = 'wholebrain' #'vtc'/'wholebrain'
brain_flag = 'MNI'
TR_shift=6
ses = 'study'

if ses=='study':
    ses_label='operation'
elif ses=='localizer':
    ses_label='category'

def group_cmatrix(subs):
    group_mean_confusion=[]
    oper_confusion_mean=[]
    for num in subs:
        print('Running sub-0%s...' % num)
        #define the subject
        sub = ('sub-0%s' % num)
        container_path='/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'
        
      
        bold_path=os.path.join(container_path,sub)
        os.chdir(bold_path)

        if ses=='study':
            sub_dict=pickle.load(open("%s-studyoperation_%s_%s_%sTR lag_data.pkl" % (sub,brain_flag,mask_flag,TR_shift),'rb'))            
            sub_dict_pred=sub_dict["L2 Predictions (No Rest)"]
            sub_dict_true=sub_dict["L2 True (No Rest)"]            
            sub_confusion_1=confusion_matrix(sub_dict_true[0],sub_dict_pred[0],normalize='true')
            sub_confusion_2=confusion_matrix(sub_dict_true[1],sub_dict_pred[1],normalize='true')            
            sub_confusion_3=confusion_matrix(sub_dict_true[2],sub_dict_pred[2],normalize='true')
            sub_confusion_mean=np.mean( np.array([ sub_confusion_1, sub_confusion_2, sub_confusion_3]), axis=0 )
   
        else:
            sub_dict=pickle.load(open("%s-preremoval_%s_%s_%sTR lag_data.pkl" % (sub,brain_flag,mask_flag,TR_shift),'rb'))                      
            sub_dict_pred=sub_dict['L2 Predictions (only On)']
            sub_dict_true=sub_dict['L2 True (only On)']
            sub_confusion_1=confusion_matrix(sub_dict_true[0],sub_dict_pred[0],normalize='true')
            sub_confusion_2=confusion_matrix(sub_dict_true[1],sub_dict_pred[1],normalize='true')
            sub_confusion_mean=np.mean( np.array([ sub_confusion_1, sub_confusion_2]), axis=0 )
        group_mean_confusion.append(sub_confusion_mean)
    return group_mean_confusion


#labels=['animals', 'food', 'tools', 'scenes', 'scrambled', 'rest']
labels=['scenes','faces']
oper_labels=['maintain', 'replace', 'suppress']

fig=plt.figure()
group_mean_confusion=group_cmatrix(subs)
plot_confusion=np.mean(group_mean_confusion,axis=0)
plt.style.use('fivethirtyeight')
ax = sns.heatmap(data=(plot_confusion*100),annot=True,cmap='jet')
if ses=='localizer':
    ax.set(xlabel='Predicted', ylabel='True', xticklabels=labels, yticklabels=labels,title='Group Mean X-Validation')
elif ses=='study':
    ax.set(xlabel='Predicted', ylabel='True', xticklabels=oper_labels, yticklabels=oper_labels,title='Group Mean X-Validation')
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
os.chdir('/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/figs')
fig.savefig('%s_xvalidation_%sclassifier_%s_TR%s.png' % (ses,ses_label,brain_flag,TR_shift), dpi=fig.dpi)
plt.show()