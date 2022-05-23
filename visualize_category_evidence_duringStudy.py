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

group_replace_evi=pd.DataFrame(columns=subs)
group_other_replace_evi=pd.DataFrame(columns=subs)
group_maintain_evi=pd.DataFrame(columns=subs)
group_suppress_evi=pd.DataFrame(columns=subs)

for num in subs:
    sub_num=num

    print('Running sub-0%s...' %sub_num)
    #define the subject
    sub = ('sub-0%s' % sub_num)
    container_path='/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'

    os.chdir(os.path.join(container_path,sub,'func'))

    evi_df=pd.read_csv("train_localizer_test_study_category_evidence.csv", delimiter=",",header=None)
    category_labels=pd.read_csv("Operation_labels.csv", delimiter=",",header=None)
    category_trials=pd.read_csv("Operation_trials.csv", delimiter=",",header=None)

    evi_df.insert(0,"operation",category_labels)
    evi_df.insert(1,"operation_trials",category_trials)

    evi_df.columns = ['operation','operation_trials','rest','scenes','faces']

    evi_df['operation'][(np.where(evi_df['operation']==0))[0]]='rest'

    evi_df['operation'][(np.where(evi_df['operation']==1))[0]]='maintain'

    evi_df['operation'][(np.where(evi_df['operation']==2))[0]]='replace'

    evi_df['operation'][(np.where(evi_df['operation']==3))[0]]='suppress'

    replace_evi=pd.DataFrame()
    other_replace_evi=pd.DataFrame()
    maintain_evi=pd.DataFrame()
    suppress_evi=pd.DataFrame()

    for i in range(1,31):
        #find the indices of the i trial for each operation
        replace_inds=np.where((evi_df['operation']=='replace') & (evi_df['operation_trials']==i))[0]
        suppress_inds=np.where((evi_df['operation']=='suppress') & (evi_df['operation_trials']==i))[0]
        maintain_inds=np.where((evi_df['operation']=='maintain') & (evi_df['operation_trials']==i))[0]
        #now we are going to add 5 indicies to the end to account for hemodynamics
        replace_inds=np.append(replace_inds, [i for i in range(replace_inds[-1]+1,replace_inds[-1]+6)])
        suppress_inds=np.append(suppress_inds, [i for i in range(suppress_inds[-1]+1,suppress_inds[-1]+6)])
        maintain_inds=np.append(maintain_inds, [i for i in range(maintain_inds[-1]+1,maintain_inds[-1]+6)])

        if (evi_df['scenes'][replace_inds].values.size) > 14:
            fills=evi_df['scenes'][replace_inds].values.size - 14
            temp = evi_df['scenes'][replace_inds].values[:-fills] #matching array lengths
            normalize=temp[:2].mean() #normalize values to first 2 TRs
            replace_evi['trial %s' % i]=temp-normalize

            temp2 = evi_df['faces'][replace_inds].values[:-fills]
            normalize2=temp2[:2].mean()
            other_replace_evi['trial %s' % i]=temp2-normalize2
        #else:
            #replace_evi['trial %s' % i]=evi_df['scenes'][replace_inds].values
            #other_replace_evi['trial %s' % i]=evi_df['faces'][replace_inds].values


        if (evi_df['scenes'][maintain_inds].values.size) > 14:
            fills = evi_df['scenes'][maintain_inds].values.size - 14
            temp = evi_df['scenes'][maintain_inds].values[:-fills]
            normalize=temp[:2].mean()
            maintain_evi['trial %s' % i]=temp-normalize

        # else:
        #     maintain_evi['trial %s' % i]=evi_df['scenes'][maintain_inds].values
        
        if (evi_df['scenes'][suppress_inds].values.size) > 14:
            fills = evi_df['scenes'][suppress_inds].values.size - 14
            temp = evi_df['scenes'][suppress_inds].values[:-fills]
            normalize=temp[:2].mean()
            suppress_evi['trial %s' % i]=temp-normalize

        # else:
        #     suppress_evi['trial %s' % i]=evi_df['scenes'][suppress_inds].values

    plot_replace_df=pd.DataFrame(columns=['x','y','l'])
    plot_suppress_df=pd.DataFrame(columns=['x','y','l'])
    plot_maintain_df=pd.DataFrame(columns=['x','y','l'])
    plot_otherreplace_df=pd.DataFrame(columns=['x','y','l'])

    plot_replace_df['x']=np.repeat(range(1,15),30)
    plot_replace_df['y']=replace_evi.values.flatten()
    plot_replace_df['l']=np.tile(replace_evi.columns,14)

    plot_otherreplace_df['x']=np.repeat(range(1,15),30)
    plot_otherreplace_df['y']=other_replace_evi.values.flatten()
    plot_otherreplace_df['l']=np.tile(other_replace_evi.columns,14)    

    plot_maintain_df['x']=np.repeat(range(1,15),30)
    plot_maintain_df['y']=maintain_evi.values.flatten()
    plot_maintain_df['l']=np.tile(maintain_evi.columns,14)

    plot_suppress_df['x']=np.repeat(range(1,15),30)
    plot_suppress_df['y']=suppress_evi.values.flatten()
    plot_suppress_df['l']=np.tile(suppress_evi.columns,14)

    ax=sns.lineplot(data=plot_replace_df,x='x',y='y',color='blue',label='Replace-old',ci=68)
    ax=sns.lineplot(data=plot_otherreplace_df,x='x',y='y',color='skyblue',label='Replace-new',ci=68)
    ax=sns.lineplot(data=plot_maintain_df,x='x',y='y',color='green',label='Maintain',ci=68)
    ax=sns.lineplot(data=plot_suppress_df,x='x',y='y',color='red',label='Suppress',ci=68)

    ax.set(xlabel='TR (unshfited)', ylabel='Category Evidence', title='%s %s Category Decoding during Operations' % (brain_flag,sub))

    plt.savefig(os.path.join(container_path,sub,'%s_%s_category_decoding_during_study.png' % (brain_flag,sub)))
    plt.clf()

    plot_diffreplace_df=pd.DataFrame(columns=['x','y','l'])
    plot_diffsuppress_df=pd.DataFrame(columns=['x','y','l'])

    plot_diffreplace_df['x']=np.repeat(range(1,15),30)
    plot_diffreplace_df['y']=replace_evi.sub(maintain_evi.mean(axis=1),axis=0).values.flatten()
    plot_diffreplace_df['l']=np.tile(replace_evi.columns,14)
    
    plot_diffsuppress_df['x']=np.repeat(range(1,15),30)
    plot_diffsuppress_df['y']=suppress_evi.sub(maintain_evi.mean(axis=1),axis=0).values.flatten()
    plot_diffsuppress_df['l']=np.tile(suppress_evi.columns,14)

    ax=sns.lineplot(data=plot_diffreplace_df,x='x',y='y',color='blue',label='Replace',ci=68)
    ax=sns.lineplot(data=plot_diffsuppress_df,x='x',y='y',color='red',label='Suppress',ci=68)

    plt.savefig(os.path.join(container_path,sub,'%s_%s_category_decoding_minusMaintain_during_study.png' % (brain_flag,sub)))
    plt.clf()

    group_replace_evi[num]=replace_evi.mean(axis=1)
    group_other_replace_evi[num]=other_replace_evi.mean(axis=1)
    group_maintain_evi[num]=maintain_evi.mean(axis=1)
    group_suppress_evi[num]=suppress_evi.mean(axis=1)


group_replace_evi.to_csv(os.path.join(container_path,'%s_group_category_decoding_replace.csv' % brain_flag))
group_other_replace_evi.to_csv(os.path.join(container_path,'%s_group_category_decoding_replace_new.csv' % brain_flag))
group_maintain_evi.to_csv(os.path.join(container_path,'%s_group_category_decoding_maintain.csv' % brain_flag))
group_suppress_evi.to_csv(os.path.join(container_path,'%s_group_category_decoding_suppress.csv' % brain_flag))   

plot_group_replace_df=pd.DataFrame(columns=['x','y','l'])
plot_group_suppress_df=pd.DataFrame(columns=['x','y','l'])
plot_group_maintain_df=pd.DataFrame(columns=['x','y','l'])
plot_group_otherreplace_df=pd.DataFrame(columns=['x','y','l'])

plot_group_replace_df['x']=np.repeat(range(1,15),len(subs))
plot_group_replace_df['y']=group_replace_evi.values.flatten()
plot_group_replace_df['l']=np.tile(group_replace_evi.columns,14)

plot_group_otherreplace_df['x']=np.repeat(range(1,15),len(subs))
plot_group_otherreplace_df['y']=group_other_replace_evi.values.flatten()
plot_group_otherreplace_df['l']=np.tile(group_other_replace_evi.columns,14)

plot_group_maintain_df['x']=np.repeat(range(1,15),len(subs))
plot_group_maintain_df['y']=group_maintain_evi.values.flatten()
plot_group_maintain_df['l']=np.tile(group_maintain_evi.columns,14)

plot_group_suppress_df['x']=np.repeat(range(1,15),len(subs))
plot_group_suppress_df['y']=group_suppress_evi.values.flatten()
plot_group_suppress_df['l']=np.tile(group_suppress_evi.columns,14)

ax=sns.lineplot(data=plot_group_replace_df,x='x',y='y',color='blue',label='Replace-old',ci=68)
ax=sns.lineplot(data=plot_group_otherreplace_df,x='x',y='y',color='skyblue',label='Replace-new',ci=68)
ax=sns.lineplot(data=plot_group_maintain_df,x='x',y='y',color='green',label='Maintain',ci=68)
ax=sns.lineplot(data=plot_group_suppress_df,x='x',y='y',color='red',label='Suppress',ci=68)

ax.set(xlabel='TR (unshfited)', ylabel='Category Evidence', title='%s Group-Level Category Decoding during Operations' % brain_flag)

plt.savefig(os.path.join(container_path,'%s_group_category_decoding_during_study.png' % brain_flag))
plt.clf()

plot_group_diffreplace_df=pd.DataFrame(columns=['x','y','l'])
plot_group_diffsuppress_df=pd.DataFrame(columns=['x','y','l'])

plot_group_diffreplace_df['x']=np.repeat(range(1,15),len(subs))
plot_group_diffreplace_df['y']=group_replace_evi.sub(group_maintain_evi.mean(axis=1),axis=0).values.flatten()
plot_group_diffreplace_df['l']=np.tile(group_replace_evi.columns,14)

plot_group_diffsuppress_df['x']=np.repeat(range(1,15),len(subs))
plot_group_diffsuppress_df['y']=group_suppress_evi.sub(group_maintain_evi.mean(axis=1),axis=0).values.flatten()
plot_group_diffsuppress_df['l']=np.tile(group_suppress_evi.columns,14)

ax=sns.lineplot(data=plot_group_diffreplace_df,x='x',y='y',color='blue',label='Replace',ci=68)
ax=sns.lineplot(data=plot_group_diffsuppress_df,x='x',y='y',color='red',label='Suppress',ci=68)

ax.set(xlabel='TR (unshfited)', ylabel='Category Evidence (removal - maintain)', title='%s Group-Level Category Decoding during Operations' % brain_flag)

plt.savefig(os.path.join(container_path,'%s_group_category_decoding_minusMaintain_during_study.png' % brain_flag))
plt.clf()