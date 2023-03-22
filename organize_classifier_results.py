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
group_m_aucs=[]
group_r_aucs=[]
group_s_aucs=[]
group_scores=[]

data_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
for subID in subIDs:

    in_file=os.path.join(data_dir,f'sub-{subID}',f'sub-{subID}_T1w_study_operation_auc.csv') #get dataframe file name

    temp_dict=pd.read_csv(in_file) #load in dataframe

    group_aucs.append(temp_dict['AUC'].mean()) #take the mean of the x-valdiation AUCs and append to group level table for stats

    group_m_aucs.append(temp_dict['AUC'][0]) #takes the first value which is maintain
    group_r_aucs.append(temp_dict['AUC'][1]) #takes the second value which is replace
    group_s_aucs.append(temp_dict['AUC'][2]) #takes the third value which is suppress

    #group_scores.append(temp_dict['scores'].mean())


#one-sample t-test against chance values: AUC=0.5, Scores=0.33

#need to run these per operation:

#maintain:
auc_t_stat_m, auc_p_value_m = ttest_1samp(group_m_aucs,0.5)
print('##### Reporting stats for Maintain AUC #####')
print(f'Mean = {np.array(group_m_aucs).mean()}')
print(f'SEM = {scipy.stats.sem(group_m_aucs)}')
print(f'Degrees of freedom = {(len(group_m_aucs)-1)}')
print(f'T-stat = {auc_t_stat_m}')
print(f'p-value = {auc_p_value_m}')
print(f'cohens d =  {calccohensd(group_m_aucs,0.5)}')
print('')

#replace:
auc_t_stat_r, auc_p_value_r = ttest_1samp(group_r_aucs,0.5)
print('##### Reporting stats for Replace AUC #####')
print(f'Mean = {np.array(group_r_aucs).mean()}')
print(f'SEM = {scipy.stats.sem(group_r_aucs)}')
print(f'Degrees of freedom = {(len(group_r_aucs)-1)}')
print(f'T-stat = {auc_t_stat_r}')
print(f'p-value = {auc_p_value_r}')
print(f'cohens d =  {calccohensd(group_r_aucs,0.5)}')
print('')

#suppress:
auc_t_stat_s, auc_p_value_s = ttest_1samp(group_s_aucs,0.5)
print('##### Reporting stats for Suppress AUC #####')
print(f'Mean = {np.array(group_s_aucs).mean()}')
print(f'SEM = {scipy.stats.sem(group_s_aucs)}')
print(f'Degrees of freedom = {(len(group_s_aucs)-1)}')
print(f'T-stat = {auc_t_stat_s}')
print(f'p-value = {auc_p_value_s}')
print(f'cohens d =  {calccohensd(group_s_aucs,0.5)}')
print('')


#Generate figure for results:
plot_df=pd.DataFrame(columns=['AUCs','Operation'])

temp_df=pd.DataFrame(columns=['AUCs','Operation'])
temp_df['AUCs']=group_m_aucs
temp_df['Operation']='maintain'

plot_df=pd.concat([plot_df,temp_df],ignore_index=True, sort=False)

temp_df=pd.DataFrame(columns=['AUCs','Operation'])
temp_df['AUCs']=group_r_aucs
temp_df['Operation']='replace'

plot_df=pd.concat([plot_df,temp_df],ignore_index=True, sort=False)

temp_df=pd.DataFrame(columns=['AUCs','Operation'])
temp_df['AUCs']=group_s_aucs
temp_df['Operation']='suppress'

plot_df=pd.concat([plot_df,temp_df],ignore_index=True, sort=False)


plt.style.use('seaborn-paper')

ax=sns.violinplot(data=plot_df,x='Operation',y='AUCs',palette=['green','blue','red'],inner='quartile')
sns.swarmplot(data=plot_df,x='Operation',y='AUCs',color= "white")
ax.set(xlabel='Operation',ylabel='AUCs')
ax.set_title('Within-Subject Operation AUCs', loc='center', wrap=True)
ax.axhline(0.5,color='k',linestyle='--')
ax.set_ylim([0.3,1.1])
plt.tight_layout()
plt.savefig(os.path.join(data_dir,'figs','Within-Subject_AUC_Operation.svg'))
plt.savefig(os.path.join(data_dir,'figs','Within-Subject_AUC_Operation.png'))
plt.clf()

###############
#collect results for between subject classification:

# collect results for the operation classifier #

group_m_aucs=[]
group_r_aucs=[]
group_s_aucs=[]

data_dir = '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'

in_file=os.path.join(data_dir,'btwnsub_MNI_study_operation_auc.csv') #get dataframe file name

temp_dict=pd.read_csv(in_file) #load in dataframe

group_m_aucs=(temp_dict['Maintain']) #takes the first column which is maintain
group_r_aucs=(temp_dict['Replace']) #takes the second column which is replace
group_s_aucs=(temp_dict['Suppress']) #takes the third column which is suppress

    #group_scores.append(temp_dict['scores'].mean())


#one-sample t-test against chance values: AUC=0.5, Scores=0.33

#need to run these per operation:

#maintain:
auc_t_stat_m, auc_p_value_m = ttest_1samp(group_m_aucs,0.5)
print('##### Reporting stats for Maintain AUC #####')
print(f'Mean = {np.array(group_m_aucs).mean()}')
print(f'SEM = {scipy.stats.sem(group_m_aucs)}')
print(f'Degrees of freedom = {(len(group_m_aucs)-1)}')
print(f'T-stat = {auc_t_stat_m}')
print(f'p-value = {auc_p_value_m}')
print(f'cohens d =  {calccohensd(group_m_aucs,0.5)}')
print('')

#replace:
auc_t_stat_r, auc_p_value_r = ttest_1samp(group_r_aucs,0.5)
print('##### Reporting stats for Replace AUC #####')
print(f'Mean = {np.array(group_r_aucs).mean()}')
print(f'SEM = {scipy.stats.sem(group_r_aucs)}')
print(f'Degrees of freedom = {(len(group_r_aucs)-1)}')
print(f'T-stat = {auc_t_stat_r}')
print(f'p-value = {auc_p_value_r}')
print(f'cohens d =  {calccohensd(group_r_aucs,0.5)}')
print('')

#suppress:
auc_t_stat_s, auc_p_value_s = ttest_1samp(group_s_aucs,0.5)
print('##### Reporting stats for Suppress AUC #####')
print(f'Mean = {np.array(group_s_aucs).mean()}')
print(f'SEM = {scipy.stats.sem(group_s_aucs)}')
print(f'Degrees of freedom = {(len(group_s_aucs)-1)}')
print(f'T-stat = {auc_t_stat_s}')
print(f'p-value = {auc_p_value_s}')
print(f'cohens d =  {calccohensd(group_s_aucs,0.5)}')
print('')

#Generate figure for results:
plot_df=pd.DataFrame(columns=['AUCs','Operation'])

temp_df=pd.DataFrame(columns=['AUCs','Operation'])
temp_df['AUCs']=group_m_aucs
temp_df['Operation']='maintain'

plot_df=pd.concat([plot_df,temp_df],ignore_index=True, sort=False)

temp_df=pd.DataFrame(columns=['AUCs','Operation'])
temp_df['AUCs']=group_r_aucs
temp_df['Operation']='replace'

plot_df=pd.concat([plot_df,temp_df],ignore_index=True, sort=False)

temp_df=pd.DataFrame(columns=['AUCs','Operation'])
temp_df['AUCs']=group_s_aucs
temp_df['Operation']='suppress'

plot_df=pd.concat([plot_df,temp_df],ignore_index=True, sort=False)


plt.style.use('seaborn-paper')

ax=sns.violinplot(data=plot_df,x='Operation',y='AUCs',palette=['green','blue','red'],inner='quartile')
sns.swarmplot(data=plot_df,x='Operation',y='AUCs',color= "white")
ax.set(xlabel='Operation',ylabel='AUCs')
ax.set_title('Between-Subject Operation AUCs', loc='center', wrap=True)
ax.axhline(0.5,color='k',linestyle='--')
ax.set_ylim([0.3,1.1])
plt.tight_layout()
plt.savefig(os.path.join(data_dir,'figs','Between-Subject_AUC_Operation.svg'))
plt.savefig(os.path.join(data_dir,'figs','Between-Subject_AUC_Operation.png'))
plt.clf()