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

#subs=['02','03','04','05','06','07','08','09','11','12','14','17','23','24','25','26'] #"subjects with no notable issues in data"

TR_shift=5
brain_flag='T1w'

#masks=['wholebrain','vtc'] #wholebrain/vtc
mask_flag='vtc'

group_replace_evi=pd.DataFrame(columns=subs)
group_other_replace_evi=pd.DataFrame(columns=subs)
group_maintain_evi=pd.DataFrame(columns=subs)
group_suppress_evi=pd.DataFrame(columns=subs)
group_baseline_evi=pd.DataFrame(columns=subs)

group_diffreplace_df=pd.DataFrame(columns=subs)
group_diffsuppress_df=pd.DataFrame(columns=subs)

group_mem_plot=pd.DataFrame(columns=['category_evi','memory','condition'])
group_stim_evi=pd.DataFrame(columns=['category_evi','memory','condition'])

param_dir =  '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs'

for num in subs:
    sub_num=num

    print('Running sub-0%s...' %sub_num)
    #define the subject
    sub = ('sub-0%s' % sub_num)
    container_path='/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'

    os.chdir(os.path.join(container_path,sub,'func'))

    evi_df=pd.read_csv("%s_train_localizer_test_study_category_evidence.csv" % brain_flag, delimiter=",",header=None)
    category_labels=pd.read_csv("%s_Operation_labels.csv" % brain_flag, delimiter=",",header=None)
    category_trials=pd.read_csv("%s_Operation_trials.csv" % brain_flag, delimiter=",",header=None)

    evi_df.insert(0,"operation",category_labels)
    evi_df.insert(1,"operation_trials",category_trials)

    evi_df.columns = ['operation','operation_trials','rest','scenes','faces']

    evi_df['operation'][(np.where(evi_df['operation']==0))[0]]='rest'

    evi_df['operation'][(np.where(evi_df['operation']==1))[0]]='maintain'

    evi_df['operation'][(np.where(evi_df['operation']==2))[0]]='replace'

    evi_df['operation'][(np.where(evi_df['operation']==3))[0]]='suppress'


    tim_path = os.path.join(param_dir, f"sub-0{sub_num}_trial_image_match.csv")

    #lets pull out the study data here:
    tim_df2 = pd.read_csv(tim_path)
    tim_df2 = tim_df2[tim_df2["phase"]==3] #phase 3 is study
    tim_df2 = tim_df2.sort_values(by=["trial_id"])
    
    study_scene_order = tim_df2[tim_df2["category"]==1][["trial_id","image_id","condition","subcategory"]]

    mem_path = os.path.join(param_dir, f"memory_and_familiar_sub-0{sub_num}.csv")
    mem_df = pd.read_csv(mem_path)

    replace_evi=pd.DataFrame()
    other_replace_evi=pd.DataFrame()
    maintain_evi=pd.DataFrame()
    suppress_evi=pd.DataFrame()
    baseline_evi=pd.DataFrame()

    r_replace_evi=pd.DataFrame()
    r_maintain_evi=pd.DataFrame()
    r_suppress_evi=pd.DataFrame()

    f_replace_evi=pd.DataFrame()
    f_maintain_evi=pd.DataFrame()
    f_suppress_evi=pd.DataFrame()

    mem_replace_evi=pd.DataFrame(columns=['category_evi','memory'])
    mem_maintain_evi=pd.DataFrame(columns=['category_evi','memory'])
    mem_suppress_evi=pd.DataFrame(columns=['category_evi','memory'])

    stim_replace_evi=pd.DataFrame(columns=['category_evi','memory'])
    stim_maintain_evi=pd.DataFrame(columns=['category_evi','memory'])
    stim_suppress_evi=pd.DataFrame(columns=['category_evi','memory'])    

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
            temp_faces = evi_df['faces'][replace_inds].values[:-fills] #matching array lengths

            #we also want to see if this item was remembered or not
            trial_id=study_scene_order['image_id'][study_scene_order['condition']==2].values[i-1]
            trial_mem=mem_df[mem_df['image_num']==trial_id]['memory'].values[0]
            if trial_mem==1:
                r_replace_evi['trial %s' % i]=temp
                temp_df={'category_evi':temp[7:11].mean(), 'memory':1,'category_diff_evi':((temp[7:11].mean()) - (temp_faces[7:11].mean()))}

                temp_stim_df={'category_evi':temp[5:7].mean(), 'memory':1, 'category_diff_evi':((temp[5:7].mean()) - (temp_faces[5:7].mean()))}

                mem_replace_evi=mem_replace_evi.append(temp_df,ignore_index=True)

                stim_replace_evi=stim_replace_evi.append(temp_stim_df,ignore_index=True)

            else:
                temp_df={'category_evi':temp[7:11].mean(), 'memory':0,'category_diff_evi':((temp[7:11].mean()) - (temp_faces[7:11].mean()))}

                temp_stim_df={'category_evi':temp[5:7].mean(), 'memory':0, 'category_diff_evi':((temp[5:7].mean()) - (temp_faces[5:7].mean()))}

                mem_replace_evi=mem_replace_evi.append(temp_df,ignore_index=True)

                stim_replace_evi=stim_replace_evi.append(temp_stim_df,ignore_index=True)

                f_replace_evi['trial %s' % i]=temp

            normalize=temp[:2].mean() #normalize values to first 2 TRs
            replace_evi['trial %s' % i]=temp-normalize
            del temp,normalize, trial_id, trial_mem

            temp2 = evi_df['faces'][replace_inds].values[:-fills]
            normalize2=temp2[:2].mean()
            other_replace_evi['trial %s' % i]=temp2-normalize2
            del temp2,normalize2
        #else:
            #replace_evi['trial %s' % i]=evi_df['scenes'][replace_inds].values
            #other_replace_evi['trial %s' % i]=evi_df['faces'][replace_inds].values


        if (evi_df['scenes'][maintain_inds].values.size) > 14:
            fills = evi_df['scenes'][maintain_inds].values.size - 14
            temp = evi_df['scenes'][maintain_inds].values[:-fills]
            temp_faces = evi_df['faces'][maintain_inds].values[:-fills] #matching array lengths

            #we also want to see if this item was remembered or not
            trial_id=study_scene_order['image_id'][study_scene_order['condition']==1].values[i-1]
            trial_mem=mem_df[mem_df['image_num']==trial_id]['memory'].values[0]
            if trial_mem==1:
                r_maintain_evi['trial %s' % i]=temp
                temp_df={'category_evi':temp[7:11].mean(), 'memory':1,'category_diff_evi':((temp[7:11].mean()) - (temp_faces[7:11].mean()))}

                temp_stim_df={'category_evi':temp[5:7].mean(), 'memory':1, 'category_diff_evi':((temp[5:7].mean()) - (temp_faces[5:7].mean()))}

                stim_maintain_evi=stim_maintain_evi.append(temp_stim_df,ignore_index=True)    

                mem_maintain_evi=mem_maintain_evi.append(temp_df,ignore_index=True)                
            else:
                f_maintain_evi['trial %s' % i]=temp
                temp_df={'category_evi':temp[7:11].mean(), 'memory':0,'category_diff_evi':((temp[7:11].mean()) - (temp_faces[7:11].mean()))}

                temp_stim_df={'category_evi':temp[5:7].mean(), 'memory':0, 'category_diff_evi':((temp[5:7].mean()) - (temp_faces[5:7].mean()))}

                stim_maintain_evi=stim_maintain_evi.append(temp_stim_df,ignore_index=True)    

                mem_maintain_evi=mem_maintain_evi.append(temp_df,ignore_index=True)

            normalize=temp[:2].mean()
            maintain_evi['trial %s' % i]=temp-normalize
            del temp,normalize, trial_id, trial_mem
        # else:
        #     maintain_evi['trial %s' % i]=evi_df['scenes'][maintain_inds].values
        
        if (evi_df['scenes'][suppress_inds].values.size) > 14:
            fills = evi_df['scenes'][suppress_inds].values.size - 14
            temp = evi_df['scenes'][suppress_inds].values[:-fills]
            temp_faces = evi_df['faces'][suppress_inds].values[:-fills]


            #we also want to see if this item was remembered or not
            trial_id=study_scene_order['image_id'][study_scene_order['condition']==3].values[i-1]
            trial_mem=mem_df[mem_df['image_num']==trial_id]['memory'].values[0]
            if trial_mem==1:
                r_suppress_evi['trial %s' % i]=temp
                temp_df={'category_evi':temp[7:11].mean(), 'memory':1,'category_diff_evi':((temp[7:11].mean()) - (temp_faces[7:11].mean()))}

                temp_stim_df={'category_evi':temp[5:7].mean(), 'memory':1, 'category_diff_evi':((temp[5:7].mean()) - (temp_faces[5:7].mean()))}

                stim_suppress_evi=stim_suppress_evi.append(temp_stim_df,ignore_index=True)  

                mem_suppress_evi=mem_suppress_evi.append(temp_df,ignore_index=True)                
            else:
                f_suppress_evi['trial %s' % i]=temp
                temp_df={'category_evi':temp[7:11].mean(), 'memory':0,'category_diff_evi':((temp[7:11].mean()) - (temp_faces[7:11].mean()))}

                temp_stim_df={'category_evi':temp[5:7].mean(), 'memory':0, 'category_diff_evi':((temp[5:7].mean()) - (temp_faces[5:7].mean()))}

                stim_suppress_evi=stim_suppress_evi.append(temp_stim_df,ignore_index=True)  

                mem_suppress_evi=mem_suppress_evi.append(temp_df,ignore_index=True)  

            normalize=temp[:2].mean()
            suppress_evi['trial %s' % i]=temp-normalize
            del temp,normalize, trial_id, trial_mem

            temp2 = evi_df['rest'][suppress_inds].values[:-fills]
            normalize2=temp2[:2].mean()
            baseline_evi['trial %s' % i]=temp2-normalize2
            del temp2,normalize2

        # else:
        #     suppress_evi['trial %s' % i]=evi_df['scenes'][suppress_inds].values
#########
    plot_replace_df=pd.DataFrame(columns=['x','y','l'])
    plot_suppress_df=pd.DataFrame(columns=['x','y','l'])
    plot_maintain_df=pd.DataFrame(columns=['x','y','l'])
    plot_otherreplace_df=pd.DataFrame(columns=['x','y','l'])
    plot_baseline_df=pd.DataFrame(columns=['x','y','l'])

    plot_replace_df['x']=np.repeat(range(0,14),30)
    plot_replace_df['y']=replace_evi.values.flatten()
    plot_replace_df['l']=np.tile(replace_evi.columns,14)

    plot_otherreplace_df['x']=np.repeat(range(0,14),30)
    plot_otherreplace_df['y']=other_replace_evi.values.flatten()
    plot_otherreplace_df['l']=np.tile(other_replace_evi.columns,14)    

    plot_maintain_df['x']=np.repeat(range(0,14),30)
    plot_maintain_df['y']=maintain_evi.values.flatten()
    plot_maintain_df['l']=np.tile(maintain_evi.columns,14)

    plot_suppress_df['x']=np.repeat(range(0,14),30)
    plot_suppress_df['y']=suppress_evi.values.flatten()
    plot_suppress_df['l']=np.tile(suppress_evi.columns,14)

    plot_baseline_df['x']=np.repeat(range(0,14),30)
    plot_baseline_df['y']=baseline_evi.values.flatten()
    plot_baseline_df['l']=np.tile(baseline_evi.columns,14)

    ax=sns.lineplot(data=plot_replace_df,x='x',y='y',color='blue',label='Replace-old',ci=68)
    ax=sns.lineplot(data=plot_otherreplace_df,x='x',y='y',color='skyblue',label='Replace-new',ci=68)
    ax=sns.lineplot(data=plot_maintain_df,x='x',y='y',color='green',label='Maintain',ci=68)
    ax=sns.lineplot(data=plot_suppress_df,x='x',y='y',color='red',label='Suppress',ci=68)
    ax=sns.lineplot(data=plot_baseline_df,x='x',y='y',color='gray',label='Baseline',ci=68)


    ax.set(xlabel='TR (unshfited)', ylabel='Category Evidence', title='%s %s Category Decoding during Operations' % (brain_flag,sub))

    plt.savefig(os.path.join(container_path,sub,'%s_%s_category_decoding_during_study.png' % (brain_flag,sub)))
    plt.clf()
############

    mem_replace_evi['condition']='replace'
    mem_maintain_evi['condition']='maintain'
    mem_suppress_evi['condition']='suppress'
    mem_plot_df=pd.concat([mem_replace_evi,mem_maintain_evi,mem_suppress_evi])

    mem_plot_df.to_csv(os.path.join(container_path,sub,'%s_%s_evidence_for_memory.csv' % (brain_flag,sub)))

    stim_replace_evi['condition']='replace'
    stim_maintain_evi['condition']='maintain'
    stim_suppress_evi['condition']='suppress'
    stim_plot_df=pd.concat([stim_replace_evi,stim_maintain_evi,stim_suppress_evi])

    stim_plot_df.to_csv(os.path.join(container_path,sub,'%s_%s_evidence_for_stimuli.csv' % (brain_flag,sub)))    

    sns.lmplot(data=mem_plot_df,y='memory',x='category_evi',hue='condition',logistic=True, y_jitter=0.03)
    plt.savefig(os.path.join(container_path,sub,'%s_%s_evidence_for_memory(by_condition).png' % (brain_flag,sub)))
    plt.clf()    

    sns.lmplot(data=mem_plot_df,y='memory',x='category_evi',logistic=True, y_jitter=0.03)
    plt.savefig(os.path.join(container_path,sub,'%s_%s_evidence_for_memory(combined).png' % (brain_flag,sub)))
    plt.clf()   

############
    plot_diffreplace_df=pd.DataFrame(columns=['x','y','l'])
    plot_diffsuppress_df=pd.DataFrame(columns=['x','y','l'])

    plot_diffreplace_df['x']=np.repeat(range(0,14),30)
    plot_diffreplace_df['y']=replace_evi.sub(maintain_evi.mean(axis=1),axis=0).values.flatten()
    plot_diffreplace_df['l']=np.tile(replace_evi.columns,14)
    
    plot_diffsuppress_df['x']=np.repeat(range(0,14),30)
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
    group_baseline_evi[num]=baseline_evi.mean(axis=1)

    group_diffreplace_df[num]=replace_evi.sub(maintain_evi.mean(axis=1),axis=0).mean(axis=1)
    group_diffsuppress_df[num]=suppress_evi.sub(maintain_evi.mean(axis=1),axis=0).mean(axis=1)

    group_mem_plot=group_mem_plot.append(mem_plot_df)
    group_stim_evi=group_stim_evi.append(stim_plot_df)
###########


group_mem_plot.to_csv(os.path.join(container_path,'%s_evidence_for_memory.csv' % (brain_flag)))
group_stim_evi.to_csv(os.path.join(container_path,'%s_evidence_for_stimuli.csv' % (brain_flag)))


sns.lmplot(data=group_mem_plot,y='memory',x='category_evi',hue='condition',logistic=True, y_jitter=0.03)
plt.savefig(os.path.join(container_path,'%s_evidence_for_memory(by_condition).png' % (brain_flag)))
plt.clf()    

sns.lmplot(data=group_mem_plot,y='memory',x='category_evi',logistic=True, y_jitter=0.03)
plt.savefig(os.path.join(container_path,'%s_evidence_for_memory(combined).png' % (brain_flag)))
plt.clf()   

group_replace_evi.to_csv(os.path.join(container_path,'%s_group_category_decoding_replace.csv' % brain_flag))
group_other_replace_evi.to_csv(os.path.join(container_path,'%s_group_category_decoding_replace_new.csv' % brain_flag))
group_maintain_evi.to_csv(os.path.join(container_path,'%s_group_category_decoding_maintain.csv' % brain_flag))
group_suppress_evi.to_csv(os.path.join(container_path,'%s_group_category_decoding_suppress.csv' % brain_flag))   
group_baseline_evi.to_csv(os.path.join(container_path,'%s_group_category_decoding_baseline.csv' % brain_flag))   


plot_group_replace_df=pd.DataFrame(columns=['x','y','l'])
plot_group_suppress_df=pd.DataFrame(columns=['x','y','l'])
plot_group_maintain_df=pd.DataFrame(columns=['x','y','l'])
plot_group_otherreplace_df=pd.DataFrame(columns=['x','y','l'])
plot_group_baseline_df=pd.DataFrame(columns=['x','y','l'])


plot_group_replace_df['x']=np.repeat(range(0,14),len(subs))
plot_group_replace_df['y']=group_replace_evi.values.flatten()
plot_group_replace_df['l']=np.tile(group_replace_evi.columns,14)

plot_group_otherreplace_df['x']=np.repeat(range(0,14),len(subs))
plot_group_otherreplace_df['y']=group_other_replace_evi.values.flatten()
plot_group_otherreplace_df['l']=np.tile(group_other_replace_evi.columns,14)

plot_group_maintain_df['x']=np.repeat(range(0,14),len(subs))
plot_group_maintain_df['y']=group_maintain_evi.values.flatten()
plot_group_maintain_df['l']=np.tile(group_maintain_evi.columns,14)

plot_group_suppress_df['x']=np.repeat(range(0,14),len(subs))
plot_group_suppress_df['y']=group_suppress_evi.values.flatten()
plot_group_suppress_df['l']=np.tile(group_suppress_evi.columns,14)

plot_group_baseline_df['x']=np.repeat(range(0,14),len(subs))
plot_group_baseline_df['y']=group_baseline_evi.values.flatten()
plot_group_baseline_df['l']=np.tile(group_baseline_evi.columns,14)

ax=sns.lineplot(data=plot_group_replace_df,x='x',y='y',color='blue',label='Replace-old',ci=68)

ax=sns.lineplot(data=plot_group_otherreplace_df,x='x',y='y',color='skyblue',label='Replace-new',ci=68)

ax=sns.lineplot(data=plot_group_maintain_df,x='x',y='y',color='green',label='Maintain',ci=68)

ax=sns.lineplot(data=plot_group_suppress_df,x='x',y='y',color='red',label='Suppress',ci=68)

ax=sns.lineplot(data=plot_group_baseline_df,x='x',y='y',color='gray',label='Baseline',ci=68)

ax.set(xlabel='TR (unshfited)', ylabel='Category Evidence', title='%s Group-Level Category Decoding during Operations' % brain_flag)


plt.savefig(os.path.join(container_path,'%s_group_category_decoding_during_study.png' % brain_flag))
plt.clf()

plot_group_diffreplace_df=pd.DataFrame(columns=['x','y','l'])
plot_group_diffsuppress_df=pd.DataFrame(columns=['x','y','l'])

plot_group_diffreplace_df['x']=np.repeat(range(0,14),len(subs))
plot_group_diffreplace_df['y']=group_diffreplace_df.values.flatten()
plot_group_diffreplace_df['l']=np.tile(group_replace_evi.columns,14)

plot_group_diffsuppress_df['x']=np.repeat(range(0,14),len(subs))
plot_group_diffsuppress_df['y']=group_diffsuppress_df.values.flatten()
plot_group_diffsuppress_df['l']=np.tile(group_suppress_evi.columns,14)

ax=sns.lineplot(data=plot_group_diffreplace_df,x='x',y='y',color='blue',label='Replace',ci=68)
ax=sns.lineplot(data=plot_group_diffsuppress_df,x='x',y='y',color='red',label='Suppress',ci=68)

ax.set(xlabel='TR (unshfited)', ylabel='Category Evidence (removal - maintain)', title='%s Group-Level Category Decoding during Operations' % brain_flag)

plt.savefig(os.path.join(container_path,'%s_group_category_decoding_minusMaintain_during_study.png' % brain_flag))
plt.clf()