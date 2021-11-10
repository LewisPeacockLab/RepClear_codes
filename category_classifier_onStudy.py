#code to train on localizer data and then test on the study data

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

subs=['02','03','04']

TR_shift=5
brain_flag='T1w'

#masks=['wholebrain','vtc'] #wholebrain/vtc
masks=['vtc']

for num in range(len(subs)):
    sub_num=subs[num]

    print('Running sub-0%s...' %sub_num)
    #define the subject
    sub = ('sub-0%s' % sub_num)
    container_path='/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'
  
    bold_path=os.path.join(container_path,sub,'func/')
    os.chdir(bold_path)
  
    #set up the path to the files and then moved into that directory
    
    #find the proper nii.gz files
    def find(pattern, path): #find the pattern we're looking for
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
            return result

    localizer_files=find('*preremoval*bold*.nii.gz',bold_path)
    wholebrain_mask_path=find('*preremoval*mask*.nii.gz',bold_path)
    
    if brain_flag=='MNI':
        pattern = '*MNI*'
        pattern2= '*MNI152NLin2009cAsym*preproc*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        localizer_files = fnmatch.filter(localizer_files,pattern2)
    elif brain_flag=='T1w':
        pattern = '*T1w*'
        pattern2 = '*T1w*preproc*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        localizer_files = fnmatch.filter(localizer_files,pattern2)
        
    brain_mask_path.sort()
    localizer_files.sort()
    if mask_flag=='vtc':
        vtc_mask_path=os.path.join('/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/',sub,'new_mask','VVS_preremoval_%s_mask.nii.gz' % brain_flag)
        
        vtc_mask=nib.load(vtc_mask_path)
    


    if os.path.exists(os.path.join(bold_path,'sub-0%s_%s_preremoval_%s_masked.npy' % (sub_num,brain_flag,mask_flag))):
        
        localizer_bold=np.load(os.path.join(bold_path,'sub-0%s_%s_preremoval_%s_masked.npy' % (sub_num,brain_flag,mask_flag)))
        print('%s %s Localizer Loaded...' % (brain_flag,mask_flag))
        run1_length=int((len(localizer_bold)/6))
        run2_length=int((len(localizer_bold)/6))
        run3_length=int((len(localizer_bold)/6))
        run4_length=int((len(localizer_bold)/6))
        run5_length=int((len(localizer_bold)/6))
        run6_length=int((len(localizer_bold)/6)) 
    else:
        #select the specific file
        localizer_run1=nib.load(localizer_files[0])
        localizer_run2=nib.load(localizer_files[1])
        localizer_run3=nib.load(localizer_files[2])
        localizer_run4=nib.load(localizer_files[3])
        localizer_run5=nib.load(localizer_files[4])
        localizer_run6=nib.load(localizer_files[5])          
          
        #to be used to filter the data
        #First we are removing the confounds
        #get all the folders within the bold path
        #confound_folders=[x[0] for x in os.walk(bold_path)]
        localizer_confounds_1=find('*preremoval*1*confounds*.tsv',bold_path)
        localizer_confounds_2=find('*preremoval*2*confounds*.tsv',bold_path)
        localizer_confounds_3=find('*preremoval*3*confounds*.tsv',bold_path)
        localizer_confounds_4=find('*preremoval*4*confounds*.tsv',bold_path)
        localizer_confounds_5=find('*preremoval*5*confounds*.tsv',bold_path)
        localizer_confounds_6=find('*preremoval*6*confounds*.tsv',bold_path)

        
        confound_run1 = pd.read_csv(localizer_confounds_1[0],sep='\t')
        confound_run2 = pd.read_csv(localizer_confounds_2[0],sep='\t')
        confound_run3 = pd.read_csv(localizer_confounds_3[0],sep='\t')
        confound_run4 = pd.read_csv(localizer_confounds_4[0],sep='\t')
        confound_run5 = pd.read_csv(localizer_confounds_5[0],sep='\t')
        confound_run6 = pd.read_csv(localizer_confounds_6[0],sep='\t')            

        confound_run1=confound_run1.fillna(confound_run1.mean())
        confound_run2=confound_run2.fillna(confound_run2.mean())
        confound_run3=confound_run3.fillna(confound_run3.mean())
        confound_run4=confound_run4.fillna(confound_run4.mean())
        confound_run5=confound_run5.fillna(confound_run5.mean())
        confound_run6=confound_run6.fillna(confound_run6.mean())                                    

        wholebrain_mask1=nib.load(brain_mask_path[0])
        wholebrain_mask2=nib.load(brain_mask_path[1])
        wholebrain_mask3=nib.load(brain_mask_path[2])
        wholebrain_mask4=nib.load(brain_mask_path[3])
        wholebrain_mask5=nib.load(brain_mask_path[4])
        wholebrain_mask6=nib.load(brain_mask_path[5])

        
        def apply_mask(mask=None,target=None):
            coor = np.where(mask == 1)
            values = target[coor]
            if values.ndim > 1:
                values = np.transpose(values) #swap axes to get feature X sample
            return values

        if mask_flag=='wholebrain':
            localizer_run1=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run1.get_data()))
            localizer_run2=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run2.get_data()))
            localizer_run3=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run3.get_data()))
            localizer_run4=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run4.get_data()))
            localizer_run5=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run5.get_data()))
            localizer_run6=apply_mask(mask=(wholebrain_mask1.get_data()),target=(localizer_run6.get_data()))                        

        elif mask_flag=='vtc':
            localizer_run1=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run1.get_data()))
            localizer_run2=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run2.get_data()))
            localizer_run3=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run3.get_data()))
            localizer_run4=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run4.get_data()))
            localizer_run5=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run5.get_data()))   
            localizer_run6=apply_mask(mask=(vtc_mask.get_data()),target=(localizer_run6.get_data()))   

        preproc_1 = clean(localizer_run1,t_r=1,detrend=False,standardize='zscore')
        preproc_2 = clean(localizer_run2,t_r=1,detrend=False,standardize='zscore')
        preproc_3 = clean(localizer_run3,t_r=1,detrend=False,standardize='zscore')
        preproc_4 = clean(localizer_run4,t_r=1,detrend=False,standardize='zscore')
        preproc_5 = clean(localizer_run5,t_r=1,detrend=False,standardize='zscore')
        preproc_6 = clean(localizer_run6,t_r=1,detrend=False,standardize='zscore')                       


        localizer_bold=np.concatenate((preproc_1,preproc_2,preproc_3,preproc_4,preproc_5,preproc_6))
        #save this data if we didn't have it saved before
        os.chdir(bold_path)
        np.save('sub-0%s_%s_preremoval_%s_masked' % (sub_num,brain_flag,mask_flag), localizer_bold)
        print('%s %s masked data...saved' % (mask_flag,brain_flag))
    #create run array
        run1_length=int((len(localizer_run1)))
        run2_length=int((len(localizer_run2)))
        run3_length=int((len(localizer_run3)))
        run4_length=int((len(localizer_run4)))
        run5_length=int((len(localizer_run5)))
        run6_length=int((len(localizer_run6)))            
    #fill in the run array with run number
    run1=np.full(run1_length,1)
    run2=np.full(run2_length,2)
    run3=np.full(run3_length,3)
    run4=np.full(run4_length,4)
    run5=np.full(run5_length,5)    
    run6=np.full(run6_length,6)    

    run_list=np.concatenate((run1,run2,run3,run4,run5,run6)) #now combine

    #load regs / labels

    #Categories: 1 Scenes, 2 Faces 

    params_dir='/scratch1/06873/zbretton/repclear_dataset/BIDS/params'
    #find the mat file, want to change this to fit "sub"
    param_search='preremoval*events*.csv'
    param_file=find(param_search,params_dir)
    # if os.path.exists(os.path.join(params_dir,'preremoval_category_array.txt')):#look for the saved stim_list
    #     os.chdir(params_dir)
    #     with open('preremoval_category_array.txt', 'rb') as fp:
    #         stim_list=pickle.load(fp)
    #     print('Stim-Category List Loaded')#load it in if it exists
    #     with open('preremoval_category_on_array.txt', 'rb') as fp:
    #         stim_on=pickle.load(fp)
    #     print('Stim-Category List Loaded')#load it in if it exists            
    # else: #if not we need to create the stim list
    # #gotta index to 0 since it's a list
    reg_matrix = pd.read_csv(param_file[0])
    reg_category=reg_matrix["category"].values
    reg_stim_on=reg_matrix["stim_present"].values
    reg_run=reg_matrix["run"].values

    run1_index=np.where(reg_run==1)
    run2_index=np.where(reg_run==2)
    run3_index=np.where(reg_run==3)
    run4_index=np.where(reg_run==4)
    run5_index=np.where(reg_run==5)
    run6_index=np.where(reg_run==6)            

    stim1_index=len(run1)
    stim2_index=(stim1_index+len(run2))
    stim3_index=stim2_index+len(run3)
    stim4_index=stim3_index+len(run4)
    stim5_index=stim4_index+len(run5)

#extract times where stimuli is on for both categories:
    #stim_on=np.where((reg_stim_on==1) & ((reg_category==1) | (reg_category==2)))
    stim_on=reg_stim_on
#need to convert this list to 1-d
    stim_list=np.empty(len(localizer_bold))
    stim_list=reg_category
#this list is now 1d list, need to add a dimentsionality to it
    stim_list=stim_list[:,None]
    stim_on=stim_on[:,None]
    # os.chdir(params_dir)
        # with open('preremoval_category_array.txt', 'wb') as fp:
        #     pickle.dump(stim_list,fp)
        # print('Stim-Category List saved')
        # with open('preremoval_category_on_array.txt', 'wb') as fp:
        #     pickle.dump(stim_on,fp)
        # print('Stim-Category-ON List saved')            

        # Create a function to shift the size, and will do the rest tag
    def shift_timing(label_TR, TR_shift_size, tag):
        # Create a short vector of extra zeros or whatever the rest label is
        zero_shift = np.full(TR_shift_size, tag)
        # Zero pad the column from the top
        zero_shift = np.vstack(zero_shift)
        label_TR_shifted = np.vstack((zero_shift, label_TR))
        # Don't include the last rows that have been shifted out of the time line
        label_TR_shifted = label_TR_shifted[0:label_TR.shape[0],0]
      
        return label_TR_shifted

# Apply the function
    shift_size = TR_shift #this is shifting by 10TR
    tag = 0 #rest label is 0
    stim_list_shift = shift_timing(stim_list, shift_size, tag) #rest is label 0
    stim_on_shift= shift_timing(stim_on, shift_size, tag)
    import random
    

    #Here I need to balance the trials of the categories / rest. I will be removing rest, but Scenes have twice the data of faces, so need to decide how to handle
    rest_times=np.where(stim_list_shift==0)
    rest_times_on=np.where(stim_on_shift==0)
    #these should be the same, but doing it just in case


    #rest_times_filt=np.concatenate((rest_p1,rest_p2,rest_p3,rest_p4,rest_p5))
    #rest_times_index=rest_times[rest_times_filt]


    # def Diff(li1,li2):
    #     return (list(set(li1)-set(li2)))
    # tr_to_remove=Diff(rest_times,stim_list_shift)

    #currently doing this like a boxcar style, but over the course of the runs. So once the data was shifted and then rest times removed
    #I am taking the rest of the data to represent that category

    #An alternative would be filtering based on the stimuli being only on (those 2 TR, which is filtered by stim_on, but that need to be shifted as well)
    stim_list_nr=np.delete(stim_list_shift, rest_times)
    stim_list_nr=stim_list_nr.flatten()
    stim_list_nr=stim_list_nr[:,None]
    localizer_bold_nr=np.delete(localizer_bold, rest_times, axis=0)
    run_list_nr=np.delete(run_list, rest_times)

    #sorted only for stimuli being on
    stim_on_filt=stim_list_shift[np.where((stim_on_shift==1) & ((stim_list_shift==1) | (stim_list_shift==2)))]
    stim_on_filt=stim_on_filt.flatten()
    stim_on_filt=stim_on_filt[:,None]
    localizer_bold_on_filt=localizer_bold[np.where((stim_on_shift==1) & ((stim_list_shift==1) | (stim_list_shift==2)))]
    run_list_on_filt=run_list[np.where((stim_on_shift==1) & ((stim_list_shift==1) | (stim_list_shift==2)))]


    bold_files=find('*study*bold*.nii.gz',bold_path)
    wholebrain_mask_path=find('*study*mask*.nii.gz',bold_path)
    pattern = '*T1w*'
    pattern2 = '*T1w*preproc*'
    brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
    study_files = fnmatch.filter(bold_files,pattern2)

    brain_mask_path.sort()
    study_files.sort()


    if os.path.exists(os.path.join(bold_path,'sub-0%s_%s_study_vtc_masked.npy' % (sub_num,brain_flag))):
        
        study_bold=np.load(os.path.join(bold_path,'sub-0%s_%s_study_vtc_masked.npy' % (sub_num,brain_flag)))
        print('%s wholebrain study Loaded...' % (brain_flag))
        run1_length=int((len(study_bold)/3))
        run2_length=int((len(study_bold)/3))
        run3_length=int((len(study_bold)/3))           
    else:
        #select the specific file
        study_run1=nib.load(study_files[0])
        study_run2=nib.load(study_files[1])
        study_run3=nib.load(study_files[2])

        #to be used to filter the data
        #First we are removing the confounds
        #get all the folders within the bold path
        #confound_folders=[x[0] for x in os.walk(bold_path)]
        study_confounds_1=find('*study*1*confounds*.tsv',bold_path)
        study_confounds_2=find('*study*2*confounds*.tsv',bold_path)
        study_confounds_3=find('*study*3*confounds*.tsv',bold_path)
        
        confound_run1 = pd.read_csv(study_confounds_1[0],sep='\t')
        confound_run2 = pd.read_csv(study_confounds_2[0],sep='\t')
        confound_run3 = pd.read_csv(study_confounds_3[0],sep='\t')

        wholebrain_mask1=nib.load(brain_mask_path[0])
        
        def apply_mask(mask=None,target=None):
            coor = np.where(mask == 1)
            values = target[coor]
            if values.ndim > 1:
                values = np.transpose(values) #swap axes to get feature X sample
            return values
        if mask_flag=='wholebrain':
            study_run1=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run1.get_data()))
            study_run2=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run2.get_data()))
            study_run3=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run3.get_data()))                    

        elif mask_flag=='vtc':
            whole_study_run1=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run1.get_data()))
            whole_study_run2=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run2.get_data()))
            whole_study_run3=apply_mask(mask=(wholebrain_mask1.get_data()),target=(study_run3.get_data()))

            study_run1=apply_mask(mask=(vtc_mask.get_data()),target=(study_run1.get_data()))
            study_run2=apply_mask(mask=(vtc_mask.get_data()),target=(study_run2.get_data()))
            study_run3=apply_mask(mask=(vtc_mask.get_data()),target=(study_run3.get_data()))

        preproc_1 = clean(study_run1,t_r=1,detrend=False,standardize='zscore')
        preproc_2 = clean(study_run2,t_r=1,detrend=False,standardize='zscore')
        preproc_3 = clean(study_run3,t_r=1,detrend=False,standardize='zscore')


        study_bold=np.concatenate((preproc_1,preproc_2,preproc_3))
        # whole_study_bold=np.concatenate((whole_preproc_1,whole_preproc_2,whole_preproc_3,whole_preproc_4,whole_preproc_5,whole_preproc_6))
        #save this data if we didn't have it saved before
        os.chdir(bold_path)
        np.save('sub-0%s_%s_study_vtc_masked' % (sub_num,brain_flag), study_bold)
        print('%s %s masked data...saved' % (mask_flag,brain_flag))

    params_dir='/scratch1/06873/zbretton/repclear_dataset/BIDS/params'
    #find the mat file, want to change this to fit "sub"
    param_search='study*events*.csv'
    param_file=find(param_search,params_dir)
    # if os.path.exists(os.path.join(params_dir,'study_stim_array.txt')):#look for the saved stim_list
    #     os.chdir(params_dir)
    #     with open('study_stim_array.txt', 'rb') as fp:
    #         stim_list=pickle.load(fp)
    #     print('Stim List Loaded')#load it in if it exists
    #     with open('study_oper_array.txt', 'rb') as fp:
    #         oper_list=pickle.load(fp)
    #     print('Operation List Loaded')#load it in if it exists            
    # else: #if not we need to create the stim list
    #gotta index to 0 since it's a list
    study_matrix = pd.read_csv(param_file[0])
    study_operation=study_matrix["condition"].values
    study_run=study_matrix["run"].values
    study_present=study_matrix["stim_present"].values

    run1_index=np.where(study_run==1)
    run2_index=np.where(study_run==2)
    run3_index=np.where(study_run==3)
          
#need to convert this list to 1-d                           
#this list is now 1d list, need to add a dimentsionality to it
#Condition:
#1. maintain
#2. replace_category
#3. suppress
    study_stim_list=np.full(len(study_bold),0)
    maintain_list=np.where((study_operation==1) & ((study_present==1) |(study_present==2) | (study_present==3)))
    suppress_list=np.where((study_operation==3) & ((study_present==1) |(study_present==2) | (study_present==3)))
    replace_list=np.where((study_operation==2) & ((study_present==1) |(study_present==2) | (study_present==3)))
    study_stim_list[maintain_list]=1
    study_stim_list[suppress_list]=3
    study_stim_list[replace_list]=2

    oper_list=study_operation
    oper_list=oper_list[:,None]

    study_stim_list=study_stim_list[:,None]            

    #do a L2 estimator
    def CLF(train_data, train_labels, test_data, test_labels, k_best):
        scores = []
        predicts = []
        trues = []
        decisions=[]
        evidences=[]
        clf=LinearSVC()
        X_train, X_test = train_data, test_data
        y_train, y_test = train_labels, test_labels

        selectedvoxels=SelectKBest(f_classif,k=1500).fit(X_train,y_train)
        #selectedvoxels=SelectFpr(f_classif,alpha=0.05).fit(X_train,y_train) #I compared this method to taking ALL k items in the F-test and filtering by p-value, so I assume this is a better feature selection method

        X_train=selectedvoxels.transform(X_train)
        X_test=selectedvoxels.transform(X_test)

        # fit the model
        clf.fit(X_train, y_train)
        
        #output decision values
        decisions=clf.decision_function(X_test)

        evidence=(1. / (1. + np.exp(-clf.decision_function(X_test))))
        evidences.append(evidence)
        
        # score the model, but we care more about values
        score=clf.score(X_test, y_test)
        predict = clf.predict(X_test)
        predicts.append(predict)
        true = y_test
        scores.append(score)
        trues.append(true)
        return clf, scores, predicts, trues, decisions, evidence

    L2_models_nr, L2_scores_nr, L2_predicts_nr, L2_trues_nr, L2_decisions_nr, L2_evidence_nr = CLF(localizer_bold_on_filt, stim_on_filt, study_bold, study_stim_list, 1500)
    L2_subject_score_nr_mean = np.mean(L2_scores_nr)                                        

    np.savetxt("train_local_test_study_category_evidence_kbest.csv",L2_evidence_nr, delimiter=",")
    np.savetxt("train_local_test_study_category_decisions_kbest.csv",L2_decisions_nr, delimiter=",")    
    np.savetxt("Operation_labels.csv",study_stim_list, delimiter=",")

    output_table = {
        "subject" : sub,

        "CLF Average Scores from Testing" : L2_subject_score_nr_mean,
        "CLF Model Testing" : L2_models_nr,
        "CLF Model Decisions" : L2_decisions_nr,

        "CLF Category Evidece" : L2_evidence_nr,

        "CLF Operation Trues" : L2_trues_nr,

        "Category List Shifted w/o Rest" : stim_on_filt,
        "Operation List": study_stim_list,
        
        "Localizer Shifted w/o Rest": localizer_bold_on_filt,
        
        }
    
    import pickle
    os.chdir(os.path.join(container_path,sub))
    f = open("%s-train_local_test_study_%s_%sTR lag_data_kbest.pkl" % (sub,brain_flag,TR_shift),"wb")
    pickle.dump(output_table,f)
    f.close()    