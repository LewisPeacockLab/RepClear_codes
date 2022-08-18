#This code is to load in the representations and then either weight them (category vs. item) and then perform RSA
# This will also handle performing this and then comparing Pre-Localizer to Study, Pre-Localizer to Post-Localizer, and Study to Post-Localizer 

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


subs=['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','20','23','24','25','26']
brain_flag='T1w'
stim_labels = {0: "Rest",
                1: "Scenes",
                2: "Faces"}
sub_cates = {
            #"face": ["male", "female"],         #60
            "scene": ["manmade", "natural"],    #120
            } #getting rid of faces for now so I can focus on scenes

def mkdir(path,local=False):
    if not os.path.exists(path):
        os.makedirs(path)

def apply_mask(mask=None,target=None):
    coor = np.where(mask == 1)
    values = target[coor]
    if values.ndim > 1:
        values = np.transpose(values) #swap axes to get feature X sample
    return values   

def find(pattern, path): #find the pattern we're looking for
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result         


container_path='/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep'
param_dir =  '/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs'

#the subject's list of image number to trial numbers are in the "subject_designs" folder

for subID in subs:
    print('Running sub-0%s...' %subID)
    #define the subject
    sub = ('sub-0%s' % subID)

    """
    Input: 
    subID: 3 digit string
    phase: single digit int. 1: "pre-exposure", 2: "pre-localizer", 3: "study", 4: "post-localizer"

    Output: 
    face_order: trial numbers ordered by ["female", "male"]. 
    scene_order: trial numbers ordered by ["manmade", "natural"]

    (It's hard to sort the RDM matrix once that's been computed, so the best practice would be to sort the input to MDS before we run it)
    """
    #lets pull out the pre-localizer data here:
    tim_path = os.path.join(param_dir, f"sub-0{subID}_trial_image_match.csv")

    tim_df = pd.read_csv(tim_path)
    tim_df = tim_df[tim_df["phase"]==2] #phase 2 is pre-localizer
    tim_df = tim_df.sort_values(by=["category", "subcategory", "trial_id"])
    
    pre_scene_order = tim_df[tim_df["category"]==1][["trial_id","image_id","condition","subcategory"]]
    pre_face_order = tim_df[tim_df["category"]==2][["trial_id","image_id","condition"]]   

    #lets pull out the study data here:
    tim_df2 = pd.read_csv(tim_path)
    tim_df2 = tim_df2[tim_df2["phase"]==3] #phase 3 is study
    tim_df2 = tim_df2.sort_values(by=["category", "subcategory", "trial_id"])
    
    study_scene_order = tim_df2[tim_df2["category"]==1][["trial_id","image_id","condition","subcategory"]]


    print(f"Running RSA - Proactive Interference (Version 1 & 2) analysis for sub0{subID}...")

    # ===== load mask for BOLD
    if brain_flag=='MNI':

        mask_path = os.path.join(container_path, "group_MNI_VTC_mask.nii.gz")
    else:
        mask_path=os.path.join('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/',sub,'new_mask','VVS_preremoval_%s_mask.nii.gz' % brain_flag)
    
    mask = nib.load(mask_path)

    print("mask shape: ", mask.shape)

    # ===== load ready BOLD for each trial of prelocalizer
    print(f"Loading preprocessed BOLDs for pre-localizer...")
    bold_dir_1 = os.path.join(container_path, f"sub-0{subID}", "item_representations_%s" % brain_flag)

    all_bolds_1 = {}  # {cateID: {trialID: bold}}
    bolds_arr_1 = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames_1 = glob.glob(f"{bold_dir_1}/*pre*{cateID}*")
        cate_bolds_1 = {}
        
        for fname in cate_bolds_fnames_1:
            trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
            trialID = int(trialID[5:])
            cate_bolds_1[trialID] = nib.load(fname).get_fdata()  #.flatten()
        cate_bolds_1 = {i: cate_bolds_1[i] for i in sorted(cate_bolds_1.keys())}
        all_bolds_1[cateID] = cate_bolds_1

        bolds_arr_1.append(np.stack( [ cate_bolds_1[i] for i in sorted(cate_bolds_1.keys()) ] ))

    bolds_arr_1 = np.vstack(bolds_arr_1)
    print("bolds for prelocalizer - shape: ", bolds_arr_1.shape)

    # ===== load ready BOLD for each trial of study
    print(f"Loading preprocessed BOLDs for the study operation...")
    bold_dir_2 = os.path.join(container_path, f"sub-0{subID}", "item_representations_%s" % brain_flag)

    all_bolds_2 = {}  # {cateID: {trialID: bold}}
    bolds_arr_2 = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames_2 = glob.glob(f"{bold_dir_2}/*study_{cateID}*")
        cate_bolds_2 = {}
        try:
            for fname in cate_bolds_fnames_2:
                trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
                trialID = int(trialID[5:])
                cate_bolds_2[trialID] = nib.load(fname).get_fdata()  #.flatten()
            cate_bolds_2 = {i: cate_bolds_2[i] for i in sorted(cate_bolds_2.keys())}
            all_bolds_2[cateID] = cate_bolds_2

            bolds_arr_2.append(np.stack( [ cate_bolds_2[i] for i in sorted(cate_bolds_2.keys()) ] ))


        except:
            print('no %s trials' % cateID)
    bolds_arr_2 = np.vstack(bolds_arr_2)
    #this now means that the study data is a stack of arrays, from trial 1 to trial 90 in order
    print("bolds for study - shape: ", bolds_arr_2.shape)

    # apply VTC mask on prelocalizer BOLD
    masked_bolds_arr_1 = []
    for bold in bolds_arr_1:
        masked_bolds_arr_1.append(apply_mask(mask=mask.get_fdata(), target=bold).flatten())
    masked_bolds_arr_1 = np.vstack(masked_bolds_arr_1)
    print("masked prelocalizer bold array shape: ", masked_bolds_arr_1.shape)

    # apply mask on study BOLD
    masked_bolds_arr_2 = []
    for bold in bolds_arr_2:
        masked_bolds_arr_2.append(apply_mask(mask=mask.get_fdata(), target=bold).flatten())
    masked_bolds_arr_2 = np.vstack(masked_bolds_arr_2)
    print("masked study bold array shape: ", masked_bolds_arr_2.shape)  


    sorted_study_scene_order=study_scene_order.sort_values(by=['trial_id']) #this organizes the trials based on the trial_id, since we will need this to assess the condition on the N-1 trial to sort


    ####### FOR NOW WE ARE DROPPING THE WEIGHTS AND WILL ADD THEM BACK IN LATER #############
    # # ===== load weights
    # print(f"Loading weights...")
    # # prelocalizer
    # if brain_flag=='MNI':
    #     cate_weights_dir = os.path.join(f'/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-0{subID}','preremoval_lvl1_%s/scene_stimuli_MNI_zmap.nii.gz' % brain_flag)
    #     item_weights_dir = os.path.join(container_path, f"sub-0{subID}", "preremoval_item_level_MNI")
    # else:
    #     cate_weights_dir = os.path.join(f'/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-0{subID}','preremoval_lvl1_%s/scene_stimuli_T1w_zmap.nii.gz' % brain_flag)
    #     item_weights_dir = os.path.join(container_path, f"sub-0{subID}", "preremoval_item_level_T1w")        

    # #prelocalizer weights (category and item) get applied to study/post representations

    # all_weights={}
    # weights_arr=[]

    # #load in all the item specific weights, which come from the LSA contrasts per subject
    # for cateID in sub_cates.keys():
    #     item_weights_fnames = glob.glob(f"{item_weights_dir}/{cateID}*full*zmap*")
    #     print(cateID, len(item_weights_fnames))
    #     item_weights = {}

    #     for fname in item_weights_fnames:
    #         trialID = fname.split("/")[-1].split("_")[1]  # "trial1"
    #         trialID = int(trialID[5:])
    #         item_weights[trialID] = nib.load(fname).get_fdata()
    #     item_weights = {i: item_weights[i] for i in sorted(item_weights.keys())}
    #     all_weights[cateID] = item_weights  
    #     weights_arr.append(np.stack( [ item_weights[i] for i in sorted(item_weights.keys()) ] ))

    # #this now masks the item weights to ensure that they are all in the same ROI (group VTC):
    # weights_arr = np.vstack(weights_arr)
    # print("weights shape: ", weights_arr.shape)
    # # apply mask on BOLD
    # masked_weights_arr = []
    # for weight in weights_arr:
    #     masked_weights_arr.append(apply_mask(mask=mask.get_fdata(), target=weight).flatten())
    # masked_weights_arr = np.vstack(masked_weights_arr)
    # print("masked item weights arr shape: ", masked_weights_arr.shape)          

    # #this masks the category weight from the group results:
    # cate_weights_arr = nib.load(cate_weights_dir)
    # masked_cate_weights_arr = apply_mask(mask=mask.get_fdata(), target=cate_weights_arr.get_fdata()).flatten()
    # print("masked category weight arr shape: ", masked_cate_weights_arr.shape) #so this is just 1D of the voxels in the VTC mask

    # # ===== multiply
    # #prelocalizer patterns and prelocalizer item weights
    # item_repress_pre = np.multiply(masked_bolds_arr_1, masked_weights_arr) #these are lined up since the trials goes to correct trials
    
    # #study patterns, prelocalizer item weights and postlocalizer    
    # #this has to come in the loop below to make sure I am weighting the correct trial with the correct item weights

    # print("item representations pre shape: ", item_repress_pre.shape)

    # #these work right out the box since there is only 1 "category" weighting we are using, and that can be applied to all scene trials in both pre and study (and post)
    # cate_repress_pre = np.multiply(masked_bolds_arr_1,masked_cate_weights_arr) #these are multiplied elementwise
    # cate_repress_study = np.multiply(masked_bolds_arr_2,masked_cate_weights_arr) #since there is only 1 cate_weight, this multiplies all of masked_bold_arr_2 with the cate_weights
    # cate_repress_post = np.multiply(masked_bolds_arr_3,masked_cate_weights_arr) #weight the post representations with category weights

    # print("category representations pre shape: ", cate_repress_pre.shape)
    # print("category representations study shape: ", cate_repress_study.shape)
    # print("category representations post shape: ", cate_repress_post.shape)


    #the way the data is currently sorted is by the index:
    #pre-localizer: 0-59 = Face trial 1 - 60 | 60-179 = Scene trial 1 - 120
    #study: 0-89 = Scene trial 1 - 90
    #post-localizer: 0-179 = Scene trial 1-180 (but 60 of these are novel)

    #the specific output of this "order" DataFrame is by subcategory within the category, but we can still use this to sort by trial since the conditions are in order

    #key arrays being used: pre_scene_order (which is the order of the images in the prelocalizer, after sorted for subcate)
                            #study_scene_order (again sorted for subcate)
   

    item_repress_study_comp=np.zeros_like(masked_bolds_arr_1[:90,:]) #set up this array in the same size as the pre, so I can size things in the correct trial order
    item_repress_pre_comp=np.zeros_like(item_repress_study_comp)

    non_weight_study_comp=np.zeros_like(item_repress_study_comp)
    non_weight_pre_comp=np.zeros_like(item_repress_study_comp)


    #these are used to hold the fidelity changes from pre to study (item-weighted)
    iw_maintain_dict={}  
    iw_replace_dict={}
    iw_suppress_dict={}

    #these are used to hold the fidelity changes from pre to study (unweighted)
    uw_maintain_dict={}  
    uw_replace_dict={}
    uw_suppress_dict={}  

    counter=0

    m_counter=0
    s_counter=0
    r_counter=0

    m_item_repress_study_comp=np.zeros_like(masked_bolds_arr_1[:30,:])
    m_item_repress_pre_comp=np.zeros_like(masked_bolds_arr_1[:30,:])

    r_item_repress_study_comp=np.zeros_like(masked_bolds_arr_1[:30,:])
    r_item_repress_pre_comp=np.zeros_like(masked_bolds_arr_1[:30,:])

    s_item_repress_study_comp=np.zeros_like(masked_bolds_arr_1[:30,:])
    s_item_repress_pre_comp=np.zeros_like(masked_bolds_arr_1[:30,:])



    for trial in sorted_study_scene_order['trial_id'].values: 

        #Bolds_arr_2 is in trial order
        study_image_id=sorted_study_scene_order[sorted_study_scene_order['trial_id']==trial]['image_id'].tolist()[0] #this now uses the trial_id (study) to find the image_id (study)
        if (trial==1) or (trial==31) or (trial==61):
            continue
        image_condition=sorted_study_scene_order[sorted_study_scene_order['trial_id']==(trial-1)]['condition'].tolist()[0]

        pre_trial_index=pre_scene_order.index[pre_scene_order['image_id']==study_image_id].tolist()[0] #find the index in the pre for this study trial
        pre_trial_num = pre_scene_order.loc[pre_trial_index,'trial_id']
        pre_trial_subcat = pre_scene_order.loc[pre_trial_index,'subcategory']

        if image_condition==1:
            m_item_repress_study_comp[m_counter]=masked_bolds_arr_2[trial-1,:]
            m_item_repress_pre_comp[m_counter]=masked_bolds_arr_1[pre_trial_num-1,:]
            m_counter=m_counter+1
        elif image_condition==2:
            r_item_repress_study_comp[r_counter]=masked_bolds_arr_2[trial-1,:]
            r_item_repress_pre_comp[r_counter]=masked_bolds_arr_1[pre_trial_num-1,:]
            r_counter=r_counter+1    
        elif image_condition==3:
            s_item_repress_study_comp[s_counter]=masked_bolds_arr_2[trial-1,:]
            s_item_repress_pre_comp[s_counter]=masked_bolds_arr_1[pre_trial_num-1,:]
            s_counter=s_counter+1                  

    #so based on the design, and sorting by the operation on the previous trials we are left with 30 suppress trials, 29 replace and 28 maintain
    # for now I am going to trim these all to 28 just so that theyre all even, though I dont think thats completely necessary
    m_item_repress_study_comp=m_item_repress_study_comp[:28]

    m_item_repress_pre_comp=m_item_repress_pre_comp[:28]

    r_item_repress_study_comp=r_item_repress_study_comp[:28]

    r_item_repress_pre_comp=r_item_repress_pre_comp[:28]

    s_item_repress_study_comp=s_item_repress_study_comp[:28]

    s_item_repress_pre_comp=s_item_repress_pre_comp[:28]

    #also want to combine across the conditions so we can look at corr across all items and not just same operation

    all_item_repress_study_comp=np.vstack((m_item_repress_study_comp,r_item_repress_study_comp,s_item_repress_study_comp))
    all_item_repress_pre_comp=np.vstack((m_item_repress_pre_comp,r_item_repress_pre_comp,s_item_repress_pre_comp))

    all_item_pre_study_comp=np.corrcoef(all_item_repress_pre_comp,all_item_repress_study_comp)
    all_uw_proactive_interference=np.zeros(84)

    all_uw_maintain_target_RSA=np.zeros(28)
    all_uw_replace_target_RSA=np.zeros(28)
    all_uw_suppress_target_RSA=np.zeros(28)

    trials=list(range(0,84))
    for i in trials:
        index_interest=i+84
        temp_same=all_item_pre_study_comp[i][index_interest]
        diff_array=np.append((all_item_pre_study_comp[i][84:index_interest]), (all_item_pre_study_comp[i][index_interest+1:]))
        temp_different=diff_array.mean()
        pro_intr=temp_same-temp_different
        all_uw_proactive_interference[i]=pro_intr


    all_pro_inter_df=pd.DataFrame(data=all_uw_proactive_interference,columns=['Fidelity'])
    all_pro_inter_df['Operation']=np.repeat(['Maintain','Replace','Suppress'],28)    

    m_item_pre_study_comp=np.corrcoef(m_item_repress_pre_comp,m_item_repress_study_comp)
    m_uw_proactive_interference=np.zeros(28)
    trials=list(range(0,28))
    for i in trials:
        index_interest=i+28
        temp_same=m_item_pre_study_comp[i][index_interest]
        diff_array=np.append((m_item_pre_study_comp[i][28:index_interest]), (m_item_pre_study_comp[i][index_interest+1:]))
        temp_different=diff_array.mean()
        pro_intr=temp_same-temp_different
        m_uw_proactive_interference[i]=pro_intr
        all_uw_maintain_target_RSA[i]=temp_same        

    r_item_pre_study_comp=np.corrcoef(r_item_repress_pre_comp,r_item_repress_study_comp)
    r_uw_proactive_interference=np.zeros(28)
    trials=list(range(0,28))
    for i in trials:
        index_interest=i+28
        temp_same=r_item_pre_study_comp[i][index_interest]
        diff_array=np.append((r_item_pre_study_comp[i][28:index_interest]), (r_item_pre_study_comp[i][index_interest+1:]))
        temp_different=diff_array.mean()
        pro_intr=temp_same-temp_different
        r_uw_proactive_interference[i]=pro_intr        
        all_uw_replace_target_RSA[i]=temp_same        

    s_item_pre_study_comp=np.corrcoef(s_item_repress_pre_comp,s_item_repress_study_comp)
    s_uw_proactive_interference=np.zeros(28)
    trials=list(range(0,28))
    for i in trials:
        index_interest=i+28
        temp_same=s_item_pre_study_comp[i][index_interest]
        diff_array=np.append((s_item_pre_study_comp[i][28:index_interest]), (s_item_pre_study_comp[i][index_interest+1:]))
        temp_different=diff_array.mean()
        pro_intr=temp_same-temp_different        
        s_uw_proactive_interference[i]=temp_different   
        all_uw_suppress_target_RSA[i]=temp_same

    temp_df=pd.DataFrame(data=m_uw_proactive_interference,columns=['Fidelity'])
    temp_df['Sub']=subID
    if not os.path.exists(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag)): os.makedirs(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag),exist_ok=True)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag,'maintain_proactive_interference_unweighted.csv'))
    del temp_df   

    temp_df=pd.DataFrame(data=r_uw_proactive_interference,columns=['Fidelity'])
    temp_df['Sub']=subID
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag,'replace_proactive_interference_unweighted.csv'))
    del temp_df   

    temp_df=pd.DataFrame(data=s_uw_proactive_interference,columns=['Fidelity'])
    temp_df['Sub']=subID
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag,'suppress_proactive_interference_unweighted.csv'))
    del temp_df             

    temp_df=pd.DataFrame(data=all_uw_maintain_target_RSA,columns=['Fidelity'])
    temp_df['Sub']=subID
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag,'maintain_unweighted_target_RSA.csv'))
    del temp_df   

    temp_df=pd.DataFrame(data=all_uw_replace_target_RSA,columns=['Fidelity'])
    temp_df['Sub']=subID
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag,'replace_unweighted_target_RSA.csv'))
    del temp_df   

    temp_df=pd.DataFrame(data=all_uw_suppress_target_RSA,columns=['Fidelity'])
    temp_df['Sub']=subID
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag,'suppress_unweighted_target_RSA.csv'))
    del temp_df  

    all_pro_inter_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag,'all_proactive_interference_unweighted.csv'))
    all_pro_inter_mean_df=all_pro_inter_df.groupby(by='Operation').mean()
    all_pro_inter_mean_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag,'all_proactive_interference_mean_unweighted.csv'))

    temp_df=pd.DataFrame(data=m_item_pre_study_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag,'maintain_corrcoef_unweighted.csv'))
    del temp_df   

    temp_df=pd.DataFrame(data=r_item_pre_study_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag,'replace_corrcoef_unweighted.csv'))
    del temp_df   

    temp_df=pd.DataFrame(data=s_item_pre_study_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag,'suppress_corrcoef_unweighted.csv'))
    del temp_df 



    print("Subject is done... saving everything")
    print("===============================================================================")    

#now I want to take the subject level "fidelity" results and look at it on a group level
group_fidelity=pd.DataFrame()
group_all_fidelity=pd.DataFrame()
group_all_target_RSA=pd.DataFrame()

for subID in subs:
    data_path=os.path.join(container_path,"sub-0%s"  % subID,"Proactive_interference_%s" % brain_flag)

    all_mean_proactive=os.path.join(data_path,'all_proactive_interference_mean_unweighted.csv')
    temp_df=pd.read_csv(all_mean_proactive)
    group_all_fidelity=group_all_fidelity.append(temp_df,ignore_index=True)
    del temp_df

    maintain_proactive=os.path.join(data_path,'maintain_proactive_interference_unweighted.csv')
    temp_df=pd.read_csv(maintain_proactive,usecols=[1])
    temp_df2=temp_df.mean()  
    temp_df2['Operation']='Maintain'  
    group_fidelity=group_fidelity.append(temp_df2,ignore_index=True)
    del temp_df, temp_df2

    replace_proactive=os.path.join(data_path,'replace_proactive_interference_unweighted.csv')
    temp_df=pd.read_csv(replace_proactive,usecols=[1])
    temp_df2=temp_df.mean()  
    temp_df2['Operation']='Replace'  
    group_fidelity=group_fidelity.append(temp_df2,ignore_index=True)
    del temp_df, temp_df2    

    suppress_proactive=os.path.join(data_path,'suppress_proactive_interference_unweighted.csv')
    temp_df=pd.read_csv(suppress_proactive,usecols=[1])
    temp_df2=temp_df.mean()  
    temp_df2['Operation']='Suppress'  
    group_fidelity=group_fidelity.append(temp_df2,ignore_index=True)
    del temp_df, temp_df2        

    maintain_target=os.path.join(data_path,'maintain_unweighted_target_RSA.csv')
    temp_df=pd.read_csv(maintain_target,usecols=[1])
    temp_df2=temp_df.mean()  
    temp_df2['Operation']='Maintain'  
    group_all_target_RSA=group_all_target_RSA.append(temp_df2,ignore_index=True)
    del temp_df, temp_df2

    replace_target=os.path.join(data_path,'replace_unweighted_target_RSA.csv')
    temp_df=pd.read_csv(replace_target,usecols=[1])
    temp_df2=temp_df.mean()  
    temp_df2['Operation']='Replace'  
    group_all_target_RSA=group_all_target_RSA.append(temp_df2,ignore_index=True)
    del temp_df, temp_df2    

    suppress_target=os.path.join(data_path,'suppress_unweighted_target_RSA.csv')
    temp_df=pd.read_csv(suppress_target,usecols=[1])
    temp_df2=temp_df.mean()  
    temp_df2['Operation']='Suppress'  
    group_all_target_RSA=group_all_target_RSA.append(temp_df2,ignore_index=True)
    del temp_df, temp_df2      

if not os.path.exists(os.path.join(container_path,"group_model","Proactive_interference_%s" % brain_flag)): os.makedirs(os.path.join(container_path,"group_model","Proactive_interference_%s" % brain_flag),exist_ok=True)
#plot and save the figures 
fig=sns.barplot(data=group_fidelity,x='Operation',y='Fidelity',ci=95,palette=['green','blue','red'])
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity (Target - Related (within condition))')
fig.set_title('Unweighted (Group Level) - Proactive Interference')  
fig, test_results = add_stat_annotation(fig, data=group_fidelity, x='Operation', y='Fidelity',
                                   box_pairs=[("Maintain", "Replace"), ("Maintain", "Suppress"), ("Suppress", "Replace")],
                                   test='t-test_ind', text_format='star',loc='inside', verbose=2) 
plt.savefig(os.path.join(container_path,"group_model","Proactive_interference_%s" % brain_flag,'Group_unweighted_proactive_interference.png'))
plt.clf() 

fig=sns.barplot(data=group_all_fidelity,x='Operation',y='Fidelity',ci=95,palette=['green','blue','red'])
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity (Target - Related (all other stims))')
fig.set_title('Unweighted (Group Level) - Proactive Interference (compared to all stims)')  
fig, test_results = add_stat_annotation(fig, data=group_all_fidelity, x='Operation', y='Fidelity',
                                   box_pairs=[("Maintain", "Replace"), ("Maintain", "Suppress"), ("Suppress", "Replace")],
                                   test='t-test_ind', text_format='star',loc='inside', verbose=2) 
plt.savefig(os.path.join(container_path,"group_model","Proactive_interference_%s" % brain_flag,'Group_unweighted_proactive_interference_all_stims.png'))
plt.clf() 

fig=sns.barplot(data=group_all_target_RSA,x='Operation',y='Fidelity',ci=95,palette=['green','blue','red'])
fig.set_xlabel('Operations')
fig.set_ylabel('Target-RSA')
fig.set_title('Unweighted (Group Level) - target-RSA')  
fig, test_results = add_stat_annotation(fig, data=group_all_target_RSA, x='Operation', y='Fidelity',
                                   box_pairs=[("Maintain", "Replace"), ("Maintain", "Suppress"), ("Suppress", "Replace")],
                                   test='t-test_ind', text_format='star',loc='inside', verbose=2) 
plt.savefig(os.path.join(container_path,"group_model","Proactive_interference_%s" % brain_flag,'Group_unweighted_target_RSA.png'))
plt.clf() 
