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

    #lets pull out the postlocalizer data here:
    tim_df3 = pd.read_csv(tim_path)
    tim_df3 = tim_df3[tim_df3["phase"]==4] #phase 4 is post-localizer
    tim_df3 = tim_df3.sort_values(by=["category", "subcategory", "trial_id"])

    post_scene_order = tim_df3[tim_df3["category"]==1][["trial_id","image_id","condition"]]


    print(f"Running RSA for sub0{subID}...")

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
    print("bolds for study - shape: ", bolds_arr_2.shape)


    # ===== load ready BOLD for each removal phase of study
    print(f"Loading preprocessed BOLDs for the removal part of study operation...")
    bold_dir_4 = os.path.join(container_path, f"sub-0{subID}", "item_representations_%s" % brain_flag)

    all_bolds_4 = {}  # {cateID: {trialID: bold}}
    bolds_arr_4 = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames_4 = glob.glob(f"{bold_dir_4}/*study_removal_timecourse_{cateID}*")
        cate_bolds_4 = {}
        try:
            for fname in cate_bolds_fnames_4:
                trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
                trialID = int(trialID[5:])
                cate_bolds_4[trialID] = nib.load(fname).get_fdata()  #.flatten()
            cate_bolds_4 = {i: cate_bolds_4[i] for i in sorted(cate_bolds_4.keys())}
            all_bolds_4[cateID] = cate_bolds_4

            bolds_arr_4.append(np.stack( [ cate_bolds_4[i] for i in sorted(cate_bolds_4.keys()) ] ))
        except:
            print('no %s trials' % cateID)
    bolds_arr_4 = np.vstack(bolds_arr_4)
    print("bolds for study - shape: ", bolds_arr_4.shape)    

    # ===== load ready BOLD for each trial of postlocalizer
    print(f"Loading preprocessed BOLDs for post-localizer...")
    bold_dir_3 = os.path.join(container_path, f"sub-0{subID}", "item_representations_%s" % brain_flag)

    all_bolds_3 = {}  # {cateID: {trialID: bold}}
    bolds_arr_3 = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames_3 = glob.glob(f"{bold_dir_1}/*post*{cateID}*")
        cate_bolds_3 = {}
        
        for fname in cate_bolds_fnames_3:
            trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
            trialID = int(trialID[5:])
            cate_bolds_3[trialID] = nib.load(fname).get_fdata()  #.flatten()
        cate_bolds_3 = {i: cate_bolds_3[i] for i in sorted(cate_bolds_3.keys())}
        all_bolds_3[cateID] = cate_bolds_3

        bolds_arr_3.append(np.stack( [ cate_bolds_3[i] for i in sorted(cate_bolds_3.keys()) ] ))

    bolds_arr_3 = np.vstack(bolds_arr_3)
    print("bolds for prelocalizer - shape: ", bolds_arr_3.shape)


    #when comparing pre vs. study, the pre scene have 120 trials, while study has 90 trials
    #when comparing pre vs. post, the pre scene have 120 trials, while post has 180 (120 old, 60 new)

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

    # apply mask on study removal BOLD
    masked_bolds_arr_4 = []
    for bold in bolds_arr_4:
        masked_bolds_arr_4.append(apply_mask(mask=mask.get_fdata(), target=bold))
    masked_bolds_arr_4 = np.dstack(masked_bolds_arr_4)
    print("masked study removal phase bold array shape: ", masked_bolds_arr_4.shape) #with this method, to access the trials you do: masked_bolds_arr_4[:,:,trial#] 

    # apply VTC mask on postlocalizer BOLD
    masked_bolds_arr_3 = []
    for bold in bolds_arr_3:
        masked_bolds_arr_3.append(apply_mask(mask=mask.get_fdata(), target=bold).flatten())
    masked_bolds_arr_3 = np.vstack(masked_bolds_arr_3)
    print("masked postlocalizer bold array shape: ", masked_bolds_arr_3.shape)

    # ===== load weights
    print(f"Loading weights...")
    # prelocalizer
    if brain_flag=='MNI':
        cate_weights_dir = os.path.join(f'/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-0{subID}','preremoval_lvl1_%s/scene_stimuli_MNI_zmap.nii.gz' % brain_flag)
        item_weights_dir = os.path.join(container_path, f"sub-0{subID}", "preremoval_item_level_MNI")
    else:
        cate_weights_dir = os.path.join(f'/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-0{subID}','preremoval_lvl1_%s/scene_stimuli_T1w_zmap.nii.gz' % brain_flag)
        item_weights_dir = os.path.join(container_path, f"sub-0{subID}", "preremoval_item_level_T1w")        

    #prelocalizer weights (category and item) get applied to study/post representations

    all_weights={}
    weights_arr=[]

    #load in all the item specific weights, which come from the LSA contrasts per subject
    for cateID in sub_cates.keys():
        item_weights_fnames = glob.glob(f"{item_weights_dir}/{cateID}*full*zmap*")
        print(cateID, len(item_weights_fnames))
        item_weights = {}

        for fname in item_weights_fnames:
            trialID = fname.split("/")[-1].split("_")[1]  # "trial1"
            trialID = int(trialID[5:])
            item_weights[trialID] = nib.load(fname).get_fdata()
        item_weights = {i: item_weights[i] for i in sorted(item_weights.keys())}
        all_weights[cateID] = item_weights  
        weights_arr.append(np.stack( [ item_weights[i] for i in sorted(item_weights.keys()) ] ))

    #this now masks the item weights to ensure that they are all in the same ROI (group VTC):
    weights_arr = np.vstack(weights_arr)
    print("weights shape: ", weights_arr.shape)
    # apply mask on BOLD
    masked_weights_arr = []
    for weight in weights_arr:
        masked_weights_arr.append(apply_mask(mask=mask.get_fdata(), target=weight).flatten())
    masked_weights_arr = np.vstack(masked_weights_arr)
    print("masked item weights arr shape: ", masked_weights_arr.shape)          

    #this masks the category weight from the group results:
    cate_weights_arr = nib.load(cate_weights_dir)
    masked_cate_weights_arr = apply_mask(mask=mask.get_fdata(), target=cate_weights_arr.get_fdata()).flatten()
    print("masked category weight arr shape: ", masked_cate_weights_arr.shape) #so this is just 1D of the voxels in the VTC mask

    # ===== multiply
    #prelocalizer patterns and prelocalizer item weights
    item_repress_pre = np.multiply(masked_bolds_arr_1, masked_weights_arr) #these are lined up since the trials goes to correct trials
    
    #study patterns, prelocalizer item weights and postlocalizer    
    #this has to come in the loop below to make sure I am weighting the correct trial with the correct item weights

    print("item representations pre shape: ", item_repress_pre.shape)

    #these work right out the box since there is only 1 "category" weighting we are using, and that can be applied to all scene trials in both pre and study (and post)
    cate_repress_pre = np.multiply(masked_bolds_arr_1,masked_cate_weights_arr) #these are multiplied elementwise
    cate_repress_study = np.multiply(masked_bolds_arr_2,masked_cate_weights_arr) #since there is only 1 cate_weight, this multiplies all of masked_bold_arr_2 with the cate_weights
    cate_repress_post = np.multiply(masked_bolds_arr_3,masked_cate_weights_arr) #weight the post representations with category weights

    print("category representations pre shape: ", cate_repress_pre.shape)
    print("category representations study shape: ", cate_repress_study.shape)
    print("category representations post shape: ", cate_repress_post.shape)


    #the way the data is currently sorted is by the index:
    #pre-localizer: 0-59 = Face trial 1 - 60 | 60-179 = Scene trial 1 - 120
    #study: 0-89 = Scene trial 1 - 90
    #post-localizer: 0-179 = Scene trial 1-180 (but 60 of these are novel)

    #the specific output of this "order" DataFrame is by subcategory within the category, but we can still use this to sort by trial since the conditions are in order

    #key arrays being used: pre_scene_order (which is the order of the images in the prelocalizer, after sorted for subcate)
                            #study_scene_order (again sorted for subcate)
   

    item_repress_study_comp=np.zeros_like(item_repress_pre[:90,:]) #set up this array in the same size as the pre, so I can size things in the correct trial order
    item_repress_removal_comp=np.zeros_like(item_repress_study_comp)
    #now we are setting up a pre_item_weighted array to correlate to the study one
    item_repress_pre_comp=np.zeros_like(item_repress_study_comp)
    item_repress_post_comp = np.zeros_like(item_repress_study_comp)

    non_weight_study_comp=np.zeros_like(item_repress_study_comp)
    non_weight_pre_comp=np.zeros_like(item_repress_study_comp)
    non_weight_post_comp=np.zeros_like(item_repress_study_comp)

    category_weight_study_comp=np.zeros_like(item_repress_study_comp)
    category_weight_pre_comp=np.zeros_like(item_repress_study_comp)
    category_weight_post_comp=np.zeros_like(item_repress_study_comp)

    #these are used to hold the fidelity changes from pre to post (item-weighted)
    change_iw_maintain_dict={}  
    change_iw_replace_dict={}
    change_iw_suppress_dict={}
    change_iw_preexp_dict={}

    #these are used to hold the fidelity changes from pre to post (unweighted)
    change_uw_maintain_dict={}  
    change_uw_replace_dict={}
    change_uw_suppress_dict={}   
    change_uw_preexp_dict={} 

    #these are used to hold the fidelity changes from pre to post (scene-weighted)
    change_cw_maintain_dict={}  
    change_cw_replace_dict={}
    change_cw_suppress_dict={}    

    #these are used to hold the fidelity changes from study to post (item-weighted)
    modify_iw_maintain_dict={}  
    modify_iw_replace_dict={}
    modify_iw_suppress_dict={}

    #these are used to hold the fidelity changes from removal to post (item-weighted)
    removal_iw_maintain_dict={}  
    removal_iw_replace_dict={}
    removal_iw_suppress_dict={}    

    #these are used to hold the fidelity changes from pre to removal (item-weighted)
    preremoval_iw_maintain_dict={}  
    preremoval_iw_replace_dict={}
    preremoval_iw_suppress_dict={}     

    #these are used to hold the fidelity changes from study to post (unweighted)
    modify_uw_maintain_dict={}  
    modify_uw_replace_dict={}
    modify_uw_suppress_dict={}

    #these are used to hold the fidelity changes from study to post (scene-weighted)
    modify_cw_maintain_dict={}  
    modify_cw_replace_dict={}
    modify_cw_suppress_dict={}      

    #these are used to hold the fidelity changes from pre to study (item-weighted)
    iw_maintain_dict={}  
    iw_replace_dict={}
    iw_suppress_dict={}

    #these are used to hold the fidelity changes from pre to study (unweighted)
    uw_maintain_dict={}  
    uw_replace_dict={}
    uw_suppress_dict={}

    #these are used to hold the fidelity changes from pre to study (scene-weighted)
    cw_maintain_dict={}  
    cw_replace_dict={}
    cw_suppress_dict={}      

    counter=0

    m_counter=0
    s_counter=0
    r_counter=0

    m_item_repress_study_comp=np.zeros_like(item_repress_pre[:30,:])
    m_item_repress_pre_comp=np.zeros_like(item_repress_pre[:30,:])
    m_item_repress_removal_comp={}
    r_item_repress_study_comp=np.zeros_like(item_repress_pre[:30,:])
    r_item_repress_pre_comp=np.zeros_like(item_repress_pre[:30,:])
    r_item_repress_removal_comp={}
    s_item_repress_study_comp=np.zeros_like(item_repress_pre[:30,:])
    s_item_repress_pre_comp=np.zeros_like(item_repress_pre[:30,:])
    s_item_repress_removal_comp={}


    for trial in study_scene_order['trial_id'].values: 

        study_trial_index=study_scene_order.index[study_scene_order['trial_id']==trial].tolist()[0] #find the order 
        study_image_id=study_scene_order.loc[study_trial_index,'image_id'] #this now uses the index of the dataframe to find the image_id
        #study_trial_num=study_scene_order.loc[study_trial_index,'trial_id'] #now we used the image ID to find the proper trial in the post condition to link to

        pre_trial_index=pre_scene_order.index[pre_scene_order['image_id']==study_image_id].tolist()[0] #find the index in the pre for this study trial
        pre_trial_num = pre_scene_order.loc[pre_trial_index,'trial_id']
        image_condition=  pre_scene_order.loc[pre_scene_order['trial_id']==pre_trial_num]['condition'].values[0]
        pre_trial_subcat = pre_scene_order.loc[pre_trial_index,'subcategory']

        if image_condition==1:
            m_item_repress_study_comp[m_counter]=np.multiply(masked_bolds_arr_2[trial-1,:], masked_weights_arr[pre_trial_num-1,:])
            m_item_repress_removal_comp[m_counter]=np.multiply(masked_bolds_arr_4[:,:,trial-1], masked_weights_arr[pre_trial_num-1,:].reshape((1, masked_weights_arr[pre_trial_num-1,:].size)))
            m_item_repress_pre_comp[m_counter]=item_repress_pre[pre_trial_num-1,:]
            m_counter=m_counter+1
        elif image_condition==2:
            r_item_repress_study_comp[r_counter]=np.multiply(masked_bolds_arr_2[trial-1,:], masked_weights_arr[pre_trial_num-1,:])
            r_item_repress_removal_comp[r_counter]=np.multiply(masked_bolds_arr_4[:,:,trial-1], masked_weights_arr[pre_trial_num-1,:].reshape((1, masked_weights_arr[pre_trial_num-1,:].size)))
            r_item_repress_pre_comp[r_counter]=item_repress_pre[pre_trial_num-1,:]
            r_counter=r_counter+1    
        elif image_condition==3:
            s_item_repress_study_comp[s_counter]=np.multiply(masked_bolds_arr_2[trial-1,:], masked_weights_arr[pre_trial_num-1,:])
            s_item_repress_removal_comp[s_counter]=np.multiply(masked_bolds_arr_4[:,:,trial-1], masked_weights_arr[pre_trial_num-1,:].reshape((1, masked_weights_arr[pre_trial_num-1,:].size)))   
            s_item_repress_pre_comp[s_counter]=item_repress_pre[pre_trial_num-1,:]
            s_counter=s_counter+1                  


    m_item_pre_study_comp=np.corrcoef(m_item_repress_pre_comp,m_item_repress_study_comp)
    temp_df=pd.DataFrame(data=m_item_pre_study_comp)
    if not os.path.exists(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag)): os.makedirs(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag),exist_ok=True)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'maintain_item_weight_pre_study_RSA.csv'))
    del temp_df   

    r_item_pre_study_comp=np.corrcoef(r_item_repress_pre_comp,r_item_repress_study_comp)
    temp_df=pd.DataFrame(data=r_item_pre_study_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'replace_item_weight_pre_study_RSA.csv'))
    del temp_df   

    s_item_pre_study_comp=np.corrcoef(s_item_repress_pre_comp,s_item_repress_study_comp)
    temp_df=pd.DataFrame(data=s_item_pre_study_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'suppress_item_weight_pre_study_RSA.csv'))
    del temp_df             

    # m_item_pre_removal_comp=np.corrcoef(m_item_repress_pre_comp,m_item_repress_removal_comp)
    # temp_df=pd.DataFrame(data=m_item_pre_removal_comp)
    # if not os.path.exists(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag)): os.makedirs(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag),exist_ok=True)
    # temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'maintain_item_weight_pre_removal_RSA.csv'))
    # del temp_df   

    # r_item_pre_removal_comp=np.corrcoef(r_item_repress_pre_comp,r_item_repress_removal_comp)
    # temp_df=pd.DataFrame(data=r_item_pre_removal_comp)
    # temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'replace_item_weight_pre_removal_RSA.csv'))
    # del temp_df   

    # s_item_pre_removal_comp=np.corrcoef(s_item_repress_pre_comp,s_item_repress_removal_comp)
    # temp_df=pd.DataFrame(data=s_item_pre_removal_comp)
    # temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'suppress_item_weight_pre_removal_RSA.csv'))
    # del temp_df 


    #this loop is limited by the smaller index, so thats the study condition (only 90 stims)
    for trial in study_scene_order['trial_id'].values: 

        study_trial_index=study_scene_order.index[study_scene_order['trial_id']==trial].tolist()[0] #find the order 
        study_image_id=study_scene_order.loc[study_trial_index,'image_id'] #this now uses the index of the dataframe to find the image_id
        #study_trial_num=study_scene_order.loc[study_trial_index,'trial_id'] #now we used the image ID to find the proper trial in the post condition to link to

        pre_trial_index=pre_scene_order.index[pre_scene_order['image_id']==study_image_id].tolist()[0] #find the index in the pre for this study trial
        pre_trial_num = pre_scene_order.loc[pre_trial_index,'trial_id']

        image_condition=pre_scene_order.loc[pre_trial_index,'condition']

        post_trial_index=post_scene_order.index[post_scene_order['image_id']==study_image_id].tolist()[0]
        post_trial_num = post_scene_order.loc[post_trial_index,'trial_id']
        #this mean we now know both the trial #, the image id and we can also grab the condition to help sort

        #now that I have the link between prelocalizer, study, and postlocalizer I can get that representation weighted with the item weight
        item_repress_study_comp[counter]=np.multiply(masked_bolds_arr_2[trial-1,:], masked_weights_arr[pre_trial_num-1,:])
        item_repress_removal_comp[counter]=np.multiply(masked_bolds_arr_4[trial-1,:], masked_weights_arr[pre_trial_num-1,:])        
        item_repress_pre_comp[counter]=item_repress_pre[pre_trial_num-1,:]
        item_repress_post_comp[counter]=np.multiply(masked_bolds_arr_3[post_trial_num-1,:],masked_weights_arr[pre_trial_num-1,:])

        #we will also want to filter the results like above for the two other iterations we want, non-weighted and category
        non_weight_study_comp[counter]=masked_bolds_arr_2[trial-1,:]
        non_weight_pre_comp[counter]=masked_bolds_arr_1[pre_trial_num-1,:]
        non_weight_post_comp[counter]=masked_bolds_arr_3[post_trial_num-1,:]

        if brain_flag=='MNI':
            category_weight_study_comp[counter]=cate_repress_study[trial-1,:]
            category_weight_pre_comp[counter]=cate_repress_pre[pre_trial_num-1,:]
            category_weight_post_comp[counter]=cate_repress_post[post_trial_num-1,:]

        #I am using this counter to preserve the ordering that results from the csv's sorting at the top
        #that had the trials in order but segmented by subcate, which I think is a better organization since actual trial number is not needed
        #this code was used to link the correlation of the patterns from pre to post, edited and revived

        #This is to get the fidelity of the current item/trial from pre to post (item_weighted)
        pre_post_trial_iw_fidelity=np.corrcoef(item_repress_pre_comp[counter,:],item_repress_post_comp[counter,:])
        #This is to get the fidelity of the current item/trial from pre to removal (item_weighted)
        pre_removal_trial_iw_fidelity=np.corrcoef(item_repress_pre_comp[counter,:],item_repress_removal_comp[counter,:])        
        #This is to get the fidelity of the current item/trial from study to post (item_weighted)
        study_post_trial_iw_fidelity=np.corrcoef(item_repress_study_comp[counter,:],item_repress_post_comp[counter,:])
        #This is to get the fidelity of the current item/removal from study to post (item_weighted)
        removal_post_trial_iw_fidelity=np.corrcoef(item_repress_removal_comp[counter,:],item_repress_post_comp[counter,:])        
        #This is to get the fidelity of the current item/trial from pre to study (item_weighted)
        pre_study_trial_iw_fidelity=np.corrcoef(item_repress_pre_comp[counter,:],item_repress_study_comp[counter,:])


        #This is to get the fidelity of the current item/trial from pre to post (unweighted)
        pre_post_trial_uw_fidelity=np.corrcoef(non_weight_pre_comp[counter,:],non_weight_post_comp[counter,:])
        #This is to get the fidelity of the current item/trial from study to post (unweighted)
        study_post_trial_uw_fidelity=np.corrcoef(non_weight_study_comp[counter,:],non_weight_post_comp[counter,:])    
        #This is to get the fidelity of the current item/trial from pre to study (unweighted)
        pre_study_trial_uw_fidelity=np.corrcoef(non_weight_pre_comp[counter,:],non_weight_study_comp[counter,:])            

        if brain_flag=='MNI':
            #This is to get the fidelity of the current item/trial from pre to post (scene-weighted)
            pre_post_trial_cw_fidelity=np.corrcoef(category_weight_pre_comp[counter,:],category_weight_post_comp[counter,:])
            #This is to get the fidelity of the current item/trial from study to post (scene-weighted)
            study_post_trial_cw_fidelity=np.corrcoef(category_weight_study_comp[counter,:],category_weight_post_comp[counter,:])        
            #This is to get the fidelity of the current item/trial from pre to study (scene-weighted)
            pre_study_trial_cw_fidelity=np.corrcoef(category_weight_pre_comp[counter,:],category_weight_study_comp[counter,:])  


        #unoperated is being dropped for now since this loop is focused on the study stims
        # if image_condition==0:
        #     LSS_unoperated_dict['image ID: %s' % image_id] = LSS_trial_fidelity[1][0]
        #     LSA_unoperated_dict['image ID: %s' % image_id] = LSA_trial_fidelity[1][0]            
        if image_condition==1:

            change_iw_maintain_dict['image ID: %s' % study_image_id] = pre_post_trial_iw_fidelity[1][0]
            change_uw_maintain_dict['image ID: %s' % study_image_id] = pre_post_trial_uw_fidelity[1][0]

            modify_iw_maintain_dict['image ID: %s' % study_image_id] = study_post_trial_iw_fidelity[1][0]
            removal_iw_maintain_dict['image ID: %s' % study_image_id] = removal_post_trial_iw_fidelity[1][0]
            preremoval_iw_maintain_dict['image ID: %s' % study_image_id] = pre_removal_trial_iw_fidelity[1][0]

            modify_uw_maintain_dict['image ID: %s' % study_image_id] = study_post_trial_uw_fidelity[1][0]
            if brain_flag=='MNI':
                modify_cw_maintain_dict['image ID: %s' % study_image_id] = study_post_trial_cw_fidelity[1][0]  
                change_cw_maintain_dict['image ID: %s' % study_image_id] = pre_post_trial_cw_fidelity[1][0]
                cw_maintain_dict['image ID: %s' % study_image_id] = pre_study_trial_cw_fidelity[1][0]    

            iw_maintain_dict['image ID: %s' % study_image_id] = pre_study_trial_iw_fidelity[1][0]
            uw_maintain_dict['image ID: %s' % study_image_id] = pre_study_trial_uw_fidelity[1][0]

        elif image_condition==2:

            change_iw_replace_dict['image ID: %s' % study_image_id] = pre_post_trial_iw_fidelity[1][0]
            change_uw_replace_dict['image ID: %s' % study_image_id] = pre_post_trial_uw_fidelity[1][0]

            modify_iw_replace_dict['image ID: %s' % study_image_id] = study_post_trial_iw_fidelity[1][0]

            removal_iw_replace_dict['image ID: %s' % study_image_id] = removal_post_trial_iw_fidelity[1][0]
            preremoval_iw_replace_dict['image ID: %s' % study_image_id] = pre_removal_trial_iw_fidelity[1][0]            

            modify_uw_replace_dict['image ID: %s' % study_image_id] = study_post_trial_uw_fidelity[1][0]

            iw_replace_dict['image ID: %s' % study_image_id] = pre_study_trial_iw_fidelity[1][0]
            uw_replace_dict['image ID: %s' % study_image_id] = pre_study_trial_uw_fidelity[1][0]

            if brain_flag=='MNI':
                change_cw_replace_dict['image ID: %s' % study_image_id] = pre_post_trial_cw_fidelity[1][0]
                modify_cw_replace_dict['image ID: %s' % study_image_id] = study_post_trial_cw_fidelity[1][0] 
                cw_replace_dict['image ID: %s' % study_image_id] = pre_study_trial_cw_fidelity[1][0]              


        elif image_condition==3:

            change_iw_suppress_dict['image ID: %s' % study_image_id] = pre_post_trial_iw_fidelity[1][0]
            change_uw_suppress_dict['image ID: %s' % study_image_id] = pre_post_trial_uw_fidelity[1][0]

            modify_iw_suppress_dict['image ID: %s' % study_image_id] = study_post_trial_iw_fidelity[1][0]
            removal_iw_suppress_dict['image ID: %s' % study_image_id] = removal_post_trial_iw_fidelity[1][0]
            preremoval_iw_suppress_dict['image ID: %s' % study_image_id] = pre_removal_trial_iw_fidelity[1][0]            

            modify_uw_suppress_dict['image ID: %s' % study_image_id] = study_post_trial_uw_fidelity[1][0]

            iw_suppress_dict['image ID: %s' % study_image_id] = pre_study_trial_iw_fidelity[1][0]
            uw_suppress_dict['image ID: %s' % study_image_id] = pre_study_trial_uw_fidelity[1][0]
            
            if brain_flag=='MNI':
                change_cw_suppress_dict['image ID: %s' % study_image_id] = pre_post_trial_cw_fidelity[1][0]
                modify_cw_suppress_dict['image ID: %s' % study_image_id] = study_post_trial_cw_fidelity[1][0]
                cw_suppress_dict['image ID: %s' % study_image_id] = pre_study_trial_cw_fidelity[1][0]  


        counter=counter+1

    #need to add a loop here about the unoperated items. I added "old_novel" to the sorting list
    #so what we're looking for are the "3s" during Pre and Post. These are the items that are not operated and thus are the true "baseline"  

    #pull out only the pre-exp (condition 0 in the pre_scene_order)

    pre_exp_df=pre_scene_order[pre_scene_order['condition']==0]

    #now we are setting up a pre_item_weighted array to correlate to the study one
    item_preexp_pre_comp=np.zeros_like(item_repress_pre[:30,:])
    item_preexp_post_comp = np.zeros_like(item_repress_pre[:30,:])    
    non_weight_preexp_pre_comp=np.zeros_like(item_repress_pre[:30,:])
    non_weight_preexp_post_comp=np.zeros_like(item_repress_pre[:30,:])    

    counter=0
    for trial in pre_exp_df['trial_id'].values: 

        #need to sort and grab the images that were pre-exposed... Then I can get the corrcoef from pre-post, and then save into a dict so I can add to the figures below

        pre_img_id = pre_exp_df['image_id'][pre_exp_df['trial_id']==trial].tolist()[0]

        post_trial_index=post_scene_order.index[post_scene_order['image_id']==pre_img_id].tolist()[0]
        post_trial_num = post_scene_order.loc[post_trial_index,'trial_id']
        #this mean we now know both the trial #, the image id and we can also grab the condition to help sort

        #now that I have the link between prelocalizer, study, and postlocalizer I can get that representation weighted with the item weight
        item_preexp_pre_comp[counter]=item_repress_pre[trial-1,:]
        item_preexp_post_comp[counter]=np.multiply(masked_bolds_arr_3[post_trial_num-1,:],masked_weights_arr[trial-1,:])

        #we will also want to filter the results like above for the two other iterations we want, non-weighted and category
        non_weight_preexp_pre_comp[counter]=masked_bolds_arr_1[trial-1,:]
        non_weight_preexp_post_comp[counter]=masked_bolds_arr_3[post_trial_num-1,:]

        #This is to get the fidelity of the current item/trial from pre to post (item_weighted)
        pre_post_trial_iw_fidelity=np.corrcoef(item_repress_pre_comp[counter,:],item_repress_post_comp[counter,:])
        #This is to get the fidelity of the current item/trial from pre to post (unweighted)
        pre_post_trial_uw_fidelity=np.corrcoef(non_weight_pre_comp[counter,:],non_weight_post_comp[counter,:])        

        change_iw_preexp_dict['image ID: %s' % pre_img_id] = pre_post_trial_iw_fidelity[1][0]
        change_uw_preexp_dict['image ID: %s' % pre_img_id] = pre_post_trial_uw_fidelity[1][0]

        counter=counter+1
    #now take the corrcoef of un-weighted to unweighted, then category weight to category weight, finally item to item
    #if everything I did above is correct (i have sanity checked it a couple times), then the order of the pre and the study are in the same trial order, which is in order but by subcategory
    #now I just need to get the corrcoefs, plot them and save... will add quanitification of changes back
    #also the pre only's will have all 120 images shown, while the comps are only the 90 shown in the study phase

    if not os.path.exists(os.path.join(container_path,"sub-0%s" % subID,"Representational_Changes_%s" % brain_flag)): os.makedirs(os.path.join(container_path,"sub-0%s" % subID,"Representational_Changes_%s" % brain_flag),exist_ok=True)

    unweighted_pre_only=np.corrcoef(masked_bolds_arr_1)
    temp_df=pd.DataFrame(data=unweighted_pre_only)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'unweighted_pre_only_RSA.csv'))
    del temp_df

    unweighted_pre_study_comp=np.corrcoef(non_weight_pre_comp,non_weight_study_comp)
    temp_df=pd.DataFrame(data=unweighted_pre_study_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'unweighted_pre_study_RSA.csv'))
    del temp_df

    unweighted_pre_post_comp=np.corrcoef(non_weight_pre_comp,non_weight_post_comp)
    temp_df=pd.DataFrame(data=unweighted_pre_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'unweighted_pre_post_RSA.csv'))
    del temp_df

    unweighted_study_post_comp=np.corrcoef(non_weight_study_comp,non_weight_post_comp)
    temp_df=pd.DataFrame(data=unweighted_study_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'unweighted_study_post_RSA.csv'))
    del temp_df    

    category_pre_only=np.corrcoef(cate_repress_pre)
    temp_df=pd.DataFrame(data=category_pre_only)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'scene_weight_pre_only_RSA.csv'))
    del temp_df  

    category_pre_study_comp=np.corrcoef(category_weight_pre_comp,category_weight_study_comp)
    temp_df=pd.DataFrame(data=category_pre_study_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'scene_weight_pre_study_RSA.csv'))
    del temp_df  

    category_pre_post_comp=np.corrcoef(category_weight_pre_comp,category_weight_post_comp)
    temp_df=pd.DataFrame(data=category_pre_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'scene_weight_pre_post_RSA.csv'))
    del temp_df  

    category_study_post_comp=np.corrcoef(category_weight_study_comp,category_weight_post_comp)    
    temp_df=pd.DataFrame(data=category_study_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'scene_weight_study_post_RSA.csv'))
    del temp_df     

    item_pre_only=np.corrcoef(item_repress_pre)
    temp_df=pd.DataFrame(data=item_pre_only)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'item_weight_pre_only_RSA.csv'))
    del temp_df  

    item_pre_study_comp=np.corrcoef(item_repress_pre_comp,item_repress_study_comp)
    temp_df=pd.DataFrame(data=item_pre_study_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'item_weight_pre_study_RSA.csv'))
    del temp_df  

    item_pre_post_comp=np.corrcoef(item_repress_pre_comp,item_repress_post_comp)
    temp_df=pd.DataFrame(data=item_pre_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'item_weight_pre_post_RSA.csv'))
    del temp_df 

    item_study_post_comp=np.corrcoef(item_repress_study_comp,item_repress_post_comp)
    temp_df=pd.DataFrame(data=item_study_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'item_weight_study_post_RSA.csv'))
    del temp_df     

    item_removal_post_comp=np.corrcoef(item_repress_removal_comp,item_repress_post_comp)
    temp_df=pd.DataFrame(data=item_removal_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'item_weight_removal_post_RSA.csv'))
    del temp_df  

    item_pre_removal_comp=np.corrcoef(item_repress_pre_comp,item_repress_removal_comp)
    temp_df=pd.DataFrame(data=item_pre_removal_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'item_weight_pre_removal_RSA.csv'))
    del temp_df 

    item_pre_post_preexp_comp=np.corrcoef(item_preexp_pre_comp,item_preexp_post_comp)
    temp_df=pd.DataFrame(data=item_pre_post_preexp_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'preexp_item_weight_pre_post_RSA.csv'))
    del temp_df     

    unweighted_pre_post_preexp_comp=np.corrcoef(non_weight_preexp_pre_comp,non_weight_preexp_post_comp)
    temp_df=pd.DataFrame(data=unweighted_pre_post_preexp_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'preexp_unweighted_pre_post_RSA.csv'))
    del temp_df

    #now lets plot all of these and save:
    if not os.path.exists(os.path.join(container_path,"sub-0%s" % subID,"RSA_%s" % brain_flag)): os.makedirs(os.path.join(container_path,"sub-0%s" % subID,"RSA_%s" % brain_flag),exist_ok=True)

    fig=sns.heatmap(unweighted_pre_only)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Unweighted - Prelocalizer only')    
    fig.axhline([60])
    fig.axvline([60])
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'unweighted_pre_only.png'))
    plt.clf() 

    fig=sns.heatmap(unweighted_pre_study_comp)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Unweighted - Prelocalizer vs. Study') 
    fig.axhline([45])
    fig.axhline([90])
    fig.axhline([135])
    fig.axvline([45])
    fig.axvline([90])
    fig.axvline([135])       
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'unweighted_pre_vs_study.png'))
    plt.clf() 

    fig=sns.heatmap(unweighted_pre_post_comp)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Unweighted - Prelocalizer vs. Postlocalizer') 
    fig.axhline([45])
    fig.axhline([90])
    fig.axhline([135])
    fig.axvline([45])
    fig.axvline([90])
    fig.axvline([135])       
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'unweighted_pre_vs_post.png'))
    plt.clf()  

    fig=sns.heatmap(unweighted_study_post_comp)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Unweighted - Study vs. Postlocalizer') 
    fig.axhline([45])
    fig.axhline([90])
    fig.axhline([135])
    fig.axvline([45])
    fig.axvline([90])
    fig.axvline([135])       
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'unweighted_study_vs_post.png'))
    plt.clf()         

    fig=sns.heatmap(category_pre_only)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Scene weighted - Prelocalizer only')
    fig.axhline([60])
    fig.axvline([60])
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'scene_weighted_pre_only.png'))
    plt.clf() 

    fig=sns.heatmap(category_pre_study_comp)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Scene weighted - Prelocalizer vs. Study')    
    fig.axhline([45])
    fig.axhline([90])
    fig.axhline([135])
    fig.axvline([45])
    fig.axvline([90])
    fig.axvline([135])       
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'scene_weighted_pre_vs_study.png'))
    plt.clf() 

    fig=sns.heatmap(category_pre_post_comp)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Scene weighted - Prelocalizer vs. Postlocalizer')    
    fig.axhline([45])
    fig.axhline([90])
    fig.axhline([135])
    fig.axvline([45])
    fig.axvline([90])
    fig.axvline([135])       
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'scene_weighted_pre_vs_post.png'))
    plt.clf()

    fig=sns.heatmap(category_study_post_comp)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Scene weighted - Study vs. Postlocalizer')    
    fig.axhline([45])
    fig.axhline([90])
    fig.axhline([135])
    fig.axvline([45])
    fig.axvline([90])
    fig.axvline([135])       
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'scene_weighted_study_vs_post.png'))
    plt.clf()        

    fig=sns.heatmap(item_pre_only)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Item weighted - Prelocalizer only')    
    fig.axhline([60])
    fig.axvline([60])    
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'item_weighted_pre_only.png'))
    plt.clf() 

    fig=sns.heatmap(item_pre_study_comp)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Item weighted - Prelocalizer vs. Study')    
    fig.axhline([45])
    fig.axhline([90])
    fig.axhline([135])
    fig.axvline([45])
    fig.axvline([90])
    fig.axvline([135])       
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'item_weighted_pre_vs_study.png'))
    plt.clf()                         

    fig=sns.heatmap(item_pre_post_comp)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Item weighted - Prelocalizer vs. Post')    
    fig.axhline([45])
    fig.axhline([90])
    fig.axhline([135])
    fig.axvline([45])
    fig.axvline([90])
    fig.axvline([135])       
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'item_weighted_pre_vs_post.png'))
    plt.clf() 

    fig=sns.heatmap(item_study_post_comp)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Item weighted - Study vs. Postlocalizer')    
    fig.axhline([45])
    fig.axhline([90])
    fig.axhline([135])
    fig.axvline([45])
    fig.axvline([90])
    fig.axvline([135])       
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'item_weighted_study_vs_post.png'))
    plt.clf()     

    fig=sns.heatmap(item_pre_post_preexp_comp)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Item weighted - PreExp Prelocalizer vs. Post')    
    fig.axhline([45])
    fig.axhline([90])
    fig.axhline([135])
    fig.axvline([45])
    fig.axvline([90])
    fig.axvline([135])       
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'preexp_item_weighted_pre_vs_post.png'))
    plt.clf() 

    fig=sns.heatmap(unweighted_pre_post_preexp_comp)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Unweighted - Prelocalizer vs. Postlocalizer') 
    fig.axhline([45])
    fig.axhline([90])
    fig.axhline([135])
    fig.axvline([45])
    fig.axvline([90])
    fig.axvline([135])       
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA_%s" % brain_flag,'preexp_unweighted_pre_vs_post.png'))
    plt.clf() 

    # print('Average LSS Fidelity of unoperated items: %s | std: %s' % (np.array(list(LSS_unoperated_dict.values())).mean(),np.array(list(LSS_unoperated_dict.values())).std()))
    # print('Average LSS Fidelity of maintained items: %s | std: %s' % (np.array(list(LSS_maintain_dict.values())).mean(),np.array(list(LSS_maintain_dict.values())).std()))
    # print('Average LSS Fidelity of replaced items: %s | std: %s' % (np.array(list(LSS_replace_dict.values())).mean(),np.array(list(LSS_replace_dict.values())).std()))
    # print('Average LSS Fidelity of suppressed items: %s | std: %s' % (np.array(list(LSS_suppress_dict.values())).mean(),np.array(list(LSS_suppress_dict.values())).std()))

    #organize this data into dataframes (which may be the best way to handle this data):

    #here is where I need to add in some code to sort these dictionaries by the memory result. For each sub I have a file I can load: 'memory_and_familiar_sub-00x.csv'
    #then I just need to take the imageID, and see if the memory column is a 1 or a 0... this will allow me to split up the data better and visualize better
    memory_csv = pd.read_csv('/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/subject_designs/memory_and_familiar_sub-0%s.csv' % subID)     


    #pre vs. post changes
    itemw_pre_df=pd.DataFrame()
    itemw_pre_df['maintain']=np.array(list(change_iw_maintain_dict.values()))
    itemw_pre_df['replace']=np.array(list(change_iw_replace_dict.values()))
    itemw_pre_df['suppress']=np.array(list(change_iw_suppress_dict.values()))
    itemw_pre_df['preexposed']=np.array(list(change_iw_preexp_dict.values()))
    itemw_pre_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'itemweighted_pre_post_fidelity.csv'))

    catew_pre_df=pd.DataFrame()
    catew_pre_df['maintain']=np.array(list(change_cw_maintain_dict.values()))
    catew_pre_df['replace']=np.array(list(change_cw_replace_dict.values()))
    catew_pre_df['suppress']=np.array(list(change_cw_suppress_dict.values()))
    catew_pre_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'sceneweighted_pre_post_fidelity.csv'))

    unw_pre_df=pd.DataFrame()
    unw_pre_df['maintain']=np.array(list(change_uw_maintain_dict.values()))
    unw_pre_df['replace']=np.array(list(change_uw_replace_dict.values()))
    unw_pre_df['suppress']=np.array(list(change_uw_suppress_dict.values()))
    unw_pre_df['preexposed']=np.array(list(change_uw_preexp_dict.values()))
    unw_pre_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'unweighted_pre_post_fidelity.csv'))    

    itemw_pre_df=pd.DataFrame()
    itemw_pre_df['maintain']=np.array(list(preremoval_iw_maintain_dict.values()))
    itemw_pre_df['replace']=np.array(list(preremoval_iw_replace_dict.values()))
    itemw_pre_df['suppress']=np.array(list(preremoval_iw_suppress_dict.values()))
    itemw_pre_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'itemweighted_pre_removal_fidelity.csv'))


    itemw_remembered_df=pd.DataFrame(columns=['maintain','replace','suppress','preexp'],index=range(0,30))
    itemw_forgot_df=pd.DataFrame(columns=['maintain','replace','suppress','preexp'],index=range(0,30))

    itemw_study_post_forgot_df=pd.DataFrame(columns=['maintain','replace','suppress'],index=range(0,30))
    itemw_removal_post_forgot_df=pd.DataFrame(columns=['maintain','replace','suppress'],index=range(0,30))
    itemw_removal_post_remember_df=pd.DataFrame(columns=['maintain','replace','suppress'],index=range(0,30))

    itemw_pre_removal_df=pd.DataFrame(columns=['maintain','replace','suppress'],index=range(0,30))

    indexer_r=0
    indexer_f=0
    for img_key in change_iw_maintain_dict.keys():
        img_num = re.split('(\d+)', img_key) 
        img_num=int(img_num[1])
        if (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==1:
            itemw_remembered_df.loc[indexer_r]['maintain']=change_iw_maintain_dict['image ID: %s' % img_num]
            itemw_removal_post_remember_df.loc[indexer_r]['maintain']=removal_iw_maintain_dict['image ID: %s' % img_num]

            indexer_r=indexer_r+1
        elif (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==0:
            itemw_forgot_df.loc[indexer_f]['maintain']=change_iw_maintain_dict['image ID: %s' % img_num]
            itemw_study_post_forgot_df.loc[indexer_f]['maintain']=modify_iw_maintain_dict['image ID: %s' % img_num]
            itemw_removal_post_forgot_df.loc[indexer_f]['maintain']=removal_iw_maintain_dict['image ID: %s' % img_num]
            indexer_f=indexer_f+1

    indexer_r=0
    indexer_f=0
    for img_key in change_iw_replace_dict.keys():
        img_num = re.split('(\d+)', img_key) 
        img_num=int(img_num[1])

        if (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==1:
            itemw_remembered_df.loc[indexer_r]['replace']=change_iw_replace_dict['image ID: %s' % img_num]
            itemw_removal_post_remember_df.loc[indexer_r]['replace']=removal_iw_replace_dict['image ID: %s' % img_num]            
            indexer_r=indexer_r+1

        elif (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==0:
            itemw_forgot_df.loc[indexer_f]['replace']=change_iw_replace_dict['image ID: %s' % img_num]
            itemw_study_post_forgot_df.loc[indexer_f]['replace']=modify_iw_replace_dict['image ID: %s' % img_num]   
            itemw_removal_post_forgot_df.loc[indexer_f]['replace']=removal_iw_replace_dict['image ID: %s' % img_num]                     
            indexer_f=indexer_f+1

    indexer_r=0
    indexer_f=0
    for img_key in change_iw_suppress_dict.keys():
        img_num = re.split('(\d+)', img_key) 
        img_num=int(img_num[1])

        if (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==1:
            itemw_remembered_df.loc[indexer_r]['suppress']=change_iw_suppress_dict['image ID: %s' % img_num]
            itemw_removal_post_remember_df.loc[indexer_r]['suppress']=removal_iw_suppress_dict['image ID: %s' % img_num]                        
            indexer_r=indexer_r+1

        elif (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==0:
            itemw_forgot_df.loc[indexer_f]['suppress']=change_iw_suppress_dict['image ID: %s' % img_num]
            itemw_study_post_forgot_df.loc[indexer_f]['suppress']=modify_iw_suppress_dict['image ID: %s' % img_num]   
            itemw_removal_post_forgot_df.loc[indexer_f]['suppress']=removal_iw_suppress_dict['image ID: %s' % img_num]                     
            indexer_f=indexer_f+1

    indexer_r=0
    indexer_f=0
    for img_key in change_iw_preexp_dict.keys():
        img_num = re.split('(\d+)', img_key) 
        img_num=int(img_num[1])

        if (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==1:
            itemw_remembered_df.loc[indexer_r]['preexp']=change_iw_preexp_dict['image ID: %s' % img_num]
            indexer_r=indexer_r+1

        elif (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==0:
            itemw_forgot_df.loc[indexer_f]['preexp']=change_iw_preexp_dict['image ID: %s' % img_num]
            indexer_f=indexer_f+1                        

    uw_remembered_df=pd.DataFrame(columns=['maintain','replace','suppress','preexp'],index=range(0,30))
    uw_forgot_df=pd.DataFrame(columns=['maintain','replace','suppress','preexp'],index=range(0,30))

    indexer_r=0
    indexer_f=0
    for img_key in change_uw_maintain_dict.keys():
        img_num = re.split('(\d+)', img_key) 
        img_num=int(img_num[1])

        if (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==1:
            uw_remembered_df.loc[indexer_r]['maintain']=change_uw_maintain_dict['image ID: %s' % img_num]
            indexer_r=indexer_r+1

        elif (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==0:
            uw_forgot_df.loc[indexer_f]['maintain']=change_uw_maintain_dict['image ID: %s' % img_num]
            indexer_f=indexer_f+1

    indexer_r=0
    indexer_f=0
    for img_key in change_uw_replace_dict.keys():
        img_num = re.split('(\d+)', img_key) 
        img_num=int(img_num[1])

        if (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==1:
            uw_remembered_df.loc[indexer_r]['replace']=change_uw_replace_dict['image ID: %s' % img_num]
            indexer_r=indexer_r+1

        elif (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==0:
            uw_forgot_df.loc[indexer_f]['replace']=change_uw_replace_dict['image ID: %s' % img_num]
            indexer_f=indexer_f+1

    indexer_r=0
    indexer_f=0
    for img_key in change_uw_suppress_dict.keys():
        img_num = re.split('(\d+)', img_key) 
        img_num=int(img_num[1])

        if (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==1:
            uw_remembered_df.loc[indexer_r]['suppress']=change_uw_suppress_dict['image ID: %s' % img_num]
            indexer_r=indexer_r+1


        elif (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==0:
            uw_forgot_df.loc[indexer_f]['suppress']=change_uw_suppress_dict['image ID: %s' % img_num]
            indexer_f=indexer_f+1

    indexer_r=0
    indexer_f=0
    for img_key in change_uw_preexp_dict.keys():
        img_num = re.split('(\d+)', img_key) 
        img_num=int(img_num[1])

        if (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==1:
            uw_remembered_df.loc[indexer_r]['preexp']=change_uw_preexp_dict['image ID: %s' % img_num]
            indexer_r=indexer_r+1


        elif (memory_csv['memory'][memory_csv['image_num']==img_num]).values[0]==0:
            uw_forgot_df.loc[indexer_f]['preexp']=change_uw_preexp_dict['image ID: %s' % img_num]
            indexer_f=indexer_f+1

    #itemw_remembered_df.dropna(inplace=True)
    #itemw_forgot_df.dropna(inplace=True)

    itemw_remembered_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'itemweighted_pre_post_remember_fidelity.csv'))
    itemw_forgot_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'itemweighted_pre_post_forgot_fidelity.csv'))
    uw_remembered_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'unweighted_pre_post_remember_fidelity.csv'))
    uw_forgot_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'unweighted_pre_post_forgot_fidelity.csv'))

    itemw_study_post_forgot_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'itemweighted_study_post_forgot_fidelity.csv'))
    itemw_removal_post_forgot_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'itemweighted_removal_post_forgot_fidelity.csv'))
    itemw_removal_post_remember_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'itemweighted_removal_post_remembered_fidelity.csv'))

    fig=sns.barplot(data=itemw_study_post_forgot_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Item Weighted - Study vs. Post - Forgot items only')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Item_Weighted_forgot_only_study_post_summary.png'))
    plt.clf()

    fig=sns.barplot(data=itemw_removal_post_forgot_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Item Weighted - Removal vs. Post - Forgot items only')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Item_Weighted_forgot_only_removal_post_summary.png'))
    plt.clf()

    fig=sns.barplot(data=itemw_removal_post_remember_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Item Weighted - Removal vs. Post - Remembered items only')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Item_Weighted_remember_only_removal_post_summary.png'))
    plt.clf()    


    #study vs. post changes
    itemw_study_df=pd.DataFrame()
    itemw_study_df['maintain']=np.array(list(modify_iw_maintain_dict.values()))
    itemw_study_df['replace']=np.array(list(modify_iw_replace_dict.values()))
    itemw_study_df['suppress']=np.array(list(modify_iw_suppress_dict.values()))
    itemw_study_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'itemweighted_study_post_fidelity.csv'))

    catew_study_df=pd.DataFrame()
    catew_study_df['maintain']=np.array(list(modify_cw_maintain_dict.values()))
    catew_study_df['replace']=np.array(list(modify_cw_replace_dict.values()))
    catew_study_df['suppress']=np.array(list(modify_cw_suppress_dict.values()))
    catew_study_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'sceneweighted_study_post_fidelity.csv'))

    unw_study_df=pd.DataFrame()
    unw_study_df['maintain']=np.array(list(modify_uw_maintain_dict.values()))
    unw_study_df['replace']=np.array(list(modify_uw_replace_dict.values()))
    unw_study_df['suppress']=np.array(list(modify_uw_suppress_dict.values()))
    unw_study_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'unweighted_study_post_fidelity.csv'))  

    #pre vs. study changes
    itemw_df=pd.DataFrame()
    itemw_df['maintain']=np.array(list(iw_maintain_dict.values()))
    itemw_df['replace']=np.array(list(iw_replace_dict.values()))
    itemw_df['suppress']=np.array(list(iw_suppress_dict.values()))
    itemw_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'itemweighted_pre_study_fidelity.csv'))

    catew_df=pd.DataFrame()
    catew_df['maintain']=np.array(list(cw_maintain_dict.values()))
    catew_df['replace']=np.array(list(cw_replace_dict.values()))
    catew_df['suppress']=np.array(list(cw_suppress_dict.values()))
    catew_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'sceneweighted_pre_study_fidelity.csv'))

    unw_df=pd.DataFrame()
    unw_df['maintain']=np.array(list(uw_maintain_dict.values()))
    unw_df['replace']=np.array(list(uw_replace_dict.values()))
    unw_df['suppress']=np.array(list(uw_suppress_dict.values()))
    unw_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'unweighted_pre_study_fidelity.csv')) 

    #plot and save the figures of the data - Pre vs Post
    fig=sns.barplot(data=itemw_pre_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Item Weighted - Pre vs. Post')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Item_Weighted_pre_post_summary.png'))
    plt.clf()  

    fig=sns.barplot(data=catew_pre_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Scene Weighted - Pre vs. Post')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Scene_Weighted_pre_post_summary.png'))
    plt.clf() 

    fig=sns.barplot(data=unw_pre_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Unweighted - Pre vs. Post')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Unweighted_pre_post_summary.png'))
    plt.clf()      


    fig=sns.barplot(data=itemw_remembered_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Item Weighted - Pre vs. Post - Remebered items only')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Item_Weighted_remember_only_pre_post_summary.png'))
    plt.clf()

    fig=sns.barplot(data=uw_remembered_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Unweighted - Pre vs. Post - Remebered items only')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Unweighted_remember_only_pre_post_summary.png'))
    plt.clf()

    fig=sns.barplot(data=itemw_forgot_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Item Weighted - Pre vs. Post - Forgot items only')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Item_Weighted_forgot_only_pre_post_summary.png'))
    plt.clf()

    fig=sns.barplot(data=uw_forgot_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Unweighted - Pre vs. Post - Forgot items only')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Unweighted_forgot_only_pre_post_summary.png'))
    plt.clf()    

    #plot and save figures of the data - Study vs Post
    fig=sns.barplot(data=itemw_study_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Item Weighted - Study vs. Post')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Item_Weighted_study_post_summary.png'))
    plt.clf()  

    fig=sns.barplot(data=catew_study_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Scene Weighted - Study vs. Post')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Scene_Weighted_study_post_summary.png'))
    plt.clf() 

    fig=sns.barplot(data=unw_study_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Unweighted - Study vs. Post')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Unweighted_study_post_summary.png'))
    plt.clf()       

    #plot and save figures of the data - Pre vs Study
    fig=sns.barplot(data=itemw_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Item Weighted - Pre vs. Study')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Item_Weighted_pre_study_summary.png'))
    plt.clf()  

    fig=sns.barplot(data=catew_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Scene Weighted - Pre vs. Study')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Scene_Weighted_pre_study_summary.png'))
    plt.clf() 

    fig=sns.barplot(data=unw_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Unweighted - Pre vs. Study')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag,'Unweighted_pre_study_summary.png'))
    plt.clf()


    print("Subject is done... saving everything")
    print("===============================================================================")    

#now I want to take the subject level "fidelity" results and look at it on a group level - I will also want to quantify the off diagonal of the RSA to random pairs
group_item_weighted_pre_post=pd.DataFrame()
group_item_weighted_study_post=pd.DataFrame()
group_item_weighted_pre_study=pd.DataFrame()

group_item_weighted_pre_removal=pd.DataFrame()

group_cate_weighted_pre_post=pd.DataFrame()
group_cate_weighted_study_post=pd.DataFrame()
group_cate_weighted_pre_study=pd.DataFrame()

group_unweighted_pre_post=pd.DataFrame()
group_unweighted_study_post=pd.DataFrame()
group_unweighted_pre_study=pd.DataFrame()

group_item_remember_pre_post=pd.DataFrame()
group_item_forgot_pre_post=pd.DataFrame()
group_unweighted_remember_pre_post=pd.DataFrame()
group_unweighted_forgot_pre_post=pd.DataFrame()

group_by_sub_item_remember_pre_post=pd.DataFrame()
group_by_sub_item_forgot_pre_post=pd.DataFrame()

group_by_sub_item_remember_removal_post=pd.DataFrame()
group_by_sub_item_forgot_removal_post=pd.DataFrame()
group_by_sub_item_pre_removal=pd.DataFrame()

group_item_forgot_study_post=pd.DataFrame()
group_item_forgot_removal_post=pd.DataFrame()
group_item_remember_removal_post=pd.DataFrame()



for subID in subs:
    data_path=os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes_%s" % brain_flag)

    item_pre_post=os.path.join(data_path,'itemweighted_pre_post_fidelity.csv')
    group_item_weighted_pre_post=group_item_weighted_pre_post.append(pd.read_csv(item_pre_post,usecols=[1,2,3,4]),ignore_index=True)

    item_study_post=find('item*study*post*fidelity*',data_path)
    group_item_weighted_study_post=group_item_weighted_study_post.append(pd.read_csv(item_study_post[0],usecols=[1,2,3]),ignore_index=True)

    item_pre_study=find('item*pre*study*fidelity*',data_path)
    group_item_weighted_pre_study=group_item_weighted_pre_study.append(pd.read_csv(item_pre_study[0],usecols=[1,2,3]),ignore_index=True)

    item_pre_removal=find('item*pre*removal*fidelity*',data_path)
    group_item_weighted_pre_removal=group_item_weighted_pre_removal.append(pd.read_csv(item_pre_removal[0],usecols=[1,2,3]),ignore_index=True)    

    cate_pre_post=find('scene*pre*post*fidelity*',data_path)
    group_cate_weighted_pre_post=group_cate_weighted_pre_post.append(pd.read_csv(cate_pre_post[0],usecols=[1,2,3]),ignore_index=True)

    cate_study_post=find('scene*study*post*fidelity*',data_path)
    group_cate_weighted_study_post=group_cate_weighted_study_post.append(pd.read_csv(cate_study_post[0],usecols=[1,2,3]),ignore_index=True)

    cate_pre_study=find('scene*pre*study*fidelity*',data_path)
    group_cate_weighted_pre_study=group_cate_weighted_pre_study.append(pd.read_csv(cate_study_post[0],usecols=[1,2,3]),ignore_index=True)

    un_pre_post=os.path.join(data_path,'unweighted_pre_post_fidelity.csv')
    group_unweighted_pre_post=group_unweighted_pre_post.append(pd.read_csv(un_pre_post,usecols=[1,2,3,4]),ignore_index=True)

    un_study_post=find('unweighted*study*post*fidelity*',data_path)
    group_unweighted_study_post=group_unweighted_study_post.append(pd.read_csv(un_study_post[0],usecols=[1,2,3]),ignore_index=True)

    un_pre_study=find('unweighted*pre*study*fidelity*',data_path)
    group_unweighted_pre_study=group_unweighted_pre_study.append(pd.read_csv(un_pre_study[0],usecols=[1,2,3]),ignore_index=True)

    #now pull the ones sorted by memory:
    item_r_pre_post=os.path.join(data_path,'itemweighted_pre_post_remember_fidelity.csv')
    group_item_remember_pre_post=group_item_remember_pre_post.append(pd.read_csv(item_r_pre_post,usecols=[1,2,3,4]),ignore_index=True)

    item_f_pre_post=os.path.join(data_path,'itemweighted_pre_post_forgot_fidelity.csv')
    group_item_forgot_pre_post=group_item_forgot_pre_post.append(pd.read_csv(item_f_pre_post,usecols=[1,2,3,4]),ignore_index=True)

    item_f_study_post=os.path.join(data_path,'itemweighted_study_post_forgot_fidelity.csv')
    group_item_forgot_study_post=group_item_forgot_study_post.append(pd.read_csv(item_f_study_post,usecols=[1,2,3]),ignore_index=True)    

    item_f_removal_post=os.path.join(data_path,'itemweighted_removal_post_forgot_fidelity.csv')
    group_item_forgot_removal_post=group_item_forgot_removal_post.append(pd.read_csv(item_f_removal_post,usecols=[1,2,3]),ignore_index=True)  

    item_r_removal_post=os.path.join(data_path,'itemweighted_removal_post_remembered_fidelity.csv')
    group_item_remember_removal_post=group_item_remember_removal_post.append(pd.read_csv(item_r_removal_post,usecols=[1,2,3]),ignore_index=True)  

    un_r_pre_post=os.path.join(data_path,'unweighted_pre_post_remember_fidelity.csv')
    group_unweighted_remember_pre_post=group_unweighted_remember_pre_post.append(pd.read_csv(un_r_pre_post,usecols=[1,2,3,4]),ignore_index=True)

    un_f_pre_post=os.path.join(data_path,'unweighted_pre_post_forgot_fidelity.csv')
    group_unweighted_forgot_pre_post=group_unweighted_forgot_pre_post.append(pd.read_csv(un_f_pre_post,usecols=[1,2,3,4]),ignore_index=True)

    group_by_sub_item_remember_pre_post=group_by_sub_item_remember_pre_post.append(pd.read_csv(item_r_pre_post,usecols=[1,2,3,4]).mean(),ignore_index=True)

    group_by_sub_item_forgot_pre_post=group_by_sub_item_forgot_pre_post.append(pd.read_csv(item_f_pre_post,usecols=[1,2,3,4]).mean(),ignore_index=True)   

    group_by_sub_item_pre_removal=group_by_sub_item_pre_removal.append(pd.read_csv(item_pre_removal[0],usecols=[1,2,3]).mean(),ignore_index=True)

    group_by_sub_item_remember_removal_post=group_by_sub_item_remember_removal_post.append(pd.read_csv(item_r_removal_post,usecols=[1,2,3]).mean(),ignore_index=True)

    group_by_sub_item_forgot_removal_post=group_by_sub_item_forgot_removal_post.append(pd.read_csv(item_f_removal_post,usecols=[1,2,3]).mean(),ignore_index=True)        

if not os.path.exists(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag)): os.makedirs(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag),exist_ok=True)
#plot and save the figures of the data - Pre vs Post
fig=sns.barplot(data=group_item_weighted_pre_post,ci=95,palette=['green','blue','red','gray'])
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Item Weighted (Group Level) - Pre vs. Post')  
# fig, test_results = add_stat_annotation(fig, data=group_item_weighted_pre_post,
#                                    box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexposed"), ("suppress", "preexposed"), ("replace", "preexposed")],
#                                    test='t-test_ind', text_format='star',loc='outside', verbose=2) 
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Item_Weighted_pre_post_summary.png'))
plt.clf() 

fig=sns.barplot(data=group_item_remember_pre_post,ci=95,palette=['green','blue','red','gray'])
# fig, test_results = add_stat_annotation(fig, data=group_item_remember_pre_post,
#                                    box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexp"), ("suppress", "preexp"), ("replace", "preexp")],
#                                    test='t-test_ind', text_format='star',loc='outside', verbose=2) 
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Item Weighted (Group Level) - Pre vs. Post - Only Remembered')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Item_Weighted_pre_post_remembered_summary.png'))
plt.clf() 

fig=sns.barplot(data=group_item_forgot_pre_post,ci=95,palette=['green','blue','red','gray'])
# fig, test_results = add_stat_annotation(fig, data=group_item_forgot_pre_post,
#                                    box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexp"), ("suppress", "preexp"), ("replace", "preexp")],
#                                    test='t-test_ind', text_format='star',loc='outside', verbose=2)  
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Item Weighted (Group Level) - Pre vs. Post - Only Forgot')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Item_Weighted_pre_post_forgot_summary.png'))
plt.clf() 

fig=sns.barplot(data=group_item_forgot_study_post,ci=95,palette=['green','blue','red'])
# fig, test_results = add_stat_annotation(fig, data=group_item_forgot_pre_post,
#                                    box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexp"), ("suppress", "preexp"), ("replace", "preexp")],
#                                    test='t-test_ind', text_format='star',loc='outside', verbose=2)  
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Item Weighted (Group Level) - Study vs. Post - Only Forgot')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Item_Weighted_study_post_forgot_summary.png'))
plt.clf() 


fig=sns.barplot(data=group_item_forgot_removal_post,ci=95,palette=['green','blue','red'])
# fig, test_results = add_stat_annotation(fig, data=group_item_forgot_pre_post,
#                                    box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexp"), ("suppress", "preexp"), ("replace", "preexp")],
#                                    test='t-test_ind', text_format='star',loc='outside', verbose=2)  
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Item Weighted (Group Level) - Removal vs. Post - Only Forgot')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Item_Weighted_removal_post_forgot_summary.png'))
plt.clf() 


fig=sns.barplot(data=group_item_remember_removal_post,ci=95,palette=['green','blue','red'])
# fig, test_results = add_stat_annotation(fig, data=group_item_forgot_pre_post,
#                                    box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexp"), ("suppress", "preexp"), ("replace", "preexp")],
#                                    test='t-test_ind', text_format='star',loc='outside', verbose=2)  
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Item Weighted (Group Level) - Removal vs. Post - Only Remembered')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Item_Weighted_removal_post_remember_summary.png'))
plt.clf() 


#plot and save the figures of the data - Pre vs Removal
fig=sns.barplot(data=group_item_weighted_pre_removal,ci=95,palette=['green','blue','red'])
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Item Weighted (Group Level) - Pre vs. Removal')  
# fig, test_results = add_stat_annotation(fig, data=group_item_weighted_pre_post,
#                                    box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexposed"), ("suppress", "preexposed"), ("replace", "preexposed")],
#                                    test='t-test_ind', text_format='star',loc='outside', verbose=2) 
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Item_Weighted_pre_removal_summary.png'))
plt.clf() 

#######################

grouped_by_sub_item_difference_removal_post=group_by_sub_item_remember_removal_post.subtract(group_by_sub_item_forgot_removal_post)
grouped_by_sub_item_difference_removal_post.dropna(inplace=True)
non_nan_subs=np.take(subs,grouped_by_sub_item_difference_removal_post.index)
grouped_by_sub_item_difference_removal_post=grouped_by_sub_item_difference_removal_post.melt()
grouped_by_sub_item_difference_removal_post['sub']=np.tile(non_nan_subs,3)
grouped_by_sub_item_difference_removal_post.rename(columns={'variable':'operation','value':'fidelity','sub':'sub'},inplace=True)
fig=sns.barplot(data=grouped_by_sub_item_difference_removal_post,x='operation',y='fidelity',ci=95,palette=['green','blue','red'])
# fig, test_results = add_stat_annotation(fig, data=grouped_by_sub_item_difference_pre_post, x='operation',y='fidelity',
#                                     box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexp"), ("suppress", "preexp"), ("replace", "preexp")],
#                                     test='t-test_paired', text_format='star',loc='inside', verbose=2)  
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity Difference (Remembered - Forgot)')
fig.set_title('Item Weighted (Group Level) - Removal vs. Post Changes - Remembered items minus Forgotten items',loc='center', wrap=True)
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Item_Weighted_removal_post_remember_minus_forgot_summary.png'))
plt.clf() 

# print(AnovaRM(data=grouped_by_sub_item_difference_removal_post, depvar='fidelity', subject='sub',within=['operation']).fit())


#######################
grouped_by_sub_item_difference_pre_post=group_by_sub_item_remember_pre_post.subtract(group_by_sub_item_forgot_pre_post)
temp=grouped_by_sub_item_difference_pre_post.pop('preexp')
grouped_by_sub_item_difference_pre_post=pd.concat([grouped_by_sub_item_difference_pre_post,temp],1)
grouped_by_sub_item_difference_pre_post.dropna(inplace=True)
non_nan_subs=np.take(subs,grouped_by_sub_item_difference_pre_post.index)
grouped_by_sub_item_difference_pre_post=grouped_by_sub_item_difference_pre_post.melt()
grouped_by_sub_item_difference_pre_post['sub']=np.tile(non_nan_subs,4)
grouped_by_sub_item_difference_pre_post.rename(columns={'variable':'operation','value':'fidelity','sub':'sub'},inplace=True)
fig=sns.barplot(data=grouped_by_sub_item_difference_pre_post,x='operation',y='fidelity',ci=95,palette=['green','blue','red','gray'])
# fig, test_results = add_stat_annotation(fig, data=grouped_by_sub_item_difference_pre_post, x='operation',y='fidelity',
#                                     box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexp"), ("suppress", "preexp"), ("replace", "preexp")],
#                                     test='t-test_paired', text_format='star',loc='inside', verbose=2)  
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity Difference (Remembered - Forgot)')
fig.set_title('Item Weighted (Group Level) - Pre vs. Post Changes - Remembered items minus Forgotten items',loc='center', wrap=True)
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Item_Weighted_pre_post_remember_minus_forgot_summary.png'))
plt.clf() 

print(AnovaRM(data=grouped_by_sub_item_difference_pre_post, depvar='fidelity', subject='sub',within=['operation']).fit())


grouped_by_sub_item_remember_pre_post=group_by_sub_item_remember_pre_post.melt()
grouped_by_sub_item_forgot_pre_post=group_by_sub_item_forgot_pre_post.melt()

grouped_by_sub_item_remember_pre_post['sub']=np.tile(subs,4)
grouped_by_sub_item_forgot_pre_post['sub']=np.tile(subs,4)

# print(AnovaRM(data=grouped_by_sub_item_difference_pre_post, depvar='fidelity', subject='sub',within=['operation']).fit())

# fig=sns.barplot(data=group_unweighted_remember_pre_post,ci=95,palette=['green','blue','red','gray'])
# fig.set_xlabel('Operations')
# fig.set_ylabel('Fidelity')
# fig.set_title('Unweighted (Group Level) - Pre vs. Post - Only Remembered')
# fig, test_results = add_stat_annotation(fig, data=group_unweighted_remember_pre_post,
#                                    box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexposed"), ("suppress", "preexposed"), ("replace", "preexposed")],
#                                    test='t-test_paired', text_format='star', loc='inside', verbose=2)  
# plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Unweighted_pre_post_remembered_summary.png'))
# plt.clf() 

# fig=sns.barplot(data=group_unweighted_forgot_pre_post,ci=95,palette=['green','blue','red','gray'])
# fig.set_xlabel('Operations')
# fig.set_ylabel('Fidelity')
# fig.set_title('Unweighted (Group Level) - Pre vs. Post - Only Forgot')
# fig, test_results = add_stat_annotation(fig, data=group_unweighted_forgot_pre_post,
#                                    box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexposed"), ("suppress", "preexposed"), ("replace", "preexposed")],
#                                    test='t-test_paired', text_format='star', loc='inside', verbose=2)  
# plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Unweighted_pre_post_forgot_summary.png'))
# plt.clf() 

# fig=sns.barplot(data=group_item_weighted_study_post,ci=95,palette=['green','blue','red','gray'])
# fig.set_xlabel('Operations')
# fig.set_ylabel('Fidelity')
# fig.set_title('Item Weighted (Group Level) - Study vs. Post')
# fig, test_results = add_stat_annotation(fig, data=group_item_weighted_study_post,
#                                    box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexposed"), ("suppress", "preexposed"), ("replace", "preexposed")],
#                                    test='t-test_paired', text_format='star', loc='inside', verbose=2)  
# plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Item_Weighted_study_post_summary.png'))
# plt.clf() 

# fig=sns.barplot(data=group_item_weighted_pre_study,ci=95,palette=['green','blue','red','gray'])
# fig.set_xlabel('Operations')
# fig.set_ylabel('Fidelity')
# fig.set_title('Item Weighted (Group Level) - Pre vs. Study')
# fig, test_results = add_stat_annotation(fig, data=group_item_weighted_pre_study,
#                                    box_pairs=[("maintain", "replace"), ("maintain", "suppress"), ("maintain", "preexposed"), ("suppress", "preexposed"), ("replace", "preexposed")],
#                                    test='t-test_paired', text_format='star', loc='inside', verbose=2)  
# plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Item_Weighted_pre_study_summary.png'))
# plt.clf() 

# fig=sns.barplot(data=group_cate_weighted_pre_post,ci=95)
# fig.set_xlabel('Operations')
# fig.set_ylabel('Fidelity')
# fig.set_title('Scene Weighted (Group Level) - Pre vs. Post')
# plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Scene_Weighted_pre_post_summary.png'))
# plt.clf() 

# fig=sns.barplot(data=group_cate_weighted_study_post,ci=95)
# fig.set_xlabel('Operations')
# fig.set_ylabel('Fidelity')
# fig.set_title('Scene Weighted (Group Level) - Study vs. Post')
# plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Scene_Weighted_study_post_summary.png'))
# plt.clf() 

# fig=sns.barplot(data=group_cate_weighted_pre_study,ci=95)
# fig.set_xlabel('Operations')
# fig.set_ylabel('Fidelity')
# fig.set_title('Scene Weighted (Group Level) - Pre vs. Study')
# plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Scene_Weighted_pre_study_summary.png'))
# plt.clf() 

# fig=sns.barplot(data=group_unweighted_pre_post,ci=95)
# fig.set_xlabel('Operations')
# fig.set_ylabel('Fidelity')
# fig.set_title('Unweighted (Group Level) - Pre vs. Post')
# plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Unweighted_pre_post_summary.png'))
# plt.clf() 

# fig=sns.barplot(data=group_unweighted_study_post,ci=95)
# fig.set_xlabel('Operations')
# fig.set_ylabel('Fidelity')
# fig.set_title('Unweighted (Group Level) - Study vs. Post')
# plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Unweighted_study_post_summary.png'))
# plt.clf() 

# fig=sns.barplot(data=group_unweighted_pre_study,ci=95)
# fig.set_xlabel('Operations')
# fig.set_ylabel('Fidelity')
# fig.set_title('Unweighted (Group Level) - Pre vs. study')
# plt.savefig(os.path.join(container_path,"group_model","Representational_Changes_%s" % brain_flag,'Group_Unweighted_pre_study_summary.png'))
# plt.clf() 


#I now need to add similar code as above, but I go into each subjects RSA results, and take the average fidelity of the off-diagonal (so items 1-90 matched to presentations in study or post) to random pairs
#for the random pairs, I can pick (0-45 comp to 135-180) and (45-90 comp to 90-135), these would be "incorrect pairings" (accross subcategory, manmade to natural)
#then I would get either 90 data points per column (Same vs Other) and could then look at the group level by taking the 3 averages and plotting that
#subject level would still be useful as a simple summary for the RSA plots 