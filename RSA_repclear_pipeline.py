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

subs=['02','03','04']
brain_flag='MNI'
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


container_path='/scratch/06873/zbretton/fmriprep'
param_dir =  '/scratch/06873/zbretton/fmriprep/subject_designs'

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
    
    pre_scene_order = tim_df[tim_df["category"]==1][["trial_id","image_id","condition"]]
    pre_face_order = tim_df[tim_df["category"]==2][["trial_id","image_id","condition"]]   

    #lets pull out the study data here:
    tim_df2 = pd.read_csv(tim_path)
    tim_df2 = tim_df2[tim_df2["phase"]==3] #phase 3 is study
    tim_df2 = tim_df2.sort_values(by=["category", "subcategory", "trial_id"])
    
    study_scene_order = tim_df2[tim_df2["category"]==1][["trial_id","image_id","condition"]]

    #lets pull out the postlocalizer data here:
    tim_df3 = pd.read_csv(tim_path)
    tim_df3 = tim_df3[tim_df3["phase"]==4] #phase 4 is post-localizer
    tim_df3 = tim_df3.sort_values(by=["category", "subcategory", "trial_id"])

    post_scene_order = tim_df3[tim_df3["category"]==1][["trial_id","image_id","condition"]]


    print(f"Running RSA for sub0{subID}...")

    # ===== load mask for BOLD
    mask_path = os.path.join(container_path, "group_MNI_VTC_mask.nii.gz")
    mask = nib.load(mask_path)
    print("mask shape: ", mask.shape)

    # ===== load ready BOLD for each trial of prelocalizer
    print(f"Loading preprocessed BOLDs for pre-localizer...")
    bold_dir_1 = os.path.join(container_path, f"sub-0{subID}", "item_representations")

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
    bold_dir_2 = os.path.join(container_path, f"sub-0{subID}", "item_representations")

    all_bolds_2 = {}  # {cateID: {trialID: bold}}
    bolds_arr_2 = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames_2 = glob.glob(f"{bold_dir_2}/*study*{cateID}*")
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

    # ===== load ready BOLD for each trial of postlocalizer
    print(f"Loading preprocessed BOLDs for post-localizer...")
    bold_dir_3 = os.path.join(container_path, f"sub-0{subID}", "item_representations")

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
    print("masked phase2 bold array shape: ", masked_bolds_arr_2.shape)    

    # apply VTC mask on postlocalizer BOLD
    masked_bolds_arr_3 = []
    for bold in bolds_arr_3:
        masked_bolds_arr_3.append(apply_mask(mask=mask.get_fdata(), target=bold).flatten())
    masked_bolds_arr_3 = np.vstack(masked_bolds_arr_3)
    print("masked postlocalizer bold array shape: ", masked_bolds_arr_3.shape)

    # ===== load weights
    print(f"Loading weights...")
    # prelocalizer
    cate_weights_dir = "/scratch/06873/zbretton/fmriprep/group_model/group_category_lvl2/group_scene_ovr_face_MNI_zmap.nii.gz" #full path to the scene weights
    item_weights_dir = os.path.join(container_path, f"sub-0{subID}", "preremoval_item_level")

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

    #these are used to hold the fidelity changes from pre to post (unweighted)
    change_uw_maintain_dict={}  
    change_uw_replace_dict={}
    change_uw_suppress_dict={}    

    #these are used to hold the fidelity changes from pre to post (scene-weighted)
    change_cw_maintain_dict={}  
    change_cw_replace_dict={}
    change_cw_suppress_dict={}    

    #these are used to hold the fidelity changes from study to post (item-weighted)
    modify_iw_maintain_dict={}  
    modify_iw_replace_dict={}
    modify_iw_suppress_dict={}

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
        item_repress_pre_comp[counter]=item_repress_pre[pre_trial_num-1,:]
        item_repress_post_comp[counter]=np.multiply(masked_bolds_arr_3[post_trial_num-1,:],masked_weights_arr[pre_trial_num-1,:])

        #we will also want to filter the results like above for the two other iterations we want, non-weighted and category
        non_weight_study_comp[counter]=masked_bolds_arr_2[trial-1,:]
        non_weight_pre_comp[counter]=masked_bolds_arr_1[pre_trial_num-1,:]
        non_weight_post_comp[counter]=masked_bolds_arr_3[post_trial_num-1,:]

        category_weight_study_comp[counter]=cate_repress_study[trial-1,:]
        category_weight_pre_comp[counter]=cate_repress_pre[pre_trial_num-1,:]
        category_weight_post_comp[counter]=cate_repress_post[post_trial_num-1,:]


        #I am using this counter to preserve the ordering that results from the csv's sorting at the top
        #that had the trials in order but segmented by subcate, which I think is a better organization since actual trial number is not needed
        #this code was used to link the correlation of the patterns from pre to post, edited and revived

        #This is to get the fidelity of the current item/trial from pre to post (item_weighted)
        pre_post_trial_iw_fidelity=np.corrcoef(item_repress_pre_comp[counter,:],item_repress_post_comp[counter,:])
        #This is to get the fidelity of the current item/trial from study to post (item_weighted)
        study_post_trial_iw_fidelity=np.corrcoef(item_repress_study_comp[counter,:],item_repress_post_comp[counter,:])
        #This is to get the fidelity of the current item/trial from pre to study (item_weighted)
        pre_study_trial_iw_fidelity=np.corrcoef(item_repress_pre_comp[counter,:],item_repress_study_comp[counter,:])


        #This is to get the fidelity of the current item/trial from pre to post (unweighted)
        pre_post_trial_uw_fidelity=np.corrcoef(non_weight_pre_comp[counter,:],non_weight_post_comp[counter,:])
        #This is to get the fidelity of the current item/trial from study to post (unweighted)
        study_post_trial_uw_fidelity=np.corrcoef(non_weight_study_comp[counter,:],non_weight_post_comp[counter,:])    
        #This is to get the fidelity of the current item/trial from pre to study (unweighted)
        pre_study_trial_uw_fidelity=np.corrcoef(non_weight_pre_comp[counter,:],non_weight_study_comp[counter,:])            

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
            change_cw_maintain_dict['image ID: %s' % study_image_id] = pre_post_trial_cw_fidelity[1][0]

            modify_iw_maintain_dict['image ID: %s' % study_image_id] = study_post_trial_iw_fidelity[1][0]
            modify_uw_maintain_dict['image ID: %s' % study_image_id] = study_post_trial_uw_fidelity[1][0]
            modify_cw_maintain_dict['image ID: %s' % study_image_id] = study_post_trial_cw_fidelity[1][0]      

            iw_maintain_dict['image ID: %s' % study_image_id] = pre_study_trial_iw_fidelity[1][0]
            uw_maintain_dict['image ID: %s' % study_image_id] = pre_study_trial_uw_fidelity[1][0]
            cw_maintain_dict['image ID: %s' % study_image_id] = pre_study_trial_cw_fidelity[1][0]    

        elif image_condition==2:

            change_iw_replace_dict['image ID: %s' % study_image_id] = pre_post_trial_iw_fidelity[1][0]
            change_uw_replace_dict['image ID: %s' % study_image_id] = pre_post_trial_uw_fidelity[1][0]
            change_cw_replace_dict['image ID: %s' % study_image_id] = pre_post_trial_cw_fidelity[1][0]

            modify_iw_replace_dict['image ID: %s' % study_image_id] = study_post_trial_iw_fidelity[1][0]
            modify_uw_replace_dict['image ID: %s' % study_image_id] = study_post_trial_uw_fidelity[1][0]
            modify_cw_replace_dict['image ID: %s' % study_image_id] = study_post_trial_cw_fidelity[1][0] 

            iw_replace_dict['image ID: %s' % study_image_id] = pre_study_trial_iw_fidelity[1][0]
            uw_replace_dict['image ID: %s' % study_image_id] = pre_study_trial_uw_fidelity[1][0]
            cw_replace_dict['image ID: %s' % study_image_id] = pre_study_trial_cw_fidelity[1][0]              

        elif image_condition==3:

            change_iw_suppress_dict['image ID: %s' % study_image_id] = pre_post_trial_iw_fidelity[1][0]
            change_uw_suppress_dict['image ID: %s' % study_image_id] = pre_post_trial_uw_fidelity[1][0]
            change_cw_suppress_dict['image ID: %s' % study_image_id] = pre_post_trial_cw_fidelity[1][0]

            modify_iw_suppress_dict['image ID: %s' % study_image_id] = study_post_trial_iw_fidelity[1][0]
            modify_uw_suppress_dict['image ID: %s' % study_image_id] = study_post_trial_uw_fidelity[1][0]
            modify_cw_suppress_dict['image ID: %s' % study_image_id] = study_post_trial_cw_fidelity[1][0]

            iw_suppress_dict['image ID: %s' % study_image_id] = pre_study_trial_iw_fidelity[1][0]
            uw_suppress_dict['image ID: %s' % study_image_id] = pre_study_trial_uw_fidelity[1][0]
            cw_suppress_dict['image ID: %s' % study_image_id] = pre_study_trial_cw_fidelity[1][0]  
        
        counter=counter+1    

    #now take the corrcoef of un-weighted to unweighted, then category weight to category weight, finally item to item
    #if everything I did above is correct (i have sanity checked it a couple times), then the order of the pre and the study are in the same trial order, which is in order but by subcategory
    #now I just need to get the corrcoefs, plot them and save... will add quanitification of changes back
    #also the pre only's will have all 120 images shown, while the comps are only the 90 shown in the study phase


    unweighted_pre_only=np.corrcoef(masked_bolds_arr_1)
    temp_df=pd.DataFrame(data=unweighted_pre_only)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'unweighted_pre_only_RSA.csv'))
    del temp_df

    unweighted_pre_study_comp=np.corrcoef(non_weight_pre_comp,non_weight_study_comp)
    temp_df=pd.DataFrame(data=unweighted_pre_study_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'unweighted_pre_study_RSA.csv'))
    del temp_df

    unweighted_pre_post_comp=np.corrcoef(non_weight_pre_comp,non_weight_post_comp)
    temp_df=pd.DataFrame(data=unweighted_pre_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'unweighted_pre_post_RSA.csv'))
    del temp_df

    unweighted_study_post_comp=np.corrcoef(non_weight_study_comp,non_weight_post_comp)
    temp_df=pd.DataFrame(data=unweighted_study_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'unweighted_study_post_RSA.csv'))
    del temp_df    

    category_pre_only=np.corrcoef(cate_repress_pre)
    temp_df=pd.DataFrame(data=category_pre_only)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'scene_weight_pre_only_RSA.csv'))
    del temp_df  

    category_pre_study_comp=np.corrcoef(category_weight_pre_comp,category_weight_study_comp)
    temp_df=pd.DataFrame(data=category_pre_study_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'scene_weight_pre_study_RSA.csv'))
    del temp_df  

    category_pre_post_comp=np.corrcoef(category_weight_pre_comp,category_weight_post_comp)
    temp_df=pd.DataFrame(data=category_pre_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'scene_weight_pre_post_RSA.csv'))
    del temp_df  

    category_study_post_comp=np.corrcoef(category_weight_study_comp,category_weight_post_comp)    
    temp_df=pd.DataFrame(data=category_study_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'scene_weight_study_post_RSA.csv'))
    del temp_df     

    item_pre_only=np.corrcoef(item_repress_pre)
    temp_df=pd.DataFrame(data=item_pre_only)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'item_weight_pre_only_RSA.csv'))
    del temp_df  

    item_pre_study_comp=np.corrcoef(item_repress_pre_comp,item_repress_study_comp)
    temp_df=pd.DataFrame(data=item_pre_study_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'item_weight_pre_study_RSA.csv'))
    del temp_df  

    item_pre_post_comp=np.corrcoef(item_repress_pre_comp,item_repress_post_comp)
    temp_df=pd.DataFrame(data=item_pre_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'item_weight_pre_post_RSA.csv'))
    del temp_df 

    item_study_post_comp=np.corrcoef(item_repress_study_comp,item_repress_post_comp)
    temp_df=pd.DataFrame(data=item_study_post_comp)
    temp_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'item_weight_study_post_RSA.csv'))
    del temp_df     

    #now lets plot all of these and save:
    if not os.path.exists(os.path.join(container_path,"sub-0%s" % subID,"RSA")): os.makedirs(os.path.join(container_path,"sub-0%s" % subID,"RSA"),exist_ok=True)

    fig=sns.heatmap(unweighted_pre_only)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Unweighted - Prelocalizer only')    
    fig.axhline([60])
    fig.axvline([60])
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA",'unweighted_pre_only.png'))
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
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA",'unweighted_pre_vs_study.png'))
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
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA",'unweighted_pre_vs_post.png'))
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
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA",'unweighted_study_vs_post.png'))
    plt.clf()         

    fig=sns.heatmap(category_pre_only)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Scene weighted - Prelocalizer only')
    fig.axhline([60])
    fig.axvline([60])
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA",'scene_weighted_pre_only.png'))
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
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA",'scene_weighted_pre_vs_study.png'))
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
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA",'scene_weighted_pre_vs_post.png'))
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
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA",'scene_weighted_study_vs_post.png'))
    plt.clf()        

    fig=sns.heatmap(item_pre_only)
    fig.set_xlabel('Total Trial #')
    fig.set_ylabel('Total Trial #')
    fig.set_title('RSA - Item weighted - Prelocalizer only')    
    fig.axhline([60])
    fig.axvline([60])    
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA",'item_weighted_pre_only.png'))
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
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA",'item_weighted_pre_vs_study.png'))
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
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA",'item_weighted_pre_vs_post.png'))
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
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"RSA",'item_weighted_study_vs_post.png'))
    plt.clf()     

    # print('Average LSS Fidelity of unoperated items: %s | std: %s' % (np.array(list(LSS_unoperated_dict.values())).mean(),np.array(list(LSS_unoperated_dict.values())).std()))
    # print('Average LSS Fidelity of maintained items: %s | std: %s' % (np.array(list(LSS_maintain_dict.values())).mean(),np.array(list(LSS_maintain_dict.values())).std()))
    # print('Average LSS Fidelity of replaced items: %s | std: %s' % (np.array(list(LSS_replace_dict.values())).mean(),np.array(list(LSS_replace_dict.values())).std()))
    # print('Average LSS Fidelity of suppressed items: %s | std: %s' % (np.array(list(LSS_suppress_dict.values())).mean(),np.array(list(LSS_suppress_dict.values())).std()))

    if not os.path.exists(os.path.join(container_path,"sub-0%s" % subID,"Representational_Changes")): os.makedirs(os.path.join(container_path,"sub-0%s" % subID,"Representational_Changes"),exist_ok=True)

    #organize this data into dataframes (which may be the best way to handle this data):
    #pre vs. post changes
    itemw_pre_df=pd.DataFrame()
    itemw_pre_df['maintain']=np.array(list(change_iw_maintain_dict.values()))
    itemw_pre_df['replace']=np.array(list(change_iw_replace_dict.values()))
    itemw_pre_df['suppress']=np.array(list(change_iw_suppress_dict.values()))
    itemw_pre_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'itemweighted_pre_post_fidelity.csv'))

    catew_pre_df=pd.DataFrame()
    catew_pre_df['maintain']=np.array(list(change_cw_maintain_dict.values()))
    catew_pre_df['replace']=np.array(list(change_cw_replace_dict.values()))
    catew_pre_df['suppress']=np.array(list(change_cw_suppress_dict.values()))
    catew_pre_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'sceneweighted_pre_post_fidelity.csv'))

    unw_pre_df=pd.DataFrame()
    unw_pre_df['maintain']=np.array(list(change_uw_maintain_dict.values()))
    unw_pre_df['replace']=np.array(list(change_uw_replace_dict.values()))
    unw_pre_df['suppress']=np.array(list(change_uw_suppress_dict.values()))
    unw_pre_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'unweighted_pre_post_fidelity.csv'))    

    #study vs. post changes
    itemw_study_df=pd.DataFrame()
    itemw_study_df['maintain']=np.array(list(modify_iw_maintain_dict.values()))
    itemw_study_df['replace']=np.array(list(modify_iw_replace_dict.values()))
    itemw_study_df['suppress']=np.array(list(modify_iw_suppress_dict.values()))
    itemw_study_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'itemweighted_study_post_fidelity.csv'))

    catew_study_df=pd.DataFrame()
    catew_study_df['maintain']=np.array(list(modify_cw_maintain_dict.values()))
    catew_study_df['replace']=np.array(list(modify_cw_replace_dict.values()))
    catew_study_df['suppress']=np.array(list(modify_cw_suppress_dict.values()))
    catew_study_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'sceneweighted_study_post_fidelity.csv'))

    unw_study_df=pd.DataFrame()
    unw_study_df['maintain']=np.array(list(modify_uw_maintain_dict.values()))
    unw_study_df['replace']=np.array(list(modify_uw_replace_dict.values()))
    unw_study_df['suppress']=np.array(list(modify_uw_suppress_dict.values()))
    unw_study_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'unweighted_study_post_fidelity.csv'))  

    #pre vs. study changes
    itemw_df=pd.DataFrame()
    itemw_df['maintain']=np.array(list(iw_maintain_dict.values()))
    itemw_df['replace']=np.array(list(iw_replace_dict.values()))
    itemw_df['suppress']=np.array(list(iw_suppress_dict.values()))
    itemw_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'itemweighted_pre_study_fidelity.csv'))

    catew_df=pd.DataFrame()
    catew_df['maintain']=np.array(list(cw_maintain_dict.values()))
    catew_df['replace']=np.array(list(cw_replace_dict.values()))
    catew_df['suppress']=np.array(list(cw_suppress_dict.values()))
    catew_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'sceneweighted_pre_study_fidelity.csv'))

    unw_df=pd.DataFrame()
    unw_df['maintain']=np.array(list(uw_maintain_dict.values()))
    unw_df['replace']=np.array(list(uw_replace_dict.values()))
    unw_df['suppress']=np.array(list(uw_suppress_dict.values()))
    unw_df.to_csv(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'unweighted_pre_study_fidelity.csv')) 

    #plot and save the figures of the data - Pre vs Post
    fig=sns.barplot(data=itemw_pre_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Item Weighted - Pre vs. Post')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'Item_Weighted_pre_post_summary.png'))
    plt.clf()  

    fig=sns.barplot(data=catew_pre_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Scene Weighted - Pre vs. Post')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'Scene_Weighted_pre_post_summary.png'))
    plt.clf() 

    fig=sns.barplot(data=unw_pre_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Unweighted - Pre vs. Post')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'Unweighted_pre_post_summary.png'))
    plt.clf()      

    #plot and save figures of the data - Study vs Post
    fig=sns.barplot(data=itemw_study_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Item Weighted - Study vs. Post')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'Item_Weighted_study_post_summary.png'))
    plt.clf()  

    fig=sns.barplot(data=catew_study_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Scene Weighted - Study vs. Post')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'Scene_Weighted_study_post_summary.png'))
    plt.clf() 

    fig=sns.barplot(data=unw_study_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Unweighted - Study vs. Post')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'Unweighted_study_post_summary.png'))
    plt.clf()       

    #plot and save figures of the data - Pre vs Study
    fig=sns.barplot(data=itemw_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Item Weighted - Pre vs. Study')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'Item_Weighted_pre_study_summary.png'))
    plt.clf()  

    fig=sns.barplot(data=catew_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Scene Weighted - Pre vs. Study')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'Scene_Weighted_pre_study_summary.png'))
    plt.clf() 

    fig=sns.barplot(data=unw_df)
    fig.set_xlabel('Operations')
    fig.set_ylabel('Fidelity')
    fig.set_title('Unweighted - Pre vs. Study')
    plt.savefig(os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes",'Unweighted_pre_study_summary.png'))
    plt.clf()

    #quickly summarize the statistics:
    # print('One-Way ANOVA on LSS:')
    # print(f_oneway(LSS_df['unoperated'],LSS_df['maintain'],LSS_df['replace'],LSS_df['suppress']))

    # #organize this data into dataframes (which may be the best way to handle this data):
    # LSA_df=pd.DataFrame()
    # LSA_df['unoperated']=np.array(list(LSA_unoperated_dict.values()))
    # LSA_df['maintain']=np.array(list(LSA_maintain_dict.values()))
    # LSA_df['replace']=np.array(list(LSA_replace_dict.values()))
    # LSA_df['suppress']=np.array(list(LSA_suppress_dict.values()))
    # LSA_df.to_csv(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSA_fidelity.csv'))

    # #plot and save the figures of the data
    # fig=sns.barplot(data=LSA_df)
    # fig.set_xlabel('Operations')
    # fig.set_ylabel('Fidelity')
    # fig.set_title('LSA - Pre vs. Post')
    # plt.savefig(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSA_fidelity_bar_summary.png'))
    # plt.clf()

    # fig=sns.violinplot(data=LSA_df,inner='point')
    # fig.set_xlabel('Operations')
    # fig.set_ylabel('Fidelity')
    # fig.set_title('LSA - Pre vs. Post')    
    # plt.savefig(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSA_fidelity_violin_summary.png'))
    # plt.clf()    

    # #quickly summarize the statistics:
    # print('One-Way ANOVA on LSA:')
    # print(f_oneway(LSA_df['unoperated'],LSA_df['maintain'],LSA_df['replace'],LSA_df['suppress']))


    # #this is just dumping all the individual dictionaries, which is nice since each comparison is labeled with the corresponding image
    # dict_file= open(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSS_unoperated.pkl'),"wb")
    # pickle.dump(LSS_unoperated_dict,dict_file)
    # dict_file.close()
    # del dict_file

    # dict_file= open(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSA_unoperated.pkl'),"wb")
    # pickle.dump(LSA_unoperated_dict,dict_file)
    # dict_file.close()
    # del dict_file

    # dict_file= open(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSS_Maintain.pkl'),"wb")
    # pickle.dump(LSS_maintain_dict,dict_file)
    # dict_file.close()
    # del dict_file

    # dict_file= open(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSA_Maintain.pkl'),"wb")
    # pickle.dump(LSA_maintain_dict,dict_file)
    # dict_file.close()
    # del dict_file

    # dict_file= open(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSS_Replace.pkl'),"wb")
    # pickle.dump(LSS_replace_dict,dict_file)
    # dict_file.close()
    # del dict_file

    # dict_file= open(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSA_Replace.pkl'),"wb")
    # pickle.dump(LSA_replace_dict,dict_file)
    # dict_file.close()
    # del dict_file

    # dict_file= open(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSS_Suppress.pkl'),"wb")
    # pickle.dump(LSS_suppress_dict,dict_file)
    # dict_file.close()
    # del dict_file

    # dict_file= open(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSA_Suppress.pkl'),"wb")
    # pickle.dump(LSA_suppress_dict,dict_file)
    # dict_file.close()    

    print("Subject is done... saving everything")
    print("===============================================================================")    

#now I want to take the subject level "fidelity" results and look at it on a group level - I will also want to quantify the off diagonal of the RSA to random pairs
group_item_weighted_pre_post=pd.DataFrame()
group_item_weighted_study_post=pd.DataFrame()
group_item_weighted_pre_study=pd.DataFrame()

group_cate_weighted_pre_post=pd.DataFrame()
group_cate_weighted_study_post=pd.DataFrame()
group_cate_weighted_pre_study=pd.DataFrame()

group_unweighted_pre_post=pd.DataFrame()
group_unweighted_study_post=pd.DataFrame()
group_unweighted_pre_study=pd.DataFrame()

for subID in subs:
    data_path=os.path.join(container_path,"sub-0%s"  % subID,"Representational_Changes")

    item_pre_post=find('item*pre*post*',data_path)
    group_item_weighted_pre_post=group_item_weighted_pre_post.append(pd.read_csv(item_pre_post[0],usecols=[1,2,3]),ignore_index=True)

    item_study_post=find('item*study*post*',data_path)
    group_item_weighted_study_post=group_item_weighted_study_post.append(pd.read_csv(item_study_post[0],usecols=[1,2,3]),ignore_index=True)

    item_pre_study=find('item*pre*study*',data_path)
    group_item_weighted_pre_study=group_item_weighted_pre_study.append(pd.read_csv(item_pre_study[0],usecols=[1,2,3]),ignore_index=True)

    cate_pre_post=find('scene*pre*post*',data_path)
    group_cate_weighted_pre_post=group_cate_weighted_pre_post.append(pd.read_csv(cate_pre_post[0],usecols=[1,2,3]),ignore_index=True)

    cate_study_post=find('scene*study*post*',data_path)
    group_cate_weighted_study_post=group_cate_weighted_study_post.append(pd.read_csv(cate_study_post[0],usecols=[1,2,3]),ignore_index=True)

    cate_pre_study=find('scene*pre*study*',data_path)
    group_cate_weighted_pre_study=group_cate_weighted_pre_study.append(pd.read_csv(cate_study_post[0],usecols=[1,2,3]),ignore_index=True)

    un_pre_post=find('unweighted*pre*post*',data_path)
    group_unweighted_pre_post=group_unweighted_pre_post.append(pd.read_csv(un_pre_post[0],usecols=[1,2,3]),ignore_index=True)

    un_study_post=find('unweighted*study*post*',data_path)
    group_unweighted_study_post=group_unweighted_study_post.append(pd.read_csv(un_study_post[0],usecols=[1,2,3]),ignore_index=True)

    un_pre_study=find('unweighted*pre*study*',data_path)
    group_unweighted_pre_study=group_unweighted_pre_study.append(pd.read_csv(un_pre_study[0],usecols=[1,2,3]),ignore_index=True)


if not os.path.exists(os.path.join(container_path,"group_model","Representational_Changes")): os.makedirs(os.path.join(container_path,"group_model","Representational_Changes"),exist_ok=True)
#plot and save the figures of the data - Pre vs Post
fig=sns.barplot(data=group_item_weighted_pre_post)
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Item Weighted (Group Level) - Pre vs. Post')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes",'Group_Item_Weighted_pre_post_summary.png'))
plt.clf() 

fig=sns.barplot(data=group_item_weighted_study_post)
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Item Weighted (Group Level) - Study vs. Post')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes",'Group_Item_Weighted_study_post_summary.png'))
plt.clf() 

fig=sns.barplot(data=group_item_weighted_pre_study)
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Item Weighted (Group Level) - Pre vs. Study')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes",'Group_Item_Weighted_pre_study_summary.png'))
plt.clf() 

fig=sns.barplot(data=group_cate_weighted_pre_post)
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Scene Weighted (Group Level) - Pre vs. Post')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes",'Group_Scene_Weighted_pre_post_summary.png'))
plt.clf() 

fig=sns.barplot(data=group_cate_weighted_study_post)
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Scene Weighted (Group Level) - Study vs. Post')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes",'Group_Scene_Weighted_study_post_summary.png'))
plt.clf() 

fig=sns.barplot(data=group_cate_weighted_pre_study)
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Scene Weighted (Group Level) - Pre vs. Study')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes",'Group_Scene_Weighted_pre_study_summary.png'))
plt.clf() 

fig=sns.barplot(data=group_unweighted_pre_post)
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Unweighted (Group Level) - Pre vs. Post')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes",'Group_Unweighted_pre_post_summary.png'))
plt.clf() 

fig=sns.barplot(data=group_unweighted_study_post)
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Unweighted (Group Level) - Study vs. Post')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes",'Group_Unweighted_study_post_summary.png'))
plt.clf() 

fig=sns.barplot(data=group_unweighted_pre_study)
fig.set_xlabel('Operations')
fig.set_ylabel('Fidelity')
fig.set_title('Unweighted (Group Level) - Pre vs. study')
plt.savefig(os.path.join(container_path,"group_model","Representational_Changes",'Group_Unweighted_pre_study_summary.png'))
plt.clf() 