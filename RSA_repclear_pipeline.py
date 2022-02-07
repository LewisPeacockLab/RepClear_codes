#I need to develop some quick code that will take things in and create RSA matrices with ease
#I can likely extract something from Suna's MDS code to speed things up

#I will also need to pull some code to access the mat files easily, since that will speed up most of this process if I have a nice dataframe to source from

#I will need to load in each subjects design information based on the phase
# then I can do something like target the "trial" column, so simply just assign an incrimenting # to each trial and link that to the name

#the result I want to get is to be able to first take the localizer data, take the design to assign the trial #'s with the name and then organize it into a matrix (focus on the diagonal, since I will only plot the scenes)

#next I will create some code to load in the category weights for scenes, and then the item weights
# this code will be similar to the one above, just simply matching item weights, and then organize into two matrices (category weighting and item weighting)

#next part is to then load in the representations from the study phase encoding period, then will again want to set up 3 matrices... unweighted RSA, Category RSA, and then item weighted RSA. Again this will occur by making sure that each trial # is linked to the item NAME and then pair them based on the name
    # A way I think I can do that generally is to create a dictionary, and first take the localizer trial # and find the name, then save that pattern to a dict with the name as the label. Then I can do the same thing via the study phase. 

    #In general this is a good idea since it will remove the "trial #" part of this and focus on the image names... and dict's are searchable via name so thats easy to add into the dict all the representations we want

#I will need to get figures of all of this (total of 6 matrices: 3 for just localizer, 3 for localizer->study)


#After all of this, the next step is to compare the localizer to the LTM, and then summarize the "correlations" by condition. 


#---

#This code is to load in the representations and then either weight them (category vs. item) and then perform RSA
# This will also handle performing this and then comparing Localizer to Study

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
    tim_df2 = tim_df2[tim_df2["phase"]==3] #phase 2 is pre-localizer
    tim_df2 = tim_df2.sort_values(by=["category", "subcategory", "trial_id"])
    
    study_scene_order = tim_df2[tim_df2["category"]==1][["trial_id","image_id","condition"]]

    print(f"Running RSA for sub{subID}...")

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


    #when comparing pre vs. study, the pre scene have 120 trials, while study has 90 trials

    # apply VTC mask on prelocalizer BOLD
    masked_bolds_arr_1 = []
    for bold in bolds_arr_1:
        masked_bolds_arr_1.append(apply_mask(mask=mask.get_fdata(), target=bold).flatten())
    masked_bolds_arr_1 = np.vstack(masked_bolds_arr_1)
    print("masked prelocalizer bold array shape: ", masked_bolds_arr_1.shape)

    # apply mask on BOLD
    masked_bolds_arr_2 = []
    for bold in bolds_arr_2:
        masked_bolds_arr_2.append(apply_mask(mask=mask.get_fdata(), target=bold).flatten())
    masked_bolds_arr_2 = np.vstack(masked_bolds_arr_2)
    print("masked phase2 bold array shape: ", masked_bolds_arr_2.shape)    

    # ===== load weights
    print(f"Loading weights...")
    # prelocalizer
    cate_weights_dir = "/scratch/06873/zbretton/fmriprep/group_model/group_category_lvl2/group_scene_ovr_face_MNI_zmap.nii.gz" #full path to the scene weights
    item_weights_dir = os.path.join(container_path, f"sub-0{subID}", "preremoval_item_level")

    #prelocalizer weights (category and item) get applied to study representations

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

    #this now masks the item weights to ensure that they are all in the same ROI:
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
    
    #study patterns and prelocalizer item weights    
    #this has to come in the loop below to make sure I am weighting the correct trial with the correct item weights

    print("item representations pre shape: ", item_repress_pre.shape)

    #these work right out the box since there is only 1 "category" weighting we are using, and that can be applied to all scene trials in both pre and study (and post)
    cate_repress_pre = np.multiply(masked_bolds_arr_1,masked_cate_weights_arr) #these are multiplied elementwise
    cate_repress_study = np.multiply(masked_bolds_arr_2,masked_cate_weights_arr) #since there is only 1 cate_weight, this multiplies all of masked_bold_arr_2 with the cate_weights


    print("category representations pre shape: ", cate_repress_pre.shape)
    print("category representations study shape: ", cate_repress_study.shape)

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
    non_weight_study_comp=np.zeros_like(item_repress_study_comp)
    non_weight_pre_comp=np.zeros_like(item_repress_study_comp)

    category_weight_study_comp=np.zeros_like(item_repress_study_comp)
    category_weight_pre_comp=np.zeros_like(item_repress_study_comp)

    counter=0
    #this loop is limited by the smaller index, so thats the study condition (only 90 stims)
    for trial in study_scene_order['trial_id'].values: 

        study_trial_index=study_scene_order.index[study_scene_order['trial_id']==trial].tolist()[0] #find the order 
        study_image_id=study_scene_order.loc[study_trial_index,'image_id'] #this now uses the index of the dataframe to find the image_id
        #study_trial_num=study_scene_order.loc[study_trial_index,'trial_id'] #now we used the image ID to find the proper trial in the post condition to link to


        pre_trial_index=pre_scene_order.index[pre_scene_order['image_id']==study_image_id].tolist()[0] #find the index in the pre for this study trial
        pre_trial_num = pre_scene_order.loc[pre_trial_index,'trial_id']

        image_condition=pre_scene_order.loc[pre_trial_index,'condition']
        #this mean we now know both the trial #, the image id and we can also grab the condition to help sort

        #now that I have the link between prelocalizer and study I can get that representation weighted with the item weight
        item_repress_study_comp[counter]=np.multiply(masked_bolds_arr_2[trial-1,:], masked_weights_arr[pre_trial_num-1,:])
        item_repress_pre_comp[counter]=item_repress_pre[pre_trial_num-1,:]

        #we will also want to filter the results like above for the two other iterations we want, non-weighted and category
        non_weight_study_comp[counter]=masked_bolds_arr_2[trial-1,:]
        non_weight_pre_comp[counter]=masked_bolds_arr_1[pre_trial_num-1,:]

        category_weight_study_comp[counter]=cate_repress_study[trial-1,:]
        category_weight_pre_comp[counter]=cate_repress_pre[pre_trial_num-1,:]

        counter=counter+1
        #I am using this counter to preserve the ordering that results from the csv's sorting at the top
        #that had the trials in order but segmented by subcate, which I think is a better organization since actual trial number is not needed

    #now take the corrcoef of un-weighted to unweighted, then category weight to category weight, finally item to item
    #if everything I did above is correct (i have sanity checked it a couple times), then the order of the pre and the study are in the same trial order, which is in order but by subcategory
    #now I just need to get the corrcoefs, plot them and save... will add quanitification of changes back
    #also the pre only's will have all 120 images shown, while the comps are only the 90 shown in the study phase


    unweighted_pre_only=np.corrcoef(masked_bolds_arr_1)
    unweighted_pre_study_comp=np.corrcoef(non_weight_pre_comp,non_weight_study_comp)

    category_pre_only=np.corrcoef(cate_repress_pre)
    category_pre_study_comp=np.corrcoef(category_weight_pre_comp,category_weight_study_comp)

    item_pre_only=np.corrcoef(item_repress_pre)
    item_pre_study_comp=np.corrcoef(item_repress_pre_comp,item_repress_study_comp)


    #now lets plot all of these and save:
    if not os.path.exists(os.path.join(container_path,"sub-0%s" % subID,"RSA")): os.makedirs(os.path.join(container_path,"sub-0%s" % subID,"RSA"),exist_ok=True)

    fig=sns.heatmap(unweighted_pre_only)
    fig.set_xlabel('Trial #')
    fig.set_ylabel('Trial #')
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

        # #this code was used to link the correlation of the patterns from pre to post, will revive later
        # LSS_trial_fidelity=np.corrcoef(item_LSS_pre[trial+len(pre_face_order),:],item_LSS_post[(post_trial_num-1),:])
        # LSA_trial_fidelity=np.corrcoef(item_repress_pre[trial+len(pre_face_order),:],item_repress_post[(post_trial_num-1),:])        
        # if image_condition==0:
        #     LSS_unoperated_dict['image ID: %s' % image_id] = LSS_trial_fidelity[1][0]
        #     LSA_unoperated_dict['image ID: %s' % image_id] = LSA_trial_fidelity[1][0]            
        # elif image_condition==1:
        #     LSS_maintain_dict['image ID: %s' % image_id] = LSS_trial_fidelity[1][0]
        #     LSA_maintain_dict['image ID: %s' % image_id] = LSA_trial_fidelity[1][0]            
        # elif image_condition==2:
        #     LSS_replace_dict['image ID: %s' % image_id] = LSS_trial_fidelity[1][0]
        #     LSA_replace_dict['image ID: %s' % image_id] = LSA_trial_fidelity[1][0]            
        # elif image_condition==3:
        #     LSS_suppress_dict['image ID: %s' % image_id] = LSS_trial_fidelity[1][0]    
        #     LSA_suppress_dict['image ID: %s' % image_id] = LSA_trial_fidelity[1][0]                        

    # print('Average LSS Fidelity of unoperated items: %s | std: %s' % (np.array(list(LSS_unoperated_dict.values())).mean(),np.array(list(LSS_unoperated_dict.values())).std()))
    # print('Average LSS Fidelity of maintained items: %s | std: %s' % (np.array(list(LSS_maintain_dict.values())).mean(),np.array(list(LSS_maintain_dict.values())).std()))
    # print('Average LSS Fidelity of replaced items: %s | std: %s' % (np.array(list(LSS_replace_dict.values())).mean(),np.array(list(LSS_replace_dict.values())).std()))
    # print('Average LSS Fidelity of suppressed items: %s | std: %s' % (np.array(list(LSS_suppress_dict.values())).mean(),np.array(list(LSS_suppress_dict.values())).std()))

    # if not os.path.exists(os.path.join(data_dir,"sub-%s" % subID,"Representational_Changes")): os.makedirs(os.path.join(data_dir,"sub-%s" % subID,"Representational_Changes"),exist_ok=True)

    # #organize this data into dataframes (which may be the best way to handle this data):
    # LSS_df=pd.DataFrame()
    # LSS_df['unoperated']=np.array(list(LSS_unoperated_dict.values()))
    # LSS_df['maintain']=np.array(list(LSS_maintain_dict.values()))
    # LSS_df['replace']=np.array(list(LSS_replace_dict.values()))
    # LSS_df['suppress']=np.array(list(LSS_suppress_dict.values()))
    # LSS_df.to_csv(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSS_fidelity.csv'))

    # pd.DataFrame(LSS_maintain_dict,index=[0]).to_csv(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSS_maintain_fidelity.csv'))
    # pd.DataFrame(LSS_replace_dict,index=[0]).to_csv(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSS_replace_fidelity.csv'))
    # pd.DataFrame(LSS_suppress_dict,index=[0]).to_csv(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSS_suppress_fidelity.csv'))
    # pd.DataFrame(LSS_unoperated_dict,index=[0]).to_csv(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSS_unoperated_fidelity.csv'))



    # #plot and save the figures of the data
    # fig=sns.barplot(data=LSS_df)
    # fig.set_xlabel('Operations')
    # fig.set_ylabel('Fidelity')
    # fig.set_title('LSS - Pre vs. Post')
    # plt.savefig(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSS_fidelity_bar_summary.png'))
    # plt.clf()

    # fig=sns.violinplot(data=LSS_df,inner='point')
    # fig.set_xlabel('Operations')
    # fig.set_ylabel('Fidelity')
    # fig.set_title('LSS - Pre vs. Post')    
    # plt.savefig(os.path.join(data_dir,"sub-%s"  % subID,"Representational_Changes",'LSS_fidelity_violin_summary.png'))
    # plt.clf()    

    # #quickly summarize the statistics:
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