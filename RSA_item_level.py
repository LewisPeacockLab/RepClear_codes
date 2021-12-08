#RSA via nilearn for Clearmem - This will be the item level analysis to get weighting 
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
from scipy import stats
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import PredefinedSplit, cross_validate, cross_val_predict, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest, SelectFpr
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
from nilearn.image import clean_img, concat_imgs, mean_img, load_img
from nilearn.signal import clean
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.plotting import plot_stat_map, plot_anat, plot_img

subs=['01','04','06','10','11','15','18','21','23','27','34','36','42','44','45','55','61','69','77','79']
#subs=['01','06','36','77','79']

brain_flag='T1w'
mask_flag='wholebrain'

for num in range(len(subs)):
    sub_num=subs[num]

    print('Running sub-0%s...' %sub_num)
    #define the subject
    #sub = ('sub-0%s' % sub_num)
    #container_path='/scratch/06873/zbretton/clearmem/derivatives/fmriprep'
  
    #bold_path=os.path.join(container_path,sub,'func/')
    #os.chdir(bold_path)
    sub = ('sub-0%s' % sub_num)
    # container_path='/Users/zhbre/Desktop/clearmem/derivatives/fmriprep'
    container_path='/scratch/06873/zbretton/clearmem/derivatives/fmriprep'

  
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

    bold_files=find('*localizer*bold*.nii.gz',bold_path)
    wholebrain_mask_path=find('*localizer*mask*.nii.gz',bold_path)

    face_mask_path=os.path.join(container_path,sub,'results','face_v_other_mask.nii.gz')
    fruit_mask_path=os.path.join(container_path,sub,'results','fruit_v_other_mask.nii.gz')
    scene_mask_path=os.path.join(container_path,sub,'results','scene_v_other_mask.nii.gz')
    
    face_mask=nib.load(face_mask_path)
    fruit_mask=nib.load(fruit_mask_path)
    scene_mask=nib.load(scene_mask_path)
    
    if brain_flag=='MNI':
        pattern = '*MNI*'
        pattern2= '*MNI152NLin2009cAsym*preproc*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        localizer_files = fnmatch.filter(bold_files,pattern2)
    elif brain_flag=='T1w':
        pattern = '*T1w*'
        pattern2 = '*T1w*preproc*'
        brain_mask_path = fnmatch.filter(wholebrain_mask_path,pattern)
        localizer_files = fnmatch.filter(bold_files,pattern2)

    localizer_files.sort()

    localizer_run1=nib.load(localizer_files[0])
    localizer_run2=nib.load(localizer_files[1])
    localizer_run3=nib.load(localizer_files[2])
    localizer_run4=nib.load(localizer_files[3])
    localizer_run5=nib.load(localizer_files[4])
    #to be used to filter the data
    #First we are removing the confounds
    #get all the folders within the bold path
    #confound_folders=[x[0] for x in os.walk(bold_path)]

    wholebrain_mask1=nib.load(brain_mask_path[0])


    #### Found out that pre-masking was an issue here

    #fmri_img = concat_imgs([preproc_1,preproc_2,preproc_3,preproc_4,preproc_5]) #masked
    fmri_img = concat_imgs([localizer_run1,localizer_run2,localizer_run3,localizer_run4,localizer_run5]) #unmasked    
    background=concat_imgs([localizer_run1,localizer_run2,localizer_run3,localizer_run4,localizer_run5]) #take unmasked data to make background
    mean_img_temp = mean_img(fmri_img)

    del background    

    params_dir='/scratch/06873/zbretton/params'
    # params_dir='/Users/zhbre/Desktop/Grad School/clearmem_take2/'    #find the mat file, want to change this to fit "sub"
    param_search='localizer*events*item*.csv'
    param_file=find(param_search,params_dir)
    reg_matrix = pd.read_csv(param_file[0]) #load in the build events file
    events=reg_matrix #grab the last 4 columns, those are of note (onset, duration, trial_type)
    #will need to refit "events" to work for each kind of contrast I want to do
    events = events[events.category != 0].reset_index(drop=True) #drop rest times
    # to grab the run1 values --> events.loc[events['run']==1]
    fmri_glm_face = FirstLevelModel(t_r=0.46, mask_img=face_mask,
                           noise_model='ar1',
                           standardize=False,
                           hrf_model='spm',
                           drift_model='cosine',
                           high_pass=.01)
    fmri_glm_fruit = FirstLevelModel(t_r=0.46, mask_img=fruit_mask,
                           noise_model='ar1',
                           standardize=False,
                           hrf_model='spm',
                           drift_model='cosine',
                           high_pass=.01)
    fmri_glm_scene = FirstLevelModel(t_r=0.46, mask_img=scene_mask,
                           noise_model='ar1',
                           standardize=False,
                           hrf_model='spm',
                           drift_model='cosine',
                           high_pass=.01)                                                          
    fmri_glm_face=fmri_glm_face.fit(fmri_img, events)
    fmri_glm_fruit=fmri_glm_fruit.fit(fmri_img, events)
    fmri_glm_scene=fmri_glm_scene.fit(fmri_img, events)        
    #fmri_glm=fmri_glm.fit(fmri_img, events.loc[events['run']==1]) just running run 1

    design_matrix = fmri_glm_face.design_matrices_[0]
    #plot_design_matrix(design_matrix)
    #plt.show() #look at the design matrix
    outdir = os.path.join(container_path,sub,'results')
    if not os.path.exists(outdir):
         os.mkdir(outdir)

    # plot_design_matrix(
    #     design_matrix, output_file=os.path.join(outdir, '%s_design_matrix.png' % sub))

    #Create the contrasts for each item, sorted by its category
    indexes=len(design_matrix.T)
    template=np.zeros(indexes)
    items=list(range(1,55))
    face_cont={}
    fruit_cont={}
    scene_cont={}
    for x in range(1,55):
        if x<=18:
            temp=list(range(1,55))
            temp.remove(x)
            temp_array=np.zeros(indexes)
            temp_array[(x-1)]=1
            temp_array[[number - 1 for number in temp]]=-1
            face_cont['%s vs other' % x] = temp_array
            del temp, temp_array
        elif (x>18)&(x<=36):
            temp=list(range(1,55))
            temp.remove(x)
            temp_array=np.zeros(indexes)
            temp_array[(x-1)]=1
            temp_array[[number - 1 for number in temp]]=-1
            fruit_cont['%s vs other' % x] = temp_array
            del temp, temp_array            
        elif (x>36):
            temp=list(range(1,55))
            temp.remove(x)
            temp_array=np.zeros(indexes)
            temp_array[(x-1)]=1
            temp_array[[number - 1 for number in temp]]=-1
            scene_cont['%s vs other' % x] = temp_array
            del temp, temp_array

    #face constrast
    for x in range(1,55):
        if x<=18:
            stat_map_face = fmri_glm_face.compute_contrast(face_cont['%s vs other' % x],stat_type='t',
                                          output_type='stat')
            z_map_face = fmri_glm_face.compute_contrast(face_cont['%s vs other' % x],stat_type='t',
                                          output_type='z_score')            
            # plot_stat_map(z_map_face, bg_img=mean_img_temp,
            #                display_mode='z', cut_coords=3, black_bg=True,
            #                title='Face item %s minus other' % x)
            #plt.show()
            z_map_face.to_filename(os.path.join(outdir,'item%s_v_other_face_z_map.nii.gz' % x))
            stat_map_face.to_filename(os.path.join(outdir,'item%s_v_other_face_stat_map.nii.gz' % x))
            del z_map_face, stat_map_face 
        elif (x>18)&(x<=36):
            #fruit constrast
            stat_map_fruit = fmri_glm_fruit.compute_contrast(fruit_cont['%s vs other' % x],stat_type='t',
                                          output_type='stat')
            z_map_fruit = fmri_glm_fruit.compute_contrast(fruit_cont['%s vs other' % x],stat_type='t',
                                          output_type='z_score')
            # plot_stat_map(z_map_fruit, bg_img=mean_img_temp,
            #               display_mode='z', cut_coords=3, black_bg=True,
            #               title='Fruit item %s minus other' % x)
            # plt.show()
            z_map_fruit.to_filename(os.path.join(outdir,'item%s_v_other_fruit_z_map.nii.gz' % x))
            stat_map_fruit.to_filename(os.path.join(outdir,'item%s_v_other_fruit_stat_map.nii.gz' % x))
            del z_map_fruit, stat_map_fruit
        elif (x>36):
            #fruit constrast
            stat_map_scene = fmri_glm_scene.compute_contrast(scene_cont['%s vs other' % x],stat_type='t',
                                          output_type='stat')
            z_map_scene = fmri_glm_scene.compute_contrast(scene_cont['%s vs other' % x],stat_type='t',
                                          output_type='z_score')
            # plot_stat_map(z_map_scene, bg_img=mean_img_temp,
            #               display_mode='z', cut_coords=3, black_bg=True,
            #               title='Scene item %s minus other' % x)
            # plt.show()
            z_map_scene.to_filename(os.path.join(outdir,'item%s_v_other_scene_z_map.nii.gz' % x))
            stat_map_scene.to_filename(os.path.join(outdir,'item%s_v_other_scene_stat_map.nii.gz' % x))
            del z_map_scene, stat_map_scene  

    print('Sub-0%s is now complete' % sub_num)