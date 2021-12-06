#RSA via nilearn for Clearmem
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
from nilearn.image import clean_img, concat_imgs, mean_img, load_img, math_img
from nilearn.signal import clean
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.plotting import plot_stat_map, plot_anat, plot_img

subs=['01','04','06','10','11','15','18','21','23','27','34','36','42','44','45','50','55','61','69','77','79']
#subs=['01','06','36','77','79']

brain_flag='T1w'
mask_flag='wholebrain'

for num in range(len(subs)):
    sub_num=subs[num]

    print('Running sub-0%s...' %sub_num)
    #define the subject
    #sub = ('sub-0%s' % sub_num)
    container_path='/scratch/06873/zbretton/clearmem/derivatives/fmriprep'
  
    #bold_path=os.path.join(container_path,sub,'func/')
    #os.chdir(bold_path)
    sub = ('sub-0%s' % sub_num)
    #container_path='/Users/zhbre/Desktop/clearmem/derivatives/fmriprep'
  
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

    vtc_mask_path=os.path.join(container_path,sub,'new_mask','LOC_VTC_study_%s_mask.nii.gz' % brain_flag)
    
    vtc_mask=nib.load(vtc_mask_path)
    
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
    localizer_confounds_1=find('*localizer*1*confounds*.tsv',bold_path)
    localizer_confounds_2=find('*localizer*2*confounds*.tsv',bold_path)
    localizer_confounds_3=find('*localizer*3*confounds*.tsv',bold_path)
    localizer_confounds_4=find('*localizer*4*confounds*.tsv',bold_path)
    localizer_confounds_5=find('*localizer*5*confounds*.tsv',bold_path)

    confound_run1 = pd.read_csv(localizer_confounds_1[0],sep='\t')
    confound_run2 = pd.read_csv(localizer_confounds_2[0],sep='\t')
    confound_run3 = pd.read_csv(localizer_confounds_3[0],sep='\t')
    confound_run4 = pd.read_csv(localizer_confounds_4[0],sep='\t')
    confound_run5 = pd.read_csv(localizer_confounds_5[0],sep='\t')

    confound_run1=confound_run1.fillna(confound_run1.mean())
    confound_run2=confound_run2.fillna(confound_run2.mean())
    confound_run3=confound_run3.fillna(confound_run3.mean())
    confound_run4=confound_run4.fillna(confound_run4.mean())
    confound_run5=confound_run5.fillna(confound_run5.mean())

    wholebrain_mask1=nib.load(brain_mask_path[0])


    #### Found out that pre-masking was an issue here

    #may want to take off z-scoring
    # preproc_1 = clean_img(localizer_run1,confounds=confound_run1.values,t_r=0.46,detrend=False,standardize=False,mask_img=vtc_mask)
    # preproc_2 = clean_img(localizer_run2,confounds=confound_run2.values,t_r=0.46,detrend=False,standardize=False,mask_img=vtc_mask)
    # preproc_3 = clean_img(localizer_run3,confounds=confound_run3.values,t_r=0.46,detrend=False,standardize=False,mask_img=vtc_mask)
    # preproc_4 = clean_img(localizer_run4,confounds=confound_run4.values,t_r=0.46,detrend=False,standardize=False,mask_img=vtc_mask)
    # preproc_5 = clean_img(localizer_run5,confounds=confound_run5.values,t_r=0.46,detrend=False,standardize=False,mask_img=vtc_mask)
    # if mask_flag=='vtc':
    # #VTC masked data
    #     preproc_1 = clean_img(localizer_run1,t_r=0.46,detrend=False,standardize=False,mask_img=vtc_mask)
    #     preproc_2 = clean_img(localizer_run2,t_r=0.46,detrend=False,standardize=False,mask_img=vtc_mask)
    #     preproc_3 = clean_img(localizer_run3,t_r=0.46,detrend=False,standardize=False,mask_img=vtc_mask)
    #     preproc_4 = clean_img(localizer_run4,t_r=0.46,detrend=False,standardize=False,mask_img=vtc_mask)
    #     preproc_5 = clean_img(localizer_run5,t_r=0.46,detrend=False,standardize=False,mask_img=vtc_mask)
    # #wholebrain masked data, used to troubleshooting
    # whole_preproc_1 = clean_img(localizer_run1,t_r=0.46,detrend=False,standardize=False,mask_img=wholebrain_mask1)
    # whole_preproc_2 = clean_img(localizer_run2,t_r=0.46,detrend=False,standardize=False,mask_img=wholebrain_mask1)
    # whole_preproc_3 = clean_img(localizer_run3,t_r=0.46,detrend=False,standardize=False,mask_img=wholebrain_mask1)
    # whole_preproc_4 = clean_img(localizer_run4,t_r=0.46,detrend=False,standardize=False,mask_img=wholebrain_mask1)
    # whole_preproc_5 = clean_img(localizer_run5,t_r=0.46,detrend=False,standardize=False,mask_img=wholebrain_mask1)    


    #fmri_img = concat_imgs([preproc_1,preproc_2,preproc_3,preproc_4,preproc_5]) #masked
    fmri_img = concat_imgs([localizer_run1,localizer_run2,localizer_run3,localizer_run4,localizer_run5]) #unmasked    
    background=concat_imgs([localizer_run1,localizer_run2,localizer_run3,localizer_run4,localizer_run5]) #take unmasked data to make background
    mean_img_temp = mean_img(fmri_img)

    del background    
    
    params_dir='/scratch/06873/zbretton/params'
    #params_dir='/Users/zhbre/Desktop/Grad School/clearmem_take2/'    #find the mat file, want to change this to fit "sub"
    param_search='localizer*events*.csv'
    param_file=find(param_search,params_dir)
    reg_matrix = pd.read_csv(param_file[1]) #load in the build events file
    events=reg_matrix[reg_matrix.columns[-4:]].copy() #grab the last 4 columns, those are of note (onset, duration, trial_type)
    #will need to refit "events" to work for each kind of contrast I want to do
    events = events[events.trial_type != 'rest'].reset_index() #drop rest times
    # to grab the run1 values --> events.loc[events['run']==1]
    fmri_glm = FirstLevelModel(t_r=0.46, mask_img=vtc_mask,
                           noise_model='ar1',
                           standardize=False,
                           hrf_model='spm',
                           drift_model='cosine',
                           high_pass=.01)    
    fmri_glm=fmri_glm.fit(fmri_img, events)
    #fmri_glm=fmri_glm.fit(fmri_img, events.loc[events['run']==1]) just running run 1

    design_matrix = fmri_glm.design_matrices_[0]
    #plot_design_matrix(design_matrix)
    #plt.show() #look at the design matrix
    outdir = os.path.join(container_path,sub,'results')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    plot_design_matrix(
        design_matrix, output_file=os.path.join(outdir, '%s_design_matrix.png' % sub))
    conditions = {
    'face':     np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    'fruit':   np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    'scene':   np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    }
    face_minus_other = conditions['face'] - (conditions['fruit'] + conditions['scene'])
    fruit_minus_other = conditions['fruit'] - (conditions['face'] + conditions['scene'])
    scene_minus_other = conditions['scene'] - (conditions['face'] + conditions['fruit'])

    #face constrast
    eff_map_face = fmri_glm.compute_contrast(face_minus_other,stat_type='t',
                                    output_type='effect_size')
    z_map_face = fmri_glm.compute_contrast(face_minus_other,stat_type='t',
                                  output_type='z_score')
    clean_map_face, threshold = threshold_stats_img(
        z_map_face, alpha=.05, cluster_threshold=10)
    plot_stat_map(clean_map_face, bg_img=mean_img_temp, threshold=threshold,
                  display_mode='z', cut_coords=3, black_bg=True,
                  title='Face minus {Fruit+Scene} (fdr=0.05), clusters > 10 voxels')
    plt.show()
    clean_map_face.to_filename(os.path.join(outdir,'face_v_other_clean_map.nii.gz'))
    z_map_face.to_filename(os.path.join(outdir,'face_v_other_clean_map.nii.gz'))  
    mask_face = math_img('img != 0', img=clean_map_face)
    mask_face.to_filename(os.path.join(outdir,'face_v_other_mask.nii.gz'))
    #fruit constrast
    eff_map_fruit = fmri_glm.compute_contrast(fruit_minus_other,stat_type='t',
                                    output_type='effect_size')
    z_map_fruit = fmri_glm.compute_contrast(fruit_minus_other,stat_type='t',
                                  output_type='z_score')
    clean_map_fruit, threshold = threshold_stats_img(
        z_map_fruit, alpha=.05, cluster_threshold=10)
    plot_stat_map(clean_map_fruit, bg_img=mean_img_temp, threshold=threshold,
                  display_mode='z', cut_coords=3, black_bg=True,
                  title='Fruit minus {Face+Scene} (fdr=0.05), clusters > 10 voxels')
    plt.show()
    clean_map_fruit.to_filename(os.path.join(outdir,'fruit_v_other_clean_map.nii.gz'))
    z_map_fruit.to_filename(os.path.join(outdir,'fruit_v_other_clean_map.nii.gz')) 
    mask_fruit = math_img('img != 0', img=clean_map_fruit)
    mask_fruit.to_filename(os.path.join(outdir,'fruit_v_other_mask.nii.gz'))        

    #scene constrast
    eff_map_scene = fmri_glm.compute_contrast(scene_minus_other,stat_type='t',
                                    output_type='effect_size')
    z_map_scene = fmri_glm.compute_contrast(scene_minus_other,stat_type='t',
                                  output_type='z_score')
    clean_map_scene, threshold = threshold_stats_img(
        z_map_scene, alpha=.05, cluster_threshold=10)
    plot_stat_map(clean_map_scene, bg_img=mean_img_temp, threshold=threshold,
                  display_mode='z', cut_coords=3, black_bg=True,
                  title='Scene minus {Fruit+Face} (fdr=0.05), clusters > 10 voxels')
    plt.show()
    clean_map_scene.to_filename(os.path.join(outdir,'scene_v_other_clean_map.nii.gz'))
    z_map_scene.to_filename(os.path.join(outdir,'scene_v_other_clean_map.nii.gz'))
    mask_scene = math_img('img != 0', img=clean_map_scene)
    mask_scene.to_filename(os.path.join(outdir,'scene_v_other_mask.nii.gz'))     

    print('Sub-0%s is now complete' % sub_num)  