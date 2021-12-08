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

subs=['01','04','06','10','11','15','18','21','23','27','34','36','42','44','45','61','69','77','79']
subs=['04','11','15','18','21','23','27','34','36','42','44','45','61','69','77','79']
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

    # face_mask_path=os.path.join(container_path,sub,'func','results','face_v_other_mask.nii.gz')
    # fruit_mask_path=os.path.join(container_path,sub,'func','results','fruit_v_other_mask.nii.gz')
    # scene_mask_path=os.path.join(container_path,sub,'func','results','scene_v_other_mask.nii.gz')    
    
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

    def apply_mask(mask=None,target=None):
        coor = np.where(mask == 1)
        values = target[coor]
        if values.ndim > 1:
            values = np.transpose(values) #swap axes to get feature X sample
        return values

    face_localizer_run1=apply_mask(mask=(face_mask.get_data()),target=(localizer_run1.get_data()))
    face_localizer_run2=apply_mask(mask=(face_mask.get_data()),target=(localizer_run2.get_data()))
    face_localizer_run3=apply_mask(mask=(face_mask.get_data()),target=(localizer_run3.get_data()))
    face_localizer_run4=apply_mask(mask=(face_mask.get_data()),target=(localizer_run4.get_data()))
    face_localizer_run5=apply_mask(mask=(face_mask.get_data()),target=(localizer_run5.get_data())) 

    fruit_localizer_run1=apply_mask(mask=(fruit_mask.get_data()),target=(localizer_run1.get_data()))
    fruit_localizer_run2=apply_mask(mask=(fruit_mask.get_data()),target=(localizer_run2.get_data()))
    fruit_localizer_run3=apply_mask(mask=(fruit_mask.get_data()),target=(localizer_run3.get_data()))
    fruit_localizer_run4=apply_mask(mask=(fruit_mask.get_data()),target=(localizer_run4.get_data()))
    fruit_localizer_run5=apply_mask(mask=(fruit_mask.get_data()),target=(localizer_run5.get_data()))

    scene_localizer_run1=apply_mask(mask=(scene_mask.get_data()),target=(localizer_run1.get_data()))
    scene_localizer_run2=apply_mask(mask=(scene_mask.get_data()),target=(localizer_run2.get_data()))
    scene_localizer_run3=apply_mask(mask=(scene_mask.get_data()),target=(localizer_run3.get_data()))
    scene_localizer_run4=apply_mask(mask=(scene_mask.get_data()),target=(localizer_run4.get_data()))
    scene_localizer_run5=apply_mask(mask=(scene_mask.get_data()),target=(localizer_run5.get_data()))

    wholebrain_mask1=nib.load(brain_mask_path[0])

    #unsure if I need to z-score or not, will check on this
    face_preproc_1 = clean(face_localizer_run1,t_r=0.46,detrend=False,standardize='zscore')
    face_preproc_2 = clean(face_localizer_run2,t_r=0.46,detrend=False,standardize='zscore')
    face_preproc_3 = clean(face_localizer_run3,t_r=0.46,detrend=False,standardize='zscore')
    face_preproc_4 = clean(face_localizer_run4,t_r=0.46,detrend=False,standardize='zscore')
    face_preproc_5 = clean(face_localizer_run5,t_r=0.46,detrend=False,standardize='zscore')

    fruit_preproc_1 = clean(fruit_localizer_run1,t_r=0.46,detrend=False,standardize='zscore')
    fruit_preproc_2 = clean(fruit_localizer_run2,t_r=0.46,detrend=False,standardize='zscore')
    fruit_preproc_3 = clean(fruit_localizer_run3,t_r=0.46,detrend=False,standardize='zscore')
    fruit_preproc_4 = clean(fruit_localizer_run4,t_r=0.46,detrend=False,standardize='zscore')
    fruit_preproc_5 = clean(fruit_localizer_run5,t_r=0.46,detrend=False,standardize='zscore')

    scene_preproc_1 = clean(scene_localizer_run1,t_r=0.46,detrend=False,standardize='zscore')
    scene_preproc_2 = clean(scene_localizer_run2,t_r=0.46,detrend=False,standardize='zscore')
    scene_preproc_3 = clean(scene_localizer_run3,t_r=0.46,detrend=False,standardize='zscore')
    scene_preproc_4 = clean(scene_localizer_run4,t_r=0.46,detrend=False,standardize='zscore')
    scene_preproc_5 = clean(scene_localizer_run5,t_r=0.46,detrend=False,standardize='zscore')    

    face_localizer_bold=np.concatenate((face_preproc_1,face_preproc_2,face_preproc_3,face_preproc_4,face_preproc_5))    
    fruit_localizer_bold=np.concatenate((fruit_preproc_1,fruit_preproc_2,fruit_preproc_3,fruit_preproc_4,fruit_preproc_5))    
    scene_localizer_bold=np.concatenate((scene_preproc_1,scene_preproc_2,scene_preproc_3,scene_preproc_4,scene_preproc_5))    



    params_dir='/scratch/06873/zbretton/params'
    # params_dir='/Users/zhbre/Desktop/Grad School/clearmem_take2/'    #find the mat file, want to change this to fit "sub"
    
    param_search='localizer*events*item*.csv'
    param_file=find(param_search,params_dir)
    reg_matrix = pd.read_csv(param_file[0]) #load in the build events file
    events=reg_matrix #grab the last 4 columns, those are of note (onset, duration, trial_type)
    #will need to refit "events" to work for each kind of contrast I want to do
    events = events[events.category != 0].reset_index(drop=True) #drop rest times

    face_items={}
    fruit_items={}
    scene_items={}

    face_data={}
    fruit_data={}
    scene_data={}
    for x in range(1,55):
        if x<=18:
            face_items[x]=[]
            temp={}
            for run in range(1,6):
                try:
                    face_items[x].append(events.loc[(events['trial_type']==x) & (events['run']==run)].onsets_volume.values[0]) #this is how we search for the item level times
                except:
                    pass
            for y in range(len(face_items[x])):
                beta_path=os.path.join(container_path,sub,'results','item%s_v_other_face_stat_map.nii.gz' % x)
                temp_beta=nib.load(beta_path)
                temp_beta_masked=apply_mask(mask=face_mask.get_data(),target=temp_beta.get_data())
                temp.update({'run%s' % (y+1):(face_localizer_bold[(face_items[x][y]+9):(face_items[x][y]+18)].mean(axis=0))*temp_beta_masked}) #take the mean of the 3 TR where the item is presented
                del beta_path,temp_beta_masked,temp_beta
            face_data[x] = temp
            del temp
            df_face_data = pd.DataFrame.from_dict({(i,j): face_data[i][j] 
                            for i in face_data.keys() 
                            for j in face_data[i].keys()},
                            orient='index')
            df_face_data.index = pd.MultiIndex.from_tuples(df_face_data.index)            
        elif (x>18)&(x<=36):
            fruit_items[x]=[]
            temp={}            
            for run in range(1,6):
                try:
                    fruit_items[x].append(events.loc[(events['trial_type']==x) & (events['run']==run)].onsets_volume.values[0]) #this is how we search for the item level times
                except:
                    pass
            for y in range(len(fruit_items[x])):
                beta_path=os.path.join(container_path,sub,'results','item%s_v_other_fruit_stat_map.nii.gz' % x)
                temp_beta=nib.load(beta_path)
                temp_beta_masked=apply_mask(mask=fruit_mask.get_data(),target=temp_beta.get_data())                                
                temp.update({'run%s' % (y+1):(fruit_localizer_bold[(fruit_items[x][y]+9):(fruit_items[x][y]+18)].mean(axis=0))*temp_beta_masked}) #take the mean of the 3 TR where the item is presented
                del beta_path,temp_beta_masked,temp_beta            
            fruit_data[x] = temp
            del temp
            df_fruit_data = pd.DataFrame.from_dict({(i,j): fruit_data[i][j] 
                            for i in fruit_data.keys() 
                            for j in fruit_data[i].keys()},
                            orient='index')
            df_fruit_data.index = pd.MultiIndex.from_tuples(df_fruit_data.index)            
        elif (x>36):
            scene_items[x]=[]
            temp={}
            for run in range(1,6):
                try:
                    scene_items[x].append(events.loc[(events['trial_type']==x) & (events['run']==run)].onsets_volume.values[0]) #this is how we search for the item level times
                except:
                    pass
            for y in range(len(scene_items[x])):
                beta_path=os.path.join(container_path,sub,'results','item%s_v_other_scene_stat_map.nii.gz' % x)
                temp_beta=nib.load(beta_path)
                temp_beta_masked=apply_mask(mask=scene_mask.get_data(),target=temp_beta.get_data())                                
                temp.update({'run%s' % (y+1):(scene_localizer_bold[(scene_items[x][y]+9):(scene_items[x][y]+18)].mean(axis=0))*temp_beta_masked}) #take the mean of the 3 TR where the item is presented
                del beta_path,temp_beta_masked,temp_beta            
            scene_data[x] = temp
            del temp
            df_scene_data = pd.DataFrame.from_dict({(i,j): scene_data[i][j] 
                            for i in scene_data.keys() 
                            for j in scene_data[i].keys()},
                            orient='index')
            df_scene_data.index = pd.MultiIndex.from_tuples(df_scene_data.index)             
    # np.arctanh(np.corrcoef(trials)) #this is the code to take a Fisher-Z of the pearson-r
    os.chdir(os.path.join(container_path,sub,'results'))
    with open('face_item_data_statweighted', 'wb') as fp:
        pickle.dump(face_data,fp)
    print('face data saved')
    with open('fruit_item_data_statweighted', 'wb') as fp:
        pickle.dump(fruit_data,fp)
    print('fruit data saved')
    with open('scene_item_data_statweighted', 'wb') as fp:
        pickle.dump(scene_data,fp)
    print('scene data saved') 
    df_face_data.to_csv('face_dataframe.csv')
    df_fruit_data.to_csv('fruit_dataframe.csv')   
    df_scene_data.to_csv('scene_dataframe.csv')
    print('CSVs saved')

    #Will need to break this up, because there are 6 items per subcategory, and at the moment I am doing this as a WHOLE category
    #shoulbe be straight forward since the items are in order and it can then be sliced out and then the fisher Z taken. that may also change some values since everything (I think) it between -1 and 1
    #category level
    face_fisher=np.arctanh(np.corrcoef(df_face_data))
    #sub category (I am unsure if this will change anything at all)
    actor_fisher=np.arctanh(np.corrcoef(df_face_data.loc[1:6]))
    musician_fisher=np.arctanh(np.corrcoef(df_face_data.loc[7:12]))
    politician_fisher=np.arctanh(np.corrcoef(df_face_data.loc[13:18]))        

    fruit_fisher=np.arctanh(np.corrcoef(df_fruit_data))
    #sub category (apple, grape, pear)
    apple_fisher=np.arctanh(np.corrcoef(df_fruit_data.loc[19:24]))
    grape_fisher=np.arctanh(np.corrcoef(df_fruit_data.loc[25:30]))
    pear_fisher=np.arctanh(np.corrcoef(df_fruit_data.loc[31:36]))

    scene_fisher=np.arctanh(np.corrcoef(df_scene_data))
    #sub category (beach, bridge, mountain)
    beach_fisher=np.arctanh(np.corrcoef(df_scene_data.loc[37:42]))
    bridge_fisher=np.arctanh(np.corrcoef(df_scene_data.loc[43:48]))
    mountain_fisher=np.arctanh(np.corrcoef(df_scene_data.loc[49:54]))

    #category level
    with open('face_fisher_statweighted', 'wb') as fp:
        pickle.dump(face_fisher,fp)
    print('face Fisher-Z saved')
    with open('fruit_fisher_statweighted', 'wb') as fp:
        pickle.dump(fruit_fisher,fp)
    print('fruit Fisher-Z saved')
    with open('scene_fisher_statweighted', 'wb') as fp:
        pickle.dump(scene_fisher,fp)
    print('scene Fisher-Z saved')

    #face subcategory
    with open('actor_fisher_statweighted', 'wb') as fp:
        pickle.dump(actor_fisher,fp)
    print('Actor Fisher-Z saved')
    with open('musician_fisher_statweighted', 'wb') as fp:
        pickle.dump(musician_fisher,fp)
    print('Musician Fisher-Z saved')
    with open('politician_fisher_statweighted', 'wb') as fp:
        pickle.dump(politician_fisher,fp)
    print('Politician Fisher-Z saved')

    #fruit subcategory
    with open('apple_fisher_statweighted', 'wb') as fp:
        pickle.dump(apple_fisher,fp)
    print('Apple Fisher-Z saved')
    with open('grape_fisher_statweighted', 'wb') as fp:
        pickle.dump(grape_fisher,fp)
    print('Grape Fisher-Z saved')
    with open('pear_fisher_statweighted', 'wb') as fp:
        pickle.dump(pear_fisher,fp)
    print('Pear Fisher-Z saved')

    #scene subcategory
    with open('beach_fisher_statweighted', 'wb') as fp:
        pickle.dump(beach_fisher,fp)
    print('Beach Fisher-Z saved')
    with open('bridge_fisher_statweighted', 'wb') as fp:
        pickle.dump(bridge_fisher,fp)
    print('Bridge Fisher-Z saved')
    with open('mountain_fisher_statweighted', 'wb') as fp:
        pickle.dump(mountain_fisher,fp)
    print('Mountain Fisher-Z saved')              

    print('Sub-0%s is now complete' % sub_num)


sub_face=[]
sub_fruit=[]
sub_scene=[]

sub_actor=[]
sub_musician=[]
sub_politician=[]

sub_apple=[]
sub_grape=[]
sub_pear=[]

sub_beach=[]
sub_bridge=[]
sub_mountain=[]

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
    sub_path=os.path.join(container_path,sub,'results')
    os.chdir(sub_path)

    pickle_files=find('face_fisher_statweighted',sub_path)
    face_infile = open(pickle_files[0],'rb')
    temp_face = pickle.load(face_infile)
    sub_face.append(temp_face)

    pickle_files=find('fruit_fisher_statweighted',sub_path)
    fruit_infile = open(pickle_files[0],'rb')
    temp_fruit = pickle.load(fruit_infile)
    sub_fruit.append(temp_fruit)

    pickle_files=find('scene_fisher_statweighted',sub_path)
    scene_infile = open(pickle_files[0],'rb')
    temp_scene = pickle.load(scene_infile)
    sub_scene.append(temp_scene)

    #face sub categories
    pickle_files=find('actor_fisher_statweighted',sub_path)
    actor_infile = open(pickle_files[0],'rb')
    temp_actor = pickle.load(actor_infile)
    sub_actor.append(temp_actor)

    pickle_files=find('musician_fisher_statweighted',sub_path)
    musician_infile = open(pickle_files[0],'rb')
    temp_musician = pickle.load(musician_infile)
    sub_musician.append(temp_musician)

    pickle_files=find('politician_fisher_statweighted',sub_path)
    politician_infile = open(pickle_files[0],'rb')
    temp_politician = pickle.load(politician_infile)
    sub_politician.append(temp_politician)    

    #fruit subcategories
    pickle_files=find('apple_fisher_statweighted',sub_path)
    apple_infile = open(pickle_files[0],'rb')
    temp_apple = pickle.load(apple_infile)
    sub_apple.append(temp_apple)

    pickle_files=find('grape_fisher_statweighted',sub_path)
    grape_infile = open(pickle_files[0],'rb')
    temp_grape = pickle.load(grape_infile)
    sub_grape.append(temp_grape)

    pickle_files=find('pear_fisher_statweighted',sub_path)
    pear_infile = open(pickle_files[0],'rb')
    temp_pear = pickle.load(pear_infile)
    sub_pear.append(temp_pear)

    #scene subcategories        
    pickle_files=find('beach_fisher_statweighted',sub_path)
    beach_infile = open(pickle_files[0],'rb')
    temp_beach = pickle.load(beach_infile)
    sub_beach.append(temp_beach)

    pickle_files=find('bridge_fisher_statweighted',sub_path)
    bridge_infile = open(pickle_files[0],'rb')
    temp_bridge = pickle.load(bridge_infile)
    sub_bridge.append(temp_bridge)

    pickle_files=find('mountain_fisher_statweighted',sub_path)
    mountain_infile = open(pickle_files[0],'rb')
    temp_mountain = pickle.load(mountain_infile)
    sub_mountain.append(temp_mountain) 

#category level
sub_mean_face=np.mean(sub_face,axis=0)
sub_mean_fruit=np.mean(sub_fruit,axis=0)
sub_mean_scene=np.mean(sub_scene,axis=0)

plt.style.use('fivethirtyeight')
ax = sns.heatmap(data=(np.nan_to_num(sub_mean_face, posinf=1)),cmap='jet',vmin=-0.2, vmax=0.9)
ax.set(title='Face RSA - Subject mean')
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()

plt.style.use('fivethirtyeight')
ax = sns.heatmap(data=(np.nan_to_num(sub_mean_fruit, posinf=1)),cmap='jet')
ax.set(title='Fruit RSA - Subject mean')
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()

plt.style.use('fivethirtyeight')
ax = sns.heatmap(data=(np.nan_to_num(sub_mean_scene, posinf=1)),cmap='jet')
ax.set(title='Scene RSA - Subject mean')
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()

#subcategory plots
sub_mean_actor=np.mean(sub_actor,axis=0)
sub_mean_musician=np.mean(sub_musician,axis=0)
sub_mean_politician=np.mean(sub_politician,axis=0)
plt.style.use('fivethirtyeight')
ax = sns.heatmap(data=(np.nan_to_num(sub_mean_actor, posinf=1)),cmap='jet')
ax.set(title='Actor RSA - Subject mean')
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()

plt.style.use('fivethirtyeight')
ax = sns.heatmap(data=(np.nan_to_num(sub_mean_musician, posinf=1)),cmap='jet')
ax.set(title='Musician RSA - Subject mean')
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()

plt.style.use('fivethirtyeight')
ax = sns.heatmap(data=(np.nan_to_num(sub_mean_politician, posinf=1)),cmap='jet')
ax.set(title='Politician RSA - Subject mean')
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()

sub_mean_apple=np.mean(sub_apple,axis=0)
sub_mean_grape=np.mean(sub_grape,axis=0)
sub_mean_pear=np.mean(sub_pear,axis=0)

sub_mean_beach=np.mean(sub_beach,axis=0)
sub_mean_bridge=np.mean(sub_bridge,axis=0)
sub_mean_mountain=np.mean(sub_mountain,axis=0)