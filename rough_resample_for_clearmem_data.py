#code to quickly register the clearmem data into 2.4mm voxel space - this is to allow for easy cross experiment classification

import os
import glob
import fnmatch
import nibabel as nib
import numpy as np
from nilearn import image as nimg
from nilearn import plotting as nplot
from joblib import Parallel, delayed

subIDs=['004','006','011','015','027','034','036','042','044','045','061','069','077','079']

def find(pattern, path): #find the pattern we're looking for
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result

def resample_to_repclear(subID):
    data_path='/scratch1/06873/zbretton/clearmem'
    reference_path='/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/sub-002/func/sub-002_task-study-1_space-MNI152NLin2009cAsym_boldref.nii.gz' #using this file as the reference to scale to

    target=nimg.load_img(reference_path)

    affine=target.affine
    shape=target.shape
    header=target.header

    print('starting sub-%s now...' % subID)

    bold_path=os.path.join(data_path,'sub-%s' % subID,'func/')

    localizer_files=find('*study*MNI*bold.nii.gz',bold_path)

    localizer_files.sort()
    for file in localizer_files:
        print('%s being resampled...' % file)
        source=nimg.load_img(file)

        string_split=file.split("_")
        string_split[-2]=(string_split[-2]+'_resized')
        new_name='_'.join(string_split)

        resamp=nimg.resample_img(source,affine,shape)
        resamp_final=nib.Nifti1Image(resamp.get_fdata(), affine,header)
        nib.save(resamp_final,new_name)

        source.uncache()

        print('file is done, saved as: %s' % new_name)

        del source, resamp, resamp_final, new_name
    print('sub-%s is now complete!' % subID)
    print('===============================================================')

Parallel(n_jobs=len(subIDs))(delayed(resample_to_repclear)(i) for i in subIDs)