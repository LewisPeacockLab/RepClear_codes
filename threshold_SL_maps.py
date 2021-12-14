#quick python loop to threshold the searchlight results locally
import os
import glob
import fnmatch

def find(pattern, path): #find the pattern we're looking for
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result


searchlight_path='/Users/zb3663/Desktop/repclear_preprocessed/repclearbids/derivatives/fmriprep/searchlight'

subIDs=['002','003','004']
conditions=['maintain','replace','suppress']

#this iterates and thresholds the data
for subID in subIDs:
    files=find('*result.nii.gz',os.path.join(searchlight_path,'sub-%s' % subID))
    for file in files:
        string_split=file.split("_")
        string_split[-2]=(string_split[-2]+'_thresholded')
        new_name='_'.join(string_split)
        os.system('fslmaths %s -thrP 98 %s' % (file,new_name))


#this next section will take these thresholded maps and then add the subjects together per condition
#the results will be a map that will show what voxels are more or less shared across the subjects

#I will also want to take the difference between the images, so that I have operation unique areas and shared areas
for condition in conditions:
    condition_files=[]
    for subID in subIDs:
        files=find('*%s_thresh*' % condition,os.path.join(searchlight_path,'sub-%s' % subID))
        condition_files.append(files)
    new_name=('group_%s_results.nii.gz' % condition)
    os.system('fslmaths %s -add %s -add %s %s' % (condition_files[0][0],condition_files[1][0],condition_files[2][0],new_name))