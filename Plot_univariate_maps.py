#code to plot the Univariate maps for paper:

from nilearn import datasets, surface, plotting
import os
import fnmatch
import numpy as np


data_dir = '/Users/zb3663/Desktop/School_Files/Repclear_files/manuscript/operation_GLM_results'
operations = ['maintain','replace','suppress']
views = ['medial', 'lateral']
hemis = ['right','left']

def find(pattern, path): #find the pattern we're looking for
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        return result  

for operation in operations:

    temp_t_map = find(f'group+{operation}_MNI_thresholded*nii*',data_dir) #pulls in the thresholded t-map

    fsaverage = datasets.fetch_surf_fsaverage('fsaverage') #gets the surface mesh to place the data on

    for hemi in hemis:
        if hemi=='right':
            texture = surface.vol_to_surf(temp_t_map, fsaverage.pial_right) #pull out the right side of the data

            for view in views:
                #this will now pull all these together into a plot, 
                plotting.plot_surf_stat_map(fsaverage.infl_right,
                    texture, hemi='right', colorbar=True,
                    title=f'Surface right hemisphere: {operation}',
                    threshold=3.02, vmax=10, view=view, bg_map=fsaverage.sulc_right, output_file=os.path.join(data_dir,'updated_figs',f'{operation}_surface_{hemi}_{view}.png'))

        elif hemi=='left':
            texture = surface.vol_to_surf(temp_t_map, fsaverage.pial_left) #pull out the right side of the data

            for view in views:
                #this will now pull all these together into a plot, 
                plotting.plot_surf_stat_map(fsaverage.infl_left,
                    texture, hemi='left', colorbar=True,
                    title=f'Surface left hemisphere: {operation}',
                    threshold=3.02, vmax=10, view=view, bg_map=fsaverage.sulc_left, output_file=os.path.join(data_dir,'updated_figs',f'{operation}_surface_{hemi}_{view}.png'))            
