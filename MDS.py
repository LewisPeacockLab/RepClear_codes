import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


# global consts
subIDs = ['002', '003', '004']
tasks = ['rest', 'preremoval', 'study', 'postremoval']
runs = np.arange(6) + 1  
spaces = {'T1w': 'T1w', 
            'MNI': 'MNI152NLin2009cAsym'}
descs = ['brain_mask', 'preproc_bold']
ROIs = ['VVS', 'PHG', 'FG']
shift_sizes_TR = [5, 6]

stim_labels = {0: "Rest",
                1: "Scenes",
                2: "Faces"}

workspace = 'scratch'
if workspace == 'work':
    data_dir = '/work/07365/sguo19/frontera/fmriprep/'
    event_dir = '/work/07365/sguo19/frontera/events/'
    results_dir = '/work/07365/sguo19/model_fitting_results/'
elif workspace == 'scratch':
    data_dir = '/scratch1/07365/sguo19/fmriprep/'
    event_dir = '/scratch1/07365/sguo19/events/'
    results_dir = '/scratch1/07365/sguo19/model_fitting_results/'


def get_images(nimgs=10):
    """
    Read and returns stim images

    Input: 
    nimgs: number of images to take from each subcategory

    Output: 
    all_imgs: shape (total_n_images x img_height x img_width x img_channels). 
              total_n_images = nimgs x 4 (4 subcategories; nimgs per subcategory);
              img_height & img_width: 400 x 400;
              img_channels: 3 (RGB). RGBA is transformed to RGB.
    labels: list of strings speifying the subcategories of each image.
    """

    from PIL import Image
    import glob

    def RGBA2RGB(image, color=(255, 255, 255)):
        image.load()  # needed for split()
        background = Image.new('RGB', image.size, color)
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return background

    # rn these are local paths on lab workstation
    data_dir = "/Users/sunaguo/Documents/code/repclear_data/stim"
    sub_cates = {
                "face": ["male", "female"],
                "scene": ["manmade", "natural"], 
                }
    print(f"Loading {nimgs} from each subcategory...")
    print(f"data dir: {data_dir}")
    print(f"subcates: {sub_cates}")

    all_imgs = []
    labels = []
    for cate, subs in sub_cates.items():
        for sub in subs:
            # labels
            labels.append([sub for _ in range(nimgs)])
            
            # images
            img_paths = glob.glob(os.path.join(data_dir, cate, sub, "*"))
            img_paths = list(sorted(img_paths))
            
            for imgi in range(nimgs):
                img = Image.open(img_paths[imgi])
                # face images are RGBA, need to be transformed
                if cate == "face":
                    img = RGBA2RGB(img)
                all_imgs.append(np.asarray(img))  
        
    all_imgs = np.asarray(all_imgs)
    print("all_imgs shape:", all_imgs.shape)
    labels = np.concatenate(labels)
    print("image labels: ", labels)

    return all_imgs, labels


def select_MDS(data, ncomps=np.arange(1,5), save=False):
    """
    Find best ncomps; fit model with given data & ncomps; return model

    Input: 
    data: must be 2D (sample x feature)
    ncomps: range/list of ncomps to try on the data

    Output: 
    mds: fitted model on the best ncomp
    """

    # choose best ncomp
    mdss = []
    for compi in ncomps:
        mds = MDS(n_components=compi)
        _ = mds.fit_transform(data)
        mdss.append(mds)

    stresses = np.asarray([mds.stress_ for mds in mdss])
    print(stresses)

    if save:
        out_dir = './'
        fname = "TEST"
        out_path = os.path.join(out_dir, fname)
        np.savez_compressed(out_path, mdss=mdss)

    return mdss


def item_lvel_RSA():
    """
    load ready BOLD for each trial & weights from LSA/LSS, 
    multiply, select MDS with best ncomp, save 
    """

    # load ready BOLD for each trial 
    

    # load weights from LSA/LSS, 
    # multiply
    # select MDS with best ncomp, save 