import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import nibabel as nib


# global consts
subIDs = ['002', '003', '004']
phases = ['rest', 'preremoval', 'study', 'postremoval']
runs = np.arange(6) + 1  
spaces = {'T1w': 'T1w', 
            'MNI': 'MNI152NLin2009cAsym'}
descs = ['brain_mask', 'preproc_bold']
ROIs = ['VVS', 'PHG', 'FG']
shift_sizes_TR = [5, 6]

stim_labels = {0: "Rest",
                1: "Scenes",  # 1: manmade, 2: natural
                2: "Faces"}  # 1: female, 2: male
sub_cates = {
            "face": ["male", "female"],         #60
            "scene": ["manmade", "natural"],    #120
            }

workspace = 'scratch'
if workspace == 'work':
    data_dir = '/work/07365/sguo19/frontera/fmriprep/'
    param_dir = '/work/07365/sguo19/frontera/params/'
    results_dir = '/work/07365/sguo19/model_fitting_results/'
elif workspace == 'scratch':
    data_dir = '/scratch1/07365/sguo19/fmriprep/'
    param_dir = '/scratch1/07365/sguo19/params/'
    results_dir = '/scratch1/07365/sguo19/model_fitting_results/'

# helper function 
def apply_mask(mask=None,target=None):
    coor = np.where(mask == 1)
    values = target[coor]
    if values.ndim > 1:
        values = np.transpose(values) #swap axes to get feature X sample
    return values


def sort_trials_by_categories(subID="002", phase=2):
    """
    Input: 
    subID: 3 digit string
    phase: single digit int. 1: "pre-exposure", 2: "pre-localizer", 3: "study", 4: "post-localizer"

    Output: 
    face_order: trial numbers ordered by ["female", "male"]. 
    scene_order: trial numbers ordered by ["manmade", "natural"]

    (It's hard to sort the RDM matrix once that's been computed, so the best practice would be to sort the input to MDS before we run it)
    """
    # *** change file saved location
    tim_path = os.path.join(param_dir, f"sub-{subID}_trial_image_match.csv")

    tim_df = pd.read_csv(tim_path)
    tim_df = tim_df[tim_df["phase"]==phase]
    tim_df = tim_df.sort_values(by=["category", "subcategory", "trial_id"])
    
    scene_order = tim_df[tim_df["category"]==1]["trial_id"].values
    face_order = tim_df[tim_df["category"]==2]["trial_id"].values

    return face_order, scene_order



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


def select_MDS(data, ncomps=np.arange(1,6), save=False, out_fname=results_dir):
    """
    Find best ncomps; fit model with given data & ncomps; return model

    Input: 
    data: must be 2D (sample x feature)
    ncomps: range/list of ncomps to try on the data

    Output: 
    mds: fitted model on the best ncomp
    """
    print(f"Running MDS selection with ncomps {ncomps}...")

    # choose best ncomp
    mdss = []
    for compi in ncomps:
        mds = MDS(n_components=compi)
        _ = mds.fit_transform(data)
        mdss.append(mds)

    stresses = np.asarray([mds.stress_ for mds in mdss])
    print(stresses)

    if save:
        print(f"Saving MDSs to {out_fname}...")
        np.savez_compressed(out_fname, mdss=mdss)

    return mdss


def run_MDS(subID="002", task="preremoval", space="T1w", mask_ROIS=["VVS"], ):
    # stim; bold;
    data_source = "bold"

    if data_source == "stim":
        all_images = get_images(i)

        out_fname = os.path.join(results_dir, "MDS", f"sub-{subID}_{task}_stim_mdss")
        select_MDS(all_images, save=True, out_fname=out_fname)

    elif data_source == "bold":
        from classification_replicate import get_preprocessed_data, get_shifted_labels

        # load masked & cleaned bold 
        full_data = get_preprocessed_data(subID, task, space, mask_ROIS)
        print("data shape: ", full_data.shape)
        full_label_df = get_shifted_labels(task, 5)
        print("label df shape: ", full_label_df.shape)
        # TODO: load stim mask & mask bold 

        # load event labels
        stim_on = full_label_df["stim_present"]
        stim_on_TRs = np.where(stim_on == 1)[0]  # all TRs here a stim is on
        trial_start_TRs = stim_on_TRs[::2]  # starting TRs of each trial. used for getting trial number & image_id & category

        # subsample & average bold
        avg_trial_bolds = full_data[stim_on_TRs]
        avg_trial_bolds = np.array([avg_trial_bolds[i:i+2].mean(axis=0) for i in range(0, len(avg_trial_bolds), 2)])
        print("avg trial bolds shape:", avg_trial_bolds.shape)

        # MDS
        out_fname = os.path.join(results_dir, "MDS", f"sub-{subID}_{task}_bold_vtc_mdss")
        select_MDS(avg_trial_bolds, save=True, out_fname=out_fname)


def item_level_RSA(subID="002", phase="pre", weight_source="LSA", stats_test="tmap"):
    """
    Load ready BOLD for each trial & weights from LSA/LSS, 
    multiply, select MDS with best ncomp, save 

    Input: 
    subID: 3 digit string. e.g. "002"
    phase: "pre" / "post". for loading item_represenation
    weight_source: "LSA" / "LSS". 
    stats_test: "tmap" / "zmap"
    """
    phase_dict = {"pre": 2, "post": 4}

    # sub_cates.pop("face")
    # expt_tag = "scene"
    # expt_tag = "nonweighted_"
    # expt_tag = ""

    print(f"Running item level RSA & MDS for sub{subID}, {phase}, {stats_test}, {expt_tag}...")

    # ===== load mask for BOLD
    mask_path = os.path.join(data_dir, "group_MNI_thresholded_VTC_mask.nii.gz")  # voxels chosen with GLM contrast: stim vs non-stim
    mask = nib.load(mask_path)
    print("mask shape: ", mask.shape)

    # === get order for face/scene trials
    face_order, scene_order = sort_trials_by_categories(subID, phase_dict[phase])
    print("face/scene length: ", len(face_order), len(scene_order))
    cates_order = {"face": face_order, "scene": scene_order}

    # ===== load ready BOLD for each trial 
    print(f"Loading preprocessed BOLDs for {phase} operation...")

    # item_repres: bold masked by category selective voxels
    # bold_dir = os.path.join(data_dir, f"sub-{subID}", "item_representations")   # for pre-post comparison

    # item_repres_MDS: bold masked by stim selective voxels
    bold_dir = os.path.join(data_dir, f"sub-{subID}", "item_representations_MDS")   # for (sub)cate comparison

    all_bolds = {}  # {cateID: {trialID: bold}}
    bolds_arr = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames = glob.glob(f"{bold_dir}/*{phase}*{cateID}*")
        cate_bolds = {}
        
        for fname in cate_bolds_fnames:
            trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
            trialID = int(trialID[5:])
            cate_bolds[trialID] = nib.load(fname).get_fdata()  #.flatten()
        cate_bolds = {i: cate_bolds[i] for i in sorted(cate_bolds.keys())}
        all_bolds[cateID] = cate_bolds

        # bolds_arr.append(np.stack( [ cate_bolds[i] for i in sorted(cate_bolds.keys()) ] ))
        # *** order by subcategories rather than trial number
        bolds_arr.append(np.stack( [cate_bolds[i] for i in cates_order[cateID]] ))

    bolds_arr = np.vstack(bolds_arr)
    print("bolds shape: ", bolds_arr.shape)

    # apply mask on BOLD
    masked_bolds_arr = []
    for bold in bolds_arr:
        masked_bolds_arr.append(apply_mask(mask=mask.get_fdata(), target=bold).flatten())
    masked_bolds_arr = np.vstack(masked_bolds_arr)
    print("masked bold arr shape: ", masked_bolds_arr.shape)

    # ===== load weights
    print(f"Loading {weight_source} weights ...")
    if weight_source == "LSA":
        # weights generated from GLM: target trial vs every other trial
        weights_dir = os.path.join(data_dir, f"sub-{subID}", "localizer_item_level")
    # *** weighting is done only on LSA, not LSS ***
    # elif weight_source == "LSS":
    #     # weights generated from GLM: target trial vs combination of all other trials
    #     weights_dir = os.path.join(data_dir, f"sub-{subID}", "localizer_LSS_lvl1")
    else:
        raise ValueError("Weight source must be LSA")
    
    all_weights = {}
    weights_arr= []
    for cateID in sub_cates.keys():
        # no full: weights from GLM with only category selective voxels
        # cate_weights_fnames = glob.glob(f"{weights_dir}/{cateID}*MNI_{stats_test}*")  # for pre-post comparison
        # full: weigthts from GLM with stim selective voxels
        cate_weights_fnames = glob.glob(f"{weights_dir}/{cateID}*full*{stats_test}*") # for (sub)cate comparison
        
        print(cateID, len(cate_weights_fnames))
        cate_weights = {}

        for fname in cate_weights_fnames:
            trialID = fname.split("/")[-1].split("_")[1]  # "trial1"
            trialID = int(trialID[5:])
            cate_weights[trialID] = nib.load(fname).get_fdata()
        cate_weights = {i: cate_weights[i] for i in sorted(cate_weights.keys())}
        all_weights[cateID] = cate_weights

        # weights_arr.append(np.stack( [ cate_weights[i] for i in sorted(cate_weights.keys()) ] ))
        # *** order by subcategories rather than trial number
        weights_arr.append(np.stack( [cate_weights[i] for i in cates_order[cateID]] ))
    
    weights_arr = np.vstack(weights_arr)
    print("weights shape: ", weights_arr.shape)

    # apply mask on BOLD
    masked_weights_arr = []
    for weight in weights_arr:
        masked_weights_arr.append(apply_mask(mask=mask.get_fdata(), target=weight).flatten())
    masked_weights_arr = np.vstack(masked_weights_arr)
    print("masked weights arr shape: ", masked_weights_arr.shape)

    # ===== multiply
    item_repress = np.multiply(masked_bolds_arr, masked_weights_arr)
    # item_repress = masked_bolds_arr
    print("item_repres shape: ", item_repress.shape)

    # ===== select MDS with best ncomp
    out_fname = os.path.join(results_dir, "MDS", f"sub-{subID}_{phase}_{stats_test}_{expt_tag}mdss")
    _ = select_MDS(item_repress, save=True, out_fname=out_fname)


def vis_mds(mds, labels):
    """
    example function demonstrating how to visualize mds results.

    Input: 
    mds: fitted sklearn mds object
    labels: group labels for each sample within mds embedding
    """

    embs = mds.embedding_
    assert len(embs) == len(labels), f"length of labels ({len(labels)}) do not match length of embedding ({len(embs)})"

    # ===== RDM: internally computed RDM by sklearn 
    fig, ax = plt.subplots(1,1)
    ax.imshow(mds.dissimilarity_matrix_, cmap="GnBu")
    im = ax.imshow(mds.dissimilarity_matrix_, cmap="GnBu")
    plt.colorbar(im)
    
    # ax.set_title("")
    # plt.savefig("")

    # ===== MDS scatter
    fig, ax = plt.subplots(1,1)
    # fix the label order 
    label_set = ["male", "female", "manamde", "natural"]
    for subcateID in label_set:
        inds = np.where(labels == subcateID)[0]
        ax.scatter(embs[inds, 0], embs[inds, 1], label=subcateID)
    plt.legend()

    # ax.set_title("")
    # plt.savefig("")




if __name__ == "__main__":
    
    for subID in ["003", "004"]:
        item_level_RSA(subID)
    # run_MDS()
    # f, s = sort_trials_by_categories()
    # print(f)
    # print(s)


    # item vs cateogory average
    # within category MDS
