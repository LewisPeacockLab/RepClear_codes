import os
import glob

import numpy as np
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
                1: "Scenes",
                2: "Faces"}
sub_cates = {
            "face": ["male", "female"],         #60
            "scene": ["manmade", "natural"],    #120
            }

workspace = 'scratch'
if workspace == 'work':
    data_dir = '/work/06873/zbretton/frontera/fmriprep/'
    event_dir = '/work/06873/zbretton/frontera/events/'
    results_dir = '/work/06873/zbretton/model_fitting_results/'
elif workspace == 'scratch':
    data_dir = '/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
    event_dir = '/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/'
    results_dir = '/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/model_fitting_results/'


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
    
    scene_order = tim_df[tim_df["category"]==1]["trial_id"]
    face_order = tim_df[tim_df["category"]==2]["trial_id"]

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
    #data_dir = "/Users/sunaguo/Documents/code/repclear_data/stim"
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


def item_level_RSA(subID="002", phase="pre", weight_source="LSS", stats_test="tmap"):
    """
    Load ready BOLD for each trial & weights from LSA/LSS, 
    multiply, select MDS with best ncomp, save 

    Input: 
    subID: 3 digit string. e.g. "002"
    phase: "pre" / "post". for loading item_represenation
    weight_source: "LSA" / "LSS". 
    stats_test: "tmap" / "zmap"
    """
    def apply_mask(mask=None,target=None):
        coor = np.where(mask == 1)
        values = target[coor]
        if values.ndim > 1:
            values = np.transpose(values) #swap axes to get feature X sample
        return values

    print(f"Running item level RSA & MDS for sub{subID}, {phase}, {weight_source}, {stats_test}...")

    # ===== load mask for BOLD
    mask_path = os.path.join(data_dir, "group_MNI_thresholded_VTC_mask.nii.gz")
    mask = nib.load(mask_path)
    print("mask shape: ", mask.shape)

    # ===== load ready BOLD for each trial 
    print(f"Loading preprocessed BOLDs for {phase} operation...")
    bold_dir = os.path.join(data_dir, f"sub-{subID}", "item_representations_MDS")

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

        bolds_arr.append(np.stack( [ cate_bolds[i] for i in sorted(cate_bolds.keys()) ] ))

    bolds_arr = np.vstack(bolds_arr)
    print("bolds shape: ", bolds_arr.shape)
    # print(f"category check:")
    # print("face: ", (bolds_arr[:60, :] == np.vstack(all_bolds["face"].values())).all() )
    # print("scene: ", (bolds_arr[60:, :] == np.vstack(all_bolds["scene"].values())).all() )

    # apply mask on BOLD
    masked_bolds_arr = []
    for bold in bolds_arr:
        masked_bolds_arr.append(apply_mask(mask=mask.get_fdata(), target=bold).flatten())
    masked_bolds_arr = np.vstack(masked_bolds_arr)
    print("masked bold arr shape: ", masked_bolds_arr.shape)

    # ===== load weights
    print(f"Loading {weight_source} weights ...")
    if weight_source == "LSA":
        weights_dir = os.path.join(data_dir, f"sub-{subID}", "localizer_item_level")
    elif weight_source == "LSS":
        weights_dir = os.path.join(data_dir, f"sub-{subID}", "localizer_LSS_lvl1")
    else:
        raise ValueError("Weight source must be LSA or LSS")
    
    all_weights = {}
    weights_arr= []
    for cateID in sub_cates.keys():
        if weight_source=='LSA':
            cate_weights_fnames = glob.glob(f"{weights_dir}/{cateID}*full*{stats_test}*")
        elif weight_source=='LSS':
            cate_weights_fnames = glob.glob(f"{weights_dir}/{cateID}*{stats_test}*")

        print(cateID, len(cate_weights_fnames))
        cate_weights = {}

        for fname in cate_weights_fnames:
            trialID = fname.split("/")[-1].split("_")[1]  # "trial1"
            trialID = int(trialID[5:])
            cate_weights[trialID] = nib.load(fname).get_fdata()
        cate_weights = {i: cate_weights[i] for i in sorted(cate_weights.keys())}
        all_weights[cateID] = cate_weights

        weights_arr.append(np.stack( [ cate_weights[i] for i in sorted(cate_weights.keys()) ] ))
    
    weights_arr = np.vstack(weights_arr)
    print("weights shape: ", weights_arr.shape)
    # print(f"category check:")
    # print("face: ", (weights_arr[:60, :] == np.vstack(all_weights["face"].values())).all() )
    # print("scene: ", (weights_arr[60:, :] == np.vstack(all_weights["scene"].values())).all() )

    # apply mask on BOLD
    masked_weights_arr = []
    for weight in weights_arr:
        masked_weights_arr.append(apply_mask(mask=mask.get_fdata(), target=weight).flatten())
    masked_weights_arr = np.vstack(masked_weights_arr)
    print("masked weights arr shape: ", masked_weights_arr.shape)

    # ===== multiply
    if weight_source=='LSA':
        item_repress = np.multiply(masked_bolds_arr, masked_weights_arr)
        print("item_repres shape: ", item_repress.shape)
    elif weight_source=='LSS':
        item_repress=masked_weights_arr
        print("item_repres shape: ", item_repress.shape)

    # ===== select MDS with best ncomp
    out_fname = os.path.join(results_dir, "MDS", f"sub-{subID}_{phase}_{weight_source}_{stats_test}_mdss")
    if not os.path.exists(os.path.join(results_dir, "MDS")): os.makedirs(os.path.join(results_dir, "MDS"),exist_ok=True)
    _ = select_MDS(item_repress, save=True, out_fname=out_fname)

    # ===== perform the correlation 
    corr_matrix=np.corrcoef(item_repress)
    out_name=os.path.join(results_dir,"RSM",f"sub-{subID}_{phase}_{weight_source}_{stats_test}_rsm")
    if not os.path.exists(os.path.join(results_dir, "RSM")): os.makedirs(os.path.join(results_dir, "RSM"),exist_ok=True)    
    np.save(out_name,corr_matrix)

def prepare_plot(subID):
    data_path='/scratch1/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep/model_fitting_results/MDS'
    label_names=['faces','scenes']
    labels=['' for i in range(180)]
    for i in range(180):
        if (i < 60):
            labels[i]=label_names[0]
        elif (i>=60):
            labels[i]=label_names[1]
    labels=np.array(labels)
    data=np.load(os.path.join(data_path,'sub-%s_pre_LSS_tmap_mdss.npz' % subID),allow_pickle=True)
    mds=data['mdss'][1]
    return mds,labels

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
    plt.show()
    # ax.set_title("")
    # plt.savefig("")

    # ===== MDS scatter
    fig, ax = plt.subplots(1,1)
    # fix the label order 
    #label_set = ["male", "female", "manamde", "natural"]
    label_set = ['faces','scenes']
    for subcateID in label_set:
        inds = np.where(labels == subcateID)[0]
        ax.scatter(embs[inds, 0], embs[inds, 1], label=subcateID)
    plt.legend()
    plt.show()

    # ac.set_title("")
    # plt.savefig("")

def item_RSA_compare(subID="002", phase1="pre", phase2='post', weight_source="LSS", stats_test="tmap"):
    """
    Load ready BOLD for each trial & weights from LSA or load in LSS, 
    take RSA across phases

    Input: 
    subID: 3 digit string. e.g. "002"
    phase1: "pre" / "post". for loading item_represenation
    phase2: "pre" / "study" / "post" for loading item_representation
    weight_source: "LSA" / "LSS". 
    stats_test: "tmap" / "zmap"
    """
    def apply_mask(mask=None,target=None):
        coor = np.where(mask == 1)
        values = target[coor]
        if values.ndim > 1:
            values = np.transpose(values) #swap axes to get feature X sample
        return values

    print(f"Running item level RSA & MDS for sub{subID}, {phase1} comapred to {phase2}, {weight_source}, {stats_test}...")

    # ===== load mask for BOLD
    mask_path = os.path.join(data_dir, "group_MNI_thresholded_VTC_mask.nii.gz")
    mask = nib.load(mask_path)
    print("mask shape: ", mask.shape)

    # ===== load ready BOLD for each trial 
    print(f"Loading preprocessed BOLDs for {phase1} operation...")
    bold_dir_1 = os.path.join(data_dir, f"sub-{subID}", "item_representations")

    all_bolds_1 = {}  # {cateID: {trialID: bold}}
    bolds_arr_1 = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames_1 = glob.glob(f"{bold_dir}/*{phase1}*{cateID}*")
        cate_bolds_1 = {}
        
        for fname in cate_bolds_fnames_1:
            trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
            trialID = int(trialID[5:])
            cate_bolds_1[trialID] = nib.load(fname).get_fdata()  #.flatten()
        cate_bolds_1 = {i: cate_bolds_1[i] for i in sorted(cate_bolds_1.keys())}
        all_bolds_1[cateID] = cate_bolds_1

        bolds_arr_1.append(np.stack( [ cate_bolds_1[i] for i in sorted(cate_bolds_1.keys()) ] ))

    bolds_arr_1 = np.vstack(bolds_arr_1)
    print("bolds for phase 1 - shape: ", bolds_arr_1.shape)

    # ===== load ready BOLD for each trial 
    print(f"Loading preprocessed BOLDs for {phase2} operation...")
    bold_dir_2 = os.path.join(data_dir, f"sub-{subID}", "item_representations")

    all_bolds_2 = {}  # {cateID: {trialID: bold}}
    bolds_arr_2 = []  # sample x vox
    for cateID in sub_cates.keys():
        cate_bolds_fnames_2 = glob.glob(f"{bold_dir}/*{phase2}*{cateID}*")
        cate_bolds_2 = {}
        
        for fname in cate_bolds_fnames_2:
            trialID = fname.split("/")[-1].split("_")[-2]  # "trial1"
            trialID = int(trialID[5:])
            cate_bolds_2[trialID] = nib.load(fname).get_fdata()  #.flatten()
        cate_bolds_2 = {i: cate_bolds_2[i] for i in sorted(cate_bolds_2.keys())}
        all_bolds_2[cateID] = cate_bolds_2

        bolds_arr_2.append(np.stack( [ cate_bolds_2[i] for i in sorted(cate_bolds_2.keys()) ] ))

    bolds_arr_2 = np.vstack(bolds_arr_2)
    print("bolds for phase 2 - shape: ", bolds_arr_2.shape)    

    #when comparing pre vs. post, the pre scene have 120 trials, while post has 180 (which are the 120 we saw before plus 60 novel images)


    # apply mask on BOLD
    masked_bolds_arr_1 = []
    for bold in bolds_arr_1:
        masked_bolds_arr_1.append(apply_mask(mask=mask.get_fdata(), target=bold).flatten())
    masked_bolds_arr_1 = np.vstack(masked_bolds_arr_1)
    print("masked phase1 bold array shape: ", masked_bolds_arr_1.shape)

    # apply mask on BOLD
    masked_bolds_arr_2 = []
    for bold in bolds_arr_2:
        masked_bolds_arr_2.append(apply_mask(mask=mask.get_fdata(), target=bold).flatten())
    masked_bolds_arr_2 = np.vstack(masked_bolds_arr_2)
    print("masked phase2 bold array shape: ", masked_bolds_arr_2.shape)    

    # ===== load weights
    print(f"Loading {weight_source}...")
    if phase1 == 'pre':
        if weight_source == "LSA":
            weights_dir_1 = os.path.join(data_dir, f"sub-{subID}", "localizer_item_level")
        elif weight_source == "LSS":
            weights_dir_1 = os.path.join(data_dir, f"sub-{subID}", "localizer_LSS_lvl1")
        else:
            raise ValueError("Weight source must be LSA or LSS")
    if phase2 == 'post':
        if weight_source == "LSA":
            weights_dir_2 = os.path.join(data_dir, f"sub-{subID}", "localizer_item_level")
        elif weight_source == "LSS":
            weights_dir_2 = os.path.join(data_dir, f"sub-{subID}", "post_localizer_LSS_lvl1")
        else:
            raise ValueError("Weight source must be LSA or LSS")
    
    all_weights = {}
    all_weights_1 = {}
    all_weights_2 = {}
    weights_arr= []
    weights_arr_1= []
    weights_arr_2= []    

    for cateID in sub_cates.keys():
        if weight_source=='LSA':
            cate_weights_fnames = glob.glob(f"{weights_dir_1}/{cateID}*full*{stats_test}*")
            print(cateID, len(cate_weights_fnames))
            cate_weights = {}

            for fname in cate_weights_fnames:
                trialID = fname.split("/")[-1].split("_")[1]  # "trial1"
                trialID = int(trialID[5:])
                cate_weights[trialID] = nib.load(fname).get_fdata()
            cate_weights = {i: cate_weights[i] for i in sorted(cate_weights.keys())}
            all_weights[cateID] = cate_weights

        weights_arr.append(np.stack( [ cate_weights[i] for i in sorted(cate_weights.keys()) ] ))    

        elif weight_source=='LSS':
            cate_weights_fnames_1 = glob.glob(f"{weights_dir_1}/{cateID}*{stats_test}*")            
            cate_weights_fnames_2 = glob.glob(f"{weights_dir_2}/{cateID}*{stats_test}*")
            print(cateID, len(cate_weights_fnames_1),len(cate_weights_fnames_2))
        
            cate_weights_1 = {}
            cate_weights_2 = {}            

            for fname in cate_weights_fnames_1:
                trialID = fname.split("/")[-1].split("_")[1]  # "trial1"
                trialID = int(trialID[5:])
                cate_weights_1[trialID] = nib.load(fname).get_fdata()

            for fname in cate_weights_fnames_2:
                trialID = fname.split("/")[-1].split("_")[1]  # "trial1"
                trialID = int(trialID[5:])
                cate_weights_2[trialID] = nib.load(fname).get_fdata()     

            cate_weights_1 = {i: cate_weights_1[i] for i in sorted(cate_weights_1.keys())}
            cate_weights_2 = {i: cate_weights_2[i] for i in sorted(cate_weights_2.keys())}

            all_weights_1[cateID] = cate_weights_1
            all_weights_2[cateID] = cate_weights_2            

            weights_arr_1.append(np.stack( [ cate_weights_1[i] for i in sorted(cate_weights_1.keys()) ] ))
            weights_arr_2.append(np.stack( [ cate_weights_2[i] for i in sorted(cate_weights_2.keys()) ] ))

    if weight_source=='LSA':
        weights_arr = np.vstack(weights_arr)
        print("weights shape: ", weights_arr.shape)
        # apply mask on BOLD
        masked_weights_arr = []
        for weight in weights_arr:
            masked_weights_arr.append(apply_mask(mask=mask.get_fdata(), target=weight).flatten())
        masked_weights_arr = np.vstack(masked_weights_arr)
        print("masked weights arr shape: ", masked_weights_arr.shape)        
    elif weight_source=='LSS':
        weights_arr_1 = np.vstack(weights_arr_1)
        weights_arr_2 = np.vstack(weights_arr_2)
        print("weights 1 shape: ", weights_arr_1.shape)      
        print("weights 2 shape: ", weights_arr_2.shape)
        # apply mask on BOLD
        masked_weights_arr_1 = []
        masked_weights_arr_2 = []        
        for weight in weights_arr_1:
            masked_weights_arr_1.append(apply_mask(mask=mask.get_fdata(), target=weight).flatten())
        masked_weights_arr_1 = np.vstack(masked_weights_arr_1)
        for weight in weights_arr_2:
            masked_weights_arr_2.append(apply_mask(mask=mask.get_fdata(), target=weight).flatten())
        masked_weights_arr_2 = np.vstack(masked_weights_arr_2)        
        print("masked weights 1 arr shape: ", masked_weights_arr_1.shape)
        print("masked weights 2 arr shape: ", masked_weights_arr_2.shape)

    # ===== multiply
    if weight_source=='LSA':
        item_repress = np.multiply(masked_bolds_arr, masked_weights_arr)
        print("item_repres shape: ", item_repress.shape)
    elif weight_source=='LSS':
        item_repress_1=masked_weights_arr_1
        item_repress_2=masked_weights_arr_2
        print("item_repres 1 shape: ", item_repress_1.shape)
        print("item_repres 2 shape: ", item_repress_2.shape)        

    # I will need to add the code here to use the sorting of the trials, to both organize the data... drop the novel from the post (or if study is phase2, drop the unoperated from the phase1)
    # I will also need to set up the correlation BETWEEN these phases, so pre vs. post and not a pair-wise comparison within the phase 

    #so the code below here would basically look at what is the phase1 and phase2, have the data sorted so that we know what item each trial is associated with
        #then I would loop over the pairs and collect the RSA between pre and post, and then likely save that into a dictionary or a dataframe

        #the way this would look is that the r=np.corrcoef(x,y), where x is the 1-D activity (BOLD or LSS) of the pre, and y is the 1-D activity of the post
        #the output when then be saved into its corresponding label in a dictionary: e.g., {'image_ID_100': r}

        #additionally it may help to also segment by the operation on that item since we also have that information available when sorting, all we have to do is look at the operation column to see
            #so I may just sort them into separate dictionary or dataframes

    # ===== perform the correlation 
    corr_matrix=np.corrcoef(item_repress)
    out_name=os.path.join(results_dir,"RSM",f"sub-{subID}_{phase}_{weight_source}_{stats_test}_rsm")
    if not os.path.exists(os.path.join(results_dir, "RSM")): os.makedirs(os.path.join(results_dir, "RSM"),exist_ok=True)    
    np.save(out_name,corr_matrix)    


if __name__ == "__main__":
    item_level_RSA()


    # item vs cateogory average
    # within category MDS