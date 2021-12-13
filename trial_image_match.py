"""for local use with Box access: match trial id with image number & operations"""

import numpy as np
import pandas as pd
import scipy.io


def make_tim_file():
    # ===== paths & consts
    # pre assigned image number file
    img_path = "./params/manual_pre_assign_image_v3.csv" 
    # BOX path for .mat files
    data_dir = "/Users/sunaguo/Box/LewPeaLabBox/STUDY/RepClear/v2_fmri"
    # output directory
    out_dir = "./param"

    # consts
    projID = "repclear"
    subIDs_dict = {"202110291": "004", 
                "202110221": "003", 
                "202110211": "002"}
    subIDs = ["202110291", "202110221", "202110211"]  # projID_subID
    fname = "dataMat_repclear_fmri_study_s_{}_run01.mat"
    phases = {1: "pre-exposure", 
            2: "pre-localizer", 
            3: "study", 
            4: "post-localizer"}

    # ===== loading & running
    # load image info 
    img_df = pd.read_csv(img_path)

    for subID in subIDs:
        mat_dict = scipy.io.loadmat(f"{data_dir}/{projID}_{subID}/{fname.format(subID)}", simplify_cells=True)
        # store dfs for each phase
        subdfs = []
        
        for phase in phases.keys():
            design = mat_dict['args']['design']['ph'][phase-1]['matrix']
            header = mat_dict['args']['design']['ph'][phase-1]['header']
            
            trial_df = pd.DataFrame(design, columns=header)
            print(trial_df.shape)
            # for storage
            phase_df = pd.DataFrame(columns=["phase", "trial_id", "image_id", "image_name", "category", "subcategory", 
                                "condition", "familiarity_mean", "familiarity_se"])
            
            # get ready data
            phase_df["phase"] = trial_df["phase"]
            phase_df["image_id"] = trial_df["image_id"]
            phase_df["category"] = trial_df["category"]
            phase_df["subcategory"] = trial_df["subcategory"]
            
            if phase == 1:  # no condition in phase 1
                phase_df["condition"] = (np.zeros(len(phase_df)) - 1).astype(int)
            else: 
                phase_df["condition"] = trial_df["condition"]
            
            # get image names & familiarity scores
            img_names = []
            fam_means = []
            fam_ses = []
            
            for imgid in phase_df["image_id"]:
                img_names.append(img_df.iloc[np.where(img_df["id"]==imgid)[0][0]]["image_name"])
                fam_means.append(img_df.iloc[np.where(img_df["id"]==imgid)[0][0]]["familiarity_mean"])
                fam_ses.append(img_df.iloc[np.where(img_df["id"]==imgid)[0][0]]["familiarity_se"])
                
            phase_df["image_name"] = img_names
            phase_df["familiarity_mean"] = fam_means
            phase_df["familiarity_se"] = fam_ses
            
            # trial_id for each category within a phase should be continuous
            for catei in range(1,3):
                n_cate_trial = sum(phase_df["category"]==catei)  # number of trials for this category in this phase  
                phase_df.loc[phase_df["category"]==catei,"trial_id"] = np.arange(n_cate_trial)+1
            
            subdfs.append(phase_df)
        
        sub_df = pd.concat(subdfs)

        # save file
        out_fname = f"{out_dir}/sub-{subIDs_dict[subID]}_trial_image_match.csv"
        sub_df.to_csv(out_fname)
        

if __name__ == "__main__": 
    make_tim_file()