# Imports
import os
import fnmatch

# List of subjects
subs = [
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "014",
    "015",
    "016",
    "017",
    "018",
    "020",
    "023",
    "024",
    "025",
    "026",
]

# Brain and task flags
brain_flags = ["T1w", "MNI"]
task_flags = ["preremoval", "study", "postremoval"]

# Loop through flags and subjects
for brain_flag in brain_flags:
    for task_flag in task_flags:
        for sub_num in subs:
            # Define the subject
            sub = f"sub-{sub_num}"
            container_path = (
                "/scratch/06873/zbretton/repclear_dataset/BIDS/derivatives/fmriprep"
            )
            bold_path = os.path.join(container_path, sub, "func/")
            os.chdir(bold_path)

            # Function to find relevant files
            def find(pattern, path):
                result = []
                for root, dirs, files in os.walk(path):
                    for name in files:
                        if fnmatch.fnmatch(name, pattern):
                            result.append(os.path.join(root, name))
                return result

            functional_files = find(f"*-{task_flag}_*.nii.gz", bold_path)

            if brain_flag == "MNI":
                pattern2 = "*MNI152NLin2009cAsym*aparcaseg*"
                functional_files = fnmatch.filter(functional_files, pattern2)

            elif brain_flag == "T1w":
                pattern2 = "*T1w*aparcaseg*"
                functional_files = fnmatch.filter(functional_files, pattern2)

            subject_path = os.path.join(container_path, sub)

            def new_mask(subject_path):
                outdir = os.path.join(subject_path, "new_mask")
                if not os.path.exists(outdir):
                    os.mkdir(outdir)

                aparc_aseg = functional_files[0]

                # Higher-Order Visual Processing Regions
                higher_order_visual = {
                    "lh_inferiorparietal": 1008,
                    "rh_inferiorparietal": 2008,
                    "lh_superiorparietal": 1029,
                    "rh_superiorparietal": 2029,
                    "lh_precuneus": 1025,
                    "rh_precuneus": 2025,
                }

                # Prefrontal Regions
                prefrontal = {
                    "lh_lateralorbitofrontal": 1012,
                    "rh_lateralorbitofrontal": 2012,
                    "lh_medialorbitofrontal": 1014,
                    "rh_medialorbitofrontal": 2014,
                    "lh_rostralmiddlefrontal": 1027,
                    "rh_rostralmiddlefrontal": 2027,
                    "lh_caudalmiddlefrontal": 1003,
                    "rh_caudalmiddlefrontal": 2003,
                    "lh_superiorfrontal": 1028,
                    "rh_superiorfrontal": 2028,
                    "lh_frontalpole": 1032,
                    "rh_frontalpole": 2032,
                }

                # Create individual masks
                for roi_name, roi_val in {**higher_order_visual, **prefrontal}.items():
                    os.system(
                        f"fslmaths {aparc_aseg} -thr {roi_val} -uthr {roi_val} {os.path.join(outdir, roi_name)}.nii.gz"
                    )

                # Combine into composite masks
                os.system(
                    f"fslmaths {' -add '.join([os.path.join(outdir, f'{k}.nii.gz') for k in higher_order_visual.keys()])} -bin {os.path.join(outdir, f'Higher_Order_Visual_ROI_{task_flag}_{brain_flag}.nii.gz')}"
                )
                os.system(
                    f"fslmaths {' -add '.join([os.path.join(outdir, f'{k}.nii.gz') for k in prefrontal.keys()])} -bin {os.path.join(outdir, f'Prefrontal_ROI_{task_flag}_{brain_flag}.nii.gz')}"
                )

            # Generate the new mask
            new_mask(subject_path)
            print(f"{sub} masks generated")
