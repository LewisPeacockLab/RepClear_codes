import numpy as np

# import glob
# import os

# from PIL import Image
import matplotlib.pyplot as plt

# import pandas as pd
# from sklearn.manifold import MDS

import seaborn as sns

sns.set(
    style="white",
)  # context='notebook',   #rc={"lines.linewidth": 2.5}
sns.set(palette="colorblind")


workspace = "scratch"
if workspace == "work":
    data_dir = "/work/07365/sguo19/frontera/fmriprep/"
    param_dir = "/work/07365/sguo19/frontera/params/"
    results_dir = "/work/07365/sguo19/model_fitting_results/"
elif workspace == "scratch":
    data_dir = "/scratch1/07365/sguo19/fmriprep/"
    param_dir = "/scratch1/07365/sguo19/params/"
    results_dir = "/scratch1/07365/sguo19/model_fitting_results/"


def hierarchical_clustering(data, save=True, out_fname="hc"):
    from sklearn.cluster import AgglomerativeClustering

    hc = AgglomerativeClustering(n_clusters=1, compute_distances=True)
    hc = hc.fit(data)

    if save:
        print(f"Saving to {out_fname}...")
        np.savez_compressed(out_fname, hc=hc)

    return hc


def plot_dendrogram(
    model, labels, save=False, tag="", out_fname="dendrogram", **kwargs
):
    """
    Plot sklean hc models with scipy dendrogram

    Input:
    labels: labels of each sample that's fed into the hierarchical clustering model
    """
    from scipy.cluster.hierarchy import dendrogram

    lc_dict = {
        None: "k",
        "female": "C1",
        "male": "C2",
        "manmade": "C3",
        "natural": "C4",
    }

    class Node:
        """helper class to assign branches with category color"""

        def __init__(self, ID, children, cate=None):
            """
            children: tuple of Nodes. If (None, None), this is a leaf node
            """
            self.ID = ID
            self.children = children
            self.cate_score = {
                "male": 0,
                "female": 0,
                "manmade": 0,
                "natural": 0,
                None: -1,
            }
            self.cate = cate
            if self.cate is not None:
                self.cate_score[self.cate] += 1

        def cate_from_children(self):
            for child in self.children:
                for cate in self.cate_score.keys():
                    self.cate_score[cate] += child.cate_score[cate]
            self.cate_score[None] = -1

            # get self.cate
            sscores, scates = zip(
                *(
                    sorted(
                        zip(self.cate_score.values(), self.cate_score.keys()),
                        reverse=True,
                    )
                )
            )
            if sscores[0] == sscores[1]:  # same scores
                self.cate = None
            else:
                self.cate = scates[0]

    def label2color(ID):
        if type(ID) == int:
            label = all_nodes[ID].cate
        elif type(ID) == str:
            label = ID
        else:
            print(ID)
        return lc_dict[label]

    # === get mat for scipy plot
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # === construct nodes to plot colors
    all_nodes = [
        Node(ID=i, children=[None, None], cate=lab) for i, lab in enumerate(labels)
    ]
    for i, (c1ID, c2ID) in enumerate(model.children_[:]):
        curr = Node(ID=i + 200, children=[all_nodes[c1ID], all_nodes[c2ID]])
        curr.cate_from_children()
        all_nodes.append(curr)

    # Plot the corresponding dendrogram
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    dendrogram(
        linkage_matrix, ax=ax, labels=labels, link_color_func=label2color, **kwargs
    )

    # set image label color
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_color(label2color(lbl.get_text()))

    # make legend
    from matplotlib.lines import Line2D

    eles = []
    for cate, color in lc_dict.items():
        if cate is None:
            continue
        eles.append(Line2D([0], [0], color=color, label=cate))
    ax.legend(handles=eles)

    if tag != "":
        ax.set_title(tag)

    if save:
        print(f"Saving to {out_fname}...")
        plt.savefig(out_fname)


if __name__ == "__main__":
    pass
