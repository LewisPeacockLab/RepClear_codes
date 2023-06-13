import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
import nibabel as nib
import scipy as scipy
import scipy.io as sio
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fnmatch
import pandas as pd
import pickle

os.chdir("/Users/zb3663/Documents/GitHub/repclear/fmri_v2/experiment/Prelim_data")

df = pd.read_csv("memory_accuracy_sure.csv", index_col=0)

plt.style.use("fivethirtyeight")
ax = sns.violinplot(data=df, inner="box")
ax.set(ylabel="Memory (Hit - Miss)", title='Pilot Memory Data - Only "sure" responses')
sns.swarmplot(data=df, color="white", edgecolor="gray")

plt.show()

plt.style.use("fivethirtyeight")
ax = sns.barplot(data=df, ci="sd")
ax.set(ylabel="Memory (Hit - Miss)", title='Pilot Memory Data - Only "sure" responses')
# sns.swarmplot(data=df,color="white",edgecolor="gray")

plt.show()

#####################

df = pd.read_csv("memory_accuracy_allcorrect.csv", index_col=0)

plt.style.use("fivethirtyeight")
ax = sns.violinplot(data=df, inner="box")
ax.set(ylabel="Memory (Hit - Miss)", title="Pilot Memory Data - All responses")
sns.swarmplot(data=df, color="white", edgecolor="gray")

plt.show()

plt.style.use("fivethirtyeight")
ax = sns.barplot(data=df, ci="sd")
ax.set(ylabel="Memory (Hit - Miss)", title="Pilot Memory Data - All responses")
# sns.swarmplot(data=df,color="white",edgecolor="gray")

plt.show()

#######################

df = pd.read_csv("memory_confidence_ratings.csv", index_col=0)
plt.style.use("fivethirtyeight")
ax = sns.barplot(data=df, ci="sd")
ax.set(
    ylabel="Memory Confidence (1 is low, 4 is high)",
    title="Average Confidence rating for memory test",
)
# sns.swarmplot(data=df,color="white",edgecolor="gray")

plt.show()

##########

os.chdir("/Users/zb3663/Documents/GitHub/repclear/fmri_v2/experiment/Prelim_data")

df1 = pd.read_csv("average_subject_maintain_evi.csv", index_col=0)
df2 = pd.read_csv("average_subject_replace_evi.csv", index_col=0)
df3 = pd.read_csv("average_subject_suppress_evi.csv", index_col=0)
