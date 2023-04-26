#script to take in the trained model from clearmem and the PCA, apply to subjects and export results:
OUTDATED_IGNORE=1
import os
import glob
import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime
import matplotlib

from nilearn.signal import clean
from nilearn.maskers import NiftiMasker
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

import psutil
import nibabel as nib
from scipy.signal import resample

import fnmatch
