import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt

import os
from os.path import join, exists, split
import sys
sys.path.append("/Users/kailong/Desktop/rtEnv/localize_fork/localize/analysis/GLMsingle/GLMsingle/")
import time
import urllib.request
import warnings
from tqdm import tqdm
from pprint import pprint
warnings.filterwarnings('ignore')

from glmsingle.glmsingle import GLM_single

testMode = True
# http://localhost:8970/notebooks/users/kp578/localize/MRMD-AE/archive/dataPreparation.ipynb
import os
import warnings  # Ignore sklearn future warning
import numpy as np
import pandas as pd
import argparse
# import torch
import random
from glob import glob
import subprocess
import nibabel as nib
from tqdm import tqdm
import sys
import pickle5 as pickle
import time

sys.path.append("/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/")
os.chdir("/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/")
# import phate
# from lib.fMRI import fMRIAutoencoderDataset, fMRI_Time_Subjs_Embed_Dataset
# from lib.helper import extract_hidden_reps, get_models, checkexist
# from torch.utils.data import DataLoader
# from lib.utils import set_grad_req
warnings.simplefilter(action='ignore', category=FutureWarning)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def mkdir(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)


def check(sbatch_response):
    print(sbatch_response)
    if "Exception" in sbatch_response or "Error" in sbatch_response or "Failed" in sbatch_response or "not" in sbatch_response or 'Unrecognised' in sbatch_response:
        raise Exception(sbatch_response)


def getjobID_num(sbatch_response):  # 根据subprocess.Popen输出的proc，获得sbatch的jpobID
    import re
    jobID = re.findall(r'\d+', sbatch_response)[0]
    return jobID


def kp_run(cmd):
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd)
    check(sbatch_response)
    return sbatch_response


def kp_remove(fileName):
    cmd = f"rm {fileName}"
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd)
    print(sbatch_response)


def wait(tmpFile, waitFor=0.1):
    while not os.path.exists(tmpFile):
        time.sleep(waitFor)
    return 1


def check(sbatch_response):
    print(sbatch_response)
    if "Exception" in sbatch_response or "Error" in sbatch_response or "Failed" in sbatch_response or "not" in sbatch_response:
        raise Exception(sbatch_response)


def checkEndwithDone(filename):
    with open(filename, 'r') as f:
        last_line = f.readlines()[-1]
    return last_line == "done\n"


def checkDone(jobIDs):
    completed = {}
    for jobID in jobIDs:
        filename = f"./logs/{jobID}.out"
        completed[jobID] = checkEndwithDone(filename)
    if np.mean(list(completed.values())) == 1:
        status = True
    else:
        status = False
    return completed, status


def check_jobIDs(jobIDs):
    completed, status = checkDone(jobIDs)
    if status == True:
        pass
    else:
        print(completed)
        assert status == True
    return completed


def check_jobArray(jobID='', jobarrayNumber=10):
    arrayIDrange = np.arange(1, 1 + jobarrayNumber)
    jobIDs = []
    for arrayID in arrayIDrange:
        jobIDs.append(f"{jobID}_{arrayID}")
    completed = check_jobIDs(jobIDs)
    return completed


def waitForEnd(jobID):
    while jobID_running_myjobs(jobID):
        print(f"waiting for {jobID} to end")
        time.sleep(5)
    print(f"{jobID} finished")


def jobID_running_myjobs(jobID):
    jobID = str(jobID)
    cmd = "squeue -u kp578"
    sbatch_response = subprocess.getoutput(cmd)
    if jobID in sbatch_response:
        return True
    else:
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ROI', type=str, default='early_visual')
parser.add_argument('--n_subjects', type=int, default=16)

args = parser.parse_args("")

subFolder = "/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/"
subs = glob(f"{subFolder}/sub*");
subs.sort();
subs = [sub.split("/")[-1] for sub in subs]
for sub in subs:
    funcs = glob(f"{subFolder}/{sub}/func/*.nii")
    print(f"{sub} {len(funcs)}")

"""
    localize的原始数据
    anat数据
        /gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/sub022/anat
    fMRI的数据
        /gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/sub022/func
    行为学的数据
        包括是否包含黑白斑块，是否按键了，当前的图片是哪一张
        /gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/sub022/behav/
    眼动记录仪的数据
        暂时不知道
"""

# 首先得到所有的被试的fMRI的数据以及对应的行为学数据
# run内fmap校准
# run内运动校准
# run内时间校准




# opt = dict()
#
# # 为完整性设置重要的字段（但这些字段在默认情况下会被启用）。  set important fields for completeness (but these would be enabled by default)
# opt['wantlibrary'] = 1
# opt['wantglmdenoise'] = 1
# opt['wantfracridge'] = 1
#
# # 在本例中，我们将在内存中保留相关的输出，同时也将它们保存在磁盘上。  for the purpose of this example we will keep the relevant outputs in memory and also save them to the disk
# opt['wantfileoutputs'] = [1,1,1,1]
# opt['wantmemoryoutputs'] = [1,1,1,1]
#
# glmsingle_obj=GLM_single(opt)
# results_glmsingle=glmsingle_obj.fit(
#                             design,
#                             data,
#                             stimdur,
#                             tr,
#                             outputdir=outputdir_glmsingle)

