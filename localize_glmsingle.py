import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import urllib.request
import warnings  # Ignore sklearn future warning
warnings.filterwarnings('ignore')
from tqdm import tqdm
from pprint import pprint
import os
from os.path import join, exists, split
import sys
sys.path.append('/gpfs/milgram/project/turk-browne/projects/localize/analysis/GLMsingle')
sys.path.append('/gpfs/milgram/project/turk-browne/projects/localize/analysis/fracridge')
from glmsingle.glmsingle import GLM_single
import pandas as pd
import argparse
import random
from glob import glob
import subprocess
import nibabel as nib
from tqdm import tqdm
import pickle5 as pickle
import time
# http://localhost:8383/lab/tree/analysis/GLMsingle/localize_glmsingle.ipynb

# sys.path.append("/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/")
# os.chdir("/gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/")
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
        os.makedirs(folder)


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


# parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--ROI', type=str, default='early_visual')
# parser.add_argument('--n_subjects', type=int, default=16)
# args = parser.parse_args("")

subFolder = "/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/"
subs = glob(f"{subFolder}/sub*")
subs.sort()
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
import string
alphabet = string.ascii_uppercase


def convertItemColumn(ShownImages):
    ShownImages_ = []
    for image in ShownImages:
        type(image)
        if type(image) == str:
            imageID = alphabet.index(image) + 1
            ShownImages_.append(imageID)
        elif type(image) == float:
            ShownImages_.append(0)
    return np.asarray(ShownImages_)


def getDesignMatrix(behav):
    TRimgList = convertItemColumn(np.asarray((behav['Item'])))
    ImgTR = TRimgList != 0
    trialList = TRimgList[ImgTR]  # in the current trial, which image is shown?
    greySquareTrial = np.asarray(behav['Change'])[ImgTR]  # in the current trial, is there a grey square?
    numberOfTrials = len(trialList)
    numberOfTRs = behav.shape[0]
    designMatrix = np.zeros((numberOfTRs, numberOfTrials))
    currTrial = -1
    for currTR in range(numberOfTRs):
        if TRimgList[currTR] > 0:
            currTrial += 1
            if behav.loc[currTR, 'Change'] == 0:
                designMatrix[currTR, currTrial] = 1
    designMatrix = np.array(designMatrix, dtype=bool)
    return designMatrix, trialList, greySquareTrial


def loadBrainData(sub='', run=1):
    brain = nib.load(
        f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/{sub}/preprocess/func0{run}.feat/filtered_func_data.nii.gz").get_fdata()
    brain = np.transpose(brain, (3, 0, 1, 2))
    brain = brain[3:, :]  # 时间校准， align TR for functional brain data and behavior data.
    print(f"brain.shape={brain.shape}")
    return brain


jobarrayDict = load_obj(f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/GLMsingle/localize_glmsingle_jobarrayDict")
jobarrayID = int(float(sys.argv[1]))
[sub] = jobarrayDict[jobarrayID]
# [sub, run] = jobarrayDict[jobarrayID]
print(f"sub={sub}")


def getBrainBehav(sub='', run=1):
    # 加载行为学数据
    subID = sub[3:]
    behavPath = f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/{sub}/behav/{subID}_{run}.csv"
    behav = pd.read_csv(behavPath)
    brain = loadBrainData(sub=sub, run=run)
    print(f"brain.shape[0]={brain.shape[0]} len(behav)={len(behav)}")
    if brain.shape[0] < len(behav):  # 一般来说是行为学的数据长于大脑数据，此时删除部分行为学数据
        behav = behav[:brain.shape[0]]
        print('行为学数据长')
    else:  # 偶尔也会行为学的数据短于大脑数据，此时删除部分大脑数据。
        brain = brain[:len(behav)]
    assert len(behav) == brain.shape[0]
    designMatrix, trialList, greySquareTrial = getDesignMatrix(behav)

    brain = np.transpose(brain, (1, 2, 3, 0))  # 使得brain的维度变为[:,:,:,TR]

    print(f"designMatrix.shape={designMatrix.shape}")
    print(f"brain.shape={brain.shape}")
    return brain, designMatrix, trialList, greySquareTrial


runs = glob(f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/{sub}/preprocess/func0?.feat")
brains, designMatrixs, trialLists, greySquareTrials = [], [], {}, {}
for run in range(1, 1 + len(runs)):
    brain, designMatrix, trialList, greySquareTrial = getBrainBehav(sub=sub, run=run)
    brains.append(brain)
    designMatrixs.append(designMatrix)
    trialLists[run] = trialList
    greySquareTrials[run] = greySquareTrial

outputdir_glmsingle = f"/gpfs/milgram/project/turk-browne/projects/localize/analysis/subjects/{sub}/glmsingle/"
try:
    os.rmdir(outputdir_glmsingle)
except:
    print(f"{outputdir_glmsingle} does not exist")
mkdir(outputdir_glmsingle)

# 保存方便实用的行为学数据
behaviorData = {}
behaviorData['designMatrixs'] = designMatrixs
behaviorData['trialLists'] = trialLists
behaviorData['greySquareTrials'] = greySquareTrials
save_obj([behaviorData],f"{outputdir_glmsingle}/behaviorData")

if not os.path.exists(f"{outputdir_glmsingle}/TYPEA_ONOFF.npy"):
    print("running GLMsingle")
    design = designMatrixs
    data = brains
    stimdur = 1.5
    tr = 1.5

    # 首先得到所有的被试的fMRI的数据以及对应的行为学数据
    opt = dict()
    # 为完整性设置重要的字段（但这些字段在默认情况下会被启用）。  set important fields for completeness (but these would be enabled by default)
    opt['wantlibrary'] = 1  # 对每个体素进行HRF拟合
    opt['wantglmdenoise'] = 1  # 使用GLMdenoise
    opt['wantfracridge'] = 1  # 使用ridge回归来改善β估计

    # 在本例中，我们将在内存中保留相关的输出，同时也将它们保存在磁盘上。 For the purpose of this example we will keep the relevant outputs in memory and also save them to the disk
    # wantfileoutputs是一个逻辑向量[A B C D]，表示将四种模型类型中的哪一种保存到磁盘（假设它们被计算出来）。
    # A = 0/1用于保存ONOFF模型的结果，
    # B = 0/1用于保存FITHRF模型的结果，
    # C = 0/1用于保存FITHRF_GLMdenoise模型的结果，
    # D = 0/1用于保存FITHRF_GLMdenoise_RR模型的结果。
    # [1 1 1 1] 表示将所有计算结果保存到磁盘。
    opt['wantfileoutputs'] = [1, 1, 1, 1]

    # wantmemoryoutputs是一个逻辑向量[A B C D]，表示要在输出<results>中返回四种模型类型。[0 0 0 1]这意味着只返回最终的D型模型。
    opt['wantmemoryoutputs'] = [1, 1, 1, 1]
    glmsingle_obj = GLM_single(opt)
    results_glmsingle = glmsingle_obj.fit(
        design,
        data,
        stimdur,
        tr,
        outputdir=outputdir_glmsingle)
print('done')
