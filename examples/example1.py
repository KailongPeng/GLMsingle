import os, re
print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")

import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt

import os
from os.path import join, exists, split
import time
import urllib.request
import warnings
from tqdm import tqdm
from pprint import pprint
warnings.filterwarnings('ignore')

import sys
sys.path.append('/gpfs/milgram/project/turk-browne/projects/localize/analysis/GLMsingle')
sys.path.append('/gpfs/milgram/project/turk-browne/projects/localize/analysis/fracridge')

from glmsingle.glmsingle import GLM_single


# 注意：运行这段代码还需要fracridge资源库。 note: the fracridge repository is also necessary to run this code
# for example, you could do:
#      git clone https://github.com/nrdg/fracridge.git

import sys
print(sys.executable)
print(sys.version)
print(sys.version_info)


# 获取安装GLMsingle的目录的路径 get path to the directory to which GLMsingle was installed
homedir = split(os.getcwd())[0]

# 创建保存数据的目录 create directory for saving data
datadir = join(homedir,'examples','data')
os.makedirs(datadir,exist_ok=True)

# 创建目录以保存实例1的输出 create directory for saving outputs from example 1
outputdir = join(homedir,'examples','example1outputs')

print(f'directory to save example dataset:\n\t{datadir}\n')
print(f'directory to save example1 outputs:\n\t{outputdir}\n')

# 从GLMsingle OSF库下载示例数据集 download example dataset from GLMsingle OSF repository
# 数据来自NSD数据集（subj01, nsd01扫描会话） data comes from the NSD dataset (subj01, nsd01 scan session).
# see: https://www.biorxiv.org/content/10.1101/2021.02.22.432340v1.full.pdf

datafn = join(datadir, 'nsdcoreexampledataset.mat')

# 为了节省时间，如果示例数据集已经存在于磁盘上，我们将跳过下载。 to save time, we'll skip the download if the example dataset already exists on disk
if not exists(datafn):
    print(f'Downloading example dataset and saving to:\n{datafn}')

    dataurl = 'https://osf.io/k89b2/download'

    # 下载.mat文件到指定目录 download the .mat file to the specified directory
    urllib.request.urlretrieve(dataurl, datafn)

# 加载包含示例数据集的结构  load struct containing example dataset
X = sio.loadmat(datafn)

# 包含每次运行中的Bold时间序列和设计矩阵的变量。  variables that will contain bold time-series and design matrices from each run
data = []
design = []

# 遍历每一次数据的运行  iterate through each run of data
for r in range(len(X['data'][0])):
    # 索引到结构中，将每个运行的时间序列数据追加到列表中 index into struct, append each run's timeseries data to list
    data.append(X['data'][0, r])

    # 将每个运行设计矩阵从稀疏数组转换为完整的numpy数组，并附加上  convert each run design matrix from sparse array to full numpy array, append
    design.append(scipy.sparse.csr_matrix.toarray(X['design'][0, r]))

# 为方便起见，获得数据卷的形状（XYZ）。 get shape of data volume (XYZ) for convenience
xyz = data[0].shape[:3]
xyzt = data[0].shape

# 获得关于刺激持续时间和TR的元数据  get metadata about stimulus duration and TR
stimdur = X['stimdur'][0][0]
tr = X['tr'][0][0]

# 获得识别枕叶皮层的视觉ROI掩码  get visual ROI mask identifying occipital cortex
roi = X['ROI']


# %matplotlib inline
# # plot example slice from run 1
# plt.figure(figsize=(20,6))
# plt.subplot(121)
# plt.imshow(data[0][:,:,0,0])
# plt.title('example slice from run 1',fontsize=16)
# plt.subplot(122)
# plt.imshow(data[11][:,:,0,0])
# plt.title('example slice from run 12',fontsize=16)
#
# # plot example design matrix from run 1
# plt.figure(figsize=(20,20))
# plt.imshow(design[0],interpolation='none')
# plt.title('example design matrix from run 1',fontsize=16)
# plt.xlabel('conditions',fontsize=16)
# plt.ylabel('time (TR)',fontsize=16);


# print some relevant metadata
print(f'There are {len(data)} runs in total\n')
print(f'N = {data[0].shape[3]} TRs per run\n')
print(f'The dimensions of the data for each run are: {data[0].shape}\n')
print(f'The stimulus duration is {stimdur} seconds\n')
print(f'XYZ dimensionality is: {data[0].shape[:3]} (one slice only in this example)\n')
print(f'Numeric precision of data is: {type(data[0][0,0,0,0])}\n')
print(f'There are {np.sum(roi)} voxels in the included visual ROI')


# 创建一个用于保存GLMsingle输出的目录  create a directory for saving GLMsingle outputs
outputdir_glmsingle = join(homedir,'examples','example1outputs','GLMsingle')

opt = dict()

# 为完整性设置重要的字段（但这些字段在默认情况下会被启用）。  set important fields for completeness (but these would be enabled by default)
opt['wantlibrary'] = 1
opt['wantglmdenoise'] = 1
opt['wantfracridge'] = 1

# 在本例中，我们将在内存中保留相关的输出，同时也将它们保存在磁盘上。  for the purpose of this example we will keep the relevant outputs in memory and also save them to the disk
opt['wantfileoutputs'] = [1,1,1,1]
opt['wantmemoryoutputs'] = [1,1,1,1]

# 运行python GLMsingle需要创建一个GLM_single对象，然后使用.fit()例程运行程序。  running python GLMsingle involves creating a GLM_single object and then running the procedure using the .fit() routine
glmsingle_obj = GLM_single(opt)

# 将所有超参数可视化  visualize all the hyperparameters
pprint(glmsingle_obj.params)

[i.shape for i in data]

# 这个例子将输出文件保存到 "example1outputs/GLMsingle "文件夹中，如果这些输出文件还不存在，我们将执行耗时的GLMsingle调用；否则，我们将直接从磁盘加载。
# this example saves output files to the folder  "example1outputs/GLMsingle" if these outputs don't already exist, we will perform the time-consuming call to GLMsingle; otherwise, we will just load from disk.

start_time = time.time()

if not exists(outputdir_glmsingle):

    print(f'running GLMsingle...')

    # 运行GLMsingle。 run GLMsingle
    results_glmsingle = glmsingle_obj.fit(
        design,
        data,
        stimdur,
        tr,
        outputdir=outputdir_glmsingle)

    # 我们将GLMsingle的输出分配给 "results_glmsingle "变量。注意results_glmsingle['typea']包含来自ONOFF模型的GLM估计值，其中所有图像被视为相同条件。
    # we assign outputs of GLMsingle to the "results_glmsingle" variable. note that results_glmsingle['typea'] contains GLM estimates from an ONOFF model, where all images are treated as the same condition. these estimates could be potentially used to find cortical areas that respond to visual stimuli. we want to compare beta weights between conditions therefore we are not going to include the ONOFF betas in any analyses of voxel reliability

else:
    print(f'loading existing GLMsingle outputs from directory:\n\t{outputdir_glmsingle}')

    # load existing file outputs if they exist
    results_glmsingle = dict()
    results_glmsingle['typea'] = np.load(join(outputdir_glmsingle, 'TYPEA_ONOFF.npy'), allow_pickle=True).item()
    results_glmsingle['typeb'] = np.load(join(outputdir_glmsingle, 'TYPEB_FITHRF.npy'), allow_pickle=True).item()
    results_glmsingle['typec'] = np.load(join(outputdir_glmsingle, 'TYPEC_FITHRF_GLMDENOISE.npy'),
                                         allow_pickle=True).item()
    results_glmsingle['typed'] = np.load(join(outputdir_glmsingle, 'TYPED_FITHRF_GLMDENOISE_RR.npy'),
                                         allow_pickle=True).item()

elapsed_time = time.time() - start_time

print(
    '\telapsed time: ',
    f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
)

# 我们将绘制FIT_HRF_GLMdenoise_RR GLM的几个输出，它包含了GLMsingle的全套优化。 we are going to plot several outputs from the FIT_HRF_GLMdenoise_RR GLM, which contains the full set of GLMsingle optimizations.

# 我们将绘制betas、R2、最佳HRF指数和体素frac值。  we will plot betas, R2, optimal HRF indices, and the voxel frac values
plot_fields = ['betasmd', 'R2', 'HRFindex', 'FRACvalue']
colormaps = ['RdBu_r', 'hot', 'jet', 'copper']
clims = [[-5, 5], [0, 85], [0, 20], [0, 1]]

plt.figure(figsize=(12, 8))

for i in range(len(plot_fields)):

    plt.subplot(2, 2, i + 1)

    if i == 0:
        # 当绘制betas时，为简单起见，只需在所有图像演示中取平均值即可。这将产生一个关于体素在响应实验刺激时是否倾向于增加或减少其活动的总结（类似于ONOFF GLM的输出）。
        # when plotting betas, for simplicity just average across all image presentations. this will yield a summary of whether voxels tend to increase or decrease their activity in response to the experimental stimuli (similar to outputs from an ONOFF GLM)
        plot_data = np.nanmean(np.squeeze(results_glmsingle['typed'][plot_fields[i]]), 2)
        titlestr = 'average GLM betas (750 stimuli)'

    else:
        # 绘制GLMsingle输出的所有其他体素指标。 plot all other voxel-wise metrics as outputted from GLMsingle
        plot_data = np.squeeze(results_glmsingle['typed'][plot_fields[i]].reshape(xyz))
        titlestr = plot_fields[i]

    plt.imshow(plot_data, cmap=colormaps[i], clim=clims[i])
    plt.colorbar()
    plt.title(titlestr)
    plt.axis(False)