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

# 注意：运行这段代码还需要fracridge资源库。 note: the fracridge repository is also necessary to run this code
# for example, you could do:
#      git clone https://github.com/nrdg/fracridge.git

# 获取安装GLMsingle的目录的路径 get path to the directory to which GLMsingle was installed
homedir = split(os.getcwd())[0]

# 创建保存数据的目录 create directory for saving data
datadir = join(homedir,'examples','data')
os.makedirs(datadir,exist_ok=True)

# 创建目录以保存实例1的输出 create directory for saving outputs from example 1
outputdir = join(homedir,'examples','example1outputs')

print(f'directory to save example dataset:\n\t{datadir}\n')
print(f'directory to save example1 outputs:\n\t{outputdir}\n')