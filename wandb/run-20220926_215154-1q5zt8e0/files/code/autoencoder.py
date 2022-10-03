import os

import experiment_buddy
import torch
import os 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import h5py
import hdf5plugin
import tables

import pandas as pd
import numpy as np
import scipy

import scanpy as sc
import anndata as ad

import gc, pickle, scipy.sparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import time
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import random

import getpass

experiment_buddy.register_defaults({})

buddy = experiment_buddy.deploy(
    host='',
    disabled=False,
    wandb_kwargs={
        'sync_tensorboard': True,
        'save_code': True,
        'entity': 'ionelia',
        'project': 'kaggle',
        'reinit': False
    },
    wandb_run_name=f"autoencoder",
    extra_modules=["cuda/11.1/nccl/2.10", "cudatoolkit/11.1", "cuda/11.1/cudnn/8.1"]
)

run = buddy.run

os.system('nvidia-smi')
os.system('htop')

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)

whoami = getpass.getuser()
if whoami == 'ionelia':
    base_dir = "/home/ionelia/datasets/msci-kaggle/"
elif whoami == 'ionelia.buzatu':
    base_dir = '/network/projects/_groups/grn_control/datasets/comp/'

df_eval = pd.read_csv(os.path.join(base_dir, 'evaluation_ids.csv'))
df_metadata = pd.read_csv(os.path.join(base_dir, 'metadata.csv'), index_col='cell_id')
cite_train_targets = pd.read_hdf(os.path.join(base_dir,"train_cite_targets.h5"))
df_metadata_cite_train = df_metadata[df_metadata.index.isin(cite_train_targets.index)]
cite_train_inputs = pd.read_hdf(os.path.join(base_dir,"train_cite_inputs.h5"))

print("shape train inputs:", cite_train_inputs.shape)