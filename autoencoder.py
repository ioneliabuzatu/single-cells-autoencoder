import os
import config
import experiment_buddy
import torch
import os 
from model import Autoencoder
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import h5py
import hdf5plugin
import tables

import pandas as pd
import numpy as np
import scipy
import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import MinMaxScaler
import scanpy as sc
import anndata as ad
from torch.utils.data import Dataset, DataLoader

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

np.random.seed(1234)
random.seed(1234)
torch.manual_seed(1234)

experiment_buddy.register_defaults({})
print(locals())

buddy = experiment_buddy.deploy(
    host='mila',
    disabled=False,
    wandb_kwargs={
        'sync_tensorboard': True,
        'save_code': True,
        'entity': 'ionelia',
        'project': 'kaggle',
        'reinit': False
    },
    wandb_run_name=f"autoencoder_batch#{config.batch_size}",
    extra_modules=["cuda/11.1/nccl/2.10", "cudatoolkit/11.1", "cuda/11.1/cudnn/8.1"]
)

run = buddy.run

os.system('nvidia-smi')
os.system('free -hm')

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

if whoami != 'ionelia':
    cite_train_inputs = pd.read_hdf(os.path.join(base_dir,"train_cite_inputs.h5"))
    print("shape train inputs:", cite_train_inputs.shape)
else:
    cite_train_targets = NotImplementedError
    raise cite_train_targets


class MyDataset(Dataset):

    def __init__(self, inputs, train: bool):

        if train:
            x = inputs.values[:50000]
        else:
            x = inputs.values[50000:]

        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


train_dataset = MyDataset(cite_train_inputs, train=True)
val_dataset = MyDataset(cite_train_inputs, train=False)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=True)

autoencoder = Autoencoder(in_shape=22050, enc_shape=2).float().to(device)
print("how much does the model weight?")
os.system('nvidia-smi')
error = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(autoencoder.parameters())

autoencoder.train()
n_epochs = 10
curr_step_train = 0
for epoch in range(n_epochs + 1):
    print(f"Epoch {epoch:10}")
    autoencoder.train()
    losses_epoch = []
    for x_id, x_batch in enumerate(train_loader):
        x = x_batch.to(device)
        optimizer.zero_grad()
        output = autoencoder(x)
        loss = error(output, x)
        loss.backward()
        optimizer.step()
        losses_epoch.append(loss.item())
        # run.log({"metrics/mse_steps": losses_epoch[-1]}, step=curr_step_train)
        run.log({"metrics/mse_steps": losses_epoch[-1], 'step_train': curr_step_train})
        curr_step_train += 1

    run.log({"metrics/mse_epoch": np.array(losses_epoch).mean(), 'epoch_step':epoch})
    del losses_epoch

    autoencoder.eval()
    reduced_val_x = []
    with torch.no_grad():
        for x_id, x_batch in enumerate(val_loader):
            x = x_batch.to(device)
            encoded = autoencoder.encode(x)
            decoded = autoencoder.decode(encoded)
            mse = error(decoded, x).item()
            print("val mse:", mse)
            reduced_val_x.append(encoded.cpu().detach().numpy())
            # run.log({"val/mse": mse}, step=curr_step_train)
        print("shape reduced asarray:", np.asarray(reduced_val_x).shape)
        reduced_val_x = np.column_stack(reduced_val_x)
        print("shape reduced col stack:", reduced_val_x.shape)
        svr = make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5, max_iter=5000))

torch.save(autoencoder.state_dict(), f"{base_dir}/checkpoints/comp_autoencoder.pth")