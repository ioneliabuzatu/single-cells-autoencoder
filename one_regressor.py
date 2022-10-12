import sys

import numpy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import getpass
import logging
import os
import random

import experiment_buddy
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import config
from model import Autoencoder, AutoencoderV2, VariationalAutoencoder, AutoencoderV3
from model import Regressor
from utils import correlation_score_fn, DatasetWithY, MyDataset, TestDataset
from val_utils import val_regression_only

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
    wandb_run_name=f"day2and3regressor#COMP{config.latent_space}b{config.batch_size}",
    extra_modules=["cuda/11.1/nccl/2.10", "cudatoolkit/11.1", "cuda/11.1/cudnn/8.1"]
)

run = buddy.run

os.system('nvidia-smi')
os.system('free -hm')

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)
print("Loading data...")
whoami = getpass.getuser()
if whoami == 'ionelia':
    base_dir = "/home/ionelia/datasets/msci-kaggle/"
    cite_train_inputs = np.random.random((100, 22050))
elif whoami == 'ionelia.buzatu':
    base_dir = '/network/projects/_groups/grn_control/datasets/comp/'
    cite_train_inputs = pd.read_hdf(os.path.join(base_dir, "train_cite_inputs.h5"))
    print("shape train inputs:", cite_train_inputs.shape)

df_eval = pd.read_csv(os.path.join(base_dir, 'evaluation_ids.csv'))
df_metadata = pd.read_csv(os.path.join(base_dir, 'metadata.csv'), index_col='cell_id')
cite_train_targets = pd.read_hdf(os.path.join(base_dir, "train_cite_targets.h5"))
# df_metadata_cite_train = df_metadata[df_metadata.index.isin(cite_train_targets.index)]
print("Done loading data.")

if whoami == 'ionelia':
    config.batch_size = 10
    y_train = cite_train_targets[:50]
    y_val = cite_train_targets[50:50 + 50]
    filepath_save_predictions = f"{base_dir}/predictions.npy"
else:
    # split = int(len(cite_train_targets) * 0.70)
    # y_train = cite_train_targets[:split]
    # y_val = cite_train_targets[split:]
    filepath_save_predictions = f"{base_dir}/predictions_day2and3trained.npy"

regressor = Regressor(in_shape=22050, out_shape=140).float().to(device)

MAKE_PREDICTIONS = True
if MAKE_PREDICTIONS:
    regressor.load_state_dict(torch.load(f"{base_dir}/checkpoints/safe-keeping/comp_day2and3_one_regressor_to_rule_them_all.pth"))
    logging.info(f"generating predictions now....")
    cite_test_inputs = pd.read_hdf(os.path.join(base_dir, "test_cite_inputs.h5"))
    print(f"Loaded test inputs of shape: {cite_test_inputs.shape}")
    test_loader = DataLoader(TestDataset(cite_test_inputs), batch_size=50, shuffle=False, drop_last=False)
    regressor.eval()
    predictions = []
    for x_id, x_batch in enumerate(test_loader):
        x = x_batch.to(device)
        protein_predictions = regressor(x)
        predictions.append(protein_predictions.cpu().detach().numpy())
    predictions = np.vstack(predictions)
    np.save(filepath_save_predictions, predictions)
    print(f"Predictions saves to 'comp/predictions.npy' with shape {predictions.shape} and now exiting.")
    sys.exit()

# cite_train_inputs = MinMaxScaler().fit_transform(cite_train_inputs)
# cite_train_inputs = StandardScaler().fit_transform(cite_train_inputs)

cite_train_inputs = cite_train_inputs.loc[df_metadata.loc[cite_train_inputs.index][df_metadata.day!=2].index]
cite_train_targets = cite_train_targets.loc[df_metadata.loc[cite_train_targets.index][df_metadata.day!=2].index]

split = int(len(cite_train_targets) * 0.70)
y_train = cite_train_targets[:split]
y_val = cite_train_targets[split:]

train_dataset = DatasetWithY(cite_train_inputs, cite_train_targets, train=True, whoami=whoami)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
val_dataset = DatasetWithY(cite_train_inputs, cite_train_targets, train=False, whoami=whoami)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)



print("how much does the model weight?")
os.system('nvidia-smi')
loss_fn_regressor = nn.MSELoss(reduction='sum')
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

optimizer_regressor = optim.Adam(regressor.parameters(), lr=0.05)
lambda_scheduler = lambda epoch: 0.85 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_regressor, lr_lambda=lambda_scheduler)

n_epochs = 50
if whoami == 'ionelia':
    n_epochs = 3
curr_step_train = 0

# val(run, 0, val_loader, device, autoencoder, loss_fn_autoencoder, regressor=regressor,
#     loss_fn_regressor=loss_fn_regressor, y_val=y_val)
for epoch in range(1, n_epochs):
    logging.info(f"Epoch {epoch:10}")
    regressor.train()
    losses_epoch = []
    losses_epoch_regressor = []
    pearson_epoch_regressor = []
    reduced_inputs = []
    for x_id, (x_batch, y_batch) in enumerate(train_loader):
        x = x_batch.to(device)
        y = y_batch.to(device)
        protein_predictions = regressor(x)
        optimizer_regressor.zero_grad()
        loss_regressor = loss_fn_regressor(protein_predictions, y)
        pearson_loss_regressor = cos(y - y.mean(dim=1, keepdim=True),
                                     protein_predictions - protein_predictions.mean(dim=1, keepdim=True)).mean()
        pearson_epoch_regressor.append(pearson_loss_regressor.item())
        loss_regressor.backward()
        optimizer_regressor.step()
        losses_epoch_regressor.append(loss_regressor.item())

        del x
        curr_step_train += 1

    losses_epoch_regressor = np.array(losses_epoch_regressor)
    run.log({"train/regressor_mse_epoch": losses_epoch_regressor.mean(), 'epoch': epoch})
    pearson_epoch_regressor = np.array(pearson_epoch_regressor)
    run.log({"train/pearson_regressor": pearson_epoch_regressor.mean(), 'epoch':epoch})
    run.log({"train/lr_regressor": optimizer_regressor.param_groups[0]["lr"], 'epoch': epoch})

    del losses_epoch_regressor
    del pearson_epoch_regressor

    if epoch % 5 == 0:
        scheduler.step()

    # ---------------------------------------- eval ------------------------------------------------- #
    val_regression_only(run,
        epoch,
        val_loader,
        device,
        regressor=regressor,
        loss_fn_regressor=loss_fn_regressor,
        y_val=y_val
    )

    torch.save(regressor.state_dict(), f"{base_dir}/checkpoints/comp_day2and3_one_regressor_to_rule_them_all.pth")