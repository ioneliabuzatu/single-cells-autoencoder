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
from model import SingleFeatureRegressor
from utils import correlation_score_fn, DatasetWithY
from val_utils import val

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
    wandb_run_name=f"!!!train only multi_regressors...#COMP{config.latent_space}b{config.batch_size}",
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
df_metadata_cite_train = df_metadata[df_metadata.index.isin(cite_train_targets.index)]
# df_cite_test_inputs = pd.head_hdf(os.path.join(base_dir, "test_cite_inputs.h5"))
print("Done loading data.")

if whoami == 'ionelia':
    config.batch_size = 10
    y_train = cite_train_targets[:50]
    y_val = cite_train_targets[50:50 + 50]
else:
    y_train = cite_train_targets[:50_000]
    y_val = cite_train_targets[50_000:]

autoencoder = AutoencoderV2(in_shape=22050, enc_shape=config.latent_space).float().to(device)
autoencoder.load_state_dict(torch.load(f"/home/mila/i/ionelia.buzatu/comp_autoencoder.pth"))
multiple_regressors = [SingleFeatureRegressor(in_shape=config.latent_space).float().to(device) for _ in range(140)]

# cite_train_inputs = MinMaxScaler().fit_transform(cite_train_inputs)
# cite_train_inputs = StandardScaler().fit_transform(cite_train_inputs)
train_dataset = DatasetWithY(cite_train_inputs, cite_train_targets, train=True, whoami=whoami)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
val_dataset = DatasetWithY(cite_train_inputs, cite_train_targets, train=False, whoami=whoami)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)


print("how much does the model weight?")
os.system('nvidia-smi')
loss_fn_autoencoder = nn.MSELoss(reduction='sum')
# loss_fn_regressor = nn.MSELoss(reduction='sum')
multiple_loss_fn_regressors = [nn.MSELoss(reduction='sum') for _ in range(140)]
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=config.lr)
# optimizer_regressor = optim.Adam(regressor.parameters())
optimizers_multiple_regressors = [optim.Adam(regressor.parameters()) for regressor in multiple_regressors]

lambda_scheduler = lambda epoch: 0.85 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_autoencoder, lr_lambda=lambda_scheduler)

n_epochs = 300
if whoami == 'ionelia':
    n_epochs = 3
curr_step_train = 0

# val(run, 0, val_loader, device, autoencoder, loss_fn_autoencoder, regressor=regressor,
#     loss_fn_regressor=loss_fn_regressor, y_val=y_val)
for epoch in range(1, n_epochs):
    logging.info(f"Epoch {epoch:10}")
    # autoencoder.train()
    multiple_regressors = list(map(lambda model: model.train(), multiple_regressors))
    losses_epoch = []
    losses_epoch_regressor = []
    pearson_epoch_regressor = []
    reduced_inputs = []
    for x_id, (x_batch, y_batch) in enumerate(train_loader):
        x = x_batch.to(device)
        y = y_batch.to(device)

        # train autoencoder
        # optimizer_autoencoder.zero_grad()
        reduced_x = autoencoder.encoder(x)
        output = autoencoder.decoder(reduced_x)
        # loss_autoencoder = loss_fn_autoencoder(output, x)
        # loss_autoencoder.backward()
        # optimizer_autoencoder.step()
        # losses_epoch.append(loss_autoencoder.item())

        # train multiple regressors
        for optim_regressor in optimizers_multiple_regressors:
            optim_regressor.zero_grad()
        reduced_x = torch.tensor(reduced_x, requires_grad=True)

        protein_predictions = list(map(lambda model: model(reduced_x), multiple_regressors))
        losses_regressors = list(map(lambda loss, pred, true: loss(pred, true), multiple_loss_fn_regressors, protein_predictions, y.T))
        protein_predictions = torch.stack(protein_predictions, dim=1).squeeze()
        pearson_loss_regressor = cos(y - y.mean(dim=1, keepdim=True), protein_predictions - protein_predictions.mean(dim=1, keepdim=True)).mean()
        pearson_epoch_regressor.append(pearson_loss_regressor.item())
        for loss_regressor in losses_regressors:
            loss_regressor.backward()
        for optim_regressor in optimizers_multiple_regressors:
            optim_regressor.step()

        losses_epoch_regressor.append(np.array(list(map(lambda loss: loss.item(), losses_regressors))).mean())

        del x
        del reduced_x
        curr_step_train += 1

    losses_epoch = np.array(losses_epoch)
    run.log({"train/autoencoder_mse_epoch": losses_epoch.mean(), 'epoch': epoch})
    lr_autoencoder = optimizer_autoencoder.param_groups[0]["lr"]
    run.log({"train/lr_autoencoder": lr_autoencoder, 'epoch': epoch})

    losses_epoch_regressor = np.array(losses_epoch_regressor)
    run.log({"train/regressor_mse_epoch": losses_epoch_regressor.mean(), 'epoch': epoch})
    pearson_epoch_regressor = np.array(pearson_epoch_regressor)
    run.log({"train/pearson_regressor": pearson_epoch_regressor.mean(), 'epoch':epoch})

    del losses_epoch
    del losses_epoch_regressor
    del pearson_epoch_regressor

    if epoch % 5 == 0:
        scheduler.step()

    # ---------------------------------------- eval ------------------------------------------------- #
    val(run,
        epoch,
        val_loader,
        device,
        autoencoder,
        loss_fn_autoencoder,
        # regressor=regressor,
        # loss_fn_regressor=loss_fn_regressor,
        regressor=multiple_regressors,
        loss_fn_regressor=multiple_loss_fn_regressors,
        y_val=y_val
    )

    # torch.save(autoencoder.state_dict(), f"{base_dir}/checkpoints/comp_autoencoder.pth")