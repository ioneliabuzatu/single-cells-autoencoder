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
from model import Autoencoder, Regressor, AutoencoderV2, VariationalAutoencoder, AutoencoderV3
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
    wandb_run_name=f"!!!V2...#COMP{config.latent_space}b{config.batch_size}",
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


class MyDataset(Dataset):

    def __init__(self, inputs, train: bool):
        if type(inputs) == pd.DataFrame: inputs = inputs.values

        if whoami == 'ionelia':
            if train:
                x = inputs[:50]
            else:
                x = inputs[50:]
        else:
            if train:
                x = inputs[:50_000]
            else:
                x = inputs[50_000:]

        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


# cite_train_inputs = MinMaxScaler().fit_transform(cite_train_inputs)
# cite_train_inputs = StandardScaler().fit_transform(cite_train_inputs)
train_dataset = DatasetWithY(cite_train_inputs, cite_train_targets, train=True, whoami=whoami)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
val_dataset = DatasetWithY(cite_train_inputs, cite_train_targets, train=False, whoami=whoami)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

# autoencoder = VariationalAutoencoder(config.latent_space, device).to(device)
# autoencoder = Autoencoder(in_shape=22050, enc_shape=config.latent_space).float().to(device)
autoencoder = AutoencoderV2(in_shape=22050, enc_shape=config.latent_space).float().to(device)
regressor = Regressor(in_shape=config.latent_space).float().to(device)
print("how much does the model weight?")
os.system('nvidia-smi')
loss_fn_autoencoder = nn.MSELoss(reduction='sum')
loss_fn_regressor = nn.MSELoss(reduction='sum')
# loss_fn_total = nn.MSELoss(reduction='sum')
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=config.lr)
# optimizer_encoder = optim.Adam(autoencoder.encoder.parameters())
# optimizer_decoder = optim.Adam(autoencoder.decoder.parameters())
optimizer_regressor = optim.Adam(regressor.parameters())
# optimizer_autoencoder = optim.SGD(autoencoder.parameters(), lr=0.01, momentum=0.95)
# optimizer_regressor = optim.SGD(regressor.parameters(), lr=0.01, momentum=0.95)

# optimizer_total = optim.Adam(list(autoencoder.parameters() + regressor.parameters()))

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
    autoencoder.train()
    regressor.train()
    losses_epoch = []
    losses_epoch_regressor = []
    pearson_epoch_regressor = []
    reduced_inputs = []
    for x_id, (x_batch, y_batch) in enumerate(train_loader):
        x = x_batch.to(device)
        y = y_batch.to(device)

        # train autoencoder
        optimizer_autoencoder.zero_grad()
        # optimizer_encoder.zero_grad()
        # optimizer_decoder.zero_grad()
        reduced_x = autoencoder.encoder(x)
        output = autoencoder.decoder(reduced_x)
        loss_autoencoder = loss_fn_autoencoder(output, x)
        # loss_autoencoder = ((x - output) ** 2).sum() + autoencoder.encoder.kl
        loss_autoencoder.backward()
        optimizer_autoencoder.step()
        # optimizer_encoder.step()
        # optimizer_decoder.step()
        losses_epoch.append(loss_autoencoder.item())

        # train regressor
        optimizer_regressor.zero_grad()
        reduced_x = torch.tensor(reduced_x, requires_grad=True)
        protein_predictions = regressor(reduced_x)
        loss_regressor = loss_fn_regressor(protein_predictions, y)
        pearson_loss_regressor = cos(y - y.mean(dim=1, keepdim=True), protein_predictions - protein_predictions.mean(dim=1, keepdim=True)).mean()
        pearson_epoch_regressor.append(pearson_loss_regressor.item())
        loss_regressor.backward()
        optimizer_regressor.step()
        losses_epoch_regressor.append(loss_regressor.item())

        del x
        del reduced_x
        curr_step_train += 1

    losses_epoch = np.array(losses_epoch)
    run.log({"train/autoencoder_mse_epoch": losses_epoch.mean(), 'epoch': epoch})
    lr_autoencoder = optimizer_autoencoder.param_groups[0]["lr"]
    run.log({"train/lr_autoencoder": lr_autoencoder, 'epoch': epoch})

    losses_epoch_regressor = np.array(losses_epoch_regressor)
    pearson_epoch_regressor = np.array(pearson_epoch_regressor)
    run.log({"train/regressor_mse_epoch": losses_epoch_regressor.mean(), 'epoch': epoch})
    run.log({"train/pearson_regressor": pearson_epoch_regressor.mean(), 'epoch':epoch})
    print(f"train/pearson:{pearson_epoch_regressor.mean():.3f}")

    del losses_epoch
    del losses_epoch_regressor
    del pearson_epoch_regressor

    if epoch % 10 == 0:
        scheduler.step()

    # ---------------------------------------- eval ------------------------------------------------- #
    val(run,
        epoch,
        val_loader,
        device,
        autoencoder,
        loss_fn_autoencoder,
        regressor=regressor,
        loss_fn_regressor=loss_fn_regressor,
        y_val=y_val
    )

torch.save(autoencoder.state_dict(), f"{base_dir}/checkpoints/comp_autoencoder.pth")