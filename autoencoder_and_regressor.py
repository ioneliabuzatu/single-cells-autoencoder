#  TODO
# combine models both reduction and prediction
#          first try to have 2 lossses
#         then merge them together where the loss is just the correlation function
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
from model import Autoencoder, Regressor, AutoencoderV2
from utils import correlation_score_fn, DatasetWithY

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
    wandb_run_name=f"100latent#CoupledRegressor#sumMSE#50latent#batchNORM+SELU#autoencoder_batch#{config.batch_size}",
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


#### cite_train_inputs = MinMaxScaler().fit_transform(cite_train_inputs)
train_dataset = DatasetWithY(cite_train_inputs, cite_train_targets, train=True, whoami=whoami)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
val_dataset = DatasetWithY(cite_train_inputs, cite_train_targets, train=False, whoami=whoami)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

autoencoder = AutoencoderV2(in_shape=22050, enc_shape=config.latent_space).float().to(device)
regressor = Regressor(in_shape=config.latent_space).float().to(device)
print("how much does the model weight?")
os.system('nvidia-smi')
loss_fn_autoencoder = nn.MSELoss(reduction='sum')
loss_fn_regressor = nn.MSELoss(reduction='sum')
optimizer_autoencoder = optim.Adam(autoencoder.parameters())
optimizer_regressor = optim.Adam(regressor.parameters())

n_epochs = 20
if whoami == 'ionelia':
    n_epochs = 1
curr_step_train = 0
for epoch in range(n_epochs):
    logging.info(f"Epoch {epoch:10}")
    autoencoder.train()
    regressor.train()
    losses_epoch = []
    losses_epoch_regressor = []
    reduced_inputs = []
    for x_id, (x_batch, y_batch) in enumerate(train_loader):
        x = x_batch.to(device)
        y = y_batch.to(device)

        # train autoencoder
        optimizer_autoencoder.zero_grad()
        reduced_x = autoencoder.encode(x)
        output = autoencoder.decode(reduced_x)
        loss_autoencoder = loss_fn_autoencoder(output, x)
        loss_autoencoder.backward()  # retain_graph=False)
        optimizer_autoencoder.step()
        losses_epoch.append(loss_autoencoder.item())
        run.log({"train/autoencoder_mse_steps": losses_epoch[-1], 'step': curr_step_train})

        # train regressor
        optimizer_regressor.zero_grad()
        reduced_x = torch.tensor(reduced_x, requires_grad=True)
        protein_predictions = regressor(reduced_x)
        loss_regressor = loss_fn_regressor(protein_predictions, y)
        loss_regressor.backward()
        optimizer_regressor.step()
        losses_epoch_regressor.append(loss_regressor.item())
        logging.info(f"regressor mse step: {losses_epoch_regressor[-1]}")
        run.log({"train/regressor_mse_steps": losses_epoch_regressor[-1], 'step': curr_step_train})

        total_loss = loss_autoencoder + loss_regressor
        run.log({"train/tot_mse_steps": total_loss.item(), 'step': curr_step_train})

        del x
        del reduced_x
        curr_step_train += 1

    losses_epoch = np.array(losses_epoch)
    losses_epoch_regressor = np.array(losses_epoch_regressor)
    run.log({"train/autoencoder_mse_epoch": losses_epoch.mean(), 'epoch': epoch})
    run.log({"train/regressor_mse_epoch": losses_epoch_regressor.mean(), 'epoch': epoch})
    del losses_epoch
    del losses_epoch_regressor
    # ---------------------------------------- eval ------------------------------------------------- #
    autoencoder.eval()
    regressor.eval()
    epoch_val_mse_autoencoder = []
    epoch_val_mse_regressor = []
    va_preds = []
    with torch.no_grad():
        for x_id, (x_batch, y_batch) in enumerate(val_loader):
            x = x_batch.to(device)
            y = y_batch.to(device)
            encoded = autoencoder.encode(x)
            decoded = autoencoder.decode(encoded)
            mse_autoencoder = loss_fn_autoencoder(decoded, x).item()
            epoch_val_mse_autoencoder.append(mse_autoencoder)
            del decoded
            del x

            val_protein_predictions = regressor(encoded)
            val_mse_regressor = loss_fn_regressor(val_protein_predictions, y).item()
            va_preds.append(val_protein_predictions.cpu().detach().numpy())
            epoch_val_mse_regressor.append(val_mse_regressor)
            del encoded

        epoch_val_mse_autoencoder = np.array(epoch_val_mse_autoencoder)
        epoch_val_mse_regressor = np.array(epoch_val_mse_regressor)
        run.log({"val/mse_autoencoder": epoch_val_mse_autoencoder.mean(), "epoch": epoch})
        run.log({"val/mse_regressor": epoch_val_mse_regressor.mean(), "epoch": epoch})
        del epoch_val_mse_autoencoder
        del epoch_val_mse_regressor

        va_preds = np.vstack(va_preds)
        logging.info(f"shape y_true {y_val.shape} vs y_preds: {va_preds.shape}")
        correlation_score = correlation_score_fn(y_val, va_preds)
        logging.info(f"Epoch[{epoch}]correlation[{correlation_score:.3f}]")
        run.log({"val/correlation": correlation_score, "epoch": epoch})

torch.save(autoencoder.state_dict(), f"{base_dir}/checkpoints/comp_autoencoder_and_regressor.pth")