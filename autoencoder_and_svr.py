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
from model import Autoencoder, Regressor
from utils import correlation_score_fn

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
    wandb_run_name=f"sumMSE#50latent#MinMaxX#batchNORM+SELU#autoencoder_batch#{config.batch_size}",
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
df_cite_test_inputs = pd.head_hdf(os.path.join(base_dir, "test_cite_inputs.h5"))
print("Done loading data.")

if whoami == 'ionelia':
    config.batch_size = 10
    y_train = cite_train_targets[:50]
    y_val = cite_train_targets[50:50+50]
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

cite_train_inputs = MinMaxScaler().fit_transform(cite_train_inputs)
train_dataset = MyDataset(cite_train_inputs, train=True)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
val_dataset = MyDataset(cite_train_inputs, train=False)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

autoencoder = Autoencoder(in_shape=22050, enc_shape=config.latent_space).float().to(device)
print("how much does the model weight?")
os.system('nvidia-smi')
error = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(autoencoder.parameters())

autoencoder.train()
n_epochs = 20
if whoami == 'ionelia':
    n_epochs = 1
curr_step_train = 0
for epoch in range(n_epochs):
    logging.info(f"Epoch {epoch:10}")
    autoencoder.train()
    losses_epoch = []
    reduced_inputs = []
    svrs_models = []
    for x_id, x_batch in enumerate(train_loader):
        x = x_batch.to(device)
        optimizer.zero_grad()

        reduced_x = autoencoder.encode(x)
        output = autoencoder.decode(reduced_x)

        loss = error(output, x)
        loss.backward()
        optimizer.step()
        losses_epoch.append(loss.item())
        run.log({"train/mse_steps": losses_epoch[-1], 'step': curr_step_train})
        curr_step_train += 1

        reduced_x = reduced_x.cpu().detach().numpy()
        reduced_inputs.append(reduced_x)
        del x
        del reduced_x

    losses_epoch = np.array(losses_epoch)
    run.log({"train/mse_epoch": losses_epoch.mean(), 'epoch': epoch})
    del losses_epoch
    reduced_inputs = np.vstack(reduced_inputs)
    logging.info(f"shape reduced_inputs: {reduced_inputs.shape}")
    for protein in range(140):
        model = make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5, max_iter=5000))
        model.fit(reduced_inputs, y_train.iloc[:, protein].copy())
        svrs_models.append(model)
    del reduced_inputs

    autoencoder.eval()
    gathered_reduced_val_inputs = []
    epoch_val_mse = []
    with torch.no_grad():
        for x_id, x_batch in enumerate(val_loader):
            x = x_batch.to(device)
            encoded = autoencoder.encode(x)
            decoded = autoencoder.decode(encoded)
            mse = error(decoded, x).item()
            epoch_val_mse.append(mse)
            gathered_reduced_val_inputs.append(encoded.cpu().detach().numpy())
            del encoded
            del decoded
            del x

        epoch_val_mse = np.array(epoch_val_mse)
        logging.info(f"val mse: {epoch_val_mse.mean()}")
        run.log({"val/mse": epoch_val_mse.mean(), "epoch": epoch})
        del epoch_val_mse

        gathered_reduced_val_inputs = np.vstack(gathered_reduced_val_inputs)
        logging.info(f"shape reduced val inputs: {gathered_reduced_val_inputs.shape}")
        va_preds = []
        for protein in range(140):
            model = svrs_models[protein]
            prediction = model.predict(gathered_reduced_val_inputs)
            va_preds.append(prediction)
        va_preds = np.column_stack(va_preds)
        logging.info(f"shape y_true {y_val.shape} vs y_preds: {va_preds.shape}")
        correlation_score = correlation_score_fn(y_val, va_preds)
        logging.info(f"Epoch[{epoch}]correlation[{correlation_score:.3f}]")
        run.log({"val/correlation": correlation_score, "epoch": epoch})
        del gathered_reduced_val_inputs

    del svrs_models

torch.save(autoencoder.state_dict(), f"{base_dir}/checkpoints/comp_autoencoder.pth")