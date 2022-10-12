import numpy as np
import torch
from utils import correlation_score_fn
import logging


def val_multi_models(run, epoch, val_loader, device, autoencoder, loss_fn_autoencoder, regressor=None, loss_fn_regressor=None, y_val=None):
    autoencoder.eval()

    if type(regressor) == list:
        regressor = list(map(lambda model: model.eval(), regressor))
    else:
        regressor.eval()

    epoch_val_mse_autoencoder = []
    epoch_val_mse_regressor = []
    va_preds = []

    with torch.no_grad():
        for x_id, (x_batch, y_batch) in enumerate(val_loader):
            x = x_batch.to(device)
            y = y_batch.to(device)
            encoded = autoencoder.encoder(x)
            decoded = autoencoder.decoder(encoded)
            mse_autoencoder = loss_fn_autoencoder(decoded, x)
            epoch_val_mse_autoencoder.append(mse_autoencoder.item())
            del decoded
            del x

            val_protein_predictions = list(map(lambda model: model(encoded), regressor))
            val_mse_regressor = list(map(lambda loss, pred, true: loss(pred, true).item(), loss_fn_regressor, val_protein_predictions, y.T))
            val_protein_predictions = torch.stack(val_protein_predictions, dim=1).squeeze()
            va_preds.append(val_protein_predictions.cpu().detach().numpy())
            epoch_val_mse_regressor.append(np.array(val_mse_regressor).mean())

            del encoded

    epoch_val_mse_autoencoder = np.array(epoch_val_mse_autoencoder)
    run.log({"val/mse_autoencoder": epoch_val_mse_autoencoder.mean(), "epoch": epoch})

    va_preds = np.vstack(va_preds)
    correlation_score = correlation_score_fn(y_val, va_preds)
    epoch_val_mse_regressor = np.array(epoch_val_mse_regressor)
    run.log({"val/mse_regressor": epoch_val_mse_regressor.mean(), "epoch": epoch})
    logging.info(f"Epoch[{epoch:10}]correlation[{correlation_score:.3f}]")
    run.log({"val/correlation": correlation_score, "epoch": epoch})

    del epoch_val_mse_autoencoder
    del epoch_val_mse_regressor

    return


def val_regression_only(
        run, epoch, val_loader, device, regressor=None, loss_fn_regressor=None, y_val=None):
    multi_regressors=False
    if type(regressor) == list:
        regressor = list(map(lambda model: model.eval(), regressor))
        multi_regressors = True
    else:
        regressor.eval()

    epoch_val_mse_regressor = []
    va_preds = []

    with torch.no_grad():
        for x_id, (x_batch, y_batch) in enumerate(val_loader):
            x = x_batch.to(device)
            y = y_batch.to(device)

            if multi_regressors:
                val_protein_predictions = list(map(lambda model: model(x), regressor))
                val_mse_regressor = list(
                    map(lambda loss, pred, true: loss(pred, true).item(), loss_fn_regressor, val_protein_predictions, y.T))
                val_protein_predictions = torch.stack(val_protein_predictions, dim=1).squeeze()
                epoch_val_mse_regressor.append(np.array(val_mse_regressor).mean())
                va_preds.append(val_protein_predictions.cpu().detach().numpy())
            else:
                val_protein_predictions = regressor(x)
                val_mse_regressor = loss_fn_regressor(val_protein_predictions, y)
                epoch_val_mse_regressor.append(val_mse_regressor.item())
                va_preds.append(val_protein_predictions.cpu().detach().numpy())

    va_preds = np.vstack(va_preds)
    correlation_score = correlation_score_fn(y_val, va_preds)
    epoch_val_mse_regressor = np.array(epoch_val_mse_regressor)
    run.log({"val/mse_regressor": epoch_val_mse_regressor.mean(), "epoch": epoch})
    logging.info(f"Epoch[{epoch:10}]correlation[{correlation_score:.3f}]")
    run.log({"val/correlation": correlation_score, "epoch": epoch})

    del epoch_val_mse_regressor

    return