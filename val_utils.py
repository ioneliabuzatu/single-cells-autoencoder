import numpy as np
import torch
from utils import correlation_score_fn


def val(val_loader, device, autoencoder, loss_fn_autoencoder, regressor=None, loss_fn_regressor=None, y_val=None):
    autoencoder.eval()
    epoch_val_mse_autoencoder = []

    regressor.eval()
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
            
    va_preds = np.vstack(va_preds)
    correlation_score = correlation_score_fn(y_val, va_preds)
    logging.info(f"Epoch[{epoch}]correlation[{correlation_score:.3f}]")
    run.log({"val/correlation": correlation_score, "epoch": epoch})

    epoch_val_mse_autoencoder = np.array(epoch_val_mse_autoencoder)
    epoch_val_mse_regressor = np.array(epoch_val_mse_regressor)
    return epoch_val_mse_autoencoder, epoch_val_mse_regressor, correlation_score