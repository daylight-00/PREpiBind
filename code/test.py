import sys, os
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import importlib.util
import argparse
from dataprovider import DataProvider
from torch.utils.data import DataLoader
from tqdm import tqdm
from plot import plot_general_curves, plot_top_10_curves
import h5py
import re
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, precision_score, precision_recall_curve, auc, matthews_corrcoef
import pandas as pd

def parse_lr(s):
    import re
    match = re.match(r"(?:(\d*)e(\d+))", s)
    if not match: raise ValueError(f"Invalid format: {s}")
    base = match.group(1)
    exp = match.group(2)
    base = int(base) if base else 1
    exponent = int(exp)
    return base * (10 ** -exponent)

def format_lr(value):
    if value <= 0:
        raise ValueError("Value must be positive.")
    import math
    exponent = -math.floor(math.log10(value))
    base = round(value * (10 ** exponent))
    if base == 1:
        return f"e{exponent}"
    else:
        return f"{base}e{exponent}"

def load_config(config_path, seed=None, fold=None, lr=None, batch_size=None, chkp_path=None, chkp_name=None, plot_path=None, epi_path=None, hla_path=None, test_path=None, bulk_path=False):
    """Dynamically import the config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    # Update config with provided parameters
    if seed is not None:
        config["seed"] = seed
    if fold is not None:
        config["CrossValidation"]["val_fold"] = fold
    if lr is not None:
        config["Train"]["optimizer_args"]["lr"] = parse_lr(lr)
    if batch_size is not None:
        config["Train"]["batch_size"] = batch_size
    if chkp_path is not None:
        config["chkp_path"] = chkp_path
    if chkp_name is not None:
        config["chkp_name"] = chkp_name
    if plot_path is not None:
        config["plot_path"] = plot_path
    if epi_path is not None:
        config["Data"]["epi_path"] = epi_path
    if hla_path is not None:
        config["Data"]["hla_path"] = hla_path
    if test_path is not None:
        config["Data"]["test_path"] = test_path
    if bulk_path:
        config["chkp_path"] = config["chkp_path"] + "/" + format_lr(config["Train"]["optimizer_args"]["lr"]) + "_s" + str(config["seed"])
        config["chkp_name"] = config["chkp_name"] + "_fold" + str(config["CrossValidation"]["val_fold"])
        config["plot_path"] = config["plot_path"] + "/" + format_lr(config["Train"]["optimizer_args"]["lr"]) + "_s" + str(config["seed"])
        print(f"Checkpoint path: {config['chkp_path']}")
        print(f"Checkpoint name: {config['chkp_name']}")
        print(f"Plot path: {config['plot_path']}")
    return config

def test_model(model, dataloader, device, target_layer=None):
    model.eval()
    # print(model)
    all_preds = []
    all_features = []
    if target_layer is not None:
        features = {}
        def hook_fn(module, input, output):
            # Detach and move to CPU, then convert to numpy
            features['feature'] = output.detach().cpu().numpy()
        # Register the hook
        target_layer = model.self_attn
        hook = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            X = batch[:-1]
            X = [x.to(device) for x in X]
            y_pred = model(*X)
            all_preds.append(y_pred.cpu().numpy())
            if target_layer is not None:
                if features.get('feature') is not None:
                    all_features.append(np.squeeze(features.get('feature')))
    if target_layer is not None:
        hook.remove()

    print(f'Length of all_features: {len(all_features)}')
    print(f'Length of all_preds: {len(all_preds)}')

    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_features

def calculate_metrics(predictions, targets, threshold=0.5):
    """    Calculate classification metrics: AUC, F1 Score, Balanced Accuracy (BACC), and Precision."""
    predictions = predictions.flatten()
    targets = targets.flatten()
    binary_preds = (predictions >= threshold).astype(int)
    roc_auc  = roc_auc_score(targets, predictions)
    precision_vals, recall_vals, _ = precision_recall_curve(targets, predictions)
    pr_auc = auc(recall_vals, precision_vals)
    f1 = f1_score(targets, binary_preds)
    bacc = balanced_accuracy_score(targets, binary_preds)
    precision = precision_score(targets, binary_preds)
    mcc = matthews_corrcoef(targets, binary_preds)  # MCC 계산
    return {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "F1": f1,
        "BACC": bacc,
        "Precision": precision,
        "MCC": mcc  # 반환값에 추가
    }

def main(config):
    model_name = config['chkp_name']
    plot_path = config['plot_path']
    os.makedirs(plot_path, exist_ok=True)

    DATA_PROVIDER_ARGS = {
        "epi_path": config['Data']['test_path'],
        "epi_args": config['Data']['test_args'],
        "hla_path": config['Data']['hla_path'],
        "hla_args": config['Data']['hla_args'],
    }

    model = config["model"](**config["model_args"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(config["chkp_path"], f"{model_name}-{config['Test']['chkp_prefix']}.pt")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f'Model loaded on {device}')

    data_provider = DataProvider(**DATA_PROVIDER_ARGS)
    print(f"Samples in test set: {len(data_provider)}")

    dataset = config["encoder"](data_provider, **config["encoder_args"])
    batch_size = config["Test"]["batch_size"] if "batch_size" in config["Test"] else len(dataset)
    num_workers = config["Data"]["num_workers"]
    collate_fn = config.get("collate_fn", None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    target_layer = None
    if config['Test']['feat_extract']:
        target_layer_name = config["Test"]["target_layer"]
        target_layer = dict([*model.named_modules()])[target_layer_name]
    y_trgt = np.array([sample[-1] for sample in data_provider.samples])  # Optimized data access
    y_pred, features = test_model(model, dataloader, device, target_layer)

    ############ Metrics Calculation ############

    metrics = calculate_metrics(y_pred, y_trgt)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    pd.DataFrame([metrics]).to_csv(f"{plot_path}/metrics-{model_name}.csv", index=False)

    ############ Plotting ############

    if config['Test']['plot']:
        print("Plotting is enabled.")
        plot_general_curves(y_trgt, y_pred, plot_path, model_name)
        # plot_top_10_curves(data_provider, model, config, device, plot_path, model_name, DATA_PROVIDER_ARGS)
    else:
        print("Plotting is disabled.")

    ######### Save Predictions ############

    if config['Test']['feat_extract']:
        print("Feature extraction is enabled.")
        epi_list = [sample[1] for sample in data_provider.samples]
        hla_list = [sample[0] for sample in data_provider.samples]
        epi_hla_list = [f'{epi_list[i]}_{hla_list[i]}' for i in range(len(epi_list))]
        save_path = config["Test"]["feat_path"]
        features_path = os.path.join(save_path, f"{model_name}-feat.h5")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with h5py.File(features_path, 'a') as f:
            for i, epi in enumerate(epi_hla_list):
                try:
                    if epi in f:
                        print(f'Dataset {epi} already exists. Skipping.')
                        continue
                    f.create_dataset(epi, data=features[i])
                except Exception as e:
                    print(f'Error: {e}')
                    print(f'Failed to save dataset for {epi}')
                    continue
        print(f"Features saved to {features_path}")

    if config['Test']['save_pred']:
        print("Saving predictions")
        y_pred_save = f'{plot_path}/pred-{model_name}.csv'
        np.savetxt(y_pred_save, y_pred, delimiter=',')

def cli_main():
    parser = argparse.ArgumentParser(description="Train model with specified config.")
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    parser.add_argument("--seed", type=int, help="Random seed to use.")
    parser.add_argument("--fold", type=int, help="Validation fold to use.")
    parser.add_argument("--lr", type=str, help="Learning rate (e.g., 1e4 for 1e-4).")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--chkp_path", type=str, help="Checkpoint path.")
    parser.add_argument("--chkp_name", type=str, help="Checkpoint name.")
    parser.add_argument("--plot_path", type=str, help="Path to save plots.")
    parser.add_argument("--epi_path", type=str, help="Path to epitope data.")
    parser.add_argument("--hla_path", type=str, help="Path to HLA data.")
    parser.add_argument("--test_path", type=str, help="Path to test data.")
    parser.add_argument("--bulk_path", action="store_true", help="Flag to bulk checkpoint path and name with lr and seed.")
    
    args = parser.parse_args()

    config = load_config(
        config_path=args.config_path,
        seed=args.seed,
        fold=args.fold,
        lr=args.lr,
        batch_size=args.batch_size,
        chkp_path=args.chkp_path,
        chkp_name=args.chkp_name,
        plot_path=args.plot_path,
        epi_path=args.epi_path,
        hla_path=args.hla_path,
        test_path=args.test_path,
        bulk_path=args.bulk_path
    )

    main(config)

if __name__ == "__main__":
    cli_main()
