import sys, os
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import importlib.util
import argparse
from dataprovider import DataProvider
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from model import UnifiedModel
from esm.tokenization import get_esmc_model_tokenizers
from esm.models.esmc import ESMC

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

def load_config(config_path, batch_size=None, chkp_path=None, chkp_name=None, out_path=None, hla_path=None, test_path=None, num_workers=None, use_compile=False, plot=False, hla_emb_path=None):
    """Dynamically import the config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    # Update config with provided parameters
    if batch_size is not None:
        config["Test"]["batch_size"] = batch_size
    if num_workers is not None:
        config["Data"]["num_workers"] = num_workers
    if chkp_path is not None:
        config["chkp_path"] = chkp_path
    if chkp_name is not None:
        config["chkp_name"] = chkp_name
    if out_path is not None:
        config["out_path"] = out_path
    if plot:
        config["Test"]["plot"] = True
    if use_compile:
        config["Test"]["use_compile"] = True
    if hla_path is not None:
        config["Data"]["hla_path"] = hla_path
    if test_path is not None:
        config["Data"]["test_path"] = test_path
    if hla_emb_path is not None:
        config["encoder_args"]["hla_emb_path"] = hla_emb_path
    return config

def load_unified_model(config, device, use_compile=False):
    model_esm = ESMC(
        d_model=960,
        n_heads=15,
        n_layers=30,
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=True,
    )
    model_esm.load_state_dict(torch.load(config['Test']['esm_chkp_path'], map_location=device))
    model_esm.to(device, dtype=torch.bfloat16).eval()
    print(f'ESM model loaded on {device}')
    model = config["model"](**config["model_args"])
    model.load_state_dict(torch.load(config['Test']['chkp_path'], map_location=device)['model_state_dict'])
    model.to(device, dtype=torch.bfloat16).eval()
    print(f'Model loaded on {device}')
    unified_model = UnifiedModel(model_esm, model).to(device).eval()
    if use_compile:
        print("Compiling unified model...")
        unified_model = torch.compile(unified_model)
    return unified_model

def test_model(model, dataloader, device):
    all_preds = []
    torch.backends.cudnn.benchmark = True
    with torch.no_grad(), torch.amp.autocast(device.type):
        for batch in tqdm(dataloader, desc="Testing"):
            batch = [item.to(device) for item in batch]
            y_pred = model(*batch)
            all_preds.append(y_pred.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    return all_preds

def main(config):
    out_path = config['out_path']
    os.makedirs(out_path, exist_ok=True)
    use_compile = config.get("use_compile", False)

    DATA_PROVIDER_ARGS = {
        "epi_path": config['Data']['test_path'],
        "epi_args": config['Data']['test_args'],
        "hla_path": config['Data']['hla_path'],
        "hla_args": config['Data']['hla_args'],
    }

    data_provider = DataProvider(**DATA_PROVIDER_ARGS)
    print(f"Datapoints in dataset: {len(data_provider)}")

    dataset = config["encoder"](data_provider, **config["encoder_args"])
    batch_size = config["Test"]["batch_size"] if "batch_size" in config["Test"] else len(dataset)
    num_workers = config["Data"]["num_workers"]
    collate_fn = config.get("collate_fn", None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_unified_model(config, device, use_compile=use_compile)
    y_pred = test_model(model, dataloader, device)

    ############ Plotting ############
    df_epi = data_provider.df_epi
    df_epi['Logits'] = y_pred
    df_epi['Score'] = df_epi['Logits'].apply(lambda x: 1 / (1 + np.exp(-x)))  # Sigmoid function
    df_epi.to_csv(os.path.join(out_path, f'prediction.csv'), index=False)
    df_top_10 = df_epi.nlargest(10, 'Score')
    df_top_10['Score'] = df_top_10['Score'].apply(lambda x: f"{x:.4f}")
    df_top_10['Logits'] = df_top_10['Logits'].apply(lambda x: f"{x:.4f}")

    if not config["Test"].get("plot", False):
        print("Plotting is disabled in the config.")
        return df_top_10
    plt.figure(figsize=(6, 6))
    sns.kdeplot(df_epi['Score'], fill=True, color='#29BDFD', alpha=0.6, linewidth=0)
    plt.title('Kernel Density Plot of Predictions')
    plt.xlabel('Predictions')
    plt.ylabel('Density')
    plt.xlim(-0.18, 1.18)
    plt.xticks(np.arange(0, 1.01, 0.1))  # x축 tick 지정
    plt.axvline(x=0.5, color='#F53255', linestyle='--', label='Threshold (0.5)')
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"plot.png"))
    plt.show()
    return df_top_10

def cli_main():
    parser = argparse.ArgumentParser(description="Train model with specified config.")
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    parser.add_argument("--seed", type=int, help="Random seed to use.")
    parser.add_argument("--fold", type=int, help="Validation fold to use.")
    parser.add_argument("--lr", type=str, help="Learning rate (e.g., 1e4 for 1e-4).")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--chkp_path", type=str, help="Checkpoint path.")
    parser.add_argument("--chkp_name", type=str, help="Checkpoint name.")
    parser.add_argument("--out_path", type=str, help="Path to save plots.")
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
        out_path=args.out_path,
        epi_path=args.epi_path,
        hla_path=args.hla_path,
        test_path=args.test_path,
        bulk_path=args.bulk_path
    )

    main(config)

if __name__ == "__main__":
    cli_main()
