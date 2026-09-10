import os
import sys
import torch
import numpy as np
import importlib.util
import argparse
from prepibind.dataprovider import DataProvider_inf as DataProvider
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from prepibind.model import UnifiedModel
from prepibind.esmc import ESMC, get_esmc_model_tokenizers

def load_config(config_path, batch_size=None, chkp_path=None, out_path=None, hla_path=None, test_path=None, num_workers=None, use_compile=None, plot=None, hla_emb_path=None, esm_chkp_path=None):
    """Import the config module at `config_path`, then apply any CLI overrides."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    overrides = {
        ("Test", "batch_size"): batch_size,
        ("Test", "chkp_path"): chkp_path,
        ("Test", "esm_chkp_path"): esm_chkp_path,
        ("Test", "out_path"): out_path,
        ("Test", "plot"): plot,
        ("Test", "use_compile"): use_compile,
        ("Data", "num_workers"): num_workers,
        ("Data", "hla_path"): hla_path,
        ("Data", "test_path"): test_path,
        ("encoder_args", "hla_emb_path"): hla_emb_path,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
    return config

#: How the two halves are run, as (ESMC dtype, head dtype).
#:
#: `as-trained` reproduces how the numbers in the paper were produced: the stored embeddings came
#: out of a bfloat16 ESMC forward pass, and the head was trained and evaluated in float32.
#:
#: `fp16` is for the demo. Encoding epitopes on the fly cannot be bit-exact against a precomputed
#: store anyway, so the demo takes the memory and speed instead. Scores move in the last decimals.
PRECISION = {
    "as-trained": (torch.bfloat16, torch.float32),
    "fp16": (torch.float16, torch.float16),
}


def resolve_runtime(device, precision):
    """flash-attn and the two dtypes, cut down to what this device can actually do.

    ESMC's flash-attn path needs Ampere or newer -- Colab's free tier is a T4, which is not -- and
    neither half format is worth running on a CPU. Whatever gets dropped is printed: a silently
    downgraded run is one whose numbers cannot be compared with anything. flash-attn also has to
    be installed; load_unified_model reports it when it is not.
    """
    esm_dtype, head_dtype = PRECISION[precision]
    if device.type != "cuda":
        return False, torch.float32, torch.float32, "no CUDA: float32 throughout, no flash-attn"
    major = torch.cuda.get_device_capability(device)[0]
    if major >= 8:
        return True, esm_dtype, head_dtype, None
    note = f"compute capability {major}.x is pre-Ampere: flash-attn off"
    if esm_dtype is torch.bfloat16:
        esm_dtype, note = torch.float16, note + ", bfloat16 -> float16"
    return False, esm_dtype, head_dtype, note


def load_unified_model(config, device, use_compile=False):
    flash, esm_dtype, head_dtype, note = resolve_runtime(
        torch.device(device), config["Test"].get("precision", "as-trained"))
    if note:
        print(f'Runtime adjusted -- {note}')
    model_esm = ESMC(
        d_model=960,
        n_heads=15,
        n_layers=30,
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=flash,
    )
    if flash and not model_esm._use_flash_attn:
        print('Runtime adjusted -- flash-attn is not installed: plain attention')
    model_esm.load_state_dict(torch.load(config["Test"]["esm_chkp_path"], map_location=device))
    model_esm.to(device, dtype=esm_dtype).eval()
    print(f'ESM model loaded on {device} ({esm_dtype})')
    model = config["model"](**config["model_args"])
    model.load_state_dict(torch.load(config["Test"]["chkp_path"], map_location=device)['model_state_dict'])
    model.to(device, dtype=head_dtype).eval()
    print(f'Model loaded on {device}')
    unified_model = UnifiedModel(model_esm, model).to(device).eval()
    if use_compile:
        print("Compiling unified model...")
        unified_model = torch.compile(unified_model)
    return unified_model

def test_model(model, dataloader, device):
    all_preds = []
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            batch = [item.to(device) for item in batch]
            y_pred = model(*batch)
            all_preds.append(y_pred.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    return all_preds

def main(config):
    out_path = config["Test"]['out_path']
    os.makedirs(out_path, exist_ok=True)
    use_compile = config['Test'].get("use_compile", False)

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

    df_epi = data_provider.df_epi
    df_epi['Logits'] = y_pred
    df_epi['Score'] = df_epi['Logits'].apply(lambda x: 1 / (1 + np.exp(-x)))
    df_epi.to_csv(os.path.join(out_path, 'prediction.csv'), index=False)

    if not config["Test"].get("plot", False):
        print("Plotting is disabled in the config.")
        return df_epi
    plt.figure(figsize=(6, 6))
    sns.kdeplot(df_epi['Score'], fill=True, color='#29BDFD', alpha=0.6, linewidth=0)
    plt.title('Kernel Density Plot of Predictions')
    plt.xlabel('Predictions')
    plt.ylabel('Density')
    plt.xlim(-0.18, 1.18)
    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.axvline(x=0.5, color='#F53255', linestyle='--', label='Threshold (0.5)')
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "plot.png"))
    plt.show()
    return df_epi

def cli_main():
    parser = argparse.ArgumentParser(description="Run inference with specified config.")
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--chkp_path", type=str, help="Path to PREpiBind model checkpoint.")
    parser.add_argument("--out_path", type=str, help="Path to save output files.")
    parser.add_argument("--hla_path", type=str, help="Path to HLA mapping CSV.")
    parser.add_argument("--test_path", type=str, help="Path to input data CSV.")
    parser.add_argument("--num_workers", type=int, help="Number of workers for DataLoader.")
    parser.add_argument("--use_compile", action='store_true', help="Use torch.compile for the model.")
    parser.add_argument("--plot", action='store_true', help="Enable plotting of results.")
    parser.add_argument("--hla_emb_path", type=str, help="Path to HLA embedding file.")
    parser.add_argument("--esm_chkp_path", type=str, help="Path to ESM model checkpoint.")
    args = parser.parse_args()

    config = load_config(
        config_path=args.config_path,
        batch_size=args.batch_size,
        chkp_path=args.chkp_path,
        out_path=args.out_path,
        hla_path=args.hla_path,
        test_path=args.test_path,
        num_workers=args.num_workers,
        use_compile=args.use_compile,
        plot=args.plot,
        hla_emb_path=args.hla_emb_path,
        esm_chkp_path=args.esm_chkp_path
    )

    main(config)

if __name__ == "__main__":
    cli_main()
