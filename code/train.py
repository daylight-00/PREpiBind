# train.py

import sys, os
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import logging
import time
import importlib.util
import argparse
from dataprovider import DataProvider
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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

def roc_auc_score(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    sorted_indices = torch.argsort(outputs, descending=True)
    sorted_labels = labels[sorted_indices]
    total_positive = torch.sum(sorted_labels).item()
    total_negative = len(sorted_labels) - total_positive
    if total_positive == 0 or total_negative == 0:
        return float('nan')
    tps = torch.cumsum(sorted_labels, dim=0)
    fps = torch.arange(1, len(sorted_labels) + 1, device=sorted_labels.device) - tps
    tpr = tps / total_positive
    fpr = fps / total_negative
    auc = torch.trapz(tpr, fpr).item()
    return auc

def save_checkpoint(model, optimizer, epoch, model_path, prefix="best"):
    """Save the model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    save_path = f"{model_path}-{prefix}.pt"
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, model_path, config):
    transfer = config["Train"].get("transfer", False)
    resume = config["Train"].get("resume", False)
    finetune = config["Train"].get("finetune", False)
    save_path = config["Train"].get("resume_path", f"{model_path}-best.pt")
    if os.path.exists(save_path):
        print(f"Checkpoint found at {save_path}")
        checkpoint = torch.load(save_path)
        if transfer or resume or finetune:
            print(f"Loading checkpoint from {save_path}")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Overwriting model")
        if resume:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {epoch}")
        else:
            epoch = 1
        if transfer:
            print(f"Transfer learning mode")
            fixed_layers = config["Train"]["transfer_args"].get("fixed_layers", [])
            train_layers = config["Train"]["transfer_args"].get("train_layers", [])
            if len(fixed_layers) > 0 and len(train_layers) > 0:
                raise ValueError("Both fixed_layers and train_layers cannot be specified at the same time.")
            elif len(fixed_layers) > 0:
                print(f"Fixed layers: {fixed_layers}")
                for name, param in model.named_parameters():
                    if name in fixed_layers:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                optimizer = config["Train"]["optimizer"](
                    filter(lambda p: p.requires_grad, model.parameters()),
                    **config["Train"]["optimizer_args"]
                )
            elif len(train_layers) > 0:
                print(f"Trainable layers: {train_layers}")
                for name, param in model.named_parameters():
                    if name in train_layers:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                optimizer = config["Train"]["optimizer"](
                    filter(lambda p: p.requires_grad, model.parameters()),
                    **config["Train"]["optimizer_args"]
                )
    else:
        print(f"No checkpoint found. Starting training from scratch.")
        epoch = 1
    return model, optimizer, epoch

def train_model(model, train_loader, val_loader, criterion, optimizer, device, log_file, num_epochs, patience, model_path, regularize=False, scheduler=None, start_epoch=1):
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - [Training process]: %(message)s'
    )

    for epoch in range(start_epoch, num_epochs+1):

        start_time = time.time()
        epoch_loss = 0
        val_loss = 0
        correct_train = 0
        total_train = 0
        correct_val = 0
        total_val = 0

        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{num_epochs}")

                X, Y = batch[:-1], batch[-1]
                X = [x.to(device) for x in X]
                Y = Y.to(device)
                outputs = model(*X)
                loss = criterion(outputs, Y)
                if regularize:
                    loss = model.regularize(loss, device)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(Y)
                predicted = (outputs > 0.5).float()
                correct_train += (predicted == Y).sum().item()
                total_train += Y.size(0)

                tepoch.set_postfix(
                    loss=loss.item(), 
                    acc = correct_train / total_train, 
                    lr = optimizer.param_groups[0]['lr']
                )
        avg_train_loss = epoch_loss / total_train
        train_acc = correct_train / total_train

        scheduler.step() if scheduler is not None else None

        model.eval()
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as vepoch:
                all_outputs = torch.tensor([]).to(device)
                all_targets = torch.tensor([]).to(device)
                for batch in vepoch:
                    vepoch.set_description(f"Validation Epoch {epoch}/{num_epochs}")
                    
                    X, Y = batch[:-1], batch[-1]
                    X = [x.to(device) for x in X]
                    Y = Y.to(device)
                    
                    outputs = model(*X)
                    loss = criterion(outputs, Y)
                    val_loss += loss.item() * len(Y)

                    predicted = (outputs > 0.5).float()
                    correct_val += (predicted == Y).sum().item()
                    total_val += Y.size(0)

                    vepoch.set_postfix(
                        loss=loss.item(), 
                        acc=correct_val / total_val, 
                        lr=optimizer.param_groups[0]['lr']
                    )
                    all_outputs = torch.cat((all_outputs, outputs), dim=0)
                    all_targets = torch.cat((all_targets, Y), dim=0)

        avg_val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        val_roc_auc = roc_auc_score(all_outputs.view(-1), all_targets.view(-1))
        epoch_time = int(time.time() - start_time)
        logging.info(f'\
[{model_path}]-[Epoch {epoch:03d}/{num_epochs:03d}] - \
Time: {epoch_time} s, \
Train Acc: {train_acc:.5f}, \
Val Acc: {val_acc:.5f}, \
Train Loss: {avg_train_loss:.5f}, \
Val Loss: {avg_val_loss:.5f}, \
Val ROC-AUC: {val_roc_auc:.5f}'\
)
        # if (epoch+1 >= 30 ) and (epoch+1 % 5 == 0):
        #     save_checkpoint(model, optimizer, epoch, model_path, prefix=f"epoch_{epoch+1}")
        
        # Early stopping and best model saving
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            save_checkpoint(model, optimizer, epoch, model_path, prefix="best")
            print(f"Best model updated at epoch {epoch}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

def main(config):
    set_seed(config["seed"])
    print(f"Seed set to {config['seed']}")
    num_folds = config["CrossValidation"].get("num_folds", 5)
    data_provider  = DataProvider(
        epi_path=config["Data"]["epi_path"],
        epi_args=config["Data"]["epi_args"],
        hla_path=config["Data"]["hla_path"],
        hla_args=config["Data"]["hla_args"],
        num_folds=num_folds,
    )
    print(f"Total samples: {len(data_provider)}")

    if config["CrossValidation"].get("cv", False):
        val_idx = config["CrossValidation"].get("val_fold", 0)
        print (f"Cross-validation fold: {val_idx}")
        fold_indices = data_provider.fold_indices
        val_indices = fold_indices[val_idx]
        train_indices = [idx for j, fold in enumerate(fold_indices) if j != val_idx for idx in fold]

    else:
        y = np.array([data_provider.samples[i][2] for i in range(len(data_provider))])
        train_indices, val_indices = train_test_split(
            np.arange(len(data_provider)),
            stratify=y,
            test_size=config["Data"]["val_size"],
            random_state=config["seed"],
            
        )

    full_dataset = config["encoder"](data_provider, **config["encoder_args"])
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print(f"Samples in train set: {len(train_dataset)}")
    print(f"Samples in validation set: {len(val_dataset)}")

    batch_size_t = config["Train"].get("batch_size", 128)
    batch_size_v = config["Train"].get("batch_size_val", batch_size_t)
    num_workers = config["Data"]["num_workers"]
    collate_fn = config.get("collate_fn", None)
    g = torch.Generator()
    g.manual_seed(config["seed"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size_t, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_v, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # Model, optimizer, and criterion setup
    model = config["model"](**config["model_args"])
    criterion = config["Train"]["criterion"]()
    optimizer = config["Train"]["optimizer"](
        model.parameters(),
        **config["Train"]["optimizer_args"]
        )
    num_epochs = config["Train"]["num_epochs"]
    model_path = os.path.join(config["chkp_path"], config["chkp_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, model_path, config)
    print(f'Model loaded on {device}')
    if config["Train"]["use_scheduler"]:
        from utils.scheduler import CosineAnnealingWarmUpRestarts
        optimizer = torch.optim.Adam(model.parameters(), lr=0)
        total_training_steps    = num_epochs*len(train_loader)
        cycle_len               = total_training_steps//10
        warmup_steps            = cycle_len//10
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0     = cycle_len,    # cycle length
            T_mult  = 1,            # cycle length multiplier after each cycle.
            T_up    = warmup_steps, # warmup length
            eta_max = 0.1,          # max learning rate
            gamma   = 0.8           # max learning rate decay
            )
        print(f"Scheduling with cycle length {cycle_len} and warmup steps {warmup_steps}")
        print("Optimizer is changed to Adam with lr=0 due to scheduler.")
    else:
        scheduler = None
    os.makedirs(config["chkp_path"], exist_ok=True)
    # Start training
    train_model(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        criterion       = criterion,
        optimizer       = optimizer,
        device          = device,
        log_file        = config["log_file"],
        num_epochs      = num_epochs,
        patience        = config["Train"]["patience"],
        regularize      = config["Train"]["regularize"],
        model_path      = model_path,
        scheduler       = scheduler,
        start_epoch     = start_epoch,
        )

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
