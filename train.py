# train.py
# ICH Detection Training Script (EfficientNet-B4)
# Usage: python train.py --dicom_dir /path/to/jpgs --csv /path/to/labels.csv --epochs 50

import os
import argparse
import random
import time
import json
import platform
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from shutil import copy2
import timm

# Silence harmless scheduler warning during fast-forward
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

try:
    import wandb
except ImportError:
    wandb = None

from preprocess import patient_split
from dataset import init_wandb, get_dataloaders, compute_class_weights, get_device

# =============================================================================
# 1. CONFIGURATION & ARGUMENTS
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="ICH Detection Training")
    parser.add_argument('--dicom_dir', type=str, required=True,
                        help='Directory (or comma-separated list) of JPEG files')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to stage_2_train.csv')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--num_workers', type=int, default=-1,
                        help='-1 for auto (0 on macOS, 4 on Linux/Kaggle)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--save_to', type=str, default='./output_model',
                        help='Directory to export final model and metrics')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to specific checkpoint to resume from (overrides last_checkpoint)')
    parser.add_argument('--last_checkpoint', type=str, default=None,
                        help='Path to save/load last checkpoint. If relative, saved in checkpoint_dir. '
                             'Default: last_model.pth inside checkpoint_dir.')
    return parser.parse_args()

# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# 3. DATA PREPARATION
# =============================================================================
def prepare_data(csv_path, dicom_dir):
    """Prepare slice-level multi-label DataFrame and patient split."""
    raw = pd.read_csv(csv_path)
    if 'ID' not in raw.columns or 'Label' not in raw.columns:
        raise KeyError("CSV must contain 'ID' and 'Label' columns")
    split = raw['ID'].str.rsplit('_', n=1, expand=True)
    raw['slice_id'] = split[0]
    raw['subtype'] = split[1]
    pivot = raw.pivot_table(index='slice_id', columns='subtype', values='Label',
                            aggfunc='max', fill_value=0).reset_index()
    for col in ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.rename(columns={'any': 'any_ich'})
    pivot['image_id'] = pivot['slice_id'] + '.dcm'
    label_cols = ['any_ich', 'epidural', 'intraparenchymal',
                  'intraventricular', 'subarachnoid', 'subdural']
    df = pivot[['image_id'] + label_cols]
    train, val, test = patient_split(df, random_state=42)
    return train, val, test, label_cols

# =============================================================================
# 4. MODEL DEFINITION
# =============================================================================
def build_model(pretrained=True):
    model = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
    in_features = model.num_features
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 6))
    # Handle different naming conventions in timm
    if hasattr(model, 'head'):
        model.head = head
    elif hasattr(model, 'classifier'):
        model.classifier = head
    else:
        model.fc = head
    return model

# =============================================================================
# 5. TRAINING & EVALUATION LOOPS
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return epoch_loss, all_preds, all_labels

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in tqdm(loader, desc="Validating", leave=False):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        all_preds.append(torch.sigmoid(outputs).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return epoch_loss, all_preds, all_labels

def compute_metrics(y_true, y_pred, label_cols, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    per_class_auc = []
    per_class_sens = []
    per_class_spec = []
    for i in range(len(label_cols)):
        if len(np.unique(y_true[:, i])) < 2:
            auc = 0.0
        else:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                if np.isnan(auc):
                    auc = 0.0
            except ValueError:
                auc = 0.0
        per_class_auc.append(auc)
        tp = ((y_pred_bin[:, i] == 1) & (y_true[:, i] == 1)).sum()
        tn = ((y_pred_bin[:, i] == 0) & (y_true[:, i] == 0)).sum()
        fp = ((y_pred_bin[:, i] == 1) & (y_true[:, i] == 0)).sum()
        fn = ((y_pred_bin[:, i] == 0) & (y_true[:, i] == 1)).sum()
        per_class_sens.append(round(tp / (tp + fn + 1e-6), 4))
        per_class_spec.append(round(tn / (tn + fp + 1e-6), 4))
    return {
        'macro_auc': float(np.mean(per_class_auc)),
        'macro_sensitivity': float(np.mean(per_class_sens)),
        'macro_specificity': float(np.mean(per_class_spec)),
        'per_class_auc': per_class_auc,
        'per_class_sens': per_class_sens,
        'per_class_spec': per_class_spec,
    }

# =============================================================================
# 6. EARLY STOPPING & CHECKPOINTING
# =============================================================================
class EarlyStopping:
    def __init__(self, patience, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.early_stop = False

    def step(self, metric):
        """Returns True if early stop triggered, False otherwise."""
        if self.best is None:
            self.best = metric
            self.counter = 0
        elif (self.mode == 'max' and metric > self.best) or (self.mode == 'min' and metric < self.best):
            self.best = metric
            self.counter = 0  # reset on improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def save_full_checkpoint(model, optimizer, scheduler, epoch, val_macro_auc, best_auc, patience_counter, path):
    """Save complete checkpoint with all fields needed for resumption."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_macro_auc': val_macro_auc,
        'best_auc': best_auc,
        'patience_counter': patience_counter,
    }, path)

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    print("Loading and splitting data...")
    train_df, val_df, test_df, label_cols = prepare_data(args.csv, args.dicom_dir)
    pos_weight = compute_class_weights(train_df).to(device)

    # Comma-separated JPEG directories
    jpg_dirs = [d.strip() for d in args.dicom_dir.split(',')]
    print(f"JPEG directories: {jpg_dirs}")

    # DataLoaders
    if args.num_workers == -1:
        num_workers = 0 if platform.system() == 'Darwin' else 4
    else:
        num_workers = args.num_workers
    loaders = get_dataloaders(train_df, val_df, test_df, jpg_dirs,
                              batch_size=args.batch_size)
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    # W&B initialisation
    if not args.no_wandb and wandb:
        init_wandb(project='ich-detection', config=vars(args))
        wandb.log({'train_size': len(train_df), 'val_size': len(val_df), 'test_size': len(test_df)})

    # Model
    model = build_model(pretrained=True).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)

    # Checkpoint paths
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if args.last_checkpoint:
        last_checkpoint_path = args.last_checkpoint
        if not os.path.isabs(last_checkpoint_path):
            last_checkpoint_path = os.path.join(args.checkpoint_dir, last_checkpoint_path)
    else:
        last_checkpoint_path = os.path.join(args.checkpoint_dir, 'last_model.pth')
    print(f"Last checkpoint will be saved to: {last_checkpoint_path}")

    # Resume logic
    start_epoch = 1
    best_val_auc = -1.0
    patience_counter = 0

    if args.resume:
        resume_path = args.resume
        print(f"Resuming from specified checkpoint: {resume_path}")
    elif os.path.exists(last_checkpoint_path):
        resume_path = last_checkpoint_path
        print(f"Resuming from last checkpoint: {resume_path}")
    else:
        resume_path = None

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        # Load model state
        state = checkpoint['model_state_dict']
        if hasattr(model, 'module'):
            model.module.load_state_dict(state)
        else:
            try:
                model.load_state_dict(state)
            except RuntimeError:
                from collections import OrderedDict
                new_state = OrderedDict()
                for k, v in state.items():
                    name = k.replace('module.', '')
                    new_state[name] = v
                model.load_state_dict(new_state)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Safe scheduler loading – fallback for old checkpoints
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  Loaded scheduler state from checkpoint.")
        else:
            print("WARNING: No scheduler_state_dict in checkpoint – fast-forwarding scheduler.")
            resumed_epoch = checkpoint.get('epoch', 0)
            for _ in range(resumed_epoch):
                scheduler.step()
            print(f"  Scheduler fast-forwarded {resumed_epoch} steps to match resumed epoch.")

        start_epoch = checkpoint.get('epoch', 0) + 1
        # FIX: fallback to val_macro_auc for old checkpoints
        best_val_auc = checkpoint.get('best_auc', checkpoint.get('val_macro_auc', -1.0))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resumed at epoch {start_epoch}, best AUC so far: {best_val_auc:.4f}, patience counter: {patience_counter}")

    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    early_stopper = EarlyStopping(patience=args.patience, mode='max')
    if resume_path and 'patience_counter' in checkpoint:
        early_stopper.counter = patience_counter
        early_stopper.best = best_val_auc   # FIX: restore best metric for early stopping

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_preds, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.grad_clip
        )
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)

        train_metrics = compute_metrics(train_labels, train_preds, label_cols)
        train_metrics['loss'] = train_loss
        val_metrics = compute_metrics(val_labels, val_preds, label_cols)
        val_metrics['loss'] = val_loss

        scheduler.step()

        # Logging to W&B
        if not args.no_wandb and wandb and wandb.run:
            for prefix, mets in [('train', train_metrics), ('val', val_metrics)]:
                wandb.log({f'{prefix}/loss': mets['loss'], f'{prefix}/macro_auc': mets['macro_auc']})
                for i, col in enumerate(label_cols):
                    wandb.log({f'{prefix}/auc_{col}': mets['per_class_auc'][i]})
            wandb.log({'lr': optimizer.param_groups[0]['lr']})

        current_auc = val_metrics['macro_auc']

        # ---- Early stopping step (this updates counter and best) ----
        early_stop_flag = early_stopper.step(current_auc)

        # ---- Update best model if improved ----
        if not np.isnan(current_auc) and current_auc > best_val_auc:
            best_val_auc = current_auc
            model_to_save = model.module if hasattr(model, 'module') else model
            save_full_checkpoint(
                model_to_save, optimizer, scheduler, epoch, current_auc,
                best_val_auc, early_stopper.counter, best_model_path
            )
            print(f"Saved best model with val macro AUC: {best_val_auc:.4f}")

        # ---- Save last checkpoint (always, after early stop step) ----
        model_to_save = model.module if hasattr(model, 'module') else model
        save_full_checkpoint(
            model_to_save, optimizer, scheduler, epoch, current_auc,
            best_val_auc, early_stopper.counter, last_checkpoint_path
        )

        if early_stop_flag:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

        elapsed = time.time() - epoch_start
        eta_seconds = elapsed * (args.epochs - epoch)
        eta_min = int(eta_seconds // 60)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val AUC: {current_auc:.4f} | "
              f"Sens: {val_metrics['macro_sensitivity']:.4f} | Spec: {val_metrics['macro_specificity']:.4f} | "
              f"Time: {elapsed/60:.1f}m | ETA: {eta_min}m")

    # ================= EXPORT FINAL MODEL & METRICS =================
    print("\nTraining finished. Exporting final assets...")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("WARNING: No best model checkpoint found – using last model weights from memory.")

    test_loss, test_preds, test_labels = validate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, test_preds, label_cols)
    test_metrics['loss'] = test_loss
    print(f"Test Loss: {test_loss:.4f} | AUC: {test_metrics['macro_auc']:.4f} | "
          f"Sens: {test_metrics['macro_sensitivity']:.4f} | Spec: {test_metrics['macro_specificity']:.4f}")

    if not args.no_wandb and wandb and wandb.run:
        wandb.log({'test/macro_auc': test_metrics['macro_auc']})

    # Save to export directory
    os.makedirs(args.save_to, exist_ok=True)
    if os.path.exists(best_model_path):
        copy2(best_model_path, os.path.join(args.save_to, 'best_model.pth'))
    else:
        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                   os.path.join(args.save_to, 'best_model.pth'))

    metrics_summary = {
        'test_loss': test_loss,
        'test_macro_auc': test_metrics['macro_auc'],
        'test_macro_sensitivity': test_metrics['macro_sensitivity'],
        'test_macro_specificity': test_metrics['macro_specificity'],
        'per_class_auc': dict(zip(label_cols, test_metrics['per_class_auc'])),
        'per_class_sensitivity': dict(zip(label_cols, test_metrics['per_class_sens'])),
        'per_class_specificity': dict(zip(label_cols, test_metrics['per_class_spec'])),
        'class_weights': pos_weight.cpu().tolist(),
        'best_val_auc': best_val_auc,
        'epochs_trained': epoch,
        'label_columns': label_cols
    }
    with open(os.path.join(args.save_to, 'metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics and model exported to {args.save_to}")

    print("Training complete.")