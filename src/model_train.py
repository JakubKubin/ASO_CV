# /src/model_train.py
import os, time, datetime
import torch, argparse, json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

from dataset_pipeline import DatasetConfig, get_dataloaders
from dataset_pipeline import CLASSES
from evaluation import evaluate_model, print_evaluation_results
from model_implementation import initialize_model, save_model


def _train_one_epoch(model, optimizer, scheduler, loader, device,
                     epoch, tensorboard=None, print_freq=20, loss_threshold=1000.0):
    """
    Train one epoch with gradient clipping and OneCycleLR scheduling per batch.
    AMP is temporarily disabled to avoid NaN issues.
    """
    model.train()
    iters = len(loader)
    running_loss = 0.0
    valid_batches = 0

    for i, (imgs, tgts) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
        step = epoch * iters + i + 1
        imgs = [img.to(device) for img in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

        # Forward
        loss_dict = model(imgs, tgts)
        # Check NaN/Inf in losses
        if any(torch.isnan(v) or torch.isinf(v) for v in loss_dict.values()):
            print(f"[WARN] Batch {i}: NaN/Inf in loss_dict, skipping.")
            continue
        loss = sum(loss_dict.values())
        loss_val = loss.item()
        if loss_val > loss_threshold or torch.isnan(loss):
            print(f"[WARN] Batch {i}: large or NaN loss {loss_val}, skipping.")
            continue

        # Backward
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=7.0)
        optimizer.step()
        # Scheduler step
        scheduler.step()

        running_loss += loss_val
        valid_batches += 1
        if i % print_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{epoch+1}][{i}/{iters}] loss={loss_val:.4f}, lr={current_lr:.2e}")
        if tensorboard:
            tensorboard.add_scalar('loss/train_total', loss_val, step)
            tensorboard.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step)

    avg_loss = running_loss / max(1, valid_batches)
    return avg_loss

def _evaluate(model, loader, device):
    was_train = model.training
    model.train()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            total_loss += sum(loss_dict.values()).item()
    if not was_train:
        model.eval()
    return total_loss / len(loader)


def train(cfg: DatasetConfig,
          model_type: str = "mask_rcnn",
          epochs: int = 300,
          lr: float = 0.005,
          wd: float = 1e-4,
          outdir: str = "checkpoints",
          patience: int = 300,
          eval_interval: int = 5):
    """
    Trains object detection model using SGD with separate LR for backbone and head,
    OneCycleLR, AMP, and logs metrics to TensorBoard.
    """
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Initialize model and data loaders
    model = initialize_model(model_type, len(CLASSES)).to(device)
    train_dl, val_dl = get_dataloaders(cfg)

    # Separate backbone and head parameters for different LRs
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.SGD(
        [
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': head_params,     'lr': lr}
        ],
        momentum=0.95,
        weight_decay=wd,
        nesterov=True
    )

    # OneCycleLR scheduler: warmup + cosine annealing
    total_steps = epochs * len(train_dl)
    warmup_epochs = 2
    pct_start = warmup_epochs / float(epochs)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[lr * 0.1, lr],
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )

    # TensorBoard setup
    tb = SummaryWriter(os.path.join(outdir, "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    history = {'train_loss': [], 'val_loss': []}
    metric_keys = [
        'mAP_coco','mAP_50','mAP_75',
        'AP_small','AP_medium','AP_large',
        'AR_1','AR_10','AR_100',
        'AR_small','AR_medium','AR_large'
    ]
    for k in metric_keys:
        history[k] = []

    best_map = 0.0
    no_improve = 0

    for ep in range(epochs):
        epoch_start = time.time()

        # Train one epoch
        train_loss = _train_one_epoch(
            model, optimizer, scheduler, train_dl, device,
            ep, tensorboard=tb, print_freq=20
        )
        history['train_loss'].append(train_loss)

        # Validation loss
        val_loss = _evaluate(model, val_dl, device)
        history['val_loss'].append(val_loss)

        # Periodic COCO evaluation
        if (ep + 1) % eval_interval == 0 or ep == epochs - 1:
            eval_results = evaluate_model(model, val_dl, device)
            for k in metric_keys:
                val = eval_results.get(k)
                history[k].append(val)
                tb.add_scalar(k, val, ep)
            inf_ms = eval_results['avg_inference_time'] * 1000.0
            tb.add_scalar('InferenceTime_ms', inf_ms, ep)
            tb.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, ep)

            print(f"--- Eval @ epoch {ep+1} ---")
            print_evaluation_results(eval_results)

            # Checkpointing
            save_model(model, os.path.join(outdir, f"{model_type}_last.pth"))
            if eval_results['mAP_coco'] > best_map:
                best_map = eval_results['mAP_coco']
                no_improve = 0
                save_model(model, os.path.join(outdir, f"{model_type}_best.pth"))
                print(f"[INFO] New best mAP: {best_map:.7f}")
            else:
                no_improve += 1

            if (ep + 1) % eval_interval == 0:
                save_model(model, os.path.join(outdir, f"{model_type}_e{ep+1}.pth"))

            if no_improve >= patience:
                print(f"[INFO] Early stopping after {patience} evals with no improvement.")
                break

        epoch_time = time.time() - epoch_start
        print(f"Epoch {ep+1}/{epochs} - time: {epoch_time:.1f}s | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

    # Save history to file
    with open(os.path.join(outdir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['faster_rcnn','mask_rcnn'], default='mask_rcnn')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--cache', action='store_true')
    args = parser.parse_args()

    cfg = DatasetConfig(cache=args.cache)
    train(cfg, model_type=args.model, epochs=args.epochs,
          patience=args.patience, eval_interval=args.eval_interval)
