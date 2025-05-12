# /src/model_train.py
import os, time, datetime
import torch, argparse, json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler

from dataset_pipeline import DatasetConfig, get_dataloaders
from dataset_pipeline import CLASSES
from evaluation import evaluate_model, print_evaluation_results
from model_implementation import initialize_model, save_model

# def _train_one_epoch(model, optimizer, loader, device, epoch, lr, warmup_iters, print_freq=20, tensorboard=None):
#     model.train()
#     is_cuda = torch.cuda.is_available()
#     scaler = GradScaler('cuda', enabled=is_cuda)

#     running_loss = 0.0
#     valid_batches = 0
#     iters = len(loader)

#     for i, (imgs, tgts) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
#         # Global step
#         step = epoch * iters + i + 1

#         # Linear warmup
#         if warmup_iters > 0 and step <= warmup_iters:
#             lr_scale = step / float(warmup_iters)
#             for pg in optimizer.param_groups:
#                 pg['lr'] = lr * lr_scale

#         # Move data to device
#         imgs = [img.to(device) for img in imgs]
#         tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

#         optimizer.zero_grad()

#         # Mixed precision forward
#         with autocast('cuda', enabled=is_cuda):
#             loss_dict = model(imgs, tgts)
#             loss = sum(loss_dict.values())

#         loss_val = loss.item()
#         # Debug poszczególnych składników (odkomentuj jeśli potrzebujesz)
#         # print({k: v.item() for k, v in loss_dict.items()})

#         # Skip extreme/nan losses
#         if torch.isnan(loss) or loss_val > 2000:
#             print(f"[WARN] Skipping batch {i}, loss={loss_val:.4f}")
#             continue

#         # Backward w/ gradient scaling
#         scaler.scale(loss).backward()
#         scaler.unscale_(optimizer)
#         clip_grad_norm_(model.parameters(), max_norm=1.0)
#         scaler.step(optimizer)
#         scaler.update()

#         running_loss += loss_val
#         valid_batches += 1

#         if i % print_freq == 0:
#             current_lr = optimizer.param_groups[0]['lr']
#             print(f"[{epoch+1}][{i}/{iters}] loss={loss_val:.4f}, lr={current_lr:.6f}")

#         # TensorBoard logging
#         if tensorboard is not None:
#             tensorboard.add_scalar('loss/total', loss_val, step)
#             tensorboard.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step)

#     avg_loss = running_loss / max(1, valid_batches)
#     return avg_loss

def _train_one_epoch(model, optimizer, loader, device, epoch, lr, warmup_iters, print_freq=20, tensorboard=None):
    model.train()
    running_loss = 0.0
    for i, (imgs, tgts) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
        imgs = [img.to(device) for img in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

        loss_dict = model(imgs, tgts)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % print_freq == 0:
            print(f"[{epoch+1}][{i}/{len(loader)}] loss={loss.item():.4f}")
    return running_loss / len(loader)

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
          model_type: str = "faster_rcnn",
          epochs: int = 50,
          lr: float = 5e-5,
          wd: float = 5e-4,
          outdir: str = "checkpoints",
          patience: int = 10,
          eval_interval: int = 5):
    """
    Trains object detection model, log loss and COCO mAP,
    saves best/last/periodic checkpoints, uses ReduceLROnPlateau,
    stops early when no mAP improvement.
    """
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Model, dataloaders
    model = initialize_model(model_type, len(CLASSES)).to(device)
    train_dl, val_dl = get_dataloaders(cfg)

    # Optimizer + scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)

    # Warmup schedule
    warmup_epochs = 2
    warmup_iters = warmup_epochs * len(train_dl)
    base_lr = lr

    # TensorBoard
    tb = SummaryWriter(os.path.join(outdir, "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    history = {
        'train_loss': [], 'val_loss': []
    }
    # prepare keys from evaluate_model
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
        # Training step
        train_loss = _train_one_epoch(model=model, optimizer=optimizer, loader=train_dl, device=device, epoch=ep, lr=base_lr, warmup_iters=warmup_iters, print_freq=20, tensorboard=tb)
        history['train_loss'].append(train_loss)

        # Simple validation loss
        val_loss = _evaluate(model, val_dl, device)
        history['val_loss'].append(val_loss)

        # Periodic full COCO evaluation
        if (ep + 1) % eval_interval == 0 or ep == epochs - 1:
            eval_results = evaluate_model(model, val_dl, device)
            # Log metrics
            for k in metric_keys:
                val = eval_results.get(k)
                history[k].append(val)
                tb.add_scalar(k, val, ep)
            # Inference time
            inf_ms = eval_results['avg_inference_time'] * 1000.0
            tb.add_scalar('InferenceTime_ms', inf_ms, ep)

            # Loss curves
            tb.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, ep)

            print(f"--- Eval @ epoch {ep+1} ---")
            print_evaluation_results(eval_results)

            # LR scheduler on plateau
            scheduler.step(eval_results['mAP_coco'])
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[LR Scheduler] New learning rate: {current_lr:.6f}")
            tb.add_scalar('learning_rate', current_lr, ep)

            # Checkpointing
            save_model(model, os.path.join(outdir, f"{model_type}_last.pth"))
            if eval_results['mAP_coco'] > best_map:
                best_map = eval_results['mAP_coco']
                no_improve = 0
                save_model(model, os.path.join(outdir, f"{model_type}_best.pth"))
                print(f"[INFO] New best mAP: {best_map:.4f}")
            else:
                no_improve += 1

            if (ep + 1) % eval_interval == 0:
                save_model(model, os.path.join(outdir, f"{model_type}_e{ep+1}.pth"))

            # early stopping
            if no_improve >= patience:
                print(f"[INFO] Early stopping after {patience} evals with no improvement.")
                break

        # Print training progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {ep+1}/{epochs} - time: {epoch_time:.1f}s | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

    # Save history
    with open(os.path.join(outdir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['faster_rcnn','mask_rcnn'], default='faster_rcnn')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--cache', action='store_true')
    args = parser.parse_args()

    cfg = DatasetConfig(cache=args.cache)
    train(cfg, model_type=args.model, epochs=args.epochs,
          patience=args.patience, eval_interval=args.eval_interval)
