# /src/model_train.py
import os, time, datetime
import torch, argparse, json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_pipeline import DatasetConfig, get_dataloaders
from dataset_pipeline import CLASSES
from evaluation import evaluate_model, print_evaluation_results
from model_implementation import initialize_model, save_model

def _train_one_epoch(model, optimizer, loader, device, epoch, print_freq=20):
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
          lr: float = 5e-4,
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
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    # TensorBoard
    tb = SummaryWriter(os.path.join(outdir, "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    history = {"train_loss": [], "val_loss": [], "mAP_coco": [], "mAP_50": [], "mAP_75": []}

    best_map = 0.0
    no_improve = 0

    for ep in range(epochs):
        epoch_start = time.time()
        # Training step
        train_loss = _train_one_epoch(model, optimizer, train_dl, device, ep)
        history['train_loss'].append(train_loss)

        # Simple validation loss
        val_loss = _evaluate(model, val_dl, device)
        history['val_loss'].append(val_loss)

        # Periodic full COCO evaluation
        if (ep + 1) % eval_interval == 0 or ep == epochs - 1:
            eval_results = evaluate_model(model, val_dl, device)
            mAP_coco = eval_results['mAP_coco']
            mAP_50   = eval_results['mAP_50']
            mAP_75   = eval_results['mAP_75']
            inf_ms   = eval_results['avg_inference_time'] * 1000.0

            history['mAP_coco'].append(mAP_coco)
            history['mAP_50'].append(mAP_50)
            history['mAP_75'].append(mAP_75)

            tb.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, ep)
            tb.add_scalar('mAP_coco', mAP_coco, ep)
            tb.add_scalar('mAP_50', mAP_50, ep)
            tb.add_scalar('mAP_75', mAP_75, ep)
            tb.add_scalar('Inference_time_ms', inf_ms, ep)

            print(f"--- Eval @ epoch {ep+1} ---")
            print_evaluation_results(eval_results)

            # LR scheduler on plateau
            scheduler.step(mAP_coco)

            # Checkpointing
            save_model(model, os.path.join(outdir, f"{model_type}_last.pth"))
            if mAP_coco > best_map:
                best_map = mAP_coco
                no_improve = 0
                save_model(model, os.path.join(outdir, f"{model_type}_best.pth"))
                print(f"[INFO] New best mAP: {mAP_coco:.4f}")
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
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--cache', action='store_true')
    args = parser.parse_args()

    cfg = DatasetConfig(cache=args.cache)
    train(cfg, model_type=args.model, epochs=args.epochs,
          patience=args.patience, eval_interval=args.eval_interval)
