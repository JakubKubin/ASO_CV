# /src/model_train.py
import os, time, datetime
import torch, argparse, json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_pipeline import DatasetConfig, get_dataloaders
from dataset_pipeline import CLASSES
from model_implementation import initialize_model, save_model

def _train_one_epoch(model, optimizer, loader, device, epoch, print_freq=20):
    model.train()
    running = 0.0
    for i, (imgs, tgts) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        imgs = [img.to(device) for img in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

        loss_dict = model(imgs, tgts)
        loss      = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()
        if i % print_freq == 0:
            print(f"[{epoch}][{i}/{len(loader)}] loss={loss.item():.4f}")

    return running / len(loader)

# def _evaluate(model, loader, device):
#     model.eval()
#     total = 0.0
#     with torch.no_grad():
#         for imgs, tgts in loader:
#             imgs = [img.to(device) for img in imgs]
#             tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
#             loss = sum(model(imgs, tgts).values())
#             total += loss.item()
#     return total / len(loader)

def _evaluate(model, loader, device):
    was_training = model.training        # zapamiętaj poprzedni stan
    model.train()                        # ← potrzebny tryb treningowy!
    total = 0.0
    with torch.no_grad():               # bez gradientów ≈ eval
        for imgs, tgts in loader:
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            losses = model(imgs, tgts)   # dict
            total += sum(losses.values()).item()
    if not was_training:                 # przywróć stan modelu
        model.eval()
    return total / len(loader)


def train(cfg: DatasetConfig,
          model_type="faster_rcnn",
          epochs=10,
          lr=5e-4,
          wd=5e-4,
          outdir="checkpoints"):

    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    model = initialize_model(model_type, len(CLASSES))
    model.to(device)

    train_dl, val_dl = get_dataloaders(cfg)

    params = [p for p in model.parameters() if p.requires_grad]
    optim  = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    sched  = torch.optim.lr_scheduler.StepLR(optim, step_size=4, gamma=0.1)

    tb = SummaryWriter(os.path.join(outdir, "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    history = {"train": [], "val": []}

    for ep in range(epochs):
        t0 = time.time()
        tr_loss = _train_one_epoch(model, optim, train_dl, device, ep)
        val_loss = _evaluate(model, val_dl, device)
        sched.step()

        history["train"].append(tr_loss)
        history["val"].append(val_loss)
        tb.add_scalars("Loss", {"train": tr_loss, "val": val_loss}, ep)

        print(f"Epoch {ep} done in {time.time()-t0:.1f}s | train {tr_loss:.4f} | val {val_loss:.4f}")

        if (ep+1) % 5 == 0 or ep == epochs-1:
            save_model(model, os.path.join(outdir, f"{model_type}_e{ep+1}.pth"))


    with open(os.path.join(outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["faster_rcnn", "mask_rcnn"], default="faster_rcnn")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    cfg = DatasetConfig(cache=False)
    train(cfg, args.model, epochs=args.epochs)
