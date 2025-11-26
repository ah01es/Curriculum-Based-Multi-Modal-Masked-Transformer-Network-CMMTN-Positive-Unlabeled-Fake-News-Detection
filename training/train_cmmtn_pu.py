import os, csv
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score

from cmmtn_pu_weibo.datasets import WeiboDataset
from cmmtn_pu_weibo.model import CMMTN_PU
from cmmtn_pu_weibo.pu_loss import nnpu_risk
from cmmtn_pu_weibo.curriculum import select_confident
from cmmtn_pu_weibo.utils import set_seed, to_device, AverageMeter, count_trainable_params

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--bert", type=str, default="bert-base-chinese")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--r", type=float, default=0.02)
    ap.add_argument("--trusted-step", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--freeze-cnn", action="store_true")
    ap.add_argument("--no-ft-bert", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    return ap.parse_args()

def split_pu(df_train, r, seed=42):
    rng = np.random.RandomState(seed)
    pos_idx = df_train.index[df_train["label"]==1].to_numpy()
    neg_idx = df_train.index[df_train["label"]==0].to_numpy()
    n_pos = len(pos_idx)
    n_labeled = max(1, int(round(r * n_pos)))
    labeled_pos = rng.choice(pos_idx, size=n_labeled, replace=False)
    rem_pos = np.setdiff1d(pos_idx, labeled_pos, assume_unique=False)
    unlabeled = np.concatenate([rem_pos, neg_idx])
    prior = float(n_pos / len(df_train))
    return labeled_pos.tolist(), unlabeled.tolist(), prior

def evaluate(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"], batch["image"])
            logits_all.append(logits.cpu())
            y_all.append(batch["label"].cpu())
    logits_all = torch.cat(logits_all)
    y_all = torch.cat(y_all).numpy()
    probs = torch.sigmoid(logits_all).numpy()
    y_pred = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_all, y_pred)
    macroF1 = f1_score(y_all, y_pred, average="macro", zero_division=0)
    weightedF1 = f1_score(y_all, y_pred, average="weighted", zero_division=0)
    return acc, macroF1, weightedF1

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = os.path.join(args.data_root, "train.csv")
    test_csv  = os.path.join(args.data_root, "test.csv")
    df_train = pd.read_csv(train_csv, header=None, names=["img","text","label"])
    df_test  = pd.read_csv(test_csv,  header=None, names=["img","text","label"])

    labeled_pos, unlabeled, prior = split_pu(df_train, r=args.r, seed=args.seed)
    print(f"[PU] prior≈{prior:.4f}, P={len(labeled_pos)} of {int((df_train['label']==1).sum())}, U={len(unlabeled)} (rest pos + all neg)")

    ds_train = WeiboDataset(train_csv, data_root=args.data_root, bert_name=args.bert, max_len=args.max_len, is_train=True)
    ds_test  = WeiboDataset(test_csv,  data_root=args.data_root, bert_name=args.bert, max_len=args.max_len, is_train=False)

    trusted_pos, trusted_neg = set(), set()

    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = CMMTN_PU(bert_name=args.bert, freeze_cnn=args.freeze_cnn, ft_bert=(not args.no_ft_bert))
    model = model.to(device)
    print("[DEBUG] trainable params:", count_trainable_params(model))

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch","trusted","acc","macroF1","weightedF1","loss_total","loss_ce","loss_pu"])

    best_macro = -1.0

    for epoch in range(1, args.epochs+1):
        model.train()

        # Build dynamic subsets
        idx_u = [i for i in unlabeled if i not in trusted_pos and i not in trusted_neg]
        idx_trusted = list(trusted_pos) + list(trusted_neg)

        loader_p = DataLoader(Subset(ds_train, labeled_pos), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        loader_u = DataLoader(Subset(ds_train, idx_u), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        loader_ce = DataLoader(Subset(ds_train, idx_trusted), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True) if len(idx_trusted)>0 else None

        loss_meter = AverageMeter()
        loss_ce_meter = AverageMeter()
        loss_pu_meter = AverageMeter()

        iters = max(len(loader_p), 1)
        ce_iter = iter(loader_ce) if loader_ce is not None else None
        p_iter = iter(loader_p)
        u_iter = iter(loader_u) if len(loader_u)>0 else None

        # create tqdm progress bar per-epoch (unit=batch)
        bar_len = 40
        bar_format = '{l_bar}{bar:'+str(bar_len)+'}{r_bar}'   # مثال: '{l_bar}{bar:40}{r_bar}'
        pbar = tqdm(total=iters, desc=f"Epoch {epoch:03d}", unit="batch",
                    ncols=None, dynamic_ncols=False, bar_format=bar_format, leave=True)


        for it in range(iters):
            ce_loss = 0.0
            if ce_iter is not None:
                try:
                    batch_ce = next(ce_iter)
                except StopIteration:
                    ce_iter = iter(loader_ce)
                    batch_ce = next(ce_iter)
                batch_ce = to_device(batch_ce, device)
                logits_ce = model(batch_ce["input_ids"], batch_ce["attention_mask"], batch_ce["image"])
                y_ce = batch_ce["label"].float()
                ce_loss = bce(logits_ce, y_ce)

            try:
                batch_p = next(p_iter)
            except StopIteration:
                p_iter = iter(loader_p)
                batch_p = next(p_iter)

            if u_iter is None:
                # no unlabeled data left -> stop this epoch early
                pbar.update(0)  # no-op but keeps intent clear
                break
            try:
                batch_u = next(u_iter)
            except StopIteration:
                u_iter = iter(loader_u)
                batch_u = next(u_iter)

            batch_p = to_device(batch_p, device)
            batch_u = to_device(batch_u, device)
            logits_p = model(batch_p["input_ids"], batch_p["attention_mask"], batch_p["image"])
            logits_u = model(batch_u["input_ids"], batch_u["attention_mask"], batch_u["image"])

            pu_loss = nnpu_risk(logits_p, logits_u, prior=prior, reduction="mean")
            loss = pu_loss + (ce_loss if isinstance(ce_loss, torch.Tensor) else 0.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_meter.update(loss.item())
            loss_pu_meter.update(pu_loss.item())
            loss_ce_meter.update(float(ce_loss.item()) if isinstance(ce_loss, torch.Tensor) else 0.0)

            # update progress bar: یک واحد جلو ببر و اطلاعات اضافی نمایش بده
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "loss_pu": f"{loss_pu_meter.avg:.4f}",
                "loss_ce": f"{loss_ce_meter.avg:.4f}",
                "trusted": len(trusted_pos)+len(trusted_neg)
            }, refresh=True)

        # make sure to close the progress bar for this epoch
        pbar.close()

        # curriculum selection
        model.eval()
        if len(idx_u) > 0:
            all_logits = []
            with torch.no_grad():
                u_eval_loader = DataLoader(Subset(ds_train, idx_u), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                for batch in u_eval_loader:
                    batch = to_device(batch, device)
                    lg = model(batch["input_ids"], batch["attention_mask"], batch["image"])
                    all_logits.append(lg.cpu())
            all_logits = torch.cat(all_logits, dim=0)
            import torch as _torch
            sel_pos, sel_neg = select_confident(all_logits, _torch.tensor(idx_u), args.trusted_step, args.trusted_step)
            trusted_pos.update(sel_pos)
            trusted_neg.update(sel_neg)

        acc, macroF1, weightedF1 = evaluate(model, test_loader, device)

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, len(trusted_pos)+len(trusted_neg),
                        f"{acc:.4f}", f"{macroF1:.4f}", f"{weightedF1:.4f}",
                        f"{loss_meter.avg:.4f}", f"{loss_ce_meter.avg:.4f}", f"{loss_pu_meter.avg:.4f}"])
        print(f"[{epoch:03d}] trusted={len(trusted_pos)+len(trusted_neg)} acc={acc:.4f} macroF1={macroF1:.4f} weightedF1={weightedF1:.4f} loss={loss_meter.avg:.4f}")

        if macroF1 > best_macro:
            best_macro = macroF1
            ckpt = {"epoch": epoch, "state_dict": model.state_dict(), "args": vars(args), "prior": prior}
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))

    torch.save({"state_dict": model.state_dict(), "args": vars(args), "prior": prior}, os.path.join(args.output_dir, "last.pt"))

if __name__ == "__main__":
    main()
