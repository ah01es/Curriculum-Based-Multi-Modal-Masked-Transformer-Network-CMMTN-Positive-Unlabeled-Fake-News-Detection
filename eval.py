

import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

from cmmtn_pu_weibo.datasets import WeiboDataset
from cmmtn_pu_weibo.model import CMMTN_PU

def load_model(run_dir):
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")
    args = ck["args"]
    model = CMMTN_PU(bert_name=args["bert"], freeze_cnn=args["freeze_cnn"], ft_bert=(not args["no_ft_bert"]))
    model.load_state_dict(ck["state_dict"], strict=False)
    return model, ck

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    model, ck = load_model(args.run_dir)
    data_root = ck["args"]["data_root"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_csv = os.path.join(data_root, "test.csv")
    ds_test = WeiboDataset(test_csv, data_root=data_root, bert_name=ck["args"]["bert"], is_train=False)
    loader = DataLoader(ds_test, batch_size=ck["args"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    all_logits, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                if hasattr(batch[k], "to"):
                    batch[k] = batch[k].to(device)
            logits = model(batch["input_ids"], batch["attention_mask"], batch["image"])
            all_logits.append(logits.cpu())
            all_y.append(batch["label"].cpu())
    logits = torch.cat(all_logits).numpy()
    y = torch.cat(all_y).numpy()
    probs = 1/(1+np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix - Test")
    plt.colorbar()
    ticks = [0,1]
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    fig.savefig(os.path.join(args.run_dir, "confusion_matrix_test.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ROC & PR
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)

    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC - Test")
    plt.legend()
    fig.savefig(os.path.join(args.run_dir, "roc_pr_test.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR - Test")
    plt.legend()
    fig.savefig(os.path.join(args.run_dir, "pr_test.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
