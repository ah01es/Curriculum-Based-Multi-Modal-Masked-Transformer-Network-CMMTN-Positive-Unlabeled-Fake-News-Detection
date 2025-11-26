import os, argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    mpath = os.path.join(args.run_dir, "metrics.csv")
    if not os.path.exists(mpath):
        print("metrics.csv not found in", args.run_dir)
        return
    df = pd.read_csv(mpath)
    fig = plt.figure()
    for col in ["acc","macroF1","weightedF1"]:
        plt.plot(df["epoch"], df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Accuracy / F1 over Epochs")
    plt.legend()
    fig.savefig(os.path.join(args.run_dir, "curves.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig = plt.figure()
    for col in ["loss_total","loss_ce","loss_pu"]:
        plt.plot(df["epoch"], df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Losses over Epochs")
    plt.legend()
    fig.savefig(os.path.join(args.run_dir, "losses.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(df["epoch"], df["trusted"])
    plt.xlabel("Epoch")
    plt.ylabel("|Trusted|")
    plt.title("Trusted Set Growth")
    fig.savefig(os.path.join(args.run_dir, "trusted_growth.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
