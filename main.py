import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.dirname(current_dir)          
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import importlib

def main():
    data_root = os.path.join(current_dir, "data", "weibo_dataset")
    output_dir = os.path.join(current_dir, "runs", "weibo_cmmtn_pu_r02")

    sys.argv = [
        "python",
        "--data-root", data_root,
        "--output-dir", output_dir,
        "--bert", "bert-base-chinese",
        "--epochs", "30",
        "--batch-size", "16",
        "--r", "0.02",
        "--lr", "1e-4",
        "--trusted-step", "30",
        "--seed", "42"
    ]

    train_module = importlib.import_module("cmmtn_pu_weibo.training.train_cmmtn_pu")
    train_module.main()

if __name__ == "__main__":
    main()
