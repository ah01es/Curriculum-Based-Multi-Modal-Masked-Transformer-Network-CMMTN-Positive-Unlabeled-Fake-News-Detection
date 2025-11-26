import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

class WeiboDataset(Dataset):
    def __init__(self, csv_path, data_root, bert_name="bert-base-chinese",
                 max_len=128, is_train=True):
        self.df = pd.read_csv(csv_path, header=None, names=["img","text","label"])
        self.data_root = data_root
        self.is_train = is_train
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name, use_fast=True)
        self.max_len = max_len

        if is_train:
            self.tf = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, str(row["img"]))
        text = str(row["text"])
        label = int(row["label"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224,224), (0,0,0))
        img_t = self.tf(img)

        toks = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {
            "input_ids": toks["input_ids"].squeeze(0),
            "attention_mask": toks["attention_mask"].squeeze(0),
            "image": img_t,
            "label": torch.tensor(label, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long)
        }
        return item
