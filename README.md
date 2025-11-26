# Curriculum-Based Multi-Modal Masked Transformer Network (CMMTN)

This repository contains my complete and independent implementation of the **Curriculum-Based Multi-Modal Masked Transformer Network (CMMTN)** for **positive–unlabeled fake news detection**.  
The model integrates **BERT (text)**, **ResNet50 (image)**, a **multi-modal masked transformer**, and **curriculum-based PU learning** to effectively detect fake news using weak supervision.

---

## 📌 Dataset (Weibo Fake News)

Download the dataset from the link below:

👉 **[Download Weibo_Dataset.rar](https://github.com/ah01es/Curriculum-Based-Multi-Modal-Masked-Transformer-Network-CMMTN-Positive-Unlabeled-Fake-News-Detection/releases/download/v1.0/Weibo_Dataset.rar)**

After extracting the RAR file, you will get two folders:

nonrumor_images/
rumor_images/

sql
Copy code

Copy both folders into:

./data/weibo_dataset/

php
Copy code

Directory structure:

data/
└── weibo_dataset/
├── nonrumor_images/
└── rumor_images/

yaml
Copy code

---

## ▶️ Run the Project

Run the model using:


