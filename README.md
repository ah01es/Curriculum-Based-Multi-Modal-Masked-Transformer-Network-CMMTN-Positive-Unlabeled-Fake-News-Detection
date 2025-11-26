# 📘 Curriculum-Based Multi-Modal Masked Transformer Network (CMMTN)

This repository contains **my complete and independent implementation**
of the *Curriculum-Based Multi-Modal Masked Transformer Network (CMMTN)*
for **positive--unlabeled fake news detection**.

The method integrates:

-   **BERT** for textual embeddings\
-   **ResNet50** for visual feature extraction\
-   **A multi-modal masked transformer**\
-   **Curriculum-based Positive--Unlabeled learning**

to detect fake news using weak supervision in a multi-modal setting.

------------------------------------------------------------------------

## 📌 Dataset --- Weibo Fake News

Download the dataset from the link below:

👉 **[Download
Weibo_Dataset.rar](https://github.com/ah01es/Curriculum-Based-Multi-Modal-Masked-Transformer-Network-CMMTN-Positive-Unlabeled-Fake-News-Detection/releases/download/v1.0/Weibo_Dataset.rar)**

After extracting, you will get the following two folders:

    nonrumor_images/
    rumor_images/

### 📂 Move both folders into:

    ./data/weibo_dataset/

### ✔ Directory Structure

    data/
    └── weibo_dataset/
        ├── nonrumor_images/
        └── rumor_images/

------------------------------------------------------------------------

## ▶️ Run the Project

Simply run:

``` bash
python main.py
```

This will start loading the data, building the multi-modal model (BERT +
ResNet + masked transformer), and training with curriculum-based PU
learning.

------------------------------------------------------------------------

## 📧 Contact

If you have any questions or issues, feel free to open an issue or
contact me.
