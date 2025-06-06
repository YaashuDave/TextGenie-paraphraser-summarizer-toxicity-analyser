# TextGenie-paraphraser-summarizer-toxicity-analyser
TextGenie offers a seamless, user-friendly  platform for paraphrasing, summarization, and toxicity detection. With the ability to process and  transform text effectively, TextGenie brings together state-of-the-art NLP models and robust architectures to provide a comprehensive toolkit for text manipulation and analysis. 

# Toxicity Detection using BERT + Kim CNN

> **Note**: This work was carried out as part of a team project.  
> I (Yaashu Dave) performed all tasks related to toxicity detection using the BERT + Kim CNN model, including data processing, modeling, training, and evaluation.  
> The other team members contributed to the summarizer and paraphraser modules of the project.

---

## 📁 Datasets Used

Subsets of the following datasets were used due to GPU memory limitations:

| Dataset                         | Source         | Train Size | Test Size | Validation Size |
|---------------------------------|----------------|------------|-----------|-----------------|
| Toxic Comment Classification    | Kaggle         | 160K       | 64K       | —               |
| Civil Comments Dataset          | Hugging Face   | 450K       | 97K       | 97K             |
| Jigsaw Toxicity Dataset         | Hugging Face   | 26K        | 8K        | 6.5K            |

> ⚠️ Note: Subsets were sampled from all datasets to ensure compatibility with Colab GPU resources.

---

## Model Architecture: BERT + Kim CNN

### 🔹 BERT
- Pretrained transformer model for generating contextual embeddings.

### 🔹 Kim CNN
- CNN with multiple kernel sizes for capturing n-gram features.

**Key Parameters:**

| Parameter         | Value              |
|------------------|--------------------|
| Kernel Sizes     | [2, 3, 4]          |
| Filters per Kernel | 100              |
| Dropout Rate     | 0.1                |
| Optimizer        | Adam               |
| Learning Rate    | 9.625e-6           |
| Loss Function    | Binary Cross-Entropy |
| Epochs           | 30                 |

---

## ⚙️ Other Hyperparameters

| Parameter              | Value        |
|------------------------|--------------|
| Embedding Dimension    | 100 (GloVe)  |
| Max Vocab Size         | 20,000       |
| Max Sequence Length    | 100          |
| Batch Size             | 128          |
| Validation Split       | 0.2          |

---

## 📈 Results (ROC-AUC Scores)

### 🧪 Dataset 1: Toxic Comment Classification (Kaggle)

| Label          | ROC-AUC Score |
|----------------|---------------|
| severe_toxic   | 0.988         |
| obscene        | 0.978         |
| insult         | 0.973         |
| identity_hate  | 0.965         |
| toxic          | 0.956         |
| threat         | 0.594         |

---

### 🧪 Dataset 2: Civil Comments (Hugging Face)

| Label           | ROC-AUC Score |
|-----------------|---------------|
| insult          | 0.931         |
| toxicity        | 0.924         |
| obscene         | 0.860         |
| identity_attack | 0.829         |
| threat          | 0.365         |

---

### 🧪 Dataset 3: Jigsaw Toxicity (Hugging Face)

| Label          | ROC-AUC Score |
|----------------|---------------|
| severe_toxic   | 0.973         |
| threat         | 0.973         |
| obscene        | 0.928         |
| identity_hate  | 0.922         |
| insult         | 0.920         |
| toxic          | 0.905         |

---

## 📊 Observations & Analysis

- The model performs consistently well on major toxicity labels like `toxic`, `insult`, and `obscene`.
- **Threat** class performs poorly across datasets — possibly due to class imbalance or fewer labeled examples.
- ROC-AUC scores from Dataset 1 (Kaggle) are the highest overall, suggesting better label quality or more balanced data.
- Combining **BERT** (semantic understanding) and **Kim CNN** (n-gram pattern detection) proves effective for multi-label text classification.

---

