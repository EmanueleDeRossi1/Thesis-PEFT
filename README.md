# Thesis-PEFT: Parameter-Efficient Domain Adaptation for Entity Matching

This repository is part of my master's thesis on Entity Matching (EM). In my project, a parameter-efficient approach - Low-Rank Adaptation (LoRA) - is used to adapt pre-trained transformers for EM tasks. By updating only a small subset of parameters, this method achieves strong performance on unlabeled target datasets by aligning feature distributions between labeled source and unlabeled target datasets.

---

## Project Overview

- **🔍 Problem:** Entity Matching (EM) requires labeled training data, which is expensive to obtain. This project explores **domain adaptation** to transfer knowledge from labeled source datasets to unlabeled target datasets.
- **💡 Solution:** Uses **LoRA** (Low-Rank Adaptation) for efficient fine-tuning and **MK-MMD** (Multiple Kernel Maximum Mean Discrepancy) for domain alignment.
- **✨ Key Features:**
  - ✅ Efficient fine-tuning with **LoRA** (only 1% additional parameters).
  - ✅ **Domain adaptation** to align source and target feature distributions.
  - ✅ **Task learning and domain adaptation trade-off**, controlled by a dynamic **α** function.

---

## 📂 Project Structure
Thesis-PEFT/
│── config/                # YAML config files for training & hyperparameter tuning
│── dataset/               # Data processing and dataloaders
│── src/
│   ├── models/            # LoRA and Fine-Tuning models
│   ├── divergences/       # Domain adaptation losses (MK-MMD, Gaussian Kernels)
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation metrics and scripts
│── scripts/
│   ├── train.sh           # Training script
│   ├── run_experiments.sh # Batch execution for different datasets
│── Dockerfile             # Docker setup for containerized execution
│── requirements.txt       # Required Python dependencies
│── setup.sh               # Setup script for dependencies (CUDA support included)
│── README.md              # Project documentation
The script:
1. Loops through source and target dataset experimented in the paper, using 3 random seeds 
4. Use the `--hparam_tuning` flag to enable hyperparameter tuning.

---

## 🏗️ Setup & Installation

### **1️⃣ Using Docker (Recommended)**
This project is containerized using **Docker** for consistency and reproducibility.  
To set up and run the model inside a Docker container:

#### **🔹 Build the Docker Image**
```bash
docker build -t thesis-peft .
```

#### **🔹 Run Training Inside a Container**
```docker run --gpus all --rm -v $(pwd):/app thesis-peft bash train.sh --src <source_dataset> --tgt <target_dataset> --model lora```

#### **🔹 Run Hyperparameter Tuning**

```docker run --gpus all --rm -v $(pwd):/app thesis-peft bash train.sh --src <source_dataset> --tgt <target_dataset> --model lora --hparam-tuning```

