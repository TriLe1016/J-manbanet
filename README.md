# J-MambaNet: Integrating Bidirectional State Space Models with Adaptive Fusion for Skin Lesion Classification

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper **"J-MambaNet: Integrating Bidirectional State Space Models with Adaptive Fusion for Skin Lesion Classification"**.

## 📌 Overview
J-MambaNet is a novel multimodal architecture designed to bridge the gap between high-performance computing and clinical workflows in dermatological diagnosis. It effectively integrates dermoscopic imagery with patient clinical metadata (age, sex, location) to resolve the "texture-context trade-off."

### Key Contributions:
1. **Resolving the Texture-Context Trade-off:** A hybrid architecture combining a DenseNet Stem (to capture high-frequency visual cues like texture and borders) with Bidirectional Mamba blocks (to assess global lesion asymmetry).
2. **Optimized Metadata Encoder with Adaptive FiLM:** A multimodal modulation mechanism allowing clinical metadata to dynamically impact the image feature extraction process with $O(N)$ complexity.
3. **Anatomically-Aware Regularization:** A novel Joint-Training V-JEPA framework with Semantic Block Masking to prevent overfitting, particularly for underrepresented classes.
4. **Adaptive Clinical Workflow Control:** A dual-track inference strategy providing a lightweight Single Model (178 FPS) for real-time screening and a high-precision Ensemble track (35 FPS) for in-depth reference diagnosis.

## 🏆 Main Results
J-MambaNet achieves state-of-the-art performance on the HAM10000 dataset and demonstrates strong cross-dataset generalization on PAD-UFES-20 and ISIC 2019.

| Dataset | Model Mode | Accuracy (%) | AUC | Weighted F1 (%) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HAM10000** | Single Model | $95.67 \pm 0.28$ | $0.969 \pm 0.003$ | $96.01 \pm 0.31$ | **178** |
| **HAM10000** | Ensemble (5-Fold + TTA) | **$96.43 \pm 0.21$** | **$0.971 \pm 0.002$** | **$96.47 \pm 0.24$** | 35 |
| **PAD-UFES-20** | Zero-shot Generalization | $89.45 \pm 0.52$ | $0.931 \pm 0.005$ | $88.67 \pm 0.55$ | - |
| **ISIC 2019** (Disjoint)| Zero-shot Generalization | $87.85 \pm 0.48$ | $0.928 \pm 0.004$ | $87.15 \pm 0.50$ | - |

## 📁 Project Structure

```text
J-manbanet/
├── scripts/
│   ├── train.py                # Main script for model training (supports V-JEPA)
│   ├── run_ensemble_TTA.py     # Inference using 5-fold ensemble with Weighted TTA
│   └── print_report.py         # Generate evaluation metrics and confusion matrices
├── src/
│   ├── config.py               # Hyperparameters and path configurations
│   ├── dataset.py              # Multimodal Dataset class for HAM10000/ISIC/PAD-UFES-20
│   ├── augmentations.py        # Semantic Block Masking and standard augmentations
│   ├── model.py                # J-MambaNet architecture (DenseNet Stem, Bi-Mamba, Adaptive FiLM)
│   ├── engine.py               # Training and validation loops with EMA updates
│   └── utils.py                # Helper functions
⚙️ Installation
Clone the repository:

Bash
git clone [https://github.com/TriLe1016/J-manbanet.git](https://github.com/TriLe1016/J-manbanet.git)
cd J-manbanet
Install dependencies:

Bash
# It is recommended to use a conda environment
conda create -n jmambanet python=3.9 -y
conda activate jmambanet

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install mamba-ssm and other requirements
pip install mamba-ssm==1.1.0
pip install -r requirements.txt 
(Note: Please create a requirements.txt file in your repository containing other necessary libraries like scikit-learn, pandas, matplotlib, etc.)

🚀 Usage
1. Data Preparation
Download the HAM10000 dataset from the official source.

Organize the data and metadata (CSV) according to the paths specified in src/config.py.

2. Training the Model
To train the J-MambaNet model from scratch using the Joint-Training V-JEPA strategy:

Bash
python scripts/train.py
3. Inference & Evaluation (Ensemble + TTA)
To reproduce the best results (96.43% Accuracy) using the 5-fold ensemble and weighted Test-Time Augmentation (TTA):

Bash
python scripts/run_ensemble_TTA.py
4. Generate Reports
To calculate detailed metrics (Precision, Recall, F1-Score per class) and visualize the confusion matrix:

Bash
python scripts/print_report.py
📜 Citation
If you find this code or research helpful, please consider citing:

Đoạn mã
@article{jmambanet2026,
  title={J-MambaNet: Integrating Bidirectional State Space Models with Adaptive Fusion for Skin Lesion Classification},
  author={Anonymous Authors},
  journal={Anonymous Journal},
  year={2026}
}
