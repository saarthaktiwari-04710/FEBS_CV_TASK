# CIFAR-10 Classification with ResNet-18 in PyTorch

This repository contains an implementation of the **ResNet-18** architecture built from scratch using PyTorch and trained on the **CIFAR-10** dataset. The model utilizes standard residual blocks with skip connections, batch normalization, and an aggressive optimization schedule to achieve strong classification performance.

---

## 🚀 How to Run the Code

### 1. Clone the Repository:
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
```

### 2. Set-up a Virtual Environment:

```bash
python -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows (Command Prompt):
.\venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```
### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Execute training and evaluation:
```bash
python train.py
```


## Chosen Hyperparameters & Paper Justification

* **Architecture:** **ResNet-18** — Standard 18-layer residual network with an adjusted initial 3×3 conv layer tailored for 32×32 inputs.

---

* **Optimizer:** **SGD** — Stochastic Gradient Descent with a momentum factor of `0.9`.

---

* **Learning Rate (LR):** **0.1** — Initial base learning rate, multiplied by a decay factor of `0.1` at key iteration milestones.

---

* **LR Scheduler:** **`MultiStepLR`** — Drop milestones calculated explicitly at **Epoch 82** and **Epoch 123** (see mathematical derivation below).

---

* **Weight Decay ($L_2$):** **5e-4** — Regularization factor optimized for standard PyTorch CIFAR-10 training to prevent overfitting.

---

* **Batch Size:** **128** — Distributed mini-batch sizing mapped onto the PyTorch data loader.

---

* **Total Epochs:** **100** — Total training iterations converted precisely to full passes over the CIFAR-10 dataset.

---

* **Regularization:** **Batch Normalization** — Applied immediately after every convolutional layer; zero Dropout layers were utilized.

---

* **Data Augmentation:** **Padding & Cropping** — 4-pixel zero-padding on all sides followed by a random 32×32 crop and random horizontal flips.



## 🏆 Final Performance Metrics

The training run yields the following performance trajectory across the 100-epoch lifecycle:

| Milestone | Training Loss | Training Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **Epoch 1** | 0.8429 | 70.37% | 68.98% |
| **Epoch 10** | 0.43333 | 85.18% | 78.78% |
| **Epoch 30** | 0.3015 | 89.68% | 84.90% |
| **Epoch 50** | 0.1987 | 93.15% | 84.95% |
| **Epoch 70** | 0.0722 | 97.64% | 92.15% |
| **Epoch 90** | 0.0030 | 99.96% | 94.96% |
| **Epoch 100** | 0.0025 | 99.98% | 95.04% |
