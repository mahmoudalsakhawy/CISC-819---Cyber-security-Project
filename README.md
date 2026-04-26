# Reproducing Results — Adversarial Attacks and Defenses on CIFAR-10

In this guide we will setup the environment and run the notebook end-to-end, reproduce all experiments, plots, andSaved Models:

---

## 1. Prerequisites

### Hardware
- **GPU (highly recommended). Each full run (3 seeds 3 models 100 epochs) takes several hours on a modern GPU (e.g. RTX 3090), and is prohibitively slow on CPU alone.
- Minimum 8GB of VRAM when using batch_size 128

### Software
- Python 3.8+
- CUDA 11.3+ (only needed for GPU execution)

---

## 2. Environment Setup

### Option A: pip (use this if you don't use conda)

```bash
 python -m venv advmlenv
 source advmlenv/bin/activate # Windows use: advmlenv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib seaborn pandas scikit-learn pillow tqdm jupyter
```
Option B: conda (use this if you already have conda installed)

```bash
conda create -n adv_ml python=3.10
conda activate adv_ml

conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy matplotlib seaborn pandas scikit-learn pillow tqdm jupyter
```

### Verify GPU is available

```python
import torch
print(torch.cuda.is_available())           # should print True
print(torch.cuda.get_device_name(0))       # e.g. NVIDIA GeForce RTX 3090
```

---

## 3. Running the Notebook

### Launch the notebook:

```bash
jupyter notebook adversarial_ml_project.ipynb
```

### Run all cells in order

Go to Kernel Restart & Run All, and run the notebook from scratch. Note: The notebook is entirely self-contained-the CIFAR-10 dataset is automatically downloaded when the notebook first runs (170MB):

### Expected cell-by-cell output

| Section | What happens |
|---------|-------------|
| **1 — Setup** | Libraries are imported, the device is printed, and the seeds are confirmed. |
| **2 — Dataset** | The CIFAR-10 dataset is downloaded andsaved to ./data/. A sample dataset is plotted in dataset_samples.png. |
| **3 — Model & Utilities** | The baseline model is trained for 100 epochs. Intermediate losses and accuracies per epoch are printed in the console. A saved model checkpoint is created at baseline_model.pth. |
| **4 — Baseline Training** | 100-epoch training loop with per-epoch logs; `baseline_model.pth` saved |
| **5 — Attack Evaluation** |  The performance of the baseline model is evaluated against an attack at 10 values of epsilon, and the accuracy is plotted in accuracyvsepsilon_baseline.png. 5 plots are saved with this. |
| **6 — Defense Training** | FGSM-AT and PGD-AT models are trained, and their corresponding saved model checkpoints are created at fgsmatmodel.pth and pgdatmodel.pth respectively.|
| **7 — Results** | The three models are evaluated over three seeds with the attack and accuracy, plotted in 6 plots and saved in two CSV files: resultsraw.csv and resultssummary.csv. |
| **8 — Analysis** | Markdown cells for analyzing the results; these contain no executable code. |

---

## 4. Key Configuration Knobs

The entire hyperparameter configuration is contained within the CFG dictionary in section 1. Some key values you might consider changing before running the notebook:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `seeds` | `[42, 123, 7]` | The total number of epochs the models will be trained for. |
| `epochs` | `100` | The total number of epochs the models will be trained for.|
| `eps_train` | `0.03` | The size of adversarial perturbation used during training.|
| `epsilons` | `[0.0 … 0.30]` |  Epsilon values that are tested during attack evaluation. |
| `batch_size` | `128` | The number of samples that are processed in each training/evaluation batch. (Reduce if you get an out-of-memory error).|
| `num_workers` | `2` | Number of worker threads to load data. (Set this to 0 for Windows if you encounter multi-processing issues). |
| `trades_beta` | `6.0` |  The weighting factor for the TRADES regularization term.|
| `pgd_adv_ratio` | `0.75` | The fraction of adversarial samples in each batch for PGD-AT.|
| `epoch_sweep` | `[10, 50, 100]` | Epoch values to consider for optional training budget analysis. |

**Quick smoke-test run** — To verify everything is running before committing to a full run, you may set:
```python
CFG["epochs"]  = 5
CFG["seeds"]   = [42]
CFG["epsilons"] = [0.0, 0.03, 0.10]
```
This should complete in around 10 minutes on a GPU, generating all plots and with lower quality numbers than a full run.
---

## 5. Expected Outputs

The following files are created in the working directory after a full run:
### Model checkpoints
```
baseline_model.pth
fgsm_at_model.pth
pgd_at_model.pth
```

### Results data
```
results_raw.csv        # per-seed accuracy for every model × attack × epsilon
results_summary.csv    # mean ± std aggregated across seeds
```

### Plots
```
dataset_samples.png
baseline_training_curves.png
accuracy_vs_epsilon_baseline.png
attack_strength_ladder.png
clean_vs_adversarial.png
failure_cases_pgd20.png
confusion_matrix.png
training_curves_comparison.png
accuracy_vs_epsilon_all_models.png
grouped_bar_accuracy.png
robustness_heatmaps.png
relative_improvement.png
seed_variance_violin.png
epoch_sweep.png
```

The last cell prints a manifest that lists all created files and their sizes, so you know everything worked:
```
✓ baseline_model.pth                          44.7 MB
✓ results_raw.csv                              0.2 MB
...
```

---

## 6. Expected Accuracy Numbers

You should expect the following numbers after a 3-seed run of the notebook:

| Model | Clean | FGSM ε=0.03 | PGD-5 ε=0.03 | PGD-20 ε=0.03 |
|-------|-------|-------------|--------------|----------------|
| Baseline | ~92% | ~56% | ~47% | ~40% |
| FGSM-AT | ~89% | ~73% | ~67% | ~60% |
| PGD-AT (TRADES) | ~87% | ~77% | ~74% | ~71% |

There are two automated checks in the notebook:

```
1. Ordering check at epsilon=0.03: We check FGSM > PGD-5 > PGD-20.
2. PGD-AT vs FGSM-AT gap under PGD-20: We calculate the increase in accuracy as a %.
```

If any of these fail, it is likely that alpha for PGD was hardcoded and should be computed adaptively from epsilon using the following relation inside pgd_attack(): alpha = epsilon * 2.5 / steps.

---

## 7. Troubleshooting

**CUDA out of memory**
Reduce the value of CFG["batch_size"] to 64 or 32.

**`multiprocessing` errors on Windows**
Try setting CFG["num_workers"] to 0.

**CIFAR-10 download fails**
Manually download the CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html, unpack it, and place the entire contents in a ./data/cifar-10-batches-py directory. Then set the download parameter to False in the relevant dataset cells.


**NaN loss during PGD-AT**
The loss calculation in adversarialtraining() is guarded by torch.isfinite(loss) to skip corrupted batches. If NaNs persist, consider reducing the CFG["tradesbeta"] value or increasing the warmup period (currently ramps 0-6 over the first 10 epochs).

**Attack ordering assertion fails (`FGSM ≥ PGD-5 ≥ PGD-20`)**
Make sure you have not hardcoded the value of alpha; it must be computed dynamically inside pgd_attack() with the equation: alpha = epsilon * 2.5 / steps.

---

## 8. Reproducibility Notes

- We ensured reproducibility of the three models by training them using the same random seeds, which were set with the help of seed_everything() (it sets Python, NumPy, and PyTorch random seeds and enables cudnn.deterministic=True). However, expect minute deviations from the stated numbers due to non-deterministic behavior of CUDA kernels on different hardware and CUDA versions.
- We implemented the Fairness Rule by providing all models with the same number of training epochs (and, thus, the same budget in compute terms) along with identical learning rates, momentum values, and weight decay hyperparameters-no model gets more computational advantage over the other.

