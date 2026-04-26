# Reproducing Results — Adversarial Attacks and Defenses on CIFAR-10

This guide explains how to set up your environment and run the notebook end-to-end to reproduce all experiments, plots, and saved models.

---

## 1. Prerequisites

### Hardware
- **GPU strongly recommended.** Each full run (3 seeds × 3 models × 100 epochs) takes several hours on a modern GPU (e.g. RTX 3090) and is prohibitively slow on CPU alone.
- Minimum **8 GB VRAM** for batch size 128.

### Software
- Python 3.8+
- CUDA 11.3+ (if using GPU)

---

## 2. Environment Setup

### Option A — pip (virtual environment)

```bash
python -m venv adv_ml_env
source adv_ml_env/bin/activate          # Windows: adv_ml_env\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib seaborn pandas scikit-learn pillow tqdm jupyter
```

### Option B — conda

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

### Launch Jupyter

```bash
jupyter notebook adversarial_ml_project_v6.ipynb
```

### Run all cells in order

Go to **Kernel → Restart & Run All** to execute every cell from scratch. The notebook is self-contained — CIFAR-10 is downloaded automatically on first run (≈170 MB).

### Expected cell-by-cell output

| Section | What happens |
|---------|-------------|
| **1 — Setup** | Libraries imported, device printed, seeds confirmed |
| **2 — Dataset** | CIFAR-10 downloaded to `./data/`, `dataset_samples.png` saved |
| **3 — Model & Utilities** | All functions defined; `✓` confirmation printed for each |
| **4 — Baseline Training** | 100-epoch training loop with per-epoch logs; `baseline_model.pth` saved |
| **5 — Attack Evaluation** | Accuracy computed at 10 epsilon values; 5 plots saved |
| **6 — Defense Training** | FGSM-AT and PGD-AT training; `fgsm_at_model.pth`, `pgd_at_model.pth` saved |
| **7 — Results** | Multi-seed evaluation (3 seeds × 3 models); 6 plots + 2 CSVs saved |
| **8 — Analysis** | Markdown analysis cells; no code output |

---

## 4. Key Configuration Knobs

All hyperparameters live in the `CFG` dictionary in **Section 1**. Common things you may want to adjust before running:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `seeds` | `[42, 123, 7]` | Random seeds for multi-seed evaluation |
| `epochs` | `100` | Training epochs per model |
| `eps_train` | `0.03` | Adversarial perturbation budget during training |
| `epsilons` | `[0.0 … 0.30]` | Epsilon values swept during attack evaluation |
| `batch_size` | `128` | Reduce to `64` if you hit OOM errors |
| `num_workers` | `2` | DataLoader workers; set to `0` on Windows if you get multiprocessing errors |
| `trades_beta` | `6.0` | TRADES regularization strength for PGD-AT |
| `pgd_adv_ratio` | `0.75` | Fraction of adversarial examples per batch in PGD-AT |
| `epoch_sweep` | `[10, 50, 100]` | Epochs used in the optional training budget experiment |

**Quick smoke-test run** — to verify everything works before committing to a full run, set:
```python
CFG["epochs"]  = 5
CFG["seeds"]   = [42]
CFG["epsilons"] = [0.0, 0.03, 0.10]
```
This completes in ~10 minutes on GPU and produces all plots with lower-quality numbers.

---

## 5. Expected Outputs

After a full run the following files will be written to the working directory:

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

The final cell prints a manifest with file sizes confirming all outputs were written:
```
✓ baseline_model.pth                          44.7 MB
✓ results_raw.csv                              0.2 MB
...
```

---

## 6. Expected Accuracy Numbers

These are the approximate results you should see after a full 3-seed run:

| Model | Clean | FGSM ε=0.03 | PGD-5 ε=0.03 | PGD-20 ε=0.03 |
|-------|-------|-------------|--------------|----------------|
| Baseline | ~92% | ~56% | ~47% | ~40% |
| FGSM-AT | ~89% | ~73% | ~67% | ~60% |
| PGD-AT (TRADES) | ~87% | ~77% | ~74% | ~71% |

The notebook includes two automatic sanity checks:

```
Ordering check at ε=0.03: FGSM > PGD-5 > PGD-20  ✓
PGD-AT vs FGSM-AT gap under PGD-20: +Δ%            ✓
```

If either assertion fails, re-check that `pgd_alpha` is **not** hardcoded — it must be computed adaptively as `epsilon * 2.5 / steps`.

---

## 7. Troubleshooting

**CUDA out of memory**
Reduce `CFG["batch_size"]` from `128` to `64` or `32`.

**`multiprocessing` errors on Windows**
Set `CFG["num_workers"] = 0`.

**CIFAR-10 download fails**
Manually download from https://www.cs.toronto.edu/~kriz/cifar.html, extract to `./data/cifar-10-batches-py/`, and set `download=False` in the dataset cells.

**NaN loss during PGD-AT**
This is guarded by the `torch.isfinite(loss)` check in `adversarial_training()`, which skips corrupt batches. If NaNs persist, lower `CFG["trades_beta"]` or increase the β warmup period (currently ramps 0→6 over the first 10 epochs).

**Attack ordering assertion fails (`FGSM ≥ PGD-5 ≥ PGD-20`)**
Ensure `alpha` is not overridden with a fixed value anywhere. The correct behaviour is `alpha = epsilon * 2.5 / steps` computed inside `pgd_attack()`.

---

## 8. Reproducibility Notes

- All three models are trained with the same random seeds via `seed_everything()`, which sets Python, NumPy, and PyTorch seeds and enables `cudnn.deterministic = True`.
- Results may vary slightly across different GPU hardware or CUDA versions due to non-deterministic CUDA kernels, even with the same seeds.
- The Fairness Rule is enforced by sharing `CFG["epochs"]`, `CFG["lr"]`, `CFG["momentum"]`, and `CFG["weight_decay"]` across all models — no model receives extra compute.
