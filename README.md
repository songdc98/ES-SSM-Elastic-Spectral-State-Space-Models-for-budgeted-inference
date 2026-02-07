# ES-SSM-Elastic-Spectral-State-Space-Models-for-budgeted-inference
The implementation of Elastic Spectral State Space Models for budgeted inference.

## What ES-SSM does 

ES-SSM represents long-range sequence operators in a **fixed spectral basis** (Hankel-derived features) and learns to **mix spectral channels** with an input-adaptive gate.
At inference time, you enforce a runtime budget by activating only the first **K** spectral channels (K ≤ K̄).
During training, **budget dropout** randomizes K to make truncation robust and to produce a single set of weights that performs well across budgets.

---

## Key mechanisms implemented here

### 1) Budgeted inference via masked normalization over the active prefix
We compute gate logits over K̄ channels, then apply a **prefix mask** so the gate normalizes only over `1..K`.
This makes the model’s runtime cost and memory naturally scale with **K**.

### 2) RMS-rescaled gate logits for budget-consistent temperature
Different budgets change the dimension of the softmax support, which can change its effective temperature.
We therefore rescale the active logits by their RMS magnitude before the masked softmax.
Practically, this stabilizes training under budget dropout and makes the **BPB vs K** curve more reliable.

### 3) Budget dropout training (single model, many budgets)
During training, we **randomly sample a training budget K_train ≤ K̄** at each update and execute the **same budgeted forward pass** as in inference (i.e., masked normalization over `1..K_train`).
Inactive channels (`k > K_train`) receive **zero gradient** on that update, while shared components (e.g., the gate network and the direct term) are updated every step.
This makes the trained weights robust to **prefix truncation at deployment**, enabling reliable accuracy–compute trade-offs from a single training run.

### 4) Spectral features computed efficiently (FFT)
Spectral convolutions are computed via FFT and chunked over channels, so runtime scales with K in a controlled way.


## Citation

If you use this code, please cite the paper:

```bibtex
@article{song2026elastic,
  title={Elastic Spectral State Space Models for Budgeted Inference},
  author={Song, Dachuan and Wang, Xuan},
  journal={arXiv preprint arXiv:2601.22488},
  year={2026}
}
