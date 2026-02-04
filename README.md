# ES-SSM-Elastic-Spectral-State-Space-Models-for-budgeted-inference
The implementation of Elastic Spectral State Space Models for budgeted inference.

**Train once at full spectral capacity K̄, deploy at any runtime budget K ≤ K̄ — with the same trained weights.**

This repo provides a **clean, auditable reference implementation** of **ES-SSM** (Elastic Spectral State Space Models) with a **PG19 byte-level language modeling** pipeline designed for **budgeted inference**:
- **one model**
- **many runtime budgets**
- **predictable compute scaling with K**

---

## What ES-SSM does (in one paragraph)

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

### 3) Spectral features computed efficiently (FFT)
Spectral convolutions are computed via FFT and chunked over channels, so runtime scales with K in a controlled way.
