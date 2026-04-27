# Subspace Comparison

Geometric comparison between the **DIM** (Difference-in-Means) refusal direction and **ActSVD** weight-delta subspaces for **Llama-3.1-8B-Instruct**, plus cross-model DIM direction similarity.

---

## Metrics

### Maximum Subspace Overlap (MSO)

Given two subspaces $A$ (dim $k_A$) and $B$ (dim $k_B$) in $\mathbb{R}^d$:

$$\text{MSO} = \frac{1}{k_A} \| U_A^\top U_B \|_F^2$$

where $U_A, U_B$ are orthonormal bases. Ranges from 0 (orthogonal subspaces) to 1 (one is fully contained in the other). The **random baseline** for two uniformly random subspaces is $k_A \cdot k_B / d$.

Here:
- $A$ = the 1-D subspace spanned by the DIM refusal direction (extracted at layer 11, position −2).
- $B$ = the column space of the low-rank weight delta $\Delta W = W_{\text{ActSVD}} - W_{\text{base}}$ for each weight matrix, computed via SVD and thresholded to the effective rank (singular values > 1 % of max).
- Three weight types are compared per layer: **Q projection**, **O projection**, and **MLP down projection** (the ones ActSVD modifies).

### Cross-Model Cosine Similarity

DIM refusal directions were pre-computed for 6 instruction-tuned models: Llama-3.1-8B-Instruct, Llama-3-8B-Instruct, Llama-2-7B-Chat, Yi-6B-Chat, Gemma-2B-IT, and Qwen-1.8B-Chat. Each direction is a unit vector in $\mathbb{R}^{d_\text{model}}$. Pairwise cosine similarity is computed for all model pairs sharing the same $d_\text{model}$; the smaller Gemma/Qwen pair is compared separately.

---

## Key Results

**DIM vs ActSVD MSO**: Per-layer MSO ranges 0.008–0.057, averaging ~0.018 — close to the random baseline. The two methods target **nearly orthogonal subspaces**.

**Layer 10 hotspot**: The only notable signal is at layer 10 (MLP down proj MSO = 0.057, 3.2× random; O proj MSO = 0.041, 2.6× random). This is the layer immediately before the DIM direction's source layer (11).

**Cross-model cosine**: Llama-3.1 ↔ Llama-3 = **0.603** (substantial). All other pairs ≈ 0. The refusal direction is model-family-specific.

---

## Plots

### 1. `mso_heatmap.png` — MSO Heatmap (Layer × Weight Type)

A 32-row × 3-column heatmap using a YlOrRd (yellow-orange-red) colourmap. Rows = transformer layers 0–31, columns = Q proj, O proj, MLP down proj. Each cell is annotated with the MSO value and its ratio to the random baseline (e.g., "0.057 (3.2×)").

- Most cells are light yellow (MSO ≈ 0.012–0.020), close to the random baseline ratio of ~1.0× — **near-random overlap**.
- **Layer 10 MLP down proj** is the clear hotspot at **0.057 (3.2× random)** — the darkest cell.
- **Layer 10 O proj** is also elevated at **0.041 (2.6× random)**.
- Layers 8–9 MLP down proj are mildly elevated (~0.035, ~2.0×).
- The overall picture: the two methods target almost completely different subspaces, with a localised mild signal around layers 8–10.

### 2. `mso_per_layer.png` — Per-Layer Average MSO Bar Chart

Dual bar chart with 32 layer positions on the x-axis. Red bars = actual DIM-vs-ActSVD MSO (averaged over the 3 weight types per layer), light blue bars = random baseline ($k_A \cdot k_B / d$).

- Most layers show red and blue bars at roughly the same height (~0.015–0.018), confirming near-random overlap.
- **Layer 10** has the tallest red bar (~0.038 average), clearly above its baseline. Layers 5–9 and 12–14 show slight elevation.
- Layers 20–31 are essentially flat at baseline level.

### 3. `cross_model_cosine.png` — Cross-Model DIM Direction Cosine Similarity

Horizontal bar chart with 7 model pairs, sorted by absolute cosine similarity. One bar is green (high similarity), the rest are blue (near-zero):

- **Llama-3.1 vs Llama-3**: **0.603** (green) — the DIM refusal direction is substantially shared within the Llama 3.x family.
- **All other pairs** ≈ 0 (range −0.017 to +0.013): Llama-3.1 vs Llama-2 (0.009), Llama-2 vs Llama-3 (0.013), and all cross-family pairs involving Yi/Gemma/Qwen.
- This demonstrates the refusal direction is **model-family-specific**: it transfers within Llama-3.x but not across architectures or even to Llama-2.

---

## Data Files

| File | Description |
|---|---|
| `comparison_results.json` | Full numerical results (per-layer/per-weight-type MSO + cross-model cosine) |
