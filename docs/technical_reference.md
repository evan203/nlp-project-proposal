# Technical Reference: Cross-Method Comparison Rationale

This document records the reasoning behind how and why we compare DIM, ActSVD,
and RCO — three methods that operate on fundamentally different mathematical
objects — and how we reconcile conflicting claims from the reference papers
about safety-utility separability.

---

## 1. The Problem: Three Methods, Three Geometric Levels

| Method | Operates on | Produces | Geometric object |
|--------|------------|----------|-----------------|
| **DIM** | Residual stream activations | A single unit vector `r̂ ∈ ℝ^d` | 1-D direction in activation space |
| **ActSVD** | Weight matrices `W` per linear layer | Modified weights `W' = W − ΔW` | A weight-space perturbation (one matrix per layer per weight type) |
| **RCO** | Residual stream activations (via nnsight) | 2 orthogonal unit vectors spanning a cone | 2-D subspace in activation space |

DIM and RCO both live in **activation space** — they are directions you ablate
from the residual stream at inference time. ActSVD lives in **weight space** —
it permanently edits the model's parameter matrices. You cannot directly take a
cosine similarity between a DIM direction vector and an ActSVD weight-delta
matrix; they are not the same type of object.

### Why comparison is still meaningful

Even though the objects are different, all three methods target the **same
behavioral outcome**: removing the model's refusal behavior. The question is
whether they arrive at that outcome by modifying the same underlying structure
or different ones. This is testable at three levels:

1. **Behavioral** — apply each method, measure the same downstream metrics
2. **Geometric (activation space)** — project all methods into a common activation-space representation and measure overlap
3. **Causal** — ablate one method's direction and measure whether it changes another method's causal effect

Note that behavioral convergence does not imply mechanistic convergence. Two
methods can achieve the same ASR by modifying entirely different pathways. Our
geometric and causal comparisons are necessary precisely because behavior alone
cannot settle mechanism.

---

## 2. How We Bridge the Activation–Weight Gap

### DIM ↔ RCO: Direct comparison

Both produce vectors in the same `ℝ^d` activation space. Cosine similarity is
the natural metric. If DIM and RCO find directions with high cosine similarity,
the gradient-optimized direction converged to approximately the same structure
the statistical mean-difference found. However, high cosine similarity is a
necessary but not sufficient condition for mechanistic equivalence — directions
can be geometrically close but causally distinct due to downstream nonlinear
interactions.

### DIM/RCO ↔ ActSVD: The SVD bridge

ActSVD modifies weight matrices. For a linear layer `y = Wx`, the output `y`
lives in activation space. The SVD of the weight change `ΔW = W_actsvd − W_base`
produces left singular vectors `U = [u₁, u₂, ..., uₖ]` that span the
**output-space subspace** that ActSVD modified. This output space IS the
activation space (residual stream). So:

```
MSO(DIM vs ActSVD at layer l, weight type t) = ‖U_B^T r̂‖²
```

where `U_B` is the truncated left-SVD basis of `ΔW` and `r̂` is the DIM/RCO
direction. This asks: "does the direction that DIM/RCO identified for ablation
lie within the subspace of output directions that ActSVD's weight edit _could_
produce changes along?"

#### Important caveat: possible ≠ actual

The column space of `ΔW` describes the set of output-space directions that the
weight edit _can_ produce changes along, but the actual effect on a given input
is `ΔW · x`, which depends on the input distribution. A direction may be in the
column space of `ΔW` but never actually activated because the relevant inputs
don't align with the corresponding right singular vectors. So high MSO between
a DIM direction and the column space of `ΔW` tells us that ActSVD's edit
_could_ affect that direction, not that it _does_ on real prompts.

We partially address this with the delta-activation bridge (below), which
measures the empirical effect rather than the theoretical capacity. But the SVD
bridge alone should be interpreted as an upper bound on mechanistic overlap.

### ActSVD → Activation-space direction: The delta-activation bridge

For safety-utility overlap analysis, we need an activation-space vector for
ActSVD too. We obtain this by running the same harmless prompts through both the
base model and the ActSVD-modified model, collecting mean activations at each
layer, and computing:

```
δ_actsvd[l] = mean_activations_modified[l] − mean_activations_base[l]
```

This per-layer vector lives in activation space and represents the mean
direction along which ActSVD shifted activations on harmless prompts.

#### Important caveat: mean ≠ full effect

This delta-activation vector compresses the entire effect of a weight edit into
one mean vector per layer. It hides several things:

- **Prompt-dependent effects**: ActSVD may shift different prompts in different
  directions. The mean may not represent any individual prompt's experience.
- **Variance changes**: ActSVD might change the shape of the activation
  distribution (spread, clustering) without changing the mean much.
- **Nonlinear downstream effects**: The mean shift at layer `l` does not capture
  how that shift propagates nonlinearly through subsequent layers.

We use this vector as a pragmatic approximation, not a complete characterization.
The behavioral benchmark remains the ground truth for whether ActSVD actually
damages utility.

---

## 3. The Conflicting Claims from the Reference Papers

The four reference papers disagree fundamentally about whether safety is
separable from utility:

### DIM (Arditi et al., 2024): "Clean separation"

> "We show that refusal is mediated by a one-dimensional subspace... resulting
> in a model that retains its original capabilities but no longer refuses
> harmful instructions."

Key claim: A single direction mediates refusal. Ablating it surgically disables
safety with minimal utility cost. MMLU/ARC/GSM8K scores stay within 99% CI of
the original.

### ActSVD (Wei et al., 2024): "Separable with effort"

> "Regions primarily contributing to safety behaviors in aligned models may also
> be crucial for its general utility."

Key finding: Naively removing top safety ranks "also leads to a drastic decrease
in the model's utility, with accuracy dropping to 0.35 from 0.58." They MUST use
an orthogonal projection `(I − Π_u)Π_s W` to disentangle. The need for this
step is itself evidence of overlap. After disentanglement, ~2.5% of ranks can be
removed to break safety while "mostly maintaining utility."

### Safety Subspaces (Ponkshe et al., 2025): "Not separable at all"

> "No selective removal is possible. Improvements in safety come only at a
> proportional cost to utility."

> "We find no evidence of a distinct safety subspace."

Direct contradiction of DIM. They show that top-k directions amplify both
helpful and harmful behaviors equally — they represent "axes of general parameter
sensitivity" rather than safety-specific structure.

### Geometry / RepInd (Wollschläger et al., 2025): "Multi-dimensional, not one cut"

> "Contrary to prior work, we uncover multiple independent directions and even
> multi-dimensional concept cones that mediate refusal."

Refusal spans up to 5-dimensional cones. DIM's single direction is insufficient
— it damages TruthfulQA, while their RDO reduces that damage by 40%. Multiple
representationally independent mechanisms exist.

### The spectrum

```
DIM              →  ActSVD              →  Geometry           →  Safety Subspaces
"clean 1-D cut"     "separable w/effort"   "multi-dimensional"   "fundamentally entangled"
```

### Why they may disagree

These papers use different models (Llama-2, Llama-3, Gemma-2, Qwen), different
datasets, different evaluation metrics, and different definitions of what
constitutes "safety" and "utility." Some of the disagreement may reflect genuine
model differences rather than contradictory truths about the same model. Our
contribution is testing all three methods on a single model under controlled
conditions, which at least removes the confound of model variation.

---

## 4. How Our Results Relate to These Claims

We frame the following as a **hypothesis supported by our analyses**, not as a
proven conclusion. The hypothesis is that the apparent contradiction between the
papers partly arises from measuring different quantities: the full safety
subspace vs. individual selected directions.

### Evidence for entanglement (supports Ponkshe et al.)

Our safety-utility MSO analysis shows that the **full DIM safety subspace**
(per-layer mean-diffs averaged over all layers) has substantially above-random
overlap with the utility PCA subspace. At rank 8:

```
Mean safety-utility MSO:  ~0.192
Random baseline:          ~0.002
```

This is ~100x above chance. The full safety signal IS geometrically entangled
with utility activations, as measured by this particular PCA-based definition
of utility.

#### Caveat: PCA utility ≠ causal utility

"High variance on harmless prompts" is not the same as "causally responsible
for utility." PCA captures dominant variation directions, not necessarily
task-relevant capability. A direction could have high PCA variance on harmless
prompts but be irrelevant to actual instruction-following quality. Conversely,
a causally important utility direction might have low variance and be missed
by PCA entirely.

So "low MSO with utility PCA" does not _prove_ low utility damage — it only
suggests low overlap with that particular harmless-activation basis. The
behavioral benchmark (perplexity, harmless compliance) is the actual test of
utility damage. The MSO analysis provides geometric context for interpreting
the behavioral results, not a replacement for them.

### Evidence for surgical separation (supports Arditi et al. / Wei et al.)

DIM's **selected direction** (the single best direction at one layer) has much
lower overlap:

```
Selected DIM direction MSO at rank 8:  ~0.078
Full-subspace mean MSO at rank 8:      ~0.192
```

#### Caveat: dimensionality confound

This comparison must be interpreted carefully. A single direction (1-D) will
naturally have lower overlap with a rank-k utility subspace than the average
over many directions across many layers. The "full-subspace mean MSO" of 0.192
is an average of 32 layer-level mean-diff directions, each independently
projected onto their layer's utility PCA basis. It is not a single high-dimensional
object being compared to a single subspace.

Still, the gap is informative if we compare like-for-like: among all 32
per-layer mean-diff directions, DIM's selection procedure picks one that is in
the lower range of utility overlap. Whether this selection is "implicitly
optimizing for low utility overlap" or simply a side effect of the refusal-score
selection criterion is not something our data can definitively distinguish. The
narrative is plausible but not proven by this comparison alone.

### Evidence for multi-dimensionality (supports Wollschläger et al.)

RepInd analysis shows that ablating the DIM direction substantially changes the
cosine profile of derived candidate directions (mean absolute change ~0.227),
while the reverse effect is much smaller (~0.002). This asymmetry indicates that
DIM captures a dominant but not exhaustive refusal component. Additional
directions with some refusal-mediating capacity exist — consistent with the
multi-dimensional cone finding, though our derived candidates are not fully
optimized RepInd vectors and may be weaker surrogates.

### Cross-method results

DIM-vs-ActSVD MSO is near the random baseline for most layers. This does NOT
mean the two methods are unrelated — it means the DIM direction does not lie
within the column space of ActSVD's weight changes at most layers. They may
achieve similar behavioral outcomes through partially distinct mechanisms, or
ActSVD's effect on the DIM direction may be mediated through input-dependent
pathways that the SVD bridge does not capture.

DIM-vs-RCO cosine similarity should be higher (both in activation space, RCO
initialized from DIM), but even high cosine similarity does not guarantee
causal equivalence.

### The hypothesis

We propose — but do not prove — the following interpretation:

1. The full safety structure is geometrically entangled with utility PCA
   directions (consistent with Ponkshe et al.)
2. Effective surgical cuts exist within that entangled space, as evidenced by
   behavioral results showing high ASR with preserved utility (consistent with
   Arditi et al.)
3. DIM's selection procedure appears to find a direction with lower utility
   overlap than average, though the extent to which this is an implicit
   optimization vs. a coincidence of the selection criterion is unclear
4. The safety structure is multi-dimensional (consistent with Wollschläger et al.)
5. ActSVD and DIM may achieve similar behavioral outcomes through partially
   distinct mechanisms; comparing their induced activation-space effects helps
   determine where, if anywhere, those mechanisms overlap

It is possible that some of the original papers' disagreements reflect genuine
model-level or methodology-level differences rather than contradictory truths
about the same underlying phenomenon. Our single-model study controls for model
variation but cannot generalize beyond LLaMA-3.1-8B-Instruct.

---

## 5. What Each Analysis Contributes to the Report

| Analysis | What it tests | Strength | Limitation |
|----------|--------------|----------|------------|
| **Behavioral benchmark** | ASR, harmless compliance, perplexity | Ground truth for safety/utility tradeoff | Doesn't explain mechanism |
| **DIM-vs-RCO cosine similarity** | Geometric convergence of activation-space methods | Direct comparison, same space | Geometry ≠ causation |
| **DIM/RCO-vs-ActSVD MSO** | Whether activation-space directions lie in weight-delta subspaces | Principled bridge via SVD | Measures capacity, not actual effect; input-dependent |
| **Safety-utility MSO (full subspace)** | Whether the full safety signal overlaps utility PCA | Above-random overlap is informative | PCA variance ≠ causal utility |
| **Safety-utility MSO (selected direction)** | Whether DIM's surgical cut avoids utility PCA | Lower than average is suggestive | Dimensionality confound; one direction vs. many |
| **Safety-utility MSO (per method)** | Comparative overlap across methods | Controlled comparison | Same PCA caveat applies to all |
| **Delta-activation for ActSVD** | Empirical activation-space effect of weight editing | Measures actual mean shift | Hides variance, prompt-dependence, nonlinear effects |
| **RepInd profiles** | Whether directions are causally independent | True causal test (intervention-based) | DIM-derived candidates are weak surrogates for optimized RepInd vectors |

---

## 6. How We Obtain Utility Directions

The "utility subspace" is derived from **harmless instruction activations** via PCA:

1. Collect residual stream activations from 128 harmless prompts at
   end-of-instruction token positions (using DIM's hook infrastructure)
2. Center the activations (subtract mean per layer)
3. Compute SVD: `U, S, V^T = SVD(centered_activations)`
4. Top-k right singular vectors form an orthonormal basis per layer

The MSO metric then measures what fraction of a safety direction's energy lies
within this basis:

```
MSO(safety, utility; rank k) = ‖basis_k^T · safety_unit‖²
```

where `basis_k` is the `[d, k]` matrix of top-k PCA vectors and `safety_unit`
is the unit-normalized safety direction.

The random baseline is `k/d` (the expected overlap of a random unit vector with
a random k-dimensional subspace in d dimensions).

### What PCA captures and what it misses

The top-k PCA directions capture the principal axes of _variation_ in how the
model processes harmless instructions. This is a reasonable proxy for "the
directions the model uses for normal instruction processing" — if most of the
activation variance on harmless prompts is concentrated in a few directions,
those directions likely carry important signal for the model's computations.

However, PCA is not a causal test. It is possible that:
- A high-variance direction is an irrelevant confound (e.g., prompt length or
  formatting artifacts) rather than a causally important utility signal.
- A causally critical utility direction has low variance across the prompt set
  (e.g., a constant instruction-following bias) and is therefore missed by PCA.
- The utility subspace is nonlinear and cannot be captured by any linear basis.

For these reasons, we treat the PCA utility subspace as a best-available linear
approximation, not as a definitive characterization of utility. The behavioral
metrics (perplexity, harmless compliance) serve as the ground truth for whether
utility is actually preserved.
