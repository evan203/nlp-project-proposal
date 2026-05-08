
// Final report for DL4NLP project.
// Builds on report.typ but reframes the narrative around the core tension
// between the reference papers' conflicting claims about safety-utility separability.

#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.1": *

#show figure: set text(size: 0.9em)
#show figure: set par(justify: false)
#show figure: set align(left)
#show figure: set block(above: 1em, below: 1.2em)

#show: doc => acl(
  doc,
  anonymous: false,
  title: [Comparing Safety-Removal Subspaces in Aligned LLMs],
  authors: make-authors(
    (
      name: "Evan Scamehorn",
      affiliation: [University of Wisconsin\ #email("scamehorn@wisc.edu")],
    ),
    (
      name: "Adam Venton",
      affiliation: [University of Wisconsin\ #email("venton@wisc.edu")],
    ),
    (
      name: "Calvin Kosmatka",
      affiliation: [University of Wisconsin\ #email("ckosmatka@wisc.edu")],
    ),
    (
      name: "Kyle Sha",
      affiliation: [University of Wisconsin\ #email("kasha2@wisc.edu")],
    ),
    (
      name: "Zeke Mackay",
      affiliation: [University of Wisconsin\ #email("afmackay@wisc.edu")],
    ),
  ),
)


#abstract[
  Four recent works make conflicting claims about whether safety can be
  cleanly separated from utility in aligned LLMs.
  #citet("arditi2024") report that a single residual-stream direction
  mediates refusal. #citet("Wei2024Brittleness") find that safety and
  utility ranks overlap and must be orthogonalized.
  #citet("Ponkshe2026Safety") argue that no selective removal is
  possible. #citet("pmlr-v267-wollschlager25a") generalize refusal to a
  multi-dimensional cone (Refusal Cone Optimization, RCO). We reproduce
  Difference-in-Means (DIM), ActSVD, and RCO on LLaMA-3.1-8B-Instruct
  and compare them at the behavioral, geometric, and causal levels.
  The full per-layer safety subspace overlaps the utility PCA basis at
  98× random (rank 8); DIM's selected 1-D direction is far less
  entangled (40×); RCO's optimized 2-D cone is essentially orthogonal
  to it (1.5×). Behaviorally, Qwen3Guard ASR ranks the methods
  *RCO 0.93 > DIM 0.90 > ActSVD 0.77*, with rank-matched
  random-direction and random-2-D-subspace baselines staying at the
  base-model floor. We extend the adversarial-suffix probe of
  #citet("arditi2024") from a single GCG suffix on Qwen 1.8B to
  WildJailbreak wrappers on Llama-3.1-8B, adding a per-method ablation
  cross-test in which RCO ablation strictly outperforms DIM on bare
  harmful requests. A benign-wrapped control shows that wrapper style
  itself perturbs the refusal subspace.
]

= Introduction

== A Contradiction in the Literature

Four recent papers make conflicting claims about whether safety is separable
from utility in aligned LLMs:

- #citet("arditi2024") show that refusal is mediated by a _single direction_ in
  activation space. Ablating it "surgically disables refusal with minimal effect
  on other capabilities." Benchmark scores stay within 99% confidence intervals
  of the original model.

- #citet("Wei2024Brittleness") find that naively removing top safety-critical
  ranks "also leads to a drastic decrease in the model's utility." They must use
  an orthogonal projection $(I - Pi^u) Pi^s W$ to disentangle safety from
  utility --- the need for this step is itself evidence of overlap. After
  disentanglement, approximately 2.5% of ranks suffice to break safety while mostly
  preserving utility.

- #citet("Ponkshe2026Safety") argue that "safety is highly entangled with the
  general learning components of the model" and that "no selective removal is
  possible." They find that improvements in safety come "only at a proportional
  cost to utility."

- #citet("pmlr-v267-wollschlager25a") generalize refusal from a single direction
  to multi-dimensional _concept cones_ of up to 5 dimensions. They show that
  DIM's single-direction ablation damages TruthfulQA, while their optimized
  directions reduce this side-effect by 40%.

These claims range from clean one-dimensional separation to fundamental
inseparability. If #citet("arditi2024") are correct, a single targeted
defense could block safety-removal attacks. If
#citet("Ponkshe2026Safety") are correct, safety removal inevitably
degrades the model. These are materially different conclusions for
alignment.

== Research Questions

We investigate whether these claims can be reconciled empirically by running all
three methods on the same model and asking:

+ *Cross-method agreement.* Do DIM, ActSVD, and RCO converge on the same
  geometric structure, despite operating at different levels (activation space
  vs.~weight space)?

+ *Safety-utility entanglement.* How much does each method's safety direction
  overlap with the model's utility activation subspace? Is the _full_ safety
  subspace entangled even if individual _selected_ directions are not?

+ *Behavioral tradeoff.* When each method removes safety, how much utility does
  it preserve? Does geometric overlap predict behavioral utility cost?

+ *Causal independence.* Are multiple refusal directions causally independent,
  or does ablating one change the effect of another?

+ *Generalizing the adversarial-suffix analysis of
  #citet("arditi2024").* Their §5.1 shows that a single GCG-optimized
  adversarial suffix suppresses the refusal direction at the EOI
  position on Qwen 1.8B Chat. The analysis is explicitly restricted to
  "a single model and a single adversarial example." We examine whether
  the same suppression appears under in-the-wild prompt-wrapping
  attacks, on a more recent and larger model (Llama-3.1-8B-Instruct),
  and whether the suppression is layer-localized or distributed.

== Key Insight

We hypothesize that these four claims may be simultaneously correct
because they describe different objects. The _full_ safety subspace
(e.g., the layer-wise stack of DIM mean-difference vectors) may be
entangled with utility, while each method's _selection procedure_
identifies a surgical direction within that entangled space with lower
utility overlap. DIM selects the direction with minimum KL divergence
on harmless prompts; ActSVD explicitly orthogonalizes safety against
utility; RCO's loss includes a retain term penalizing harmless-prompt
disruption. All three procedures implicitly optimize for low
safety-utility overlap, which may explain why the resulting
interventions preserve utility despite broad entanglement.

Some of the disagreement across papers may also reflect genuine differences in
models, datasets, or evaluation protocols rather than contradictory truths about
the same underlying phenomenon. Our controlled single-model study eliminates
model variation as a confound but cannot fully resolve this ambiguity.

= Literature Survey

== Difference-in-Means (DIM)

#citet("arditi2024") compute the mean residual-stream activation difference
between harmful and harmless prompts at each layer and token position. Each
difference vector is a candidate refusal direction. They select the single
direction that most effectively bypasses refusal when ablated and induces refusal
when added. The selected direction $hat(bold(r))$ defines a one-dimensional
safety subspace. Weight orthogonalization
$W'_("out") <- W_("out") - hat(bold(r))hat(bold(r))^T W_("out")$ permanently
removes the direction from the model. They report minimal degradation on
MMLU, ARC, and GSM8K, with the notable exception of TruthfulQA, which "veers
close to the territory of refusal." Their §5.1 also studies *one*
GCG-optimized adversarial suffix on Qwen 1.8B Chat and shows that
appending it suppresses the cosine of the last-token activation with
$hat(bold(r))$ relative to no-suffix and random-suffix conditions; this is
explicitly limited to a single model and a single suffix and is what our
probe extends.

== ActSVD

#citet("Wei2024Brittleness") perform SVD on the product of model weights and
input activations $W X_"in"$ for both safety and utility calibration datasets.
The orthogonal projection matrices $Pi^s$ and $Pi^u$ project onto the top $r^s$
safety and $r^u$ utility rank subspaces respectively. Crucially, they find that
directly removing top safety ranks destroys utility (accuracy drops from 0.58 to
0.35), so they compute the isolated safety projection
$Delta W = (I - Pi^u) Pi^s W$ to remove safety ranks _orthogonal_ to utility
ranks. This necessity of disentanglement provides rank-level evidence for
superposition: safety and utility share representational capacity.

== Safety Subspaces Are Not Linearly Distinct

#citet("Ponkshe2026Safety") demonstrate that the principal directions amplifying
safe behaviors also amplify useful ones. Using MSO between fine-tuning update
subspaces, they show that "the strongest overlap is between the useful and
harmful updates, rather than between alignment and harmful updates." They
conclude that "safety alignment is not linearly separable in LLMs" and that
"subspace filtering based on alignment directions imposes a strict tradeoff."

== The Geometry of Refusal

#citet("pmlr-v267-wollschlager25a") show that refusal is mediated by
multi-dimensional polyhedral cones rather than a single direction. They
introduce Refusal Direction Optimization (RDO) and Refusal Cone Optimization
(RCO), which use gradient descent to find one or more refusal directions.
They define _Representational Independence_ (RepInd): two directions are
representationally independent if ablating one does not change the cosine
similarity profile of the other across layers. They find multiple such
independent directions, concluding that "refusal behavior is more nuanced than
previously assumed."

= Methodology

All experiments target LLaMA-3.1-8B-Instruct. We compare three methods that
operate at different levels of the model:

#figure(
  table(
    columns: (1.2fr, 2fr, 2fr),
    inset: 6pt,
    align: (left, left, left),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { gray.lighten(80%) },
    [*Method*], [*Operates on*], [*Intervention type*],
    [DIM], [Residual stream activations], [Inference-time ablation (1-D direction)],
    [ActSVD], [Weight matrices per linear layer], [Permanent weight surgery],
    [RCO], [Residual stream activations], [Inference-time ablation (2-D subspace: both cone basis vectors orthonormalized + projected out)],
  ),
  caption: [The three methods operate at different levels of the model but target the same behavioral outcome (removing refusal).],
) <method_summary>

== Safety Subspace Extraction

*DIM.* Following #citet("arditi2024"), we compute per-layer mean activation
differences $bold(r)_i^((l)) = bold(mu)_"harmful"^((l)) - bold(mu)_"harmless"^((l))$
at end-of-instruction token positions. The selection procedure evaluates each
candidate on: ablation refusal score (lower = better at bypassing refusal),
ablation KL divergence on harmless prompts (lower = less utility damage), and
steering refusal score (higher = better at inducing refusal). The direction with
the lowest ablation refusal score, after filtering by KL and steerability
thresholds, is selected.

*ActSVD.* Following #citet("Wei2024Brittleness"), we wrap each linear layer
with activation recording, collect activations on utility data (Alpaca without
safety content) and safety data (alignment SFT data), compute
$"score" = X_"in" W^T$, perform low-rank SVD, and apply the orthogonal
projection $W' = W - (I - Pi^u) Pi^s W$ with utility rank $r^u = 3950$ and
safety rank $r^s = 4090$, matching the paper-optimal setting reported
by #citet("Wei2024Brittleness").

*RCO.* Following #citet("pmlr-v267-wollschlager25a"), we initialize from the
DIM direction and optimize a 2-dimensional refusal cone via gradient descent.
The loss combines a refusal-scaling term (refusal probability should increase
when the cone is added), a surgical-ablation term (removing the cone should
bypass refusal without affecting harmless prompts), and a KL-divergence
retain term. At inference, both cone basis vectors are orthonormalized
via QR and projected out at every block input and at every attention/MLP
output, so the intervention removes the entire 2-D cone subspace.

== Bridging Spaces, Utility Subspace, Comparison Framework

DIM and RCO operate in activation space, while ActSVD operates in
weight space. We bridge these two regimes in two complementary ways.
*Weight-delta MSO*: for each layer and weight type, we compute the
thin SVD of $Delta W = W_"actsvd" - W_"base"$ and project the DIM and
RCO directions onto its left singular basis. This is a *capacity*
measure: the realized perturbation $Delta W bold(x)$ depends on the
input distribution. *Activation-delta direction*: per-layer mean
activation deltas computed across 64 harmless prompts provide a direct
activation-space comparison between the base and ActSVD-modified models.

The *utility subspace* is the rank-$k$ PCA basis $Q_k$ of 128 harmless
EOI activations per layer. For a 1-D safety direction
$"MSO"(hat(bold(s)), Q_k) = norm(Q_k^top hat(bold(s)))^2$ with random
baseline $k/d$. For a $k_S$-D safety subspace (RCO cone) we use the
*normalized* subspace MSO $norm(Q_k^top B)_F^2 / min(k_S, k)$ with
random baseline $max(k_S, k)/d$ so the cone is not artificially
advantaged by ablating more dimensions. PCA captures *variance*, which
is a proxy for --- not equivalent to --- causal utility contribution;
behavioral perplexity is the definitive utility test.

The behavioral benchmark uses JailbreakBench
ASR #cite("jailbreakbench"), harmless compliance, and Pile/Alpaca
perplexity #cite("thepile") #cite("alpaca"). Direction-pair similarity
between activation-space methods is the top principal-angle cosine
($= sigma_1(Q_("DIM")^top Q_("RCO"))$, reduces to ordinary cosine when
both are 1-D). Causal comparison follows
#citet("pmlr-v267-wollschlager25a"): we measure each direction's
per-layer cosine profile, ablate another direction, re-measure, and
report mean absolute profile change.

== Extending the Adversarial-Suffix Analysis of #citet("arditi2024") to In-the-Wild Attacks

The §5.1 analysis of #citet("arditi2024") demonstrates that a single
GCG-optimized adversarial suffix suppresses the refusal direction at
the EOI position on Qwen 1.8B Chat. We extend this analysis along
four axes: (i) we use Llama-3.1-8B-Instruct in place of Qwen 1.8B;
(ii) we substitute WildJailbreak prompt-wrapping attacks for the
single GCG suffix; (iii) we measure projection at every layer rather
than at the DIM-selected layer alone; and (iv) we add a per-method
ablation cross-test in which DIM and RCO ablations are applied
separately to the same probe prompts. Three prompt groups are
considered: *direct_request* (HarmBench bare harmful prompts),
*adversarial_harmful* (WildJailbreak wrapped harmful prompts), and
*adversarial_benign* (WildJailbreak wrapped benign prompts, included
as a control to isolate wrapper-style effects from harmful-intent
effects). The probe inverts the ablation comparison: rather than
asking whether removing the direction enables attacks, we ask whether
successful prompt attacks remove the direction.

= Experimental Settings

All experiments target Llama-3.1-8B-Instruct on a single A100 GPU. The full
pipeline is reproducible end-to-end via `notebooks/colab_end_to_end.ipynb`.

*Datasets.* JailbreakBench #cite("jailbreakbench") provides 100 categorized
harmful prompts for ASR. Harmless compliance and perplexity use Alpaca
#cite("alpaca"); generic-text perplexity uses the Pile #cite("thepile").
ActSVD calibration uses Alpaca-cleaned-no-safety (utility) and alignment SFT
data (safety), following #citet("Wei2024Brittleness"). The prompt-attack
probe uses HarmBench standard behaviors #cite("mazeika2024harmbench") for the
direct-request baseline and WildJailbreak #cite("jiang2024wildteaming")
`adversarial_harmful` / `adversarial_benign` splits for the wrapped and
benign-control groups (streamed with shuffle buffer; gated, requires HF token).

*Method hyperparameters.*
- *DIM*: 128 harmful (Advbench split) and 128 harmless prompts; layer $l_*=11$
  selected with KL- and steerability filtering.
- *ActSVD*: 128 calibration samples; utility rank $r^u=3950$, safety rank
  $r^s=4090$, matching #citet("Wei2024Brittleness")'s reported optimum
  (effective $Delta W$ rank $approx 6$).
- *RCO*: 2-D cone, DIM-initialized, learning rate $1 times 10^(-3)$, batch
  size 16, 1500 steps; refusal-scaling + surgical-ablation + KL-retain losses.

*Evaluation harness.* JBB ASR (100 prompts) is graded by substring at
eval time, then re-graded post-hoc by *Qwen3Guard-Gen-4B*
#cite("qwen3guard"), an external response-safety classifier (1.19M-pair
training set, three-tiered safe / controversial / unsafe labels),
applied to every method's saved completions in a single consistent
pass. Using an external moderator from a different model family
removes both the cross-method confound (a method's intervention
biasing its self-judgment) and the same-family bias of using the base
model as judge. Harmless compliance (100 prompts), Pile/Alpaca
perplexity (64 each), and TruthfulQA (64 questions, substring against
`correct_answers` / `incorrect_answers`) round out the behavioral
evaluation. All rates carry 1,000-sample bootstrap 95% CIs. *Two
random baselines* ($cal(N)(0,I)$, seed 7) are added as sanity
checks: a 1-D random direction (rank-matched to DIM's ablation) and a
2-D random orthonormal subspace (rank-matched to RCO's cone).
Together they distinguish direction-specific effects from
rank-dependent ones. Safety-utility overlap uses PCA of 128 harmless
activations at ranks $k in {1,2,4,8,16,32}$. We adopt $k=8$ as the
primary rank for the headline numbers because the full DIM
mean-difference subspace's MSO-to-random-baseline ratio peaks at
$k=8$ on this model (78× at $k in {1,2}$, 71× at $k=4$, 98× at
$k=8$, 74× at $k=16$, 45× at $k=32$); $k=8$ therefore captures the
strongest entanglement signal while remaining well below the
4096-dimensional residual-stream space. The full rank sweep is
preserved in the supplementary results;
RepInd uses 32 prompt pairs with a 3-D basis; the probe (75 prompts)
adds a *layer sweep* (project at every block input) and an *ablation
cross-test* (generate each prompt under base, DIM-ablated, and
RCO-ablated conditions).

= Results

== Behavioral Benchmark

#figure(
  image("figures/benchmark_safety_utility_tradeoff.png", width: 95%),
  caption: [Safety-utility tradeoff across all methods. DIM and RCO achieve high ASR with minimal utility cost; ActSVD achieves moderate ASR with higher perplexity degradation.],
) <fig_benchmark>

@fig_benchmark shows the behavioral comparison across all methods.

#figure(
  table(
    columns: (2fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { gray.lighten(80%) },
    [*Method*], [*sub-ASR*], [*Q3G-ASR*], [*Hrmless*], [*PPL Pile*], [*PPL Alpaca*],
    [Base], [0.15], [0.00], [1.00], [13.93], [8.60],
    [DIM-Ablated (1-D)], [1.00], [0.90], [1.00], [14.17], [8.80],
    [ActSVD-Modified], [0.80], [0.77], [1.00], [20.16], [11.65],
    [RCO-Cone-2 (true 2-D ablation)], [1.00], [*0.93*], [1.00], [14.08], [8.76],
    [Random-Direction-7-1D], [0.16], [0.00], [0.98], [14.65], [8.86],
    [Random-Subspace-7-2D], [0.14], [0.00], [0.99], [14.61], [8.78],
  ),
  caption: [Behavioral benchmark results. Higher ASR = more safety removed. Higher harmless compliance = better utility preservation. Lower perplexity = better language modeling quality.],
) <tab_benchmark>

Substring ASR overestimates jailbreak success because it scores any
non-refusal --- including vague or hedged completions --- as a
jailbreak. The Qwen3Guard response-safety judge provides the
discriminating metric and yields the ranking
*RCO (0.93) > DIM (0.90) > ActSVD (0.77)*. The corresponding
refusal-rate counts agree (RCO complies on 99/100 prompts, DIM on
98/100, ActSVD on 84/100). ActSVD's lower ASR is accompanied by
substantially higher perplexity (Pile 20.16, Alpaca 11.65 vs.~baseline
13.93 / 8.60), indicating that weight-space modification is
both less effective at removing refusal and more disruptive to general
language modeling than activation-space ablation. The two random
baselines establish direction-specificity: the 1-D random direction
yields ASR 0.16 / 0.00 and the rank-matched 2-D random subspace
yields 0.14 / 0.00, ruling out the alternative hypothesis that RCO's
gain is a mechanical consequence of ablating additional dimensions.

JBB ASR is graded by *Qwen3Guard-Gen-4B*, an external safety
classifier purpose-built for response-safety classification.
Jailbroken is defined as the moderator's `Unsafe` label;
`Controversial` is conservatively counted as not jailbroken. TruthfulQA
(64 questions, substring grading) is mostly ambiguous (78--89% across
methods), so it serves only as a weak side-effect check. All rates
carry 1,000-sample bootstrap 95% CIs.

== Cross-Method Geometric Agreement

#figure(
  image("figures/subspace_mso_per_layer_avg.png", width: 95%),
  caption: [Per-layer MSO between the DIM 1-D direction (or RCO 2-D cone) and the ActSVD weight-delta column space, averaged over the three weight types. Most layers sit near the random baseline; layer~10 is the main hotspot for both methods, peaking at $0.070$ for DIM-vs-ActSVD MLP-down (48× random) and $0.023$ for RCO-vs-ActSVD attention-out (16× random) -- both methods plausibly target the same layer-10 mechanism.],
) <fig_mso_per_layer>

DIM-vs-ActSVD MSO remains near the random baseline for most layers,
with prominent peaks at layer~10's MLP down-projection
($"MSO" = 0.070$ vs.~random $0.00146$, $48×$) and attention output
projection ($0.054$ vs.~$0.00146$, $37×$). RCO-vs-ActSVD exhibits a
weaker version of the same pattern at the same layer: the
output-projection peak is $0.023$ ($16×$) and the MLP-down peak is
$0.009$ ($6×$). We treat this as *exploratory rather than
conclusive*: the SVD bridge characterizes the column space that
ActSVD's edit *can* perturb, not its realized effect
$Delta W bold(x)$ on inputs sampled from the data distribution. The
behavioral benchmark, in which all three methods elevate ASR,
provides stronger evidence that they share a common refusal mechanism,
even though the bridge cannot localize it precisely. The DIM-vs-RCO
top principal-angle cosine is $0.505$ (reducing to the ordinary
cosine because DIM is one-dimensional), indicating moderate agreement
in activation space; the gradient-optimized direction departs
substantially from the statistical mean-difference estimate,
consistent with a non-trivial difference in the loss landscape's
optimum.

== Safety-Utility Overlap: The Central Finding

#figure(
  image("figures/safety_utility_overlap_per_layer.png", width: 95%),
  caption: [Per-layer safety-utility MSO at rank 8. Bars: full DIM mean-difference direction at each layer (98× random on average). Lines: each method's safety object projected onto the same utility PCA basis. RCO's 2-D cone subspace MSO (green) tracks the random baseline at every layer.],
) <fig_safety_utility_per_layer>

#figure(
  placement: top,
  table(
    columns: (2.5fr, 1.1fr, 1fr),
    inset: 3.5pt,
    align: (left, center, center),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { gray.lighten(80%) },
    [*Safety object*], [*MSO (rank 8)*], [*vs.~Random*],
    [Full DIM mean-diffs (averaged)], [0.191], [98×],
    [DIM selected direction (layer 11)], [0.078], [40×],
    [DIM selected direction (avg.\ over layers)], [0.0165], [8.4×],
    [RCO 2-D cone (subspace MSO)], [*0.0030*], [*1.5×*],
    [ActSVD activation delta (per-layer)], [0.0605], [31×],
    [Random 1-D baseline], [0.00195], [1×],
    [Random 2-D subspace baseline ($k_S{=}2$, $k_U{=}8$)], [0.00195], [1×],
  ),
  caption: [Safety-utility MSO at rank 8 for each method's safety object. RCO uses the normalized 2-D subspace MSO formula $||U^top Q_S||_F^2 / min(k_S, k_U)$ with baseline $max(k_S,k_U)/d = 8/4096$.],
) <tab_safety_utility>

The full DIM safety subspace has MSO 0.191 (98× random), consistent
with the claim of #citet("Ponkshe2026Safety") that safety is broadly
entangled with utility. The DIM-selected direction (40× at layer~11,
8.4× averaged) occupies a substantially less entangled region of the
same space, and RCO's optimized 2-D cone has normalized subspace MSO
of just 0.0030 (1.5× random) at rank 8 --- essentially orthogonal to
the utility PCA basis, and lower than the 2-D random baseline at the
same rank. RCO ablates more dimensions than DIM yet preserves
perplexity at least as well, indicating that the dimensions it ablates
contain less utility-relevant content. The gap between the full
subspace and the selected direction partly reflects a dimensionality
asymmetry: averaging 32 per-layer directions inflates the mean. The
within-method comparison (DIM single direction at 8.4× vs.~RCO 2-D
cone at 1.5×) avoids this asymmetry, since both quantities use the
same normalized formula and the same rank-$k=8$ utility basis. We
emphasize that PCA-MSO is not a sufficient predictor of utility cost;
the perplexity column of @tab_benchmark serves as the definitive
behavioral test.

== Representational Independence

#figure(
  image("figures/repind_change_heatmap.png", width: 80%),
  caption: [RepInd profile-change matrix. Rows: ablated direction. Columns: measured direction. Lower off-diagonal values indicate greater representational independence.],
) <fig_repind>

The RepInd analysis reveals an asymmetric causal structure. Ablating
DIM changes the first RCO cone basis vector's per-layer cosine profile
substantially (mean $|Delta|=0.094$, max $|Delta|=0.218$), while
ablating that same basis vector causes a smaller change to DIM's
profile (mean $|Delta|=0.061$, max $|Delta|=0.280$). The second RCO
basis is still more independent: ablating DIM changes its profile by
mean $|Delta|=0.065$, but ablating it changes DIM by only mean
$|Delta|=0.032$. This asymmetry indicates that DIM captures a dominant
refusal component that causally influences other directions, but the
reverse is weaker --- consistent with
#citet("pmlr-v267-wollschlager25a")'s finding that multiple
representationally-independent refusal mediators coexist, with a
hierarchy of causal influence rather than fully orthogonal channels.

== Prompt-Based Jailbreaks vs.~the Refusal Direction

#figure(
  image("figures/probe_asr_and_projection_by_attack_type.png", width: 100%),
  caption: [ASR (left) and mean DIM/RCO projection (right) per prompt group.],
) <fig_probe_asr_proj>

#figure(
  image("figures/probe_layer_sweep_projection.png", width: 100%),
  caption: [*Layer sweep:* per-layer mean DIM/RCO projection per group, $plus.minus$SE bands.],
) <fig_probe_layer>

#figure(
  image("figures/probe_ablation_cross_test.png", width: 90%),
  caption: [*Ablation cross-test:* same probe prompts under base vs each
  ablated model. Both DIM (1-D direction ablation) and RCO (2-D cone subspace
  ablation) are run as separate conditions on the *same* prompts. If RCO's
  cone covers refusal directions DIM misses, RCO-ablated ASR should be at
  least as high as DIM-ablated ASR on every group.],
) <fig_probe_abl>

The probe metric for RCO is the L2 norm of the EOI activation's
projection into the optimized 2-D cone subspace, $"proj"_("RCO") = norm(B^top
bold(h))_2$, which is the natural multi-dimensional analog of DIM's signed
scalar $"proj"_("DIM") = bold(h) dot.op hat(bold(r))$. We also record the
two per-basis signed scalars $"proj"_("RCO,basis 0")$ and
$"proj"_("RCO,basis 1")$ separately for diagnostic value.

The probe yields a coherent picture. Direct harmful requests exhibit
low base ASR (0.12) and high projection onto both refusal subspaces
($"proj"_("DIM") = 3.17$, $"proj"_("RCO") = 1.89$ at layer~11;
per-basis $1.55$ and $1.08$). Adversarial-harmful wrapping increases
ASR to $0.36$ and reduces all projections in tandem
($"proj"_("DIM") = 1.11$, $"proj"_("RCO") = 0.73$; per-basis $0.60$
and $0.42$), generalizing the §5.1 pattern of #citet("arditi2024")
from a single GCG suffix on Qwen 1.8B to a population of
WildJailbreak attacks on Llama-3.1-8B. The benign-wrapped control
suppresses the projections further ($"proj"_("DIM") = 0.66$,
$"proj"_("RCO") = 0.44$; per-basis $0.33$ and $0.29$) while
producing high compliance (ASR $0.84$). The reduction in projection is
therefore not uniquely associated with harmful jailbreaks; the
wrapper style itself perturbs the refusal subspace.

The ablation cross-test applies DIM and RCO ablations separately to
the same probe prompts. DIM ablation lifts direct-request ASR from
$0.12$ to $0.96$, while RCO 2-D cone ablation lifts it to $1.00$ ---
a strict improvement on the bare-harmful subset where DIM leaves four
residual refusals out of 25 prompts. On the adversarial groups both
ablations reach $1.00$, indicating that the cone's additional coverage
manifests precisely where DIM's single direction is weakest. This is
consistent with the behavioral advantage of RCO in @tab_benchmark and
with the geometric picture: the cone covers refusal-mediating
directions that the DIM-selected vector misses while maintaining
utility entanglement at random-baseline levels. The layer
sweep localizes the DIM gap sharply around the selected layer $l_*=11$
for direct vs.~adversarial-harmful prompts; the RCO subspace norm
grows into later layers, suggesting it captures a related but more
distributed refusal geometry.

= Discussion

*Reconciling the four claims.* The four works may partly disagree
because they describe different objects:
#citet("arditi2024")'s "clean separation" concerns a *selected*
direction (KL-filtered on harmless prompts);
#citet("Ponkshe2026Safety")'s "no separation" concerns the *full*
subspace ($98×$ random PCA overlap);
#citet("Wei2024Brittleness")'s "separable with effort" is the gap
between the two; and the multi-dimensionality of
#citet("pmlr-v267-wollschlager25a") is consistent with both our
behavioral data (RCO 2-D ablation strictly outperforms DIM 1-D
ablation on direct harmful requests, $0.96 arrow.r 1.00$) and our
RepInd asymmetry. The strongest single piece of new evidence is that
RCO's 2-D cone has *lower* utility-PCA overlap than DIM's 1-D
direction (1.5× vs.~8.4× random, same rank-8 basis) --- the opposite
of what one would expect if removing additional dimensions necessarily
removed additional utility. A refined reconciliation follows: the
optimization landscape contains multi-dimensional refusal subspaces
that are more utility-orthogonal than the dominant 1-D refusal
direction.

*Probe findings.* Successful wrapping coincides with lower
refusal-subspace projection, generalizing the §5.1 result of
#citet("arditi2024") from a single GCG suffix on Qwen 1.8B to
WildJailbreak on Llama-3.1-8B. The benign-wrapped control also
suppresses the projection, so the probe supports
"wrappers perturb refusal geometry" more strongly than "harmful
intent specifically suppresses the refusal mediator;" the layer-sweep
and ablation cross-test separate those two interpretations.

== Limitations

- *Single model* (Llama-3.1-8B-Instruct); some inter-paper
  disagreements may be model-specific.
- *PCA utility $eq.not$ causal utility.* The utility subspace is
  defined by variance, not causal contribution; low MSO does not
  prove low utility damage. Behavioral perplexity is the definitive test.
- *SVD bridge measures capacity, not effect.* DIM/RCO-vs-ActSVD MSO
  does not capture $Delta W bold(x)$ on real inputs.
- *Early-layer mean-diff is partly format variance.* The full DIM
  stack at layers 0--3 has high MSO with utility PCA but those depths
  are dominated by token-embedding/template variance; the mid-layer
  subset (8--23) yields a more conservative estimate ($approx 85$× random).
- *RepInd uses the RCO cone basis as the second pair of candidates*,
  not independent vectors from RepInd's own loss; the asymmetry is
  evidence about the cone vs.~DIM, not a fully general independence test.
- *Probe sample size:* 25 prompts per group, substring refusal judge
  inside the probe (Qwen3Guard is used only on JBB).
- *Qwen3Guard model bias.* The judge is purpose-built but carries its
  own training-distribution biases and is conservative on borderline
  content; we report substring ASR alongside the Qwen3Guard score as a
  sanity check.

= Conclusion

We highlight three principal findings. *(i)* Among the three
interventions evaluated, RCO's 2-D cone ablation is the most
effective under an external safety judge (Qwen3Guard ASR
$0.93 > 0.90 > 0.77$ for RCO, DIM, and ActSVD respectively); the
rank-matched 2-D random-subspace baseline remains at the base-model
floor ($0.00$), establishing that this gain is direction-specific
rather than rank-driven. *(ii)* While the full safety subspace is
broadly entangled with utility ($98×$ random PCA overlap), the safety
objects each method actually intervenes with are not: RCO's 2-D cone
exhibits $1.5×$ random utility-PCA overlap, essentially orthogonal
to the dominant utility variance directions. *(iii)* DIM constitutes
a dominant but non-exhaustive refusal mediator: RepInd shows that
ablating DIM perturbs RCO's cone profiles more than the reverse
operation, and on the prompt-attack probe RCO ablation strictly
improves on DIM ablation for the bare-harmful subset
($0.96 arrow.r 1.00$). Our probe also extends the §5.1 analysis of
#citet("arditi2024") from a single GCG suffix on Qwen 1.8B to
WildJailbreak attacks on Llama-3.1-8B; the benign-wrapped control
demonstrates that wrapper style itself is a major driver of
refusal-direction suppression, weakening the interpretation that
harmful intent specifically suppresses the refusal mediator.

= Member Contributions

All five authors contributed to writing and research-design discussion.
Effort (totaling 100%):
*Evan Scamehorn (25%)* --- DIM/ActSVD Colab pipelines and behavioral benchmark.
*Adam Venton (15%)* --- safety-utility overlap, weight-delta SVD bridge.
*Calvin Kosmatka (15%)* --- literature survey, intro drafting.
*Kyle Sha (25%)* --- prompt-attack probe (incl.\ layer sweep + ablation
cross-test), WildJailbreak integration, Discussion.
*Zeke Mackay (20%)* --- RepInd analysis, RCO training, asymmetry interpretation.


#pagebreak()

#add-bib-resource(read("bibliography.bib"))
#print-acl-bibliography()
