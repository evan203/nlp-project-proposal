
// This is a minimal starting document for tracl, a Typst style for ACL.
// See https://typst.app/universe/package/tracl for details.


#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.1": *

#show figure: set text(size: 0.7em)
#show figure: set align(left)

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
  Modern Large Language Models (LLMs) undergo extensive alignment to ensure
  they don't produce harmful outputs, while still being as helpful as possible.
  However, these aligned models are fragile and can be jailbroken by a variety of methods. Recent research suggests
  that this fragility may stem from compact safety mechanisms whose relationship
  to general utility remains poorly understood. Several methods have been
  proposed to isolate or remove safety-related structure, including Difference
  in Means (DIM), Refusal Cone Optimization (RCO), and ActSVD.
  Conversely, other research suggests that it might be impossible to completely linearly
  separate safety and utility in current LLMs. We reproduce
  Difference-in-Means (DIM), ActSVD, and RCO on Llama-3.1-8B-Instruct
  and compare them at the behavioral, geometric, and causal levels.
  The full per-layer DIM mean-difference stack overlaps the utility PCA basis at
  98× random (rank 8); DIM's selected 1-D direction is far less
  entangled (40×); RCO's optimized 2-D cone is essentially orthogonal
  to it (1.5×). Behaviorally, Qwen3Guard ASR ranks the methods
  *RCO 0.93 > DIM 0.90 > ActSVD 0.77*, with rank-matched
  random-direction and random-2-D-subspace baselines staying at the
  base-model floor. We extend the adversarial-suffix probe of
  DIM from a single GCG suffix on Qwen 1.8B to
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

#citet("arditi2024") show that for several common chat models, a manipulation of
a single dimensional subspace is enough to both induce refusal of non-harmful
requests, and turn off the refusal of harmful requests. They use a procedure
called *difference-of-means* to identify this subspace. This technique
measures the average activations at each token position of each transformer
layer, compared between a set of harmful requests and a set of harmless
requests. Once these difference-of-means vectors are constructed, they score
each for its ability to cause the model to refuse harmless requests and respond
to harmful requests. The vector with the highest score is normed and selected as
the refusal dimension vector, denoted by $hat(r)$. This refusal dimension can
then be ablated in the residual stream at inference time to bypass refusal
behavior. This vector can also be used to create a jailbroken model by updating
model weights according to the following formula:
$W'_("out") <- W_("out") - hat(r)hat(r)^T W_("out")$. They find that this attack
method is successful on the Qwen model family with and without the system
prompt, and on the Llama model family without the system prompt, and has little
effect on model coherence.

#v(1em)

#citet("Wei2024Brittleness") introduce low-rank decomposition methods designed
to identify specific ranks within a weight matrix related to given LLM
behaviors. Their ActSVD algorithm performs Singular Value Decomposition on the
product of the model weights and input activations ($W X_"in"$), and yields an
orthogonal projection matrix ($Pi$).

Removing the top safety-critical ranks ActSVD identifies causes the model to
completely stop rejecting unsafe prompts, and the model's utility is severely
compromised. These findings suggest that safety regions in aligned models are
also crucial for its general utility. To disentangle safety from utility, the
authors remove safety ranks orthogonal to utility ranks using
$Delta W = (I - Pi^u) Pi^s W$. This yields higher attack success rate for unsafe
prompts while maintaining zero-shot accuracy for utility prompts.

The fact that naively ablating safety ranks destroys utility, whereas
surgically removing disentangled ranks preserves it, indicates that top safety
ranks and top utility ranks heavily overlap. The necessity of this orthogonal
projection matrix provides strong evidence against the hypothesis of strict
linear separability between safety and utility. Ultimately, ActSVD provides
rank-level evidence for superposition: safety and utility share representational
capacity and are not linearly distinct.

#v(1em)

#citet("Ponkshe2026Safety") demonstrates fundamental mathematical limitations of linear subspace-based
safety defenses, arguing that safety is not linearly separable from utility. Removing
any specific safety-related subspace inherently degrades the overall performance of the model.
The authors support this hypothesis through a series of empirical evaluations. Using
singular value decomposition and mode subspace overlap, the study reveals that the
principal directions amplifying safe behaviors also amplify useful ones, indicating that
these directions do not constitute a distinctly separable safety subspace. Furthermore,
attempts to mitigate harmful behaviors via orthogonal projection resulted in a proportional
drop in the model's utility. The researchers also found that providing the model with helpful
and harmful inputs produced highly overlapping activations. Collectively, these findings further
challenge the hypothesis that safety and utility operate within linearly separable subspaces.

#v(1em)

#citet("pmlr-v267-wollschlager25a") generalize the identification of safety
subspaces to conic regions of multiple basis refusal vectors as opposed to one
refusal direction. Instead of testing pairs of harmful and harmless prompts,
their methods of Refusal Direction Optimization and Refusal Cone Optimization
perform gradient descent to converge on refusal vector direction(s). They
leverage two properties of ideal refusal vectors in loss functions for
optimization:

- Given a refusal direction $r$, scalar $alpha$, initial activation $x_i$, and
  revised activation $caron(x)_i = x_i + alpha dot r$, refusal probability
  should scale with $alpha$.
- Removing the refusal direction should not affect harmless prompts while
  allowing response to harmful prompts.

This research finds significant jailbreaking performance gains using one refusal
direction, with further gains up to a four-dimensional refusal region. Testing
on Gemma-2, Llama 3, and Qwen 2.5 model families and benchmarks such as
TruthfulQA, ablation of multiple refusal vectors is shown to have better attack
success and lower side-effects on model performance.

Furthermore, noting that vector orthogonality does not guarantee causal
independence, the research defines a notion of representational independence
between vectors, in which the ablation of one direction does not impact the
effects of the other. Ablating three or more representationally independent
refusal vectors is found to have higher attack success than difference-of-means
direction ablation. This further shows that safety and utility occupy a complex
subspace within LLMs.

= Methodology

Targeting Llama-3.1-8B-Instruct, we compare three methods for extracting safety subspaces:

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { gray.lighten(80%) },
    [*Method*], [*Operates on*], [*Intervention*],
    [DIM], [Residual stream activations], [Inference-time ablation by 1 direction],

    [ActSVD], [Weight matrices and activations per linear layer], [Removal of safety ranks from weight matrix],

    [RCO], [Residual stream activations], [Inference-time ablation by subspace of 2 basis vectors],
  ),
  caption: [The three methods operate at different levels of the model but target the same behavioral outcome (removing refusal).],
)

== Safety Subspace Identification

We implement three complementary methods for extracting safety-relevant
subspaces from the model's internal representations.

*Difference-in-Means (DIM).* Following #citet("arditi2024"), we compute the mean
residual stream activations at each layer $l$ and post-instruction token
position $i$ for sets of harmful and harmless prompts. The difference-in-means
vector is $bold(r)_i^((l)) = bold(mu)_i^((l)) - bold(v)_i^((l))$, where
$bold(mu)$ and $bold(v)$ are the mean activations over harmful and harmless
prompts, respectively. We select the single most effective vector $hat(bold(r))$
by evaluating each candidate's ability to bypass refusal when ablated and to
induce refusal when added. The selected unit-norm vector defines a
one-dimensional safety subspace.

*ActSVD Safety and Utility Ranks.* Following #citet("Wei2024Brittleness"), we perform
Singular Value Decomposition on the product of model weights and input
activations $W X_"in"$ for both safety and utility calibration datasets,
yielding $U S V^top approx W X_"in"$. The orthogonal projection matrices
$Pi^s = U^s (U^s)^top$ and $Pi^u = U^u (U^u)^top$ project onto the top $r^s$
safety and top $r^u$ utility rank subspaces, respectively. To disentangle
safety from utility, we compute the isolated safety projection:
$Delta W(r^u, r^s) = (I - Pi^u) Pi^s W$. While Wei et al. evaluate on
Llama-2 7B/13B, the method operates on generic linear layers and transfers
directly to Llama-3.1 8B.

*Refusal Cone Optimization (RCO).* Following #citet("pmlr-v267-wollschlager25a"), the
Geometry codebase can use gradient-based optimization to discover multiple
refusal directions that together form a multi-dimensional conic region. The
optimization minimizes a composite loss encoding two properties: (1) monotonic
scaling of refusal probability with the magnitude of activation addition, and
(2) surgical ablation that bypasses refusal on harmful prompts while preserving
behavior on harmless prompts. A retain loss based on KL divergence ensures
minimal side effects on harmless inputs. In our current results, full optimized
RCO training is treated as an extension path; we do, however, run the RepInd
profile test on DIM-derived directions.

== Subspace Comparison

Our comparison phase addresses two questions. First, _cross-method
consistency_: do DIM, ActSVD, and RCO converge on similar safety-relevant features,
or does each capture a distinct aspect of the safety mechanism?
Second, _safety--utility overlap_: how much does each layer's safety direction
lie inside a utility activation subspace? Third, _behavioral safety-utility
tradeoff_: after removing safety behavior, how much does each method degrade
useful behavior?

DIM and RCO operate in activation space, while ActSVD operates in
weight space. We bridge these two regimes in two complementary ways.
*Weight-delta MSO*: for each layer and weight type, we compute the
thin SVD of $Delta W = W_"actsvd" - W_"base"$ and project the DIM and
RCO directions onto its left singular basis. This is a *capacity*
measure: the realized perturbation $Delta W bold(x)$ depends on the
input distribution. *Activation-delta direction*: per-layer mean
activation deltas computed across 64 harmless prompts provide a direct
activation-space comparison between the base and ActSVD-modified models.

*Mode Subspace Overlap (MSO).* Following #citet("Ponkshe2026Safety"), MSO
measures the geometric overlap between two subspaces. For two matrices
$bold(V)$ and $bold(W)$, we extract their principal directions via thin SVD and
select the smallest number of left singular vectors capturing an
$eta$-fraction of the energy. The MSO metric is defined as:
$ "MSO"(bold(V), bold(W); eta) = (||S||_F^2) / min(k_V, k_W) $
where $S = Q_V^top Q_W$ is the overlap matrix between the orthonormal bases.
MSO ranges from 0 (orthogonal subspaces) to 1 (identical spans). We compute MSO
between the DIM refusal direction and ActSVD weight-delta subspaces to assess
cross-method agreement. We also compute MSO between DIM safety directions and
matched utility PCA subspaces to quantify safety--utility entanglement
directly. Because DIM yields a single direction while ActSVD yields
multi-dimensional subspaces, cross-method MSO is bounded by the dimensionality
asymmetry; we report the random baseline
$EE["overlap"] = max(k_V, k_W) slash d$ alongside each MSO value for
calibration.

*Representational Independence (RepInd).* Following
#citet("pmlr-v267-wollschlager25a"), RepInd tests whether two individual directions are
_causally_ related, not merely geometrically similar. Two directions
$lambda, mu in RR^d$ are representationally independent if ablating one does
not change the cosine similarity profile of the other across layers:
$
  forall l in L: cos(bold(x)^((l)), lambda) = cos(tilde(bold(x))_("abl"(mu))^((l)), lambda)
$
and vice versa. MSO may report high geometric overlap between directions that
turn out to be causally independent, or low overlap between directions that are
causally entangled via non-linear interactions across layers. In this codebase,
the cones-repind implementation can train RDO directions, orthogonal directions,
independent directions, and multi-vector refusal cones. Our project-owned
RepInd script measures a direction's layerwise cosine-similarity profile before
and after ablating another direction. Without completed optimized RCO artifacts,
we derive a small cone-like basis from high-norm DIM mean-difference candidates;
with trained RCO artifacts, the same script can compare DIM, RDO, RepInd, and
cone basis vectors directly.

== Direct Activation Comparison
This section describes our methodology for comparison and analysis of the hidden layer activations
produced by ActSVD and DIM jailbroken models. We compared the activations of these jailbroken model
both with each other, and with the base model which, consistent with our other experiments, is *LLaMA 3.1 8B Instruct*.
We compare the activations using both cosine similarity and euclidean distance in order to understand the changes
in both direction and magnitude.

= Experimental Settings

All experiments target Llama-3.1-8B-Instruct on a single A100 GPU. The full
pipeline is reproducible from the script entry points documented in `readme.md`.

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
- *DIM*: 128 harmful (AdvBench split) and 128 harmless prompts; layer $l_*=11$
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
    inset: 3pt,
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { gray.lighten(80%) },
    [*Method*], [*sub-ASR*], [*Q3G-ASR*], [*Harmless*], [*PPL Pile*], [*PPL Alpaca*],
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
  caption: [Cross-method geometry. Per-layer MSO to the ActSVD weight-delta column space, averaged over Q-proj, O-proj, and MLP-down. Layer~10 is the main averaged hotspot (DIM $0.042$, RCO $0.011$).],
) <fig_mso_per_layer>

In @fig_mso_per_layer, DIM-vs-ActSVD MSO stays near the random
baseline except for a layer~10 hotspot ($0.042$ averaged over Q-proj,
O-proj, and MLP-down). RCO-vs-ActSVD shows a weaker layer~10 hotspot
($0.011$). Separately, the direct DIM-vs-RCO activation-space
comparison gives a top principal-angle cosine of $0.505$. Individual
weight matrices produce larger peaks at the same layer: DIM-vs-ActSVD reaches
$0.070$ on MLP-down ($48×$ random), while RCO-vs-ActSVD reaches
$0.023$ on attention output ($16×$). We treat this as *exploratory
rather than conclusive*: the SVD bridge characterizes the column space
that ActSVD's edit *can* perturb, not its realized effect
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
  caption: [Safety-utility MSO at rank 8 for each method's safety object. RCO uses the normalized 2-D subspace MSO formula $||U^top Q_S||_F^2 / min(k_S, k_U)$ with baseline $max(k_S, k_U)/d = 8/4096$.],
) <tab_safety_utility>

The full DIM mean-difference stack has MSO 0.191 (98× random), consistent
with the claim of #citet("Ponkshe2026Safety") that safety is broadly
entangled with utility. The DIM-selected direction (40× at layer~11,
8.4× averaged) occupies a substantially less entangled region of the
same space, and RCO's optimized 2-D cone has normalized subspace MSO
of just 0.0030 (1.5× random) at rank 8 --- essentially orthogonal to
the utility PCA basis, only modestly above the normalized random
baseline at the same rank. RCO ablates more dimensions than DIM yet preserves
perplexity at least as well, indicating that the dimensions it ablates
contain less utility-relevant content. The gap between the full
mean-difference stack and the selected direction partly reflects an object-level
asymmetry: the full measure averages all per-layer candidates,
including high-overlap early layers, while the selected direction is
KL-filtered for intervention. The method-level comparison (DIM single
direction at 8.4× vs.~RCO 2-D cone at 1.5×) uses the same rank-$k=8$
utility basis and dimension-aware normalization. We
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
  image("figures/probe_ablation_cross_test.png", width: 100%),
  caption: [*Ablation cross-test:* same probe prompts under base vs each
    ablated model. Both DIM (1-D direction ablation) and RCO (2-D cone subspace
    ablation) are run as separate conditions on the *same* prompts. If RCO's
    cone covers refusal directions DIM misses, RCO-ablated ASR should be at
    least as high as DIM-ablated ASR on every group.],
) <fig_probe_abl>

The probe metric for RCO is the L2 norm of the EOI activation's
projection into the optimized 2-D cone subspace, $"proj"_("RCO") = norm(
  B^top
  bold(h)
)_2$, which is the natural multi-dimensional analog of DIM's signed
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

== Self-Consistency

#figure(image("./figures/dim_self_consistency.png"), caption: "Difference-in-Means Self-Consistency")

The figure above shows the cosine similarity of the difference-in-means candidate vectors when trained on the reference dataset vs when trained on TwinPrompt #cite("twinbreak").
The fact that the similarity is non-uniform in part reflects that these two datasets are simply different; we don't expect all the DIM candidates all to reflect safety.
However, if the DIM process truly identifies a single universal safety vector for the model, we should expect perfect agreement on that subspace.
The reference dataset identifies the vector at layer 11 as the safety vector, and the graph above shows that the cosine similarity at layer 11 is 0.7126. This is not perfect agreement but fairly close.
The highest agreement is at layer 12 with a cosine similarity of 0.7303.

=== Layer-wise Activation Divergence: ActSVD vs. Diff-in-Means

#figure(
  image("figures/DIM_ActSVD_act_comp.png", width: 100%),
  caption: [Cosine similarity and Euclidean distance between ActSVD and DIM jailbroken model activations],
) <DIM_ActSVD_act_comp>

The first comparison examines how the two editing methods differ from each other across all 33
layers (embedding layer + 32 transformer blocks). In both cosine similarity and Euclidean
distance, we observe that the two modified models produce clearly different internal
representations. Harmful prompts are slightly more different than helpful, but not by a large margin.
These findings are counter to what we would expect if these methods are truly only affecting safety, which
would imply that each model would act similarly to the base model when prompted on helpful prompts.

=== Layer-wise Activation Divergence: Base vs. Modified Models

#figure(
  image("figures/Base_act_comp.png", width: 100%),
  caption: [Cosine similarity and Euclidean distance between ActSVD and DIM jailbroken model activations and the Base model.
    Note the different Y axis scaling between the graphs],
) <Base_act_comp>

In the second set of comparisons (Base vs. ActSVD and Base vs. DIM), we see that ActSVD
changes the base model much more than DIM, especially for helpful prompts. If a method is only
affecting a safety subspace, we would expect the Base and the modified model to be very similar
on helpful prompts while being different on Harmful prompts.

For ActSVD, this is not the case at all. We see that both harmful and helpful prompts are significantly
affected by the modification, indicating it is also affecting utility along with safety.

For DIM, helpful prompts _are_ affected much less than harmful ones, but they are still
significantly affected again indicating it is also affecting utility along with safety.

= Discussion

Our work in this paper has shown some promising new leads.
Through our comparison of the overlap of the subspaces generated by difference-in-means (DIM), ActSVD, and Refusal Cone Optimization (RCO), we have shown the existence of multiple seperate safety subspaces.
This shows that #citeg("arditi2024") titular claim that "Refusal in Language Models Is Mediated by a Single Direction" is inaccurate.
On one hand, this is promising for LLM safety as it shows that safety mechanisms are distributed throughout the model.
One the other hand, the fact that any one of these methods can be successful without touching the other methods safety subspaces shows that these safety mechanisms are brittle.
They do not function as failsafes for one another.

As far as implications for the field, the fact that methods like we have described here work so well should be concerning.
If our goal as a field is to release models that cannot be trivially modified to behave in unsafe ways, the success of a simple method like difference-in-means shows we still have a long way to go.
From a safety perspective, it is desirable for safety and utility to be entangled.
Models where safety and utility are seperable are inherently vulnerable to these kinds of attacks.
Therefore future work in jailbreak resistance research should specifically aim to entangle safety and utility, such that any attempt to make the model respond to unsafe requests makes the model useless.


== Limitations

Our work has some limitations.

- *Only one model family:* All of our experiments use Llama-3.1-8B-Instruct. Our results may not generalize to other families of models such as Qwen or Deepseek.
- *Only one model size*: We used an 8B parameter model for all our experiments. It may be the case that larger models have more deeply intertwined safety subspaces.


= Conclusion

In this paper we have replicated the results of multiple state-of-the-art jailbreak methods.
We have compared the safety subspaces generated by each of these methods and shown that they do not entirely overlap.
This shows that while safety may be distributed throughout multiple subspaces, this does not lead to a more robust safety mechanism.

= Member Contributions

All five authors contributed to writing and research-design discussion.
Effort (totaling 100%):
*Evan Scamehorn (20%)* --- DIM/ActSVD Colab pipelines and behavioral benchmark, planning and git maintenance.\
*Adam Venton (20%)* --- safety-utility overlap, activation comparison experiment.\
*Calvin Kosmatka (20%)* --- literature survey, self-consistency experiment for DIM.\
*Kyle Sha (20%)* --- prompt-attack probe (incl.\ layer sweep + ablation
cross-test), WildJailbreak integration, Discussion.\
*Zeke Mackay (20%)* --- RepInd analysis, RCO training, asymmetry interpretation, writing of data analysis in report.\

#pagebreak()

#add-bib-resource(read("bibliography.bib"))
#print-acl-bibliography()
