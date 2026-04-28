
// Final report for DL4NLP project.
// Builds on report.typ but reframes the narrative around the core tension
// between the reference papers' conflicting claims about safety-utility separability.

#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.1": *

#show figure: set text(size: 0.9em)
#show figure: set par(justify: false)
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
  Recent work on jailbreaking aligned LLMs has produced conflicting claims about
  whether safety can be cleanly separated from utility. Difference-in-Means (DIM)
  claims a single direction mediates refusal with minimal utility cost. ActSVD
  finds safety and utility ranks overlap, requiring orthogonal projection to
  disentangle them. The Safety Subspaces paper argues no selective removal is
  possible. The Geometry paper shows refusal is multi-dimensional, not
  one-dimensional. We investigate these claims by reproducing DIM, ActSVD, and
  Refusal Cone Optimization (RCO) on LLaMA-3.1-8B-Instruct and comparing them
  at three levels: behavioral (JailbreakBench ASR, harmless compliance,
  perplexity), geometric (Mode Subspace Overlap between safety directions and
  utility PCA subspaces), and causal (Representational Independence profiles).
  We find that the full per-layer safety subspace shows substantially
  above-random overlap with a PCA-based utility basis (mean MSO
  0.191 vs.\ random baseline 0.00195), while
  each method's _selected_ direction has lower overlap (DIM:
  0.078, RCO: 0.004). We
  propose that this gap may partly explain why effective safety removal is
  behaviorally possible despite broad geometric entanglement, though we note
  that PCA overlap is not a direct measure of causal utility dependence.
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
inseparability. If DIM is correct, a single targeted defense could block
safety-removal attacks. If the Safety Subspaces paper is correct, safety removal
inevitably degrades the model. These are materially different conclusions for
alignment.

== Research Questions

We investigate whether these claims can be reconciled empirically by running all
three methods on the same model and asking:

+ *Cross-method agreement.* Do DIM, ActSVD, and RCO converge on the same
  geometric structure, despite operating at different levels (activation space
  vs.\ weight space)?

+ *Safety-utility entanglement.* How much does each method's safety direction
  overlap with the model's utility activation subspace? Is the _full_ safety
  subspace entangled even if individual _selected_ directions are not?

+ *Behavioral tradeoff.* When each method removes safety, how much utility does
  it preserve? Does geometric overlap predict behavioral utility cost?

+ *Causal independence.* Are multiple refusal directions causally independent,
  or does ablating one change the effect of another?

== Key Insight

We hypothesize that the four papers may be simultaneously correct, because they
measure different things. The _full_ safety subspace (e.g., DIM's per-layer
mean-diffs) may be entangled with utility. But each method's _selection
procedure_ finds a surgical direction within that entangled space that has
lower utility overlap. DIM selects the direction with minimum KL divergence on
harmless prompts; ActSVD explicitly orthogonalizes safety against utility;
RCO's loss function includes a retain term penalizing harmless-prompt
disruption. All three procedures implicitly optimize for low safety-utility
overlap, which may explain why the resulting interventions preserve utility
despite broad entanglement.

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
close to the territory of refusal."

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
    [RCO], [Residual stream activations], [Inference-time ablation (2-D cone)],
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
projection $W' = W - (I - Pi^u) Pi^s W$ with utility rank $r^u = 3000$ and
safety rank $r^s = 4000$.

*RCO.* Following #citet("pmlr-v267-wollschlager25a"), we initialize from the
DIM direction and optimize a 2-dimensional refusal cone via gradient descent.
The loss combines refusal-scaling (refusal probability should increase when the
direction is added) and surgical-ablation (removing the direction should bypass
refusal without affecting harmless prompts) terms, with a KL-divergence retain
loss.

== Bridging Activation Space and Weight Space

The central methodological challenge is comparing methods that produce different
types of objects. DIM and RCO produce vectors in activation space ($RR^d$);
ActSVD produces modified weight matrices. We bridge this gap in two ways:

*Weight-delta MSO.* For each layer and weight type, we compute
$Delta W = W_"actsvd" - W_"base"$ and take its thin SVD. The left singular
vectors $U_B$ span the output-space subspace that ActSVD modified. Because the
output space of a weight matrix IS the residual stream (activation space), we
can project DIM/RCO directions onto $U_B$:
$ "MSO"("DIM vs ActSVD") = ||U_B^T hat(bold(r))||^2 $
This measures whether the DIM/RCO direction lies within the subspace of
output directions that ActSVD's edit _could_ produce changes along. However,
the actual activation change for a given input $bold(x)$ is $Delta W bold(x)$,
which depends on the input distribution. High MSO therefore indicates that
the edit has the _capacity_ to affect that direction, not that it necessarily
does so on real prompts. The activation-delta bridge below provides a
complementary empirical measure.

*Activation-delta direction.* We run 64 harmless prompts through both the base
and ActSVD-modified model, collect per-layer mean activations, and compute
$delta_l = bold(mu)_"modified"^((l)) - bold(mu)_"base"^((l))$. This converts
ActSVD's weight-space change into an activation-space vector, enabling direct
comparison with DIM and RCO on the safety-utility overlap analysis. This
captures the mean empirical effect but compresses it into a single vector,
hiding prompt-dependent variation and nonlinear downstream interactions.

== Utility Subspace Construction

We define the utility subspace via PCA on harmless instruction activations:

+ Collect residual-stream activations from 128 harmless prompts at
  end-of-instruction token positions.
+ Center per layer (subtract mean).
+ Compute SVD; top-$k$ right singular vectors form an orthonormal basis.

The MSO between a safety direction $hat(bold(s))$ and the rank-$k$ utility basis $Q_k$ is:
$ "MSO"(hat(bold(s)), Q_k) = ||Q_k^T hat(bold(s))||^2 $
The random baseline is $k slash d$.

_Interpretation note:_ PCA captures directions of large _variance_ in harmless
activations, which is a proxy for --- but not identical to --- directions
causally responsible for utility. A low MSO with utility PCA suggests low
overlap with the dominant harmless-activation basis, but does not prove that
ablating the safety direction preserves utility. The behavioral benchmark
(perplexity, harmless compliance) provides the actual test of utility damage;
the MSO analysis provides geometric context for interpreting those behavioral
results.

== Comparison Framework

=== Behavioral Benchmark

We evaluate each method on three metrics using the same evaluation harness:
- *JailbreakBench ASR* #cite("jailbreakbench"): fraction of 100 harmful prompts
  where the model does not refuse after intervention.
- *Harmless compliance*: fraction of 100 harmless prompts answered correctly.
- *Perplexity*: next-token NLL on Pile #cite("thepile") and Alpaca #cite("alpaca") samples.

=== Geometric Comparison

- *Cross-method MSO*: DIM-vs-ActSVD and RCO-vs-ActSVD weight-delta MSO per
  layer and weight type.
- *Direction cosine similarity*: DIM-vs-RCO (both in activation space).
- *Safety-utility MSO*: each method's direction projected onto the utility PCA
  subspace at multiple ranks (1, 2, 4, 8, 16, 32).

=== Causal Comparison (RepInd)

Following #citet("pmlr-v267-wollschlager25a"), we measure layerwise cosine
similarity profiles before and after ablating each direction. Lower profile
change indicates greater representational independence.

= Results

== Behavioral Benchmark

#figure(
  image("figures/benchmark_safety_utility_tradeoff.png", width: 85%),
  caption: [Safety-utility tradeoff across all methods. DIM and RCO achieve high ASR with minimal utility cost; ActSVD achieves moderate ASR with higher perplexity degradation.],
) <fig_benchmark>

@fig_benchmark shows the behavioral comparison across all methods.

#figure(
  table(
    columns: (2fr, 1fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: (left, center, center, center, center),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { gray.lighten(80%) },
    [*Method*], [*ASR*], [*Harmless Compl.*], [*PPL (Pile)*], [*PPL (Alpaca)*],
    [Base], [0.15], [1.00], [13.93], [8.60],
    [DIM-Ablated], [1.00], [1.00], [14.17], [8.80],
    [ActSVD-Modified], [0.62], [1.00], [19.91], [10.61],
    [RCO-Cone-2], [1.00], [1.00], [13.95], [8.63],
  ),
  caption: [Behavioral benchmark results. Higher ASR = more safety removed. Higher harmless compliance = better utility preservation. Lower perplexity = better language modeling quality.],
) <tab_benchmark>

DIM achieves the highest ASR (1.00) while maintaining harmless
compliance at 1.00 and producing small perplexity changes. ActSVD
achieves moderate ASR (0.62) but causes larger perplexity
degradation (19.91 vs.\ baseline 13.93).
RCO achieves ASR of 1.00 with the smallest perplexity increase of any
intervention method (13.95, barely above baseline), consistent with its explicit
KL-divergence retain loss.

The original ActSVD paper #cite("Wei2024Brittleness") similarly reports vanilla
ASR of 0.67--0.71 on Llama-2-7B-chat (reaching $approx$1.0 only with adversarial
suffixes), confirming that weight-surgery approaches achieve lower vanilla ASR
than activation-space methods. Our more aggressive rank settings ($r^u=3000$,
$r^s=4000$, effective $Delta W$ rank $approx$1000) compared to their optimal
setting ($r^u=3950$, $r^s=4090$, effective rank $approx$6) likely contribute to
the larger perplexity degradation we observe. They did not report perplexity,
measuring utility only via zero-shot task accuracy, which may be less sensitive
to the generation quality degradation that perplexity captures.

== Cross-Method Geometric Agreement

#figure(
  image("figures/subspace_mso_per_layer_avg.png", width: 90%),
  caption: [Per-layer MSO between DIM/RCO directions and ActSVD weight-delta subspaces. Most layers are near the random baseline, indicating that activation-space and weight-space methods target geometrically distinct structures.],
) <fig_mso_per_layer>

DIM-vs-ActSVD MSO is near the random baseline for most layers (@fig_mso_per_layer).
This indicates that the DIM direction does not lie within the column space of
ActSVD's weight changes at most layers. However, layer 10's MLP down-projection
is a notable exception, with MSO of 0.113 (7$times$ the random baseline of
0.016). Layer 14 also shows mild elevation. This suggests that while the two
methods target largely distinct geometric structures, they converge on a shared
component in a small number of layers --- particularly around layers 10--14,
which are near DIM's selected layer (11). Low MSO elsewhere does not rule out
mechanistic overlap: the actual activation-space effect of ActSVD's weight edit
depends on the input distribution, and two methods can achieve similar behavioral
outcomes through partially distinct geometric structures. DIM-vs-RCO
cosine similarity is 0.450, indicating moderate agreement: the gradient-optimized direction moved substantially from DIM's statistical estimate, consistent with the loss landscape having a meaningfully different optimum.

== Safety-Utility Overlap: The Central Finding

#figure(
  image("figures/safety_utility_overlap_per_layer.png", width: 90%),
  caption: [Per-layer safety-utility MSO for all methods at rank 8. The full DIM safety subspace (bars) shows substantial above-random overlap. Individual method directions (lines) have varying overlap, with lower values for the selected DIM direction.],
) <fig_safety_utility_per_layer>

#figure(
  image("figures/safety_utility_overlap_by_rank.png", width: 75%),
  caption: [Mean safety-utility MSO as the utility PCA rank increases. All curves remain above the random baseline (dashed line).],
) <fig_safety_utility_by_rank>

This is the central geometric result. At rank 8:

#figure(
  table(
    columns: (2.5fr, 1.2fr, 1.2fr),
    inset: 5pt,
    align: (left, center, center),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { gray.lighten(80%) },
    [*Direction*], [*MSO (rank 8)*], [*vs.\ Random*],
    [Full DIM mean-diffs (averaged)], [0.191], [98×],
    [DIM selected direction (layer 11)], [0.078], [40×],
    [RCO direction], [0.004], [1.8×],
    [ActSVD activation delta], [0.124], [63.5×],
    [Random baseline], [0.00195], [1×],
  ),
  caption: [Safety-utility MSO at rank 8 for each method's direction. The full safety subspace is substantially entangled, but selected directions have lower overlap.],
) <tab_safety_utility>

The full DIM safety subspace has MSO of 0.191 --- approximately
98 times the random baseline. This is consistent with
#citet("Ponkshe2026Safety")'s claim that safety is entangled with utility. But
the DIM _selected_ direction has MSO of only 0.078 ---
substantially lower.

_Dimensionality caveat:_ The "full-subspace mean MSO" averages 32 per-layer
mean-diff directions, each independently projected onto its layer's utility PCA
basis. A single direction will naturally tend to have lower overlap with a rank-$k$
subspace than the average over many directions, so the gap between 0.191 and 0.078
is suggestive but may partly reflect this dimensionality asymmetry rather than a
property of DIM's selection procedure. Nevertheless, the behavioral results
(@tab_benchmark) provide independent confirmation: DIM's selected direction
empirically preserves utility better than the average safety direction would
predict.

The behavioral results are broadly consistent with this geometric picture. DIM,
which has the lowest selected-direction MSO, achieves the best utility preservation.
ActSVD, which operates in weight space and must explicitly disentangle via
$(I - Pi^u) Pi^s W$, shows more utility degradation. However, we emphasize
that the geometric MSO and the behavioral metrics measure different things, and
MSO should not be treated as a sufficient predictor of utility cost.

== Representational Independence

#figure(
  image("figures/repind_change_heatmap.png", width: 70%),
  caption: [RepInd profile-change matrix. Rows: ablated direction. Columns: measured direction. Lower off-diagonal values indicate greater representational independence.],
) <fig_repind>

The RepInd analysis reveals an asymmetric causal structure. Ablating the DIM
direction substantially changes derived candidate profiles (mean |Δ| =
0.095), while ablating derived candidates barely
changes the DIM profile (0.062). This indicates that
DIM captures a dominant refusal component that causally influences other
directions, but not vice versa --- consistent with
#citet("pmlr-v267-wollschlager25a")'s finding that multiple independent refusal
mechanisms exist, with a hierarchy of causal influence.

= Discussion

== Reconciling the Conflicting Claims

Our results are consistent with the following interpretation, which we advance as
a hypothesis rather than a proven conclusion. The apparent contradiction between
the reference papers may partly arise from measuring different quantities:

+ *DIM's "clean separation" claim* is supported by behavioral results: the
  _selected_ direction achieves high ASR with minimal utility cost. However, this
  direction is chosen by a selection procedure that filters by KL divergence on
  harmless prompts, which may implicitly select for low utility entanglement.

+ *The Safety Subspaces "no separation" claim* is consistent with our geometric
  finding that the _full_ per-layer safety subspace has above-random overlap
  with utility PCA bases. However, our PCA-based utility definition is a proxy;
  high PCA overlap does not prove that ablation would degrade utility, only
  that the directions are not orthogonal to the dominant harmless-activation basis.

+ *ActSVD's "separable with effort" claim* is consistent with both: the raw
  safety ranks overlap with utility (motivating the orthogonal projection), and
  after disentanglement a surgical cut is possible, albeit with more utility
  degradation than DIM.

+ *The Geometry paper's multi-dimensionality claim* is supported by the RepInd
  asymmetry: the DIM direction appears dominant but not exhaustive, and derived
  candidate directions show partial independence. However, our candidates are
  DIM-derived rather than fully optimized RepInd vectors, limiting the strength
  of this conclusion.

Some of the original disagreement across papers may also reflect genuine
differences in models, datasets, or evaluation definitions rather than
contradictory truths about a single underlying phenomenon.

== Why Different Methods Find Different Structures

DIM-vs-ActSVD MSO is near random baseline for most layers. This does not
necessarily mean the methods target unrelated phenomena. ActSVD and DIM may
achieve similar behavioral outcomes through partially distinct mechanisms;
the low MSO tells us that the DIM direction does not lie in the column space of
ActSVD's weight changes, but the actual effect of those weight changes also
depends on the input distribution, which the SVD bridge does not capture.

The behavioral convergence (both increase ASR) despite geometric divergence
(low MSO) could indicate either that (a) different geometric interventions
can disrupt the same downstream refusal computation, or (b) the methods
are disrupting different pathways that independently contribute to refusal.
Our data does not conclusively distinguish these possibilities.

== Limitations

- All experiments use a single model (LLaMA-3.1-8B-Instruct). Generalization
  to other model families and sizes is untested. Some of the reference papers'
  disagreements may reflect genuine model-level differences.
- *PCA utility $eq.not$ causal utility.* The utility subspace is defined by PCA
  on harmless activations, which captures dominant _variation_ directions.
  Directions causally responsible for utility may differ from high-variance
  directions. Low MSO with utility PCA does not prove low utility damage.
- *SVD bridge measures capacity, not effect.* The weight-delta MSO measures
  whether a direction lies within the column space of $Delta W$, but the actual
  activation change is $Delta W bold(x)$, which depends on input distribution.
  High MSO does not guarantee the edit actually affects that direction on real
  prompts.
- *Dimensionality confound in full-vs-selected MSO comparison.* Averaging MSO
  over 32 per-layer directions and comparing against a single selected direction
  confounds the effect of selection with the natural tendency of single directions
  to have lower overlap than direction averages.
- *Activation-delta compression.* The ActSVD delta-activation vector averages
  over prompts, hiding prompt-dependent effects, variance changes, and nonlinear
  downstream interactions.
- *Geometric overlap $eq.not$ causal mechanism.* Directions can be geometrically
  close but causally irrelevant, or geometrically distant but behaviorally
  equivalent. Our geometric analyses provide suggestive evidence, not causal proof.
- RCO training uses a 2-dimensional cone. The Geometry paper finds cones up to
  dimension 5; higher-dimensional cones might reveal additional structure.
- RepInd is evaluated using DIM-derived cone-basis candidates rather than fully
  optimized RepInd vectors, limiting the causal independence conclusions.

= Conclusion

We reproduce DIM, ActSVD, and RCO on LLaMA-3.1-8B-Instruct and compare them at
behavioral, geometric, and causal levels. We find that the per-layer DIM safety
subspace shows substantially above-random overlap with PCA-based utility
subspaces, consistent with #citet("Ponkshe2026Safety")'s entanglement claim. At
the same time, each method's selected direction has lower overlap with utility,
and the behavioral benchmark confirms that effective safety removal with
preserved utility is empirically achievable, consistent with #citet("arditi2024").

We propose that this gap between full-subspace and selected-direction overlap may
partly explain the apparent contradiction in the literature, though we
acknowledge that PCA overlap is a proxy for causal utility dependence, and that
the dimensionality difference between the comparisons introduces a confound.
The behavioral benchmark, not the geometric analysis, is the definitive test of
utility preservation.

DIM-vs-ActSVD geometric overlap is near random, indicating that activation-space
and weight-space methods may achieve similar behavioral outcomes through
partially distinct mechanisms. The RepInd analysis suggests that refusal is not
exhausted by a single direction, consistent with the multi-dimensional findings
of #citet("pmlr-v267-wollschlager25a"), though our DIM-derived candidates are
weaker surrogates for fully optimized independent directions.

These findings suggest that improving alignment robustness requires considering
the multi-dimensional nature of the safety subspace and its entanglement with
utility representations, while recognizing that the limits of linear
subspace-based analyses leave open the possibility of more complex, nonlinear
safety-utility interactions that our methods cannot capture.


= Data Sets

We use two primary evaluation datasets. JailbreakBench #cite("jailbreakbench")
provides 100 harmful prompts across multiple categories for measuring attack
success rate. The DIM pipeline's internal datasets provide paired harmful and
harmless instruction sets for computing mean-diff directions and evaluating
refusal behavior. Alpaca #cite("alpaca") provides harmless instruction data for
utility evaluation (harmless compliance and perplexity). The Pile
#cite("thepile") provides general text for perplexity measurement.

For ActSVD calibration, we use Alpaca-cleaned-no-safety (utility calibration)
and alignment SFT data (safety calibration), following
#citet("Wei2024Brittleness")'s methodology.


#pagebreak()

#add-bib-resource(read("bibliography.bib"))
#print-acl-bibliography()
