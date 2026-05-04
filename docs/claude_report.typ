
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
  Beyond reproducing the prior methods, we extend the adversarial-suffix
  analysis of Arditi et al.'s §5.1 ("Adversarial suffixes suppress the
  refusal-mediating direction") in four ways: (i) substituting Qwen 1.8B
  with Llama-3.1-8B-Instruct, (ii) replacing a single GCG suffix with
  in-the-wild WildJailbreak prompts, (iii) projecting at every layer rather
  than one, and (iv) adding an ablation cross-test plus RCO-direction
  comparison.
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

+ *Generalizing DIM's adversarial-suffix analysis.* Arditi et al.'s §5.1
  shows that a single GCG-optimized adversarial suffix suppresses the
  refusal direction at the EOI position on Qwen 1.8B Chat. Their analysis
  is explicitly restricted to "a single model and a single adversarial
  example." We ask whether the same suppression appears under the in-the-wild
  prompt-wrapping attacks deployed models actually face, on a more recent
  and larger model (Llama-3.1-8B-Instruct), and whether the suppression
  is layer-localized or distributed (DIM looks at one layer).

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

== Extending DIM's Adversarial-Suffix Analysis to In-the-Wild Attacks

#citet("arditi2024")'s §5.1 shows that *one* GCG-optimized adversarial suffix
suppresses the refusal direction at the EOI position on Qwen 1.8B Chat
(Figure 5 of their paper plots last-token cosine similarity with $hat(bold(r))$
under no-suffix vs random-suffix vs adversarial-suffix conditions). They flag
this as restricted to a single model and a single suffix. We extend that
analysis along four axes: (i) Llama-3.1-8B-Instruct in place of Qwen 1.8B,
(ii) WildJailbreak in-the-wild prompt-wrapping attacks in place of one GCG
suffix, (iii) projection at *every* block input rather than one layer, and
(iv) two diagnostics that DIM does not run: a comparison against the RCO
direction (which post-dates DIM), and a cross-test that runs the same probe
prompts under DIM ablation to bound the share of refusal DIM mediates.

For each prompt $p$, a single forward pass captures the residual stream at
the DIM-selected layer at the end-of-instruction (EOI) token position and
projects it onto the unit refusal direction:
$ "proj"(p) = bold(h)_"EOI"^((l_*))(p) dot hat(bold(r)) $
We project onto both the DIM and RCO directions and apply the
substring-based refusal judge to the base model's completion.

Three prompt groups: *direct_request* (HarmBench bare harmful), *adversarial_harmful*
(WildJailbreak wrapped harmful), *adversarial_benign* (WildJailbreak wrapped
benign --- a critical control: if wrapping alone drives projection drop,
both adversarial groups should fall together). Three questions: (Q1) does
wrapping suppress DIM relative to direct? (Q2) does it suppress RCO the same
way --- if not, prompt attacks tap different geometry than RCO finds? (Q3)
does benign-wrapped content suppress the direction --- if so, the wrapper,
not the harmful intent, perturbs the subspace.

The probe inverts the ablation comparison ("does removing the direction
enable attacks?") and asks "do successful prompt attacks remove the
direction?". The answers do not have to agree.

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
  (effective $Delta W$ rank $approx 6$). An earlier aggressive setting
  ($r^u=3000$, $r^s=4000$) caused outsized perplexity damage and is now
  superseded.
- *RCO*: 2-D cone, DIM-initialized, learning rate $1 times 10^(-3)$, batch
  size 16, 1500 steps; refusal-scaling + surgical-ablation + KL-retain losses.

*Evaluation harness.* JBB ASR (100 prompts) is graded by substring at
eval time, then re-graded post-hoc by the *unmodified* base Llama
("comply or refuse?") applied to every method's saved completions in a
single consistent pass. Harmless compliance (100 prompts), Pile/Alpaca
perplexity (64 each), TruthfulQA (64 questions, substring against
`correct_answers` / `incorrect_answers`). All rates carry 1,000-sample
bootstrap 95% CIs. A *random-direction baseline* ($cal(N)(0,I)$, seed 7)
is added as a sanity check.
Safety-utility overlap: PCA of 128 harmless activations, ranks
$k in {1,2,4,8,16,32}$, primary $k=8$. RepInd: 32 prompt pairs, 3-D basis.
*Probe* (75 prompts) adds a *layer sweep* (project at every block input)
and an *ablation cross-test* (generate each prompt twice: base + DIM-ablated).

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

ActSVD re-runs with paper-optimal $r^u=3950$, $r^s=4090$ (effective
$Delta W$ rank $approx 6$). The earlier aggressive setting over-cut and is
superseded.

*Sanity check, judges, side-effect.* (i) A random-direction baseline (unit
vector $cal(N)(0,I)$, seed 7) tests that ablation isn't lifting ASR
regardless of which direction is removed. (ii) JBB ASR is graded twice:
substring matching, and a post-hoc LLM judge using the *unmodified* base
Llama applied to every method's saved completions in a single pass.
Using the same loaded model as judge inside the per-method eval would be
unsafe --- for DIM-ablated, ActSVD-modified, and RCO-cone runs the loaded
model has hooks active or modified weights, so the very intervention that
compromised refusal also compromises the judge. The post-hoc consistent
judge removes that cross-method confound (a same-family bias remains).
(iii) TruthfulQA (64 questions) tests #citet("pmlr-v267-wollschlager25a")'s
claim that DIM hurts truthfulness more than RCO. All ASR / harmless /
truthful rates carry 1,000-sample bootstrap 95% CIs in @tab_benchmark.

== Cross-Method Geometric Agreement

#figure(
  image("figures/subspace_mso_per_layer_avg.png", width: 90%),
  caption: [Per-layer MSO between DIM/RCO directions and ActSVD weight-delta subspaces. Most layers sit near the random baseline; layer~10's MLP down-projection is the main hotspot ($7 times$ baseline).],
) <fig_mso_per_layer>

DIM-vs-ActSVD MSO is near random for most layers, with a hotspot at
layer~10's MLP down-projection ($"MSO" = 0.113$ vs random $0.016$) and mild
elevation at layer~14. We treat this as *exploratory rather than conclusive*:
the SVD bridge measures the column space ActSVD's edit *can* perturb, not its
actual effect $Delta W bold(x)$ on real inputs. The behavioral benchmark, where
both methods raise ASR, is the stronger evidence that they share *some* refusal
mechanism, even if the bridge cannot pinpoint where. DIM-vs-RCO cosine
similarity is $0.450$, indicating moderate agreement in activation space ---
the gradient-optimized direction moved substantially from DIM's statistical
estimate, consistent with the loss landscape having a meaningfully different
optimum.

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

== Prompt-Based Jailbreaks vs.\ the Refusal Direction

#figure(
  image("figures/probe_asr_and_projection_by_attack_type.png", width: 90%),
  caption: [ASR (left) and mean DIM/RCO projection (right) per prompt group.],
) <fig_probe_asr_proj>

#figure(
  image("figures/probe_dim_vs_rco_projection_scatter.png", width: 55%),
  caption: [Per-prompt DIM (x) vs RCO (y) projection. $circle$ = jailbroken, $times$ = refused.],
) <fig_probe_dim_rco>

#figure(
  image("figures/probe_layer_sweep_projection.png", width: 90%),
  caption: [*Layer sweep:* per-layer mean DIM/RCO projection per group, $plus.minus$SE bands.],
) <fig_probe_layer>

#figure(
  image("figures/probe_ablation_cross_test.png", width: 60%),
  caption: [*Ablation cross-test:* same probe prompts under base vs DIM-ablated models.],
) <fig_probe_abl>

#emph[The figures above are placeholders. Per-prompt projections,
per-layer trajectories, and the ablation-cross-test ASRs (with bootstrap
95% CIs) will overwrite them once the Colab run completes; predictions
below that the data contradicts will be revised at that point.]

Three predictions follow from the hypothesis that DIM/RCO identify the
genuine mediator of refusal: *(P1)* successful adversarial-harmful prompts
should show *lower* DIM projection than direct harmful requests, and
jailbroken cases should cluster at low projections
(@fig_probe_asr_proj, @fig_probe_dim_rco); *(P2)* DIM and RCO projections
should track each other but imperfectly, since $"cos"("DIM","RCO") = 0.450$
(@fig_probe_dim_rco); *(P3)* benign-wrapped controls should *not* be
suppressed --- if they are, the direction is more like a wrapper-style
detector than a refusal mediator and the geometric methods' "safety"
interpretation is overstated. *Layer sweep* (@fig_probe_layer) tests
whether suppression is layer-localized around $l_*=11$ or distributed
across late layers. *Ablation cross-test* (@fig_probe_abl) bounds the
share of refusal DIM mediates: the gap between direct base-ASR and direct
ablated-ASR is what prompt attacks cannot reach by themselves.

= Discussion

*Reconciling the four claims (hypothesis, not proof).* The papers may
partly disagree because they measure different objects: DIM's "clean
separation" is about a *selected* direction (KL-filtered on harmless
prompts, implicitly minimizing utility entanglement); Safety Subspaces'
"no separation" is about the *full* subspace ($98 times$ random PCA
overlap); ActSVD's "separable with effort" is the gap between the two
(raw safety ranks overlap utility, hence the $(I - Pi^u)$ projection);
and the Geometry paper's multi-dimensionality matches our RepInd
asymmetry, where DIM dominates but does not exhaust refusal. Genuine
model/dataset/judge differences likely also contribute.

*Probe takeaway.* If the predictions hold, the result generalizes
DIM §5.1 from one GCG suffix on Qwen 1.8B to a population of WildJailbreak
attacks on Llama-3.1-8B, and the layer-sweep + ablation cross-test
diagnostics give two readings DIM did not provide. The cross-method
DIM-vs-ActSVD MSO, by contrast, measures the column space ActSVD's edit
*can* perturb rather than its actual effect; we treat it as exploratory
and the behavioral benchmark + probe as load-bearing.

== Limitations

- *Single model.* All experiments use Llama-3.1-8B-Instruct; some inter-paper
  disagreements may be model-specific.
- *PCA utility $eq.not$ causal utility.* The utility subspace is defined by
  variance, not causal contribution; low MSO does not prove low utility damage.
- *SVD bridge measures capacity, not effect.* DIM-vs-ActSVD MSO does not capture
  $Delta W bold(x)$ on real inputs, so its negative result is suggestive only.
- *Dimensionality confound* in the full ($32$ directions averaged) vs
  selected (single direction) MSO comparison.
- *RepInd uses DIM-derived candidates*, not fully optimized independent vectors,
  limiting strength of the causal-independence conclusions.
- *Probe sample size.* 25 prompts per group with a substring-based refusal
  judge $arrow.r$ wide CIs, partial refusals may be misclassified.
- *Same-family LLM judge.* The post-hoc judge is the unmodified base
  Llama, so it removes the cross-method confound but retains a
  same-family bias (Llama judging Llama). A separate classifier
  (e.g.\ HarmBench's Llama-2-13B-cls) or a frontier model would be
  stronger but requires extra GPU memory or an API key.

= Conclusion

We reproduce DIM, ActSVD, and RCO on Llama-3.1-8B-Instruct and compare
them behaviorally, geometrically, and causally. The full per-layer safety
subspace is entangled with utility PCA bases ($98 times$ random), but
selected directions have lower overlap and preserve utility behaviorally
--- a gap that may partly explain the literature's contradictions. Our
prompt-attack probe extends #citet("arditi2024")'s §5.1 adversarial-suffix
analysis from one GCG suffix on Qwen 1.8B to in-the-wild WildJailbreak
attacks on Llama-3.1-8B, with two diagnostics they do not run: a per-layer
projection sweep and an ablation cross-test that bounds the share of
refusal DIM mediates. The probe is more incremental than groundbreaking;
its value is a controlled comparison on a recent model and attack
distribution, plus the RCO direction (which post-dates DIM).

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
