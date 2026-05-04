#import "@preview/typslides:1.3.2": *
#import "@preview/fontawesome:0.5.0": *

#let github-tag(repo) = box(
  fill: gray.lighten(80%),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
  radius: 8pt,
  [#fa-github() #link("https://github.com/" + repo, repo)],
)

#show: typslides.with(
  ratio: "16-9",
  theme: "bluey",
  font: "Fira Sans",
  font-size: 20pt,
  link-style: "color",
  show-progress: true,
)

#front-slide(
  title: "Comparing Safety-Removal Subspaces in Aligned LLMs",
  subtitle: [_Final Presentation_],
  authors: "Group 6: Evan Scamehorn, Kyle Sha, Adam Venton, Zeke Mackay, and Calvin Kosmatka",
  info: [#link("https://github.com/evan203/nlp-project-proposal")],
)

// -------------------------------------------------------------------------
// Slide 1: Motivation — the contradiction in the literature
// -------------------------------------------------------------------------
#slide[
  *Motivation: four papers, four different stories about safety vs utility*

  - *DIM* (Arditi et al. '24): a *single direction* mediates refusal — ablate it, attacks succeed, utility intact.
  - *ActSVD* (Wei et al. '24): safety and utility *share ranks* — must orthogonalize to disentangle.
  - *Safety Subspaces* (Ponkshe et al. '25): "*no selective removal* is possible" — every gain in safety costs utility.
  - *Geometry of Refusal* (Wollschläger et al. '25): refusal is a *multi-dimensional cone*, not one direction.

  #line(length: 100%)

  *If DIM is right*, a single defense blocks all attacks. *If Safety Subspaces is right*, safety removal is fundamentally lossy. These are very different conclusions for alignment.

  #v(0.4em)
  *Question:* Can these claims be reconciled by running them on the *same model*?
]

// -------------------------------------------------------------------------
// Slide 2: Research questions
// -------------------------------------------------------------------------
#slide[
  *Research questions*

  + *Cross-method agreement.* Do DIM, ActSVD, and RCO converge on the same geometric structure?

  + *Safety–utility entanglement.* How much does each method's safety direction overlap with the model's utility activation subspace?

  + *Behavioral tradeoff.* When each method removes safety, how much utility does it preserve?

  + *Causal independence.* Are multiple refusal directions causally independent under ablation (RepInd)?

  + *Generalize DIM §5.1.* Arditi et al. show one GCG suffix suppresses the refusal direction on Qwen 1.8B. We extend to in-the-wild WildJailbreak prompts on Llama-3.1-8B, plus layer sweep, RCO comparison, ablation cross-test.
]

// -------------------------------------------------------------------------
// Slide 3: Background — the four reference methods
// -------------------------------------------------------------------------
#slide[
  *Existing methods we build on*

  #table(
    columns: (1fr, 1.3fr, 1.7fr),
    inset: 4pt,
    align: (left, left, left),
    stroke: 0.4pt + gray,
    [*Method*], [*Operates on*], [*Identifies*],
    [DIM @arditi2024],     [Residual stream], [Single direction $hat(bold(r)) in RR^d$],
    [ActSVD @Wei2024Brittleness], [Weight matrices], [Low-rank safety/utility projections],
    [RCO @pmlr-v267-wollschlager25a], [Residual stream], [Multi-D refusal cone (grad. descent)],
    [MSO @Ponkshe2026Safety], [Subspace pairs], [$"MSO"(A,B) = (||U_A^top U_B||_F^2) / min(k_A, k_B)$],
  )

  #v(0.4em)
  *RepInd* @pmlr-v267-wollschlager25a: ablate one direction, check whether the other's layerwise cosine profile changes --- a *causal* test that geometric overlap cannot give.
]

// -------------------------------------------------------------------------
// Slide 4: Proposed method — unified comparison framework
// -------------------------------------------------------------------------
#slide[
  *Our method: a unified comparison framework*

  Three independent comparisons on *one model* (Llama-3.1-8B-Instruct):

  + *Behavioral.* JBB ASR + harmless compliance + Pile/Alpaca perplexity. Same harness for all methods.

  + *Geometric.* Bridge weight space and activation space:
    - DIM/RCO produce vectors in $RR^d$.
    - ActSVD produces $Delta W$. Take $"SVD"(Delta W) = U_B Sigma V_B^top$ — $U_B$ spans the *output-space* subspace ActSVD modified.
    - Compute $"MSO"(hat(bold(r)), U_B)$ between activation- and weight-space methods.

  + *Causal.* RepInd profile changes after ablation, plus an asymmetric variant: *DIM ablated $arrow.r$ measure derived basis*, vs *derived basis ablated $arrow.r$ measure DIM*.
]

// -------------------------------------------------------------------------
// Slide 5: Probe — extending DIM §5.1 (honest framing)
// -------------------------------------------------------------------------
#slide[
  *Probe: extending DIM §5.1's adversarial-suffix analysis*

  *DIM §5.1 already does the core idea* — they project last-token activation onto $hat(bold(r))$ and show *one* GCG suffix suppresses it on Qwen 1.8B. They flag this as restricted to a single model + suffix.

  *We extend along four axes:*
  - *Different model*: Llama-3.1-8B-Instruct (not Qwen 1.8B).
  - *In-the-wild attacks*: ~25 WildJailbreak prompts (not one GCG suffix).
  - *Layer sweep*: project at every layer, not one.
  - *Two new diagnostics*: RCO-direction comparison + ablation cross-test (run same probe prompts under DIM ablation).

  Three prompt groups: `direct_request` (HarmBench), `adversarial_harmful` (WildJailbreak wrapped harmful), `adversarial_benign` (wrapped benign — our control vs DIM's random-suffix control).
]

// -------------------------------------------------------------------------
// Slide 6: Experimental setup
// -------------------------------------------------------------------------
#slide[
  *Experimental setup*

  - *Model:* Llama-3.1-8B-Instruct on a single A100 (40 GB).
  - *DIM:* layer 11 selected, KL- and steerability-filtered.
  - *ActSVD:* utility rank 3000, safety rank 4000, all linear layers.
  - *RCO:* 2-D cone, DIM-initialized, refusal-scaling + surgical-ablation + KL-retain loss.
  - *Behavioral:* JailbreakBench (100), 100 harmless Alpaca, 64 Pile + 64 Alpaca perplexity.
  - *Overlap PCA:* 128 harmless instruction activations at EOI; ranks $k in {1,2,4,8,16,32}$.
  - *RepInd:* 32 prompt pairs.
  - *Probe:* 25 + 25 + 25 = 75 prompts (HarmBench + WildJailbreak streamed).

  *Reproducibility:* one notebook end-to-end (`notebooks/colab_end_to_end.ipynb`).
]

// -------------------------------------------------------------------------
// Slide 7: Behavioral results
// -------------------------------------------------------------------------
#slide[
  *Result 1: behavioral benchmark*

  #table(
    columns: (1.4fr, 1fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: (left, center, center, center, center),
    stroke: 0.4pt + gray,
    [*Method*], [*ASR*], [*Harmless*], [*PPL Pile*], [*PPL Alpaca*],
    [Base],            [0.15], [1.00], [13.93], [8.60],
    [DIM-Ablated],     [*1.00*], [1.00], [14.17], [8.80],
    [ActSVD-Modified], [0.62], [1.00], [19.91], [10.61],
    [RCO-Cone-2],      [*1.00*], [1.00], [13.95], [8.63],
  )

  #v(0.6em)
  - *DIM and RCO* both fully break safety (ASR $1.00$) with near-zero perplexity cost.
  - *ActSVD* reaches only ASR $0.62$ but pays much higher PPL — the weight surgery is less surgical than the activation ablations.
  - All three preserve harmless compliance — the *behavioral* claim of "clean separation" holds.
]

// -------------------------------------------------------------------------
// Slide 8: Cross-method geometric MSO
// -------------------------------------------------------------------------
#slide[
  *Result 2: cross-method MSO bridge is exploratory*

  #grid(
    columns: (1.4fr, 1fr),
    gutter: 12pt,
    image("figures/subspace_mso_per_layer_avg.png", width: 100%),
    [
      #v(0.6em)
      - DIM-vs-ActSVD MSO is near random for most layers; hotspot at layers 10–14.
      - DIM-vs-RCO cosine $= 0.450$.

      *But*: the SVD bridge measures *capacity*, not *effect* on real inputs.

      *Verdict:* exploratory only. Behavioral benchmark + probe are the load-bearing comparisons.
    ],
  )
]

// -------------------------------------------------------------------------
// Slide 9: Safety-utility overlap (the central finding)
// -------------------------------------------------------------------------
#slide[
  *Result 3: safety–utility overlap is the key finding*

  #table(
    columns: (2fr, 1fr, 1fr),
    inset: 5pt,
    align: (left, center, center),
    stroke: 0.4pt + gray,
    [*Direction*], [*MSO (rank 8)*], [*vs random*],
    [Full DIM mean-diffs (avg)], [*0.191*], [98×],
    [DIM selected (layer 11)],   [0.078], [40×],
    [RCO direction],             [0.004], [1.8×],
    [ActSVD activation $delta$], [0.124], [63.5×],
    [Random baseline],           [0.00195], [1×],
  )

  #v(0.5em)
  - *Full safety subspace is entangled* with utility (98× random) — Safety Subspaces was right.
  - *Selected directions are not* — DIM 40×, RCO essentially 1.8×.
  - *Reconciliation:* selection procedures (KL filter, retain loss) implicitly minimize utility overlap *within* an entangled space.
]

// -------------------------------------------------------------------------
// Slide 10: RepInd asymmetry
// -------------------------------------------------------------------------
#slide[
  *Result 4: RepInd is asymmetric*

  #grid(
    columns: (1.1fr, 1fr),
    gutter: 12pt,
    image("figures/repind_change_heatmap.png", width: 95%),
    [
      #v(0.5em)
      - Ablate *DIM* $arrow.r$ derived bases shift (mean $|Delta| = 0.095$).
      - Ablate *derived bases* $arrow.r$ DIM barely shifts ($0.062$).

      #v(0.5em)
      *DIM is dominant but not exhaustive* --- multiple refusal mediators with a hierarchy of causal influence.

      Consistent with Geometry's multi-dimensional claim.
    ],
  )
]

// -------------------------------------------------------------------------
// Slide 11: Probe results — new experiment (predictions; numbers from Colab)
// -------------------------------------------------------------------------
#slide[
  *Result 5 (new): does prompt-based jailbreaking suppress the refusal direction?* #text(size: 0.65em)[(figure + numbers fill in from Colab run)]

  #grid(
    columns: (1.3fr, 1fr),
    gutter: 12pt,
    image("figures/probe_asr_and_projection_by_attack_type.png", width: 100%),
    [
      #v(0.4em)
      *Predictions:*
      - *Direct $arrow.r$ adv-harmful*: ASR rises, projection drops (both DIM, RCO).
      - *Adv-benign control*: projection stays near direct (wrapping alone shouldn't suppress).
      - DIM/RCO track each other imperfectly (cosine = 0.45).

      #v(0.3em)
      If predictions hold: replicates DIM §5.1's suppression effect on a different model + attack distribution.
    ],
  )
]

// -------------------------------------------------------------------------
// New: layer sweep + ablation cross-test (probe extensions)
// -------------------------------------------------------------------------
#slide[
  *Probe extensions: where attacks act + ablation bound* #text(size: 0.65em)[(figures fill in from Colab run)]

  #grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("figures/probe_layer_sweep_projection.png", width: 100%),
    image("figures/probe_ablation_cross_test.png", width: 100%),
  )

  - *Layer sweep:* tells us *where* attacks act --- is the direct-vs-adv-harmful gap layer-localized around DIM's $l_*=11$, or distributed across late layers?
  - *Ablation cross-test:* gap between direct base-ASR and direct ablated-ASR bounds the share of refusal mediated by DIM that prompt attacks can't reach.
]

// -------------------------------------------------------------------------
// New: methodological additions (random direction + judges + TruthfulQA)
// -------------------------------------------------------------------------
#slide[
  *Methodological additions (sanity checks + side-effect tests)*

  - *Random-direction baseline.* Ablate a Gaussian-sampled unit vector. ASR should stay near $0.15$ (sanity check none of the four reference papers does explicitly).
  - *LLM-graded judge.* Post-hoc pass: the *unmodified* base Llama grades every method's saved completions in one go. Avoids the cross-method confound where a modified model's intervention biases its self-judgment. Same-family bias remains, documented in limitations.
  - *TruthfulQA.* 64 questions, substring vs `correct/incorrect_answers`. Tests Wollschläger's claim that DIM hurts truthfulness more than RCO.
  - *Bootstrap 95% CI* on ASR, harmless compliance, and projection means.
  - *ActSVD re-run* with paper-optimal ($r^u=3950$, $r^s=4090$) instead of our earlier aggressive setting.
]

// -------------------------------------------------------------------------
// Slide 12: Reconciliation — answers per RQ
// -------------------------------------------------------------------------
#slide[
  *Reconciliation: how the four claims fit together*

  - DIM ✓ — *selected* direction has low utility overlap, behaviorally clean.
  - Safety Subspaces ✓ — *full* safety subspace is broadly entangled with utility.
  - ActSVD ✓ — orthogonalizing $Pi^u$ matters because raw safety ranks share utility ranks.
  - Geometry ✓ — RepInd asymmetry confirms multiple non-equivalent refusal mediators.

  *They are measuring different objects.* The methods don't contradict each other once you separate _full subspace_ from _selected direction_, and _capacity_ from _causal effect_.

  *Our probe* extends this picture beyond the four papers: real attacks tap the same direction the methods identify, but only partially.
]

// -------------------------------------------------------------------------
// Slide 13: Limitations
// -------------------------------------------------------------------------
#slide[
  *Limitations*

  - Single model — Llama-3.1-8B-Instruct only. Some inter-paper disagreement may be model-specific.
  - PCA utility $eq.not$ causal utility. Variance directions are a proxy.
  - Weight-delta MSO measures *capacity*, not *effect*: depends on input distribution.
  - Activation-delta vector is a mean — hides prompt-dependent and nonlinear effects.
  - RepInd uses DIM-derived candidates rather than fully optimized RepInd vectors.
  - Probe uses a substring-based refusal judge over 75 prompts; wide CIs.
]

// -------------------------------------------------------------------------
// Slide 14: Member contributions
// -------------------------------------------------------------------------
#slide[
  *Contributions*

  - *Evan Scamehorn (25%)* — DIM/ActSVD pipelines on Colab, behavioral benchmark harness, results integration.
  - *Adam Venton (15%)* — Safety-utility overlap, weight-delta MSO bridge.
  - *Calvin Kosmatka (15%)* — Literature survey, introduction, paper curation.
  - *Kyle Sha (25%)* — Prompt-attack probe (new experiment), WildJailbreak integration, RCO extension, discussion section.
  - *Zeke Mackay (20%)* — RepInd analysis, RCO training, asymmetry interpretation.

  All five authors contributed to writing and discussion of research design.
]

// Bibliography
#let bib = bibliography("bibliography.bib", full: true)
#bibliography-slide(bib)
