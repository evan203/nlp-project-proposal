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
    [DIM @arditi2024],     [Residual stream], [Single direction $hat(bold(r)) in RR^d$ (1-D ablation)],
    [ActSVD @Wei2024Brittleness], [Weight matrices], [Low-rank safety/utility projections],
    [RCO @pmlr-v267-wollschlager25a], [Residual stream], [Multi-D refusal cone (2-D subspace ablation)],
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
  - *ActSVD:* utility rank 3950, safety rank 4090, all linear layers.
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
    [Base],                       [0.15], [1.00], [13.93], [8.60],
    [DIM-Ablated (1-D)],          [*1.00*], [1.00], [14.17], [8.80],
    [ActSVD-Modified],            [0.77], [1.00], [19.94], [11.41],
    [RCO-Cone-2 (true 2-D)],      [_TBD_], [_TBD_], [_TBD_], [_TBD_],
    [Random-Direction-7-1D],      [0.16], [0.98], [14.65], [8.86],
    [Random-Subspace-7-2D],       [_TBD_], [_TBD_], [_TBD_], [_TBD_],
  )

  #v(0.6em)
  - *DIM* fully breaks substring safety (ASR $1.00$) with near-zero perplexity cost; *RCO* now ablates *both* cone basis vectors (true 2-D subspace), pending re-run.
  - *ActSVD* reaches ASR $0.77$ but pays much higher PPL — weight surgery is less surgical than activation ablations.
  - *Random 1-D ablation* stays near base ASR (direction-specific). *Random 2-D subspace* is the rank-matched control for RCO.
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
      - DIM-vs-ActSVD MSO is near random for most layers; layer 10 down/o projections are the clear hotspots.
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
    [*Safety object*], [*MSO (rank 8)*], [*vs random*],
    [Full DIM mean-diffs (avg)],          [*0.191*], [98×],
    [DIM selected (layer 11, 1-D)],       [0.078], [40×],
    [RCO 2-D cone (norm. subspace MSO)],  [_TBD_], [_TBD_],
    [ActSVD activation $delta$ avg.],     [0.067], [34×],
    [Random 1-D baseline],                [0.00195], [1×],
    [Random 2-D subspace baseline],       [_TBD_], [_TBD_],
  )

  #v(0.5em)
  - *Full safety subspace is entangled* with utility (98× random) — Safety Subspaces was right.
  - *Selected DIM direction* drops to 40×; *RCO 2-D cone* uses proper normalized subspace MSO ($||U^top Q_S||_F^2 / 2$, baseline $8/d$) — the earlier "0.004 / 1.8×" came from a per-layer-vs-cone-basis bug, now fixed.
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
  *Result 5 (new): prompt wrappers suppress the refusal direction, but not only for harmful prompts*

  #grid(
    columns: (1.3fr, 1fr),
    gutter: 12pt,
    image("figures/probe_asr_and_projection_by_attack_type.png", width: 100%),
    [
      #v(0.4em)
      *Observed (qualitative; numbers refresh on re-run):*
      - Direct $arrow.r$ adv-harmful: ASR rises, DIM signed projection drops, RCO cone-subspace norm $norm(B^top bold(h))_2$ drops.
      - Adv-benign control drops too — the wrapper style itself perturbs the refusal subspace.
      - DIM/RCO track each other imperfectly (top principal-angle cosine $approx 0.45$).

      #v(0.3em)
      *Takeaway:* suppression generalizes DIM §5.1, but wrapper style is a confound.
    ],
  )
]

// -------------------------------------------------------------------------
// New: layer sweep + ablation cross-test (probe extensions)
// -------------------------------------------------------------------------
#slide[
  *Probe extensions: where attacks act + ablation bound*

  #grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("figures/probe_layer_sweep_projection.png", width: 100%),
    image("figures/probe_ablation_cross_test.png", width: 100%),
  )

  - *Layer sweep:* DIM gap peaks around $l_*=11$; RCO 2-D subspace norm grows later, suggesting related but more distributed geometry.
  - *Ablation cross-test:* now run for *both* DIM (1-D) *and* RCO (2-D cone subspace). DIM previously lifted direct ASR $0.12 arrow.r 0.96$; RCO 2-D should be at least as effective if its second cone direction adds real coverage.
]

// -------------------------------------------------------------------------
// New: methodological additions (random direction + judges + TruthfulQA)
// -------------------------------------------------------------------------
#slide[
  *Methodological additions (sanity checks + side-effect tests)*

  - *Random-direction baselines (1-D + 2-D).* Ablate a Gaussian-sampled unit vector (rank-matched to DIM) *and* a random 2-D orthonormal subspace (rank-matched to RCO). Splitting them isolates "direction-specific" from "removing more dims inflates ASR" — neither baseline is foregrounded in the four reference papers.
  - *External safety judge.* Post-hoc pass with `Qwen/Qwen3Guard-Gen-4B` (purpose-built moderator, different family) replaces an earlier base-Llama judge that introduced same-family bias and unstable verdicts.
  - *TruthfulQA.* 64 questions, substring vs `correct/incorrect_answers`; mostly ambiguous, so only a weak side-effect check.
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

  *Our probe* extends this picture beyond the four papers: real wrappers tap related refusal geometry, but the benign control shows wrapper style also drives suppression.
]

// -------------------------------------------------------------------------
// Slide 13: Limitations
// -------------------------------------------------------------------------
#slide[
  *Limitations*

  - Single model — Llama-3.1-8B-Instruct only. Some inter-paper disagreement may be model-specific.
  - PCA utility $eq.not$ causal utility. Variance directions are a proxy.
  - Early-layer mean-diff (layers 0--3) is partly format/template variance; the "98×" headline averages those in. Mid-layer subset (8--23) gives the cleaner read (~85×).
  - Weight-delta MSO measures *capacity*, not *effect*: depends on input distribution.
  - Activation-delta vector is a mean — hides prompt-dependent and nonlinear effects.
  - RepInd uses DIM-derived candidates rather than fully optimized RepInd vectors.
  - Probe uses a substring-based refusal judge over 75 prompts; wide CIs.
  - Qwen3Guard judge has its own training-distribution biases; reported alongside (not instead of) substring ASR.
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
