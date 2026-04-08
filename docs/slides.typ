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
  title: "Examining the Superposition of Saftey and Utility in LLM Activation Spaces",
  subtitle: [_Midterm Project Update_],
  authors: "Group 6: Evan Scamehorn, Kyle Sha, Adam Venton, Zeke Mackay, and Calvin Kosmatka",
  info: [#link("https://github.com/evan203/nlp-project-proposal")],
)

#slide[

  *Recap:*

  - Aligned LLMs can produce harmful outputs using diverse jailbreaking techniques
  - Safety subspaces are hypothesized to be represented in several different ways
  - Alpaca (harmless prompts) and BeaverTails (harmful prompts) datasets
  - Refusal score and Attack Success Rate to evaluate performance

  // #line(length: 100%)
]

#slide[

  *Baseline Method:*

  - Difference-in-means (DIM) @arditi2024
    - Mean response activation difference between harmful and harmless prompts
  - Refusal Cone Optimization (RCO) @pmlr-v267-wollschlager25a
    - Use gradient descent to generate multiple basis vectors representing safety mechanisms
  - ActSVD safety/utility ranks @Wei2024Brittleness
    - Perform Singular Value Decomposition on model weights to identify safety/utility-critical low-rank matrices
  - Mode Subspace Overlap (MSO) @Ponkshe2026Safety
    - Performs SVD to quantify overlap between subspaces
  - Representational Independence (RepInd) @pmlr-v267-wollschlager25a
    - Performs cosine similarity on ablated model activations to test independence of multiple subspaces
]

#slide[
  *Experiment setup:*

  - Comparison of safety and utility subspaces
    - Mode Subspace Overlap (MSO) similarity test between safety subspaces @Ponkshe2026Safety
    - Representational Independence (RepInd) comparison between each safety subspace and utility subspace @pmlr-v267-wollschlager25a

]

#slide[
  *Findings — Safety Benchmarks:*

  We benchmark three variants of *Llama-3.1-8B-Instruct*:
  + *Base* — the original aligned model
  + *DIM-Ablated* — refusal direction projected out at layer 11 @arditi2024
  + *ActSVD-Modified* — low-rank safety-critical weight components removed @Wei2024Brittleness

  Evaluated on *JailbreakBench* @jailbreakbench (100 harmful prompts, 10 harm categories, 10 each) and 100 harmless *Alpaca* @alpaca prompts. Attack Success Rate (ASR) = fraction of harmful prompts where the model complies instead of refusing. Compliance on harmless prompts should remain at 1.0.

  #table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    [*Model*], [*JBB ASR*], [*Harmless*], [*Pile PPL*], [*Alpaca PPL*],
    [Base], [0.16], [1.00], [8.69], [6.01],
    [DIM-Ablated], [1.00], [1.00], [8.75], [6.13],
    [ActSVD-Modified], [0.63], [1.00], [13.09], [6.82],
  )
]

#slide[
  *Findings — Activation Comparison:*

  #grid(
    columns: (2fr, 1fr),
    gutter: 12pt,
    [
      #image("figures/activation_comparison.png", height: 50%)
    ],
    [
      #text(size: 0.75em)[
    When running both jailbroken models we see from both metrics that their activations are more different when given harmful prompts vs helpful ones.


            This supports the idea that neither of these methods fully describes the safety subspace since both jailbreaks are able to work while being different.


        _Comparison of last layer activations between actSVD and DIM jailbroken models._ 
      ]
    ],
  )
]

#slide[
  *Findings — Jailbreak ASR:*

  #grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    [
      #image("figures/benchmark_jailbreak_asr_overall.png", height: 50%)
    ],
    [
      #text(size: 0.75em)[
        DIM ablation achieves *100 % ASR* — complete safety removal across every harm category. ActSVD reaches *63 %* — partial removal only.

        The per-category breakdown reveals ActSVD is *non-uniform*: 0.9 on Disinformation, Expert advice, Malware — but only 0.2 on Harassment and 0.0 on Sexual content.

        _Benchmark: JailbreakBench @jailbreakbench — 100 prompts, 10 harm categories. Compliance via refusal-prefix matching._
      ]
    ],
  )
]

#slide[
  *Findings — Per-Category Jailbreak ASR:*

  #image("figures/benchmark_jailbreak_asr_per_category.png", width: 80%, height: 55%)

  #text(
    size: 0.7em,
  )[ActSVD's weight pruning removes safety *non-uniformly* — it largely fails on socially sensitive categories (Harassment, Sexual content) while succeeding on more technical harms (Malware, Disinformation). DIM is uniformly effective.]
]

#slide[
  *Findings — Utility Preservation:*

  #grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    [
      #image("figures/benchmark_perplexity_pile_alpaca.png", height: 45%)
    ],
    [
      #image("figures/benchmark_safety_utility_tradeoff.png", height: 45%)
    ],
  )

  #text(
    size: 0.7em,
  )[Perplexity on *The Pile* @thepile (general text) and *Alpaca* @alpaca (instructions). DIM barely impacts capability (*+0.7 %* Pile PPL). ActSVD causes *+51 %* Pile PPL degradation — its distributed weight modifications damage general language modelling. The safety–utility scatter (right) shows DIM *dominates* ActSVD on both axes.]
]

#slide[
  *Findings — MSO Heatmap (DIM vs ActSVD):*

  #grid(
    columns: (50%, 50%),
    gutter: 12pt,
    [
      #image("figures/subspace_mso_heatmap_layer_by_weight.png", height: 85%, fit: "contain")
    ],
    [
      #text(size: 0.65em)[
        Maximum Subspace Overlap (MSO) @Ponkshe2026Safety measures overlap of the DIM refusal direction with ActSVD's weight-delta subspace per layer and weight type.

        Most cells are *near random baseline* ($approx k_A dot k_B \/ d$) — the two methods find *nearly orthogonal* safety structures.

        Only notable signal: *layer 10* MLP down proj (MSO = 0.057, 3.2× random) — adjacent to DIM's source layer (11).
      ]
    ],
  )
]

#slide[
  *Findings — Per-Layer MSO & Cross-Model Cosine:*

  #grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    [
      #image("figures/subspace_mso_per_layer_avg.png", height: 50%)
    ],
    [
      #image("figures/subspace_cross_model_dim_cosine.png", height: 50%)
    ],
  )

  #text(
    size: 0.7em,
  )[*Left:* Per-layer average MSO (red) vs random baseline (blue). Layer 10 is the only layer clearly above baseline. Layers 20–31 show no signal.
    *Right:* DIM directions computed independently for 6 models. Llama-3.1 ↔ Llama-3 cosine similarity = *0.603* — all cross-family pairs $approx 0$. The refusal direction is *model-family-specific*.]
]

#slide[

  *Future Extension: Extensions of Current Work*

  - Further comparisons of the three methods
  - More datasets
    - TwinPrompt dataset from TwinBreak @twinbreak
]

#slide[

  *Future Extension: Additional Techniques*

  - Differentiated Directional Intervention @diffDirection
    - More advanced version of difference-in-means
  - Prompt optimization @hiddenDimensions
    - Avoiding words that activate the harmfulness subspace
  - Evaluate Mode Subspace Overlap (MSO) between safety subspaces
  - Implement and evaluate further jailbreaking techniques
]

#slide[
  *Contribution:*

  - Evan: Project Management, Paper reimplementation (20%)
  - Adam: Report writeup, Model analysis (20%)
  - Calvin: Model analysis, Literature review (20%)
  - Kyle: Paper reimplementation, Report writeup (20%)
  - Zeke: Model analysis, Slide writeup (20%)

]

// Bibliography
// note: report-bibliography is a link to the file ../report/bibliography.bib. don't modify that file
#let bib = bibliography("bibliography.bib", full: true)
#bibliography-slide(bib)
