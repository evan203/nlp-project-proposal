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
  title: "Comparison of Safety Alignment Subspaces in LLMs",
  subtitle: [_Final Presentation_],
  authors: "Group 6: Evan Scamehorn, Kyle Sha, Adam Venton, Zeke Mackay, and Calvin Kosmatka",
  info: [#link("https://github.com/evan203/nlp-project-proposal")],
)

#slide[

  *Problem & Motivation:*

  - Otherwise aligned LLMs can produce harmful outputs with a variety of jailbreaking techniques.
    - Since these models are publicly available, developers are obliged to ensure safe usage.
  - Mechanisms controlling query refusal are not well-understood.
    - Several distinct methods have been developed to isolate model activation/weight subspaces dictating refusal

  // #line(length: 100%)
]

#slide[

  *Refusal Subspace Generation Methods:*

  - Difference-in-means (DIM) @arditi2024
    - Mean response activation difference between harmful and harmless prompts
  - Refusal Cone Optimization (RCO) @pmlr-v267-wollschlager25a
    - Use gradient descent to generate multiple basis vectors representing safety mechanisms
  - ActSVD safety/utility ranks @Wei2024Brittleness
    - Perform Singular Value Decomposition on model weights to identify safety-critical low-rank matrices
]

#slide[
  *Comparison Experiment:*

  - Benchmark five versions of *Llama-3.1-8B-Instruct*: one base aligned model, one model with a random direction ablated, and models with each ablation method.
    - Evaluated on 100 harmful prompts from *JailbreakBench* and 100 harmless prompts from *Alpaca*.
    - *Attack Success Rate*: proportion of harmful prompts answered by model. Compliance on harmless prompts should remain at 100%.
  - Perform Mean Subspace Overlap (MSO) and Cosine Similarity between ablated model weights & activations

]

#slide[
  *Findings — Safety Benchmarks:*
  #table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    [*Model*], [*JBB ASR*], [*Harmless*], [*Pile PPL*], [*Alpaca PPL*],
    [Base], [0.15], [1.00], [13.93], [8.60],
    [Random-Dir-Ablated], [0.16], [0.98], [14.65], [8.86],
    [DIM-Ablated], [1.00], [1.00], [14.17], [8.80],
    [ActSVD-Modified], [0.77], [1.00], [19.94], [11.41],
    [RCO-Cone-Ablated], [1.00], [1.00], [13.97], [8.62]
  )

  - Random ablation remains similar to base model.
  - DIM and RCO both fully break safety with low perplexity cost.
  - ActSVD reaches ASR 0.77 but with higher PPL cost - weight ablations are less effective than activation ablations.
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
  *Findings — MSO of DIM-vs-ActSVD*

  #grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    [
      #image("figures/subspace_mso_per_layer_avg.png", height: 50%, fit: "contain")
    ]
  )

  - MSO between DIM and ActSVD is approximately random for most layers except layer 10.

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
