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

#slide(title: [_Zeke_])[

  *Problem & Motivation:*

  - Otherwise aligned LLMs can produce harmful outputs with a variety of jailbreaking techniques.
    - Since these models are publicly available, developers are obliged to ensure safe usage.
  - Mechanisms controlling query refusal are not well-understood.
    - Several distinct methods have been developed to isolate model activation/weight subspaces dictating refusal

]

#slide(title: [_Zeke_])[

  *Refusal Subspace Generation Methods:*

  - Difference-in-means (DIM) @arditi2024
    - Mean response activation difference between harmful and harmless prompts
  - Refusal Cone Optimization (RCO) @pmlr-v267-wollschlager25a
    - Use gradient descent to generate multiple basis vectors representing safety mechanisms
  - ActSVD safety/utility ranks @Wei2024Brittleness
    - Perform Singular Value Decomposition on model weights to identify safety-critical low-rank matrices
]

#slide(title: [_Zeke_])[
  *Comparison Experiment:*

  - Benchmark five versions of *Llama-3.1-8B-Instruct*: one base aligned model, one model with a random direction ablated, and models with each ablation method.
    - Evaluated on 100 harmful prompts from *JailbreakBench* and 100 harmless prompts from *Alpaca*.
    - *Attack Success Rate*: proportion of harmful prompts answered by model. Compliance on harmless prompts should remain at 100%.
  - Perform Mean Subspace Overlap (MSO) and Cosine Similarity between ablated model weights & activations

]

#slide(title: [_Kyle_])[
  *Findings — Safety Benchmarks:*
  #table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    [*Model*], [*JBB ASR*], [*Harmless*], [*Pile PPL*], [*Alpaca PPL*],
    [Base], [0.15], [1.00], [13.93], [8.60],
    [Random-Dir-Ablated], [0.16], [0.98], [14.65], [8.86],
    [DIM-Ablated], [1.00], [1.00], [14.17], [8.80],
    [ActSVD-Modified], [0.77], [1.00], [19.94], [11.41],
    [RCO-Cone-Ablated], [1.00], [1.00], [13.97], [8.62],
  )

  - Random ablation remains similar to base model.
  - DIM and RCO both fully break safety with low perplexity cost.
  - ActSVD reaches ASR 0.77 but with higher PPL cost - weight ablations are less effective than activation ablations.
]

#slide(title: [_Kyle_])[
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


#slide(title: [_Kyle_])[
  *Findings — MSO of DIM-vs-ActSVD*

  #image("figures/subspace_mso_per_layer_avg.png", width: 80%, fit: "contain")

  - MSO between DIM and ActSVD is approximately random for most layers except layer 10.
  - DIM's source is layer 11.

]


#slide(title: [_Adam_])[
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

#slide(title: [_Adam_])[
  *Findings — Activation Comparison:*

  #grid(
    columns: (2fr, 1fr),
    gutter: 12pt,
    [
      #image("figures/DIM_ActSVD_act_comp.png", height: 80%)
    ],
    [
      #text(size: 0.75em)[

        The difference between the activations increases across layers, but not linearly.

        _Comparison of activations between actSVD and DIM jailbroken models across layers._
      ]
    ],
  )
]

#slide(title: [_Adam_])[
  *Findings — Activation Comparison:*

  #grid(
    columns: (2fr, 1fr),
    gutter: 12pt,
    [
      #image("figures/Base_act_comp.png", height: 80%)
    ],
    [
      #text(size: 0.75em)[

        The difference between the harmful and helpful prompts is much greater for DIM indicating that it more cleanly seperates safety and utility.

        _Comparison of activations between actSVD and DIM jailbroken models and the Base model across layers._
      ]
    ],
  )
]

#slide(title: [_Calvin_])[
  *Findings — Self-Consistency:*

  #grid(
    columns: (2fr, 1fr),
    gutter: 12pt,
    [
      #image("figures/dim_self_consistency.png", height: 50%)
    ],
    [
      #text(size: 0.75em)[
        Overall the activations for the two datasets are positively correlated, but not identical

        They are most highly correlated in layers 12-14

        Layer 11, the ablated layer has a similarity of 0.57

        _Cosine similarity of DIM candidate vectors between reference dataset and TwinPrompt._
      ]
    ],
  )
]

#slide(title: [_Evan_])[
  *Safety-Utility Overlap*

  #table(
    columns: (2fr, 1fr, 1fr),
    inset: 5pt,
    align: (left, center, center),
    stroke: 0.4pt + gray,
    [*Direction*], [*MSO (rank 8)*], [*vs random*],
    [Full DIM mean-diffs (avg)], [*0.191*], [98×],
    [DIM selected (layer 11)], [0.078], [40×],
    [RCO direction], [0.004], [1.8×],
    [ActSVD activation $delta$ avg.], [0.067], [34×],
    [Random baseline], [0.00195], [1×],
  )

  #v(0.5em)
  - *Full safety subspace is entangled* with utility (98× random) - Safety Subspaces was right.
  - *Selected directions are not* — DIM 40×, RCO essentially 1.8×.
  - *Reconciliation:* selection procedures (KL filter, retain loss) implicitly minimize utility overlap *within* an entangled space.
]

// Bibliography
// note: report-bibliography is a link to the file ../report/bibliography.bib. don't modify that file
#let bib = bibliography("bibliography.bib", full: true)
#bibliography-slide(bib)
