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
  subtitle: [_Midterm Project Update_],
  authors: "Group 6: Evan Scamehorn, Kyle Sha, Adam Venton, Zeke Mackay, and Calvin Kosmatka",
  info: [#link("https://github.com/evan203/nlp-project-proposal")],
)

#slide[

  *Research question:*

  - Do different linear safety-removal methods recover the same refusal mechanism?
  - When a method removes refusal behavior, how much utility is preserved?
  - Do Geometry-style RepInd profiles show independent refusal directions?

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
    - Tests whether ablating one direction changes another direction's layerwise cosine profile
]

#slide[
  *Experiment setup:*

  - Target model: Llama-3.1-8B-Instruct
  - Safety eval: JailbreakBench attack success rate
  - Utility eval: harmless Alpaca compliance + Pile/Alpaca perplexity
  - Geometry eval:
    - MSO between DIM refusal direction and ActSVD weight-delta subspaces @Ponkshe2026Safety
    - MSO between DIM safety directions and harmless-instruction utility PCA subspaces
    - RepInd profile changes before/after direction ablation @pmlr-v267-wollschlager25a

]

#slide[

  *Findings:*

  - DIM ablation raises JBB ASR from 0.16 to 1.00 with little perplexity change.
  - ActSVD raises JBB ASR to 0.63 but causes larger Pile/Alpaca perplexity degradation.
  - DIM vs ActSVD MSO is near random for most layers, with a mild hotspot around layer 10.
  - Direct safety-vs-utility overlap is above random: rank-8 mean MSO = 0.192 vs 0.00195 random baseline.
  - RepInd profile test is asymmetric: ablating DIM strongly changes one derived basis profile, but ablating that basis barely changes DIM.

]

#slide[

  *Cones/RepInd Next Step:*

  - Current RepInd run uses DIM-derived cone-basis candidates.
  - Full optimized cone claim requires running `scripts/run_rco.sh`.
  - Then compare DIM, RDO, orthogonal-RDO, RepInd, and cone basis directions with the same RepInd script.
  - Evaluate cone samples against DIM and ActSVD on the same safety/utility benchmark.
]

#slide[
  /*
   * What are the GPUs required (e.g., quantity, GPU RAMs)?
   *
   * Estimate how long it will take to run one experiment. For example,
   * training a 3B model for 10 epochs takes ~5hrs. Add a reference of how you
   * derive this estimation (e.g., either you have run one experiment yourself
   * or get the estimation from an existing paper).
   */
  *Contribution:*

  - Describe each team member’s contribution.
  - Provide the percentage of contributions within the group. For example, Member 1 contributes 40% of efforts. Member 2 20%, Member 3 20%, Member 4 20%.

]

// Bibliography
// note: report-bibliography is a link to the file ../report/bibliography.bib. don't modify that file
#let bib = bibliography("bibliography.bib", full: true)
#bibliography-slide(bib)
