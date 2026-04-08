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

  *Findings:*

  - How does your method compare with existing baseline?
  - quantitative results
  - qualitative error analysis.

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
