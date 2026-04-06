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

  - What problem are you addressing?
  - What are the datasets? What are the evaluation metrics?

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

  *Future Extension:*

  - What future extension are you planning to do before the final presentation?
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
