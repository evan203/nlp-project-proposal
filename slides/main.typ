#import "@preview/typslides:1.3.2": *

#show: typslides.with(
  ratio: "16-9",
  theme: "bluey",
  font: "Fira Sans",
  font-size: 20pt,
  link-style: "color",
  show-progress: true,
)

#front-slide(
  title: "The Geometry of Refusal: A Comparative Subspace Analysis of Safety Mechanisms",
  subtitle: [_Project Proposal_],
  authors: "Group 6: Evan Scamehorn, Kyle Sha, Adam Venton, Zeke Mackay, and Calvin Kosmatka",
  info: [#link("https://github.com/evan203/nlp-project-proposal")],
)

#slide[

  // What is the problem?
  *Problem Description:*

  - *The "Geometry" Debate:*
      - *Vector:* Safety is a single direction (Arditi et al., 2024).
      - *Cone:* Safety is a multi-dimensional cone (Wollschläger et al., 2025).
      - *Rank:* Safety is a statistical variance component (Wei et al., 2024).
  - *The Gap:* No study has rigorously compared the *geometric overlap* of these shapes on the same model. Do they target the same dimensions?

  #line(length: 100%)

  /*
   * Why is it interesting and important?
   *
   * 1. What could we benefit from your research findings if your project
   * succeeds?
   *
   * 2. What scientific outcome can you offer to the community?
   *
   * here is a starting point, we can refine from here
   */
  *Significance and Research Value:*

  - *Unifying Frameworks:* We aim to map the "Refusal Landscape" of Llama-3.1 to determine if simple vectors are sufficient or if complex cones are required.
  - *Representation Independence:* Finding the subspace that is most orthogonal to "General Utility" allows for safer model unlearning and jailbreaking defenses.
]

#slide[

  /*
   * 1. Why is it hard? (E.g., why do naive approaches fail?)
   *
   * 2. Why hasn't it been solved before? Or, what's wrong with previous
   * proposed solutions?
   *
   * 3. It’d be good to use a concrete failure example from previous
   * solutions to show the challenges. It’d be great to try an LLM API to see
   * if state-of-the-art production LLMs work or not.
   */
  *Technical Challenges:*

  - *Geometric Complexity:* "Refusal" is likely high-dimensional. We must use *Refusal Direction Optimization (RDO)* to find the full cone, not just averages.
  - *Measuring Overlap:* We need rigorous metrics like *Modal Subspace Overlap (MSO)* (Ponkshe et al.) and *Principal Angles* to quantify how much safety "bleeds" into utility.

  #line(length: 100%)

  /*
   * 1. What are the key components of my approach and results? If your method
   * is a new algorithm, write a simple pseudocode or a workflow to illustrate
   * the algorithm. If your method is a new model component, clarify the key
   * modules. Try to be more specific and thoughtful if possible.
   *
   * 2. Novelty: What are the key differences between your proposed methods
   * and existing solution? Why is your method conceptually a better design?
   */
  *Proposed Methods or Explorations:*

  1.  *Extract Subspaces (Linear Spans):*
      - $S_"vec"$ (Arditi): Rank-1 subspace (Difference-of-Means).
      - $S_"cone"$ (Wollschläger): Multi-dim subspace (RDO Basis).
      - $S_"util"$ (Baseline): Unbiased variance directions (ActSVD).
  2.  *Geometric Analysis:*
      - *Comparison:* Compute Principal Angles & MSO.
      - *Hypothesis:* $S_"cone"$ minimizes entanglement with $S_"util"$.
      - *Layer-wise:* Track how safety subspaces emerge across depth.
]

#slide[

  /*
   * What are the datasets used for training, validation, and testing? Give
   * the data statistics. Add a citation to existing datasets used in this
   * project.
   *
   * How do you quantify the success of the model outputs? Add a citation if
   * you reuse a metric in some papers.
   */
  *Datasets and Evaluation Metrics:*

  - Alpaca @alpaca
    - General, safe instructional dataset containing instructions and outputs from text-davinci-003
    - Used to test refusal by ablated LLMs
  - BeaverTails @beavertails
    - QA pairs of various categories of harmful prompts
    - Used to test safety of ablated LLMs
  - Refusal score @arditi2024
    - Rate of model refusing to answer
    - Based on several common refusal phrases (I'm sorry, As an LLM, etc)
  - Attack success rate @pmlr-v267-wollschlager25a
    - Rate of model answering unsafe prompts
  #line(length: 100%)

  /*
   * Add only the most related baseline methods for experimental comparison.
   * Add a brief description of each method per bullet point. The description
   * should highlight the distinction of the method and your proposed idea.
   */
  *Baseline Methods:*

  - *Vector Baseline:* $S_"vec"$ (Difference-of-Means) - Assumes 1D safety.
  - *Rank Baseline:* $S_"svd"$ (ActSVD) - Assumes variance = safety.

  #line(length: 100%)

  /*
   * What are the GPUs required (e.g., quantity, GPU RAMs)?
   *
   * Estimate how long it will take to run one experiment. For example,
   * training a 3B model for 10 epochs takes ~5hrs. Add a reference of how you
   * derive this estimation (e.g., either you have run one experiment yourself
   * or get the estimation from an existing paper).
   */
  *Computing Estimation:*

  - *Hardware:* ~24GB VRAM (Single A100/H100 or dual 3090).
  - *Time:* ~5 hrs for RDO training, ~2 hrs for SVD extraction.
  - *Plan:*
      - *Week 1:* Extract Baselines ($S_"vec"$, $S_"svd"$). (Evan/Kyle)
      - *Week 2:* Run RDO ($S_"cone"$) & Compute Geometry. (Adam/Zeke)
      - *Week 3:* Validation Experiments & Final Report. (Calvin)

  #line(length: 100%)

  /*
   * What model checkpoints are you going to used? (e.g., LLaMA-3.3 3B)
   *
   * Provide any Github codebases that you may reused for training/prompting
   */
  *Model Checkpoints and Codebase:*

  - *Primary Model:* Llama-3.1-8B-Instruct.
  - *Comparisons:* Gemma-2-9B, Qwen-2.5-7B-Instruct.
  - *Code:* Custom PyTorch implementations of RDO and ActSVD.
]

#let bib = bibliography("report-bibliography.bib")
#bibliography-slide(bib)
