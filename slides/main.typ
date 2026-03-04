#import "@preview/typslides:1.3.2": *

// Project configuration
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
  subtitle: [_Project Proposal_],
  authors: "Group 6: Evan Scamehorn, Kyle Sha, Adam Venton, Zeke Mackay, and Calvin Kosmatka",
  info: [#link("https://github.com/evan203/nlp-project-proposal")],
)

#slide[

  // What is the problem?
  *Problem Description:*
  /*
   * Course staff slides template asks us to describe:
   *
   * 1. What is the input and output format of the tasks? If your problem
   * requires multiple datasets, try to clearly define them in math notations.
   *
   * 2. What are the objectives of the problem? (e.g., learn a classifier,
   * prevent models from forgetting fine-tuning tasks). Clearly define the
   * scope of the project. If your project idea involves building multiple ML
   * models, try to narrow the scope and pick the most critical model that is
   * still weak and has large room for improvement.
   *
   * 3. [Optional] Given a figure to illustrate an example on the right empty
   * space.
   *
   * here is a starting point, we can refine from here
   */

  - Aligned LLMs can produce *harmful outputs* using many diverse
    *jailbreaking* techniques
  - We seek to understand why safety mechanisms of LLMs are *fragile* by
    examining the *activation space* of harmful and helpful prompts
  - Recent studies demonstrate that safety and utility share
    *representational capacity* (superposition) in linear activation space

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

  - Addressing failure cases in the alignment of LLMs requires a deep
    understanding of why their safety mechanisms are fragile.
  - Mechanistic interpretability can be used to better understand how safety
    mechanisms operate, and inform creating more robust safety alignment
    methods
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

  - Refusal is known to span *multi-dimensional "concept cones"* containing *
    infinite refusal directions* and *multi-dimensional "safety residual
    spaces"*
  - The geometric mechanics of how fine-tuning destroys these structures is
    *poorly understood*

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

  - To be determined
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

  - Todo

  #line(length: 100%)

  /*
   * Add only the most related baseline methods for experimental comparison.
   * Add a brief description of each method per bullet point. The description
   * should highlight the distinction of the method and your proposed idea.
   */
  *Baseline Methods:*

  - Todo

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

  - Todo

  #line(length: 100%)

  /*
   * What model checkpoints are you going to used? (e.g., LLaMA-3.3 3B)
   *
   * Provide any Github codebases that you may reused for training/prompting
   */
  *Model Checkpoints and Codebase:*

  - Todo
]

// Bibliography
// note: report-bibliography is a link to the file ../report/bibliography.bib. don't modify that file
#let bib = bibliography("report-bibliography.bib", full: true)
#bibliography-slide(bib)
