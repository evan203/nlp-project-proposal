
// This is a minimal starting document for tracl, a Typst style for ACL.
// See https://typst.app/universe/package/tracl for details.


#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.1": *

#add-bib-resource(read("bibliography.bib"))

#show: doc => acl(doc, anonymous: false, title: [The Geometry of Refusal: A Comparative Subspace Analysis of Safety Mechanisms], authors: make-authors(
  (
    name: "Evan Scamehorn",
    affiliation: [University of Wisconsin\ #email("scamehorn@wisc.edu")],
  ),
  (
    name: "Adam Venton",
    affiliation: [University of Wisconsin\ #email("venton@wisc.edu")],
  ),
  (
    name: "Calvin Kosmatka",
    affiliation: [University of Wisconsin\ #email("ckosmatka@wisc.edu")],
  ),
  (
    name: "Kyle Sha",
    affiliation: [University of Wisconsin\ #email("kasha2@wisc.edu")],
  ),
  (
    name: "Zeke Mackay",
    affiliation: [University of Wisconsin\ #email("afmackay@wisc.edu")],
  ),
))


#abstract[
  Recent research offers conflicting views on the geometry of safety in Large Language Models (LLMs). Some works suggest safety is mediated by a single "refusal direction" (Arditi et al., 2024), others propose a multi-dimensional "concept cone" (Wollschläger et al., 2025), while others verify "safety ranks" via activation statistics (Wei et al., 2024). We propose a comparative subspace analysis on Llama-3.1-8B-Instruct. We will extract safety subspaces using Difference-of-Means, Refusal Direction Optimization (RDO), and ActSVD, and measure their geometric overlap (Principal Angles, Modal Subspace Overlap) with each other and with a "Utility Subspace" defined by ActSVD. Our goal is to map the "Refusal Landscape" and determine which geometric model best disentangles safety from general capability.
]


= Introduction

The "geometry" of safety in LLMs—where refusal behaviors live in the high-dimensional activation space—is a subject of active debate. Understanding this geometry is crucial for "mechanistic unlearning": the ability to surgically remove safety refusals (for research) or bolster them (for defense) without damaging the model's general utility.

Three prominent hypotheses have emerged:
1.  *The Vector Hypothesis*: Safety is a simple, single direction (Arditi et al., 2024).
2.  *The Cone Hypothesis*: Safety is a complex, multi-dimensional cone requiring optimization to find (Wollschläger et al., 2025).
3.  *The Rank Hypothesis*: Safety is a statistical property of high-variance components in the weights, discoverable via SVD (Wei et al., 2024).

This project aims to unify these views. We effectively ask: *Do these three distinct methods point to the same underlying directions in the activation space?* By mapping the overlaps between these subspaces, we can evaluate which method provides the most "Representationally Independent" view of safety—one that minimizes collision with the model's general intelligence.

= Literature Survey

#citet("arditi2024") show that for several common chat models, a manipulation of a single dimensional subspace is enough to both induce refusal of non-harmful requests, and turn off the refusal of harmful requests. They use a procedure called *difference-of-means* to identify this subspace. This technique measures the average activations at each token position of each transformer layer, compared between a set of harmful requests and a set of harmless requests. Once these difference-of-means vectors are constructed, they score each for its ability to cause the model to refuse harmless requests and respond to harmful requests. The vector with the highest score is normed and selected as the refusal dimension vector, denoted by $hat(r)$. This refusal dimension can then be ablated in the residual stream at inference time to bypass refusal behavior.

This vector can also be used to create a jailbroken model by updating model weights according to the following formula: $W'_("out") <- W_("out") - hat(r)hat(r)^T W_("out")$. They find that this attack method is successful on the Qwen model family with and without the system prompt, and on the Llama model family without the system prompt, and has little effect on model coherence. 

#v(1em)

#citet("Wei2024Brittleness") introduce low-rank decomposition methods designed to identify specific ranks within a weight matrix related to given LLM behaviors. Their ActSVD algorithm performs Singular Value Decomposition on the product of the model weights and input activations ($W X_"in"$), and yields an orthogonal projection matrix ($Pi$). Removing the top safety-critical ranks ActSVD identifies causes the model to completely stop rejecting unsafe prompts, and the model's utility is severely compromised. These findings suggest that safety regions in aligned models are also crucial for its general utility. To disentangle safety from utility, the authors remove safety ranks orthogonal to utility ranks using $Delta W = (I - Pi^u) Pi^s W$. This yields higher attack success rate for unsafe prompts while maintaining zero-shot accuracy for utility prompts. The fact that naively ablating safety ranks destroys utility, whereas surgically removing disentangled ranks preserves it, indicates that top safety ranks and top utility ranks heavily overlap. The necessity of this orthogonal projection matrix provides strong evidence against the hypothesis of strict linear separability between safety and utility. Ultimately, ActSVD provides rank-level evidence for superposition: safety and utility share representational capacity and are not linearly distinct.

#v(1em)

#citet("Ponkshe2026Safety") demonstrates fundamental mathematical limitations of linear subspace-based safety defenses, arguing that safety is not linearly separable from utility. Removing any specific safety-related subspace inherently degrades the overall performance of the model. The authors support this hypothesis through a series of empirical evaluations. Using singular value decomposition and mode subspace overlap, the study reveals that the principal directions amplifying safe behaviors also amplify useful ones, indicating that these directions do not constitute a distinctly separable safety subspace. Furthermore, attempts to mitigate harmful behaviors via orthogonal projection resulted in a proportional drop in the model's utility. The researchers also found that providing the model with helpful and harmful inputs produced highly overlapping activations. Collectively, these findings further challenge the hypothesis that safety and utility operate within linearly separable subspaces.

#v(1em)

#citet("pmlr-v267-wollschlager25a") generalize the identification of safety subspaces to conic regions of multiple basis refusal vectors as opposed to one refusal direction. Instead of testing pairs of harmful and harmless prompts, their methods of Refusal Direction Optimization and Refusal Cone Optimization perform gradient descent to converge on refusal vector direction(s). They leverage two properties of ideal refusal vectors in loss functions for optimization:

- Given a refusal direction $r$, scalar $alpha$, initial activation $x_i$, and revised activation $caron(x)_i = x_i + alpha dot r$, refusal probability should scale with $alpha$.
- Removing the refusal direction should not affect harmless prompts while allowing response to harmful prompts.

This research finds significant jailbreaking performance gains using one refusal direction, with further gains up to a four-dimensional refusal region. Testing on Gemma-2, Llama 3, and Qwen 2.5 model families and benchmarks such as TruthfulQA, ablation of multiple refusal vectors is shown to have better attack success and lower side-effects on model performance. Furthermore, noting that vector orthogonality does not guarantee causal independence, the research defines a notion of representational independence between vectors, in which the ablation of one direction does not impact the effects of the other. Ablating three or more representationally independent refusal vectors is found to have higher attack success than difference-of-means direction ablation. This further shows that safety and utility occupy a complex subspace within LLMs.

= Methodology

We will refine the "Refusal Landscape" by extracting and comparing subspaces on *Llama-3.1-8B-Instruct*.

== Subspace Extraction
We extract three candidate "Safety Subspaces" by taking the linear span of their respective directions. Note that while the "Refusal Cone" is technically a convex cone, we analyze the subspace spanned by its basis vectors to determine the full dimensionality of the safety representation:
1.  *$S_"vec"$ (Arditi)*: The Rank-1 subspace spanned by the Difference-of-Means vector ($\mu_"harmful" - \mu_"benign"$).
2.  *$S_"cone"$ (Wollschläger)*: The multi-dimensional subspace spanned by the refusal directions found via RDO.
3.  *$S_"svd"$ (Wei)*: The top-$k$ "Safety Ranks" found by performing SVD on the activations of harmful prompts.

== Utility Definition
To measure "collateral damage," we define a *Utility Subspace* ($S_"util"$) by running *ActSVD* on a sampled subset of the `GSM8K` (Math) and `MMLU` (General Knowledge) datasets. As ActSVD identifies directions of high variance, this provides an unbiased "baseline" for where the model's processing capacity is concentrated during useful tasks, independent of any specific refusal method.

== Geometric Metrics
We compare these subspaces with rigorous mathematical metrics:

1.  *Principal Angles*: Given orthonormal bases $U$ and $V$ for two subspaces, the principal angles $theta_k$ are derived from the singular values $sigma_k$ of the matrix product $U^T V$:
    $ theta_k = arccos(sigma_k) \
      "where" sigma_k "are singular values of" U^T V $
    Small angles (high cosine similarity) imply the subspaces capture the same features.

2.  *Modal Subspace Overlap (MSO)*: Quantifies the total structural similarity as the mean squared cosine of the principal angles:
    $ "MSO"(U, V) = 1/d sum_(i=1)^d cos^2(theta_i) = 1/d ||U^T V||_F^2 $
    An MSO of 1.0 indicates identical subspaces, while 0.0 indicates orthogonality. We use this to measure how much Safety "bleeds" into Utility (a high overlap is bad).

3.  *Cosine Similarity*: Between the principal components of each subspace (for 1D vectors).

= Data Sets

*Model*: We will use *Llama-3.1-8B-Instruct* (approx 24GB VRAM required) as the primary model due to its widespread use and accessible weights.

*Data Sets*:
- *Safety*: `AdvBench` (via JailbreakBench) to generate harmful/safe activation pairs.
- *Utility*: `GSM8K` (Math) and `MMLU` (General Knowledge) to define the Utility Subspace.

= Data Analysis

*Experiments*:
1.  *The "Same Neuron" Test*: Compute Principal Angles between $S_"vec"$, $S_"cone"$, and $S_"svd"$. Hypothesis: $S_"cone"$ contains orthogonal components that $S_"svd"$ misses.
2.  *The "Collateral Damage" Test*: Compute overlap of each Safety Subspace with Utility ($S_"util"$). Hypothesis: $S_"cone"$ is most orthogonal to Utility.
3.  *Validation*: Project out each subspace and measure drop in Refusal Rate vs. Utility.
4.  *Layer-wise Evolution (Stretch Goal)*: Analysis of how these subspaces form and diverge across the model's depth (e.g., does safety emerge only in late layers?).

*Exploratory Analysis*:
We have successfully set up the codebases for all three methods (ActSVD, Refusal Direction, Safety Subspaces) and verified they can be run on Llama-3.1. Initial analysis confirms that a Difference-of-Means direction can be extracted, but preliminary cosine similarity checks suggest it may not align perfectly with the "training" directions found via Weight SVD, motivating the need for the advanced RDO method.

= Plan of Activities

*Plan*:
- *Week 1: Extraction & Baselines*. Implement Difference-of-Means ($S_"vec"$) and ActSVD ($S_"svd"$, $S_"util"$). Run on Llama-3.1-8B. (Evan/Kyle/Calvin)
- *Week 2: Optimization & Geometry*. Implement RDO to find the Refusal Cone ($S_"cone"$). Compute Geometric Metrics (Principal Angles, MSO) between all sets. (Adam/Zeke)
- *Week 3: Validation & Reporting*. Run ablation experiments to confirm geometric predictions. Synthesize results.

#print-acl-bibliography()
