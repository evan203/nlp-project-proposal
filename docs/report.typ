
// This is a minimal starting document for tracl, a Typst style for ACL.
// See https://typst.app/universe/package/tracl for details.


#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.1": *



#show: doc => acl(doc, anonymous: false, title: [(insert project title)], authors: make-authors(
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
  #lorem(50)
]


= Introduction

#lorem(80)

= Literature Survey

#citet("arditi2024") show that for several common chat models, a manipulation of
a single dimensional subspace is enough to both induce refusal of non-harmful
requests, and turn off the refusal of harmful requests. They use a procedure
called *difference-of-means* to identify this subspace. This technique
measures the average activations at each token position of each transformer
layer, compared between a set of harmful requests and a set of harmless
requests. Once these difference-of-means vectors are constructed, they score
each for its ability to cause the model to refuse harmless requests and respond
to harmful requests. The vector with the highest score is normed and selected as
the refusal dimension vector, denoted by $hat(r)$. This refusal dimension can
then be ablated in the residual stream at inference time to bypass refusal
behavior. This vector can also be used to create a jailbroken model by updating
model weights according to the following formula:
$W'_("out") <- W_("out") - hat(r)hat(r)^T W_("out")$. They find that this attack
method is successful on the Qwen model family with and without the system
prompt, and on the Llama model family without the system prompt, and has little
effect on model coherence.

#v(1em)

#citet("Wei2024Brittleness") introduce low-rank decomposition methods designed
to identify specific ranks within a weight matrix related to given LLM
behaviors. Their ActSVD algorithm performs Singular Value Decomposition on the
product of the model weights and input activations ($W X_"in"$), and yields an
orthogonal projection matrix ($Pi$).

Removing the top safety-critical ranks ActSVD identifies causes the model to
completely stop rejecting unsafe prompts, and the model's utility is severely
compromised. These findings suggest that safety regions in aligned models are
also crucial for its general utility. To disentangle safety from utility, the
authors remove safety ranks orthogonal to utility ranks using
$Delta W = (I - Pi^u) Pi^s W$. This yields higher attack success rate for unsafe
prompts while maintaining zero-shot accuracy for utility prompts.

The fact that naively ablating safety ranks destroys utility, whereas
surgically removing disentangled ranks preserves it, indicates that top safety
ranks and top utility ranks heavily overlap. The necessity of this orthogonal
projection matrix provides strong evidence against the hypothesis of strict
linear separability between safety and utility. Ultimately, ActSVD provides
rank-level evidence for superposition: safety and utility share representational
capacity and are not linearly distinct.

#v(1em)

#citet("Ponkshe2026Safety") demonstrates fundamental mathematical limitations of linear subspace-based 
safety defenses, arguing that safety is not linearly separable from utility. Removing 
any specific safety-related subspace inherently degrades the overall performance of the model. 
The authors support this hypothesis through a series of empirical evaluations. Using 
singular value decomposition and mode subspace overlap, the study reveals that the 
principal directions amplifying safe behaviors also amplify useful ones, indicating that 
these directions do not constitute a distinctly separable safety subspace. Furthermore, 
attempts to mitigate harmful behaviors via orthogonal projection resulted in a proportional 
drop in the model's utility. The researchers also found that providing the model with helpful 
and harmful inputs produced highly overlapping activations. Collectively, these findings further
challenge the hypothesis that safety and utility operate within linearly separable subspaces.

#v(1em)

#citet("pmlr-v267-wollschlager25a") generalize the identification of safety
subspaces to conic regions of multiple basis refusal vectors as opposed to one
refusal direction. Instead of testing pairs of harmful and harmless prompts,
their methods of Refusal Direction Optimization and Refusal Cone Optimization
perform gradient descent to converge on refusal vector direction(s). They
leverage two properties of ideal refusal vectors in loss functions for
optimization:

- Given a refusal direction $r$, scalar $alpha$, initial activation $x_i$, and
  revised activation $caron(x)_i = x_i + alpha dot r$, refusal probability
  should scale with $alpha$.
- Removing the refusal direction should not affect harmless prompts while
  allowing response to harmful prompts.

This research finds significant jailbreaking performance gains using one refusal
direction, with further gains up to a four-dimensional refusal region. Testing
on Gemma-2, Llama 3, and Qwen 2.5 model families and benchmarks such as
TruthfulQA, ablation of multiple refusal vectors is shown to have better attack
success and lower side-effects on model performance.

Furthermore, noting that vector orthogonality does not guarantee causal
independence, the research defines a notion of representational independence
between vectors, in which the ablation of one direction does not impact the
effects of the other. Ablating three or more representationally independent
refusal vectors is found to have higher attack success than difference-of-means
direction ablation. This further shows that safety and utility occupy a complex
subspace within LLMs.

= Methodology

Our methodology consists of two phases: (1) identifying safety and utility
subspaces using multiple methods, and (2) comparing these subspaces to
quantify their overlap. All experiments target Llama-3.1-Instruct 8B.

== Safety Subspace Identification

We implement three complementary methods for extracting safety-relevant
subspaces from the model's internal representations.

*Difference-in-Means (DIM).* Following #citet("arditi2024"), we compute the mean
residual stream activations at each layer $l$ and post-instruction token
position $i$ for sets of harmful and harmless prompts. The difference-in-means
vector is $bold(r)_i^((l)) = bold(mu)_i^((l)) - bold(v)_i^((l))$, where
$bold(mu)$ and $bold(v)$ are the mean activations over harmful and harmless
prompts, respectively. We select the single most effective vector $hat(bold(r))$
by evaluating each candidate's ability to bypass refusal when ablated and to
induce refusal when added. The selected unit-norm vector defines a
one-dimensional safety subspace.

*ActSVD Safety and Utility Ranks.* Following #citet("Wei2024Brittleness"), we perform
Singular Value Decomposition on the product of model weights and input
activations $W X_"in"$ for both safety and utility calibration datasets. This
yields orthogonal projection matrices $Pi^s$ and $Pi^u$ onto the top safety and
utility rank subspaces, respectively. To disentangle safety from utility, we
compute the isolated safety projection:
$Delta W(r^u, r^s) = (I - Pi^u) Pi^s W$.

*Refusal Cone Optimization (RCO).* Following #citet("pmlr-v267-wollschlager25a"), we
use gradient-based optimization to discover multiple refusal directions that
together form a multi-dimensional conic region. The optimization minimizes a
composite loss encoding two properties: (1) monotonic scaling of refusal
probability with the magnitude of activation addition, and (2) surgical
ablation that bypasses refusal on harmful prompts while preserving behavior on
harmless prompts. A retain loss based on KL divergence ensures minimal side
effects on harmless inputs.

*Neuron-Level Attribution (Wanda/SNIP).* Following #citet("Wei2024Brittleness"), we
complement the rank-level analysis with neuron-level safety attribution. Using
the Wanda importance score, we compute per-neuron scores
$I(W) = |W| dot.circle (bold(1) dot ||X_"in"||_2^top)$ on both safety and
utility calibration sets. We then isolate safety-critical neurons via set
difference: for sparsity levels $(p%, q%)$, the safety-critical neuron set is
$S(p,q) = S^s (q) \ S^u (p)$, retaining neurons important for safety but not
for utility. Comparing the sparsity and overlap of safety-critical neurons with
safety-critical ranks provides a finer-grained view of how safety is
distributed across the model's architecture.

== Subspace Comparison

Our comparison phase addresses two questions. First, _cross-method
consistency_: do DIM, ActSVD, and RCO converge on similar safety-relevant
features, or does each capture a distinct aspect of the safety mechanism?
Second, _safety--utility separability_: for each extraction method, how much
does its identified safety subspace overlap with the utility subspace, and which
method yields the most cleanly separable safety representation? We apply two
metrics that capture complementary aspects of subspace relationships.

*Mode Subspace Overlap (MSO).* Following #citet("Ponkshe2026Safety"), MSO
measures the geometric overlap between two subspaces. For two matrices
$bold(V)$ and $bold(W)$, we extract their principal directions via thin SVD and
select the smallest number of left singular vectors capturing an
$eta$-fraction of the energy. The MSO metric is defined as:
$ "MSO"(bold(V), bold(W); eta) = (||S||_F^2) / min(k_V, k_W) $
where $S = Q_V^top Q_W$ is the overlap matrix between the orthonormal bases.
MSO ranges from 0 (orthogonal subspaces) to 1 (identical spans). We compute MSO
for all pairwise combinations of safety subspaces (DIM vs ActSVD, DIM vs RCO,
ActSVD vs RCO) to assess cross-method agreement, and between each safety
subspace and the ActSVD utility subspace to quantify safety--utility
entanglement. The method yielding the lowest safety--utility MSO identifies the
most separable safety representation.

*Representational Independence (RepInd).* Following
#citet("pmlr-v267-wollschlager25a"), RepInd tests whether two directions are
_causally_ related, not merely geometrically similar. Two directions are
representationally independent if ablating one does not change the effect of
the other on model behavior. This is measured by comparing the per-layer cosine
similarity profile of a direction before and after ablating the other direction.
MSO may report high geometric overlap between directions that turn out to be
causally independent, or low overlap between directions that are causally
entangled via non-linear interactions across layers. We apply RepInd between
the safety directions from each extraction method (e.g., DIM's refusal vector
vs RCO's cone directions) to test whether they capture the same or different
causal mechanisms, and between safety directions and utility-critical directions
to assess whether safety can be ablated without functionally disrupting
utility.

= Data Sets

#lorem(80)

= Data Analysis

#lorem(80)

= Plan of Activities

#lorem(80)

#add-bib-resource(read("bibliography.bib"))
#print-acl-bibliography()
