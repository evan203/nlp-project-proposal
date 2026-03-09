
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
  Modern Large Language Models (LLMs) undergo extensive alignment in order to ensure 
  they don't produce harmful outputs, while still being as helpful as possible.
  However, the methods by which LLMs navigate these two competing goals are not well
  understood. This alignment can also be very fragile, being susceptible to a variety
  of jailbreaks that seem to bypass the alignment entirely. Recent research suggests
  that this fragility stems from the superposition of safety and utility in the
  model's activation space. Several methods have been proposed to isolate safety
  and utility within the activation space, including Difference in Means (DIM) #cite("arditi2024"),
  Refusal Cone Optimization (RCO) #cite("pmlr-v267-wollschlager25a"), and ActSVD #cite("Wei2024Brittleness"). 
  Conversely, other research suggests that it might be impossible to completely linearly
  separate safety and utility in current LLMs #cite("Ponkshe2026Safety"). In this project, we will compare these
  separation methods in order to gain a better understanding of the extent to which
  utility and safety are separable, and if they are, which techniques are effective
  at separating them. We will use LLaMA-3.1-instruct along with the Alpaca and
  BeaverTails datasets as a standard environment for each of our baseline methods,
  then compare the subspaces they create using Mode Subspace Overlap (MSO) and
  Representational Independence. By quantifying the difference between these methods,
  we hope to gain a better understanding of how and why they are different and what
  each method's practical use might be when training safe and robust LLMs.
]

= Introduction

1. Context and Motivation
As LLMs have become more powerful and more accessible, LLM alignment techniques
have become significantly more important. However, despite much research in this
area, modern LLMs remain fragile, being able to be jailbroken by a variety of
methods, including special prompts and white-box methods like DIM #cite("arditi2024") and ActSVD #cite("Wei2024Brittleness").
Improving the robustness of these models will require moving beyond current
empirical methods and developing a deep theoretical understanding of how safety
and utility are represented within models.

2. The Problem: Superposition in Activation Space
Recent research into mechanistic interpretability has shown that model behavior
is determined by distinct directions within the activation space. For example, there
might be a single direction that determines model refusal, and by adjusting
that direction within the activation space, we can control whether or not the
model refuses to answer a prompt #cite("arditi2024"). However, a fundamental challenge to this way of understanding activation spaces is _superposition_.
When models need to represent more features than they have dimensions, some
dimensions must contain information about multiple features. Recent research
suggests that safety and utility share representation capacity, and thus any
attempt to adjust one of these features through linear modification may
(and probably will) degrade the other.

3. The Gap in Current Literature
Recent research has introduced several methods to identify safety and utility
subspaces. Difference in Means (DIM) #cite("arditi2024") identifies a single
vector mediating refusal, Refusal Cone Optimization (RCO) maps a multidimensional
cone space #cite("pmlr-v267-wollschlager25a"), and ActSVD isolates low-rank matrices via singular value decomposition (SVD) #cite("Wei2024Brittleness"). 
While each of these methods successfully creates a safety-related subspace, each has a different mathematical
geometry, and it is not well understood how they all relate.

4. Proposed Research
In this project, we aim to investigate the superposition of safety and utility
by comparing each of these baseline methods. We will implement DIM, RCO, and
ActSVD on the LLaMA-3.1-instruct model using the Alpaca and BeaverTails datasets.
We will evaluate the effectiveness of each model, then compute the overlap of
the subspaces using Mode Subspace Overlap (MSO) and Representational Independence.
By evaluating the overlap of these methods, we hope to clarify how each subspace
relates to the overall safety and utility subspaces within the model in order
to create a foundation for future safety and alignment methods.


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

#lorem(80)

= Data Sets

#lorem(80)

= Data Analysis

#lorem(80)

= Plan of Activities

#lorem(80)

#add-bib-resource(read("bibliography.bib"))
#print-acl-bibliography()
