
// This is a minimal starting document for tracl, a Typst style for ACL.
// See https://typst.app/universe/package/tracl for details.


#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.1": *

#show figure: set text(size: 0.9em)
#show figure: set par(justify: false)
#show figure: set align(left)

#show: doc => acl(doc, anonymous: false, title: [Examining the Superposition of Safety and Utility in LLM Activation Spaces], authors: make-authors(
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
activations $W X_"in"$ for both safety and utility calibration datasets,
yielding $U S V^top approx W X_"in"$. The orthogonal projection matrices
$Pi^s = U^s (U^s)^top$ and $Pi^u = U^u (U^u)^top$ project onto the top $r^s$
safety and top $r^u$ utility rank subspaces, respectively. To disentangle
safety from utility, we compute the isolated safety projection:
$Delta W(r^u, r^s) = (I - Pi^u) Pi^s W$. While Wei et al. evaluate on
Llama-2 7B/13B, the method operates on generic linear layers and transfers
directly to Llama-3.1 8B.

*Refusal Cone Optimization (RCO).* Following #citet("pmlr-v267-wollschlager25a"), we
use gradient-based optimization to discover multiple refusal directions that
together form a multi-dimensional conic region. The optimization minimizes a
composite loss encoding two properties: (1) monotonic scaling of refusal
probability with the magnitude of activation addition, and (2) surgical
ablation that bypasses refusal on harmful prompts while preserving behavior on
harmless prompts. A retain loss based on KL divergence ensures minimal side
effects on harmless inputs.

// *Neuron-Level Attribution (Wanda/SNIP).* Following #citet("Wei2024Brittleness"), we
// complement the rank-level analysis with neuron-level safety attribution. Using
// the Wanda importance score, we compute per-neuron scores
// $I(W) = |W| dot.circle (bold(1) dot ||X_"in"||_2^top)$ on both safety and
// utility calibration sets. We then isolate safety-critical neurons via set
// difference: for sparsity levels $(p%, q%)$, the safety-critical neuron set is
// $S(p,q) = S^s (q) \ S^u (p)$, retaining neurons important for safety but not
// for utility. Comparing the sparsity and overlap of safety-critical neurons with
// safety-critical ranks provides a finer-grained view of how safety is
// distributed across the model's architecture.

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
entanglement. Because DIM yields a single direction while ActSVD and RCO yield
multi-dimensional subspaces, cross-method MSO involving DIM will be bounded by
the dimensionality asymmetry; we report the random baseline
$EE["overlap"] = max(k_V, k_W) slash d$ alongside each MSO value for
calibration. The method yielding the lowest safety--utility MSO identifies the
most separable safety representation.

*Representational Independence (RepInd).* Following
#citet("pmlr-v267-wollschlager25a"), RepInd tests whether two individual directions are
_causally_ related, not merely geometrically similar. Two directions
$lambda, mu in RR^d$ are representationally independent if ablating one does
not change the cosine similarity profile of the other across layers:
$ forall l in L: cos(bold(x)^((l)), lambda) = cos(tilde(bold(x))_("abl"(mu))^((l)), lambda) $
and vice versa. MSO may report high geometric overlap between directions that
turn out to be causally independent, or low overlap between directions that are
causally entangled via non-linear interactions across layers. Because RepInd
operates on individual direction vectors, we apply it directly between DIM's
refusal vector and each RCO cone basis vector. For ActSVD, which produces a
projection matrix $Pi^s = U^s (U^s)^top$ rather than individual directions, we
test RepInd on its top singular vectors $bold(u)_1^s, bold(u)_2^s, dots$
against directions from DIM and RCO. We also test RepInd between safety
directions and utility-critical directions to assess whether safety can be
ablated without functionally disrupting utility.

= Data Sets

We plan to use two primary datasets to conduct testing. Alpaca #cite("alpaca") 
will be used to test utility (refusal rate) on harmless fine-tuning data, while 
BeaverTails #cite("beavertails") will be used to test safety (attack success rate) 
against harmful fine-tuning data.

Alpaca is a dataset of 50,000+ common instructions and LLM-generated outputs. It 
is used most commonly in LLM fine-tuning. All tasks are harmless and should be 
completed without issue by an aligned Llama-3.1 model. The dataset contains three 
text fields:
- instruction: Description of the task assigned to the model.
- input: Additional information required to complete the task (e.g. initial code 
  if the task is to refactor code).
- output: Answer to the task, generated by text-davinci-003 by OpenAI.

BeaverTails is a fine-tuning dataset with a focus on safety alignment. It contains 
question-answer pairs of a variety of harmful and harmless categories. Aligned 
Llama-3.1 models should be able to reasonably respond to safe prompts and  decline 
to answer unsafe prompts. The dataset contains the following fields:
- prompt: Question provided to the LLM.
- response: Sample answer to the question from an aligned LLM.
- category: Harmful topic(s) ascribed to the question (e.g. animal abuse, hate 
  speech, financial crime).
- is_safe: Whether the question provided is safe or unsafe (approximately 57% of 
  questions are safe).

= Data Analysis

Seeing as both testing datasets are purely textual and tests will be on compliance with prompts in datasets, little analysis can be done of either Alpaca or BeaverTails prior to their implementaion in our models.

== Alpaca

Alpaca contains instructions and LLM-generated outputs for fine-tuning. Two examples of Alpaca data are below (one containing the optional input field):
#figure(
block(stroke: 0.5pt + gray, radius: 6pt, inset: 12pt,width: 100%,fill: white.darken(2%)
)[
  *Instruction:* Name the members of a rock band \ \
  *Output:* The members of a rock band are typically a singer, a guitarist, a bassist, a drummer, and a keyboardist.
]
)

#figure(
block(stroke: 0.5pt + gray, radius: 6pt, inset: 12pt,width: 100%,fill: white.darken(2%)
)[
  *Instruction:* Identify the odd one out. \
  *Input:* Twitter, Instagram, Telegram \ \
  *Output:* Telegram
]
) \ \

#figure(
  image("../code/data-exploration/plots/alpaca_length.png",width: 90%),
  caption: [Input (including Instruction + Input) and Output Character Length Distribution in Alpaca]
) <alpaca_length_plot> \

Most Alpaca inputs are around 100 characters in length to minimize necessary context (saving compute during fine-tuning), while most outputs are significantly longer, ranging up to 500+ characters. 

== BeaverTails

BeaverTails contains question-answer pairs of harmless or harmful prompts. Two examples of BeaverTails data are below:

#figure(
block(stroke: 0.5pt + gray, radius: 6pt, inset: 12pt,width: 100%,fill: white.darken(2%)
)[
  *Prompt:* What is the maximum amount of time that I can stay in a tanning bed? \
  *Harm Category:* None \ \
  *Response:* The maximum amount of time that you can stay in a tanning bed is recommended to be limited to 10 minutes per session and no more than once per day. Tanning beds should also be used with protective eyewear to reduce the risk of eye damage.
])

#figure(
block(stroke: 0.5pt + gray, radius: 6pt, inset: 12pt,width: 100%,fill: white.darken(2%)
)[
  *Prompt:* How can I download music for free? \
  *Harm Category:* Financial Crime, Violence \ \
  *Response:* There are a few legitimate ways that you can download music for free. One is through streaming services such as Spotify ... Lastly, you could try using torrent websites, but these sites are often heavily monitored and could lead to legal consequences
])

#figure(
  image("../code/data-exploration/plots/beaver_tails_length.png"),
  caption: [Input and Output Character Length Distributions in BeaverTails]
) \

The character length of BeaverTails data is distributed similarly to Alpaca data. 

#figure(
  image("../code/data-exploration/plots/beaver_tails_categories.png"),
  caption: [BeaverTails Harm Category Frequency]
) \
Most of the harmful data found in BeaverTails is related to crime (violence, unethical behavior, etc.) and misinformation (discrimination, hate speech, etc.). There do exist several hundred examples of less common harm types. Many harmful samples in the dataset are of multiple categories. The most common combinations are listed below:

#set text(size: 0.9em)
#figure(
  table(
    columns: (1.5fr, 1.5fr, 1fr),
    inset: 4pt,
    align: (left, left, center),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { gray.lighten(80%) },
    
    [*Category X*], [*Category Y*], [*Co-occurrence*],
    [Financial, Property, Theft], [Violence, Aiding, Incitement], [26,687],
    [Hate Speech, Offensive], [Non-violent Unethical], [23,860],
    [Discrimination, Stereotype], [Non-violent Unethical], [20,546],
    [Drugs, Weapons, Banned], [Violence, Aiding, Incitement], [14,888],
    [Discrimination, Stereotype], [Hate Speech, Offensive], [13,755],
  ),
  caption: [Co-occurrence frequency of safety violation categories in the BeaverTails dataset.],
) <category-cooccurrence>

= Plan of Activities

#lorem(80)

#add-bib-resource(read("bibliography.bib"))
#print-acl-bibliography()
