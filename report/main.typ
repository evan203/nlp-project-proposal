
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
  )
))


#abstract[
  #lorem(50)
]


= Introduction

#lorem(80)

= Literature Survey

#citet("Wei2024Brittleness") introduce low-rank decomposition methods designed to identify
specific ranks within a weight matrix related to given LLM behaviors. Their
ActSVD algorithm performs Singular Value Decomposition on the product of the model weights and input
activations ($W X_"in"$), and yields an orthogonal projection matrix ($Pi$).
Removing the top safety-critical ranks ActSVD identifies causes the model to
completely stop rejecting unsafe prompts, and the model's utility is severely
compromised. These findings suggest that safety regions in aligned models are
also crucial for its general utility. To disentangle safety from utility, the
authors remove safety ranks orthogonal to utility ranks using
$Delta W = (I - Pi^u) Pi^s W$. This yields higher attack success rate for unsafe
prompts while maintaining zero-shot accuracy for utility prompts. The fact that
naively ablating safety ranks destroys utility, whereas surgically removing
disentangled ranks preserves it, indicates that top safety ranks and top utility
ranks heavily overlap. The necessity of this orthogonal projection matrix
provides strong evidence against the hypothesis of strict linear separability
between safety and utility. Ultimately, ActSVD provides rank-level evidence for
superposition: safety and utility share representational capacity and are not
linearly distinct.

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
