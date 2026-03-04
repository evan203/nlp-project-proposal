
// This is a minimal starting document for tracl, a Typst style for ACL.
// See https://typst.app/universe/package/tracl for details.


#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.1": *



#show: doc => acl(doc, anonymous: false, title: [(insert project title)], authors: make-authors(
  (
    name: "Evan Scamehorn",
    affiliation: [University of Wisconsin\ #email("scamehorn@wisc.edu")],
  ),
))


#abstract[
  #lorem(50)
]


= Introduction

#lorem(80)

= Literature Survey

#citet("wollschlager2025geometry") generalize the identification of safety subspaces to conic regions of multiple basis refusal vectors as opposed to one refusal direction. Instead of testing pairs of harmful and harmless prompts, their methods of Refusal Direction Optimization and Refusal Cone Optimization perform gradient descent to converge on refusal vector direction(s). They leverage two properties of ideal refusal vectors in loss functions for optimization:

- Given a refusal direction $r$, scalar $alpha$, initial activation $x_i$, and revised activation $caron(x)_i = x_i + alpha dot r$, refusal probability should scale with $alpha$.
- Removing the refusal direction should not affect harmless prompts while allowing response to harmful prompts.

This research finds significant jailbreaking performance gains using one refusal direction, with further gains up to a four-dimensional refusal region. Testing on Gemma-2, Llama 3, and Qwen 2.5 model families and benchmarks such as TruthfulQA, ablation of multiple refusal vectors is shown to have better attack success and lower side-effects on model performance.

= Methodology

#lorem(80)

= Data Sets

#lorem(80)

= Data Analysis

#lorem(80)

= Plan of Activities

#lorem(80)

// Uncomment this to include your bibliography:
#add-bib-resource(read("../bibliography.bib"))
#print-acl-bibliography()
