
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

Using low-rank modifications, B Wei et al show the brittleness of LLM safety
alignment. They find that the isolated safety-critical ranks of LLama2-chat
model corresponding to safety alignment are 2.5% of the total ranks. They
identify and isolate these regions by testing how the model changes after the
removal of ranks of the weight matrices. They propose the method ActSVD, which
performs SVD on $W X_"in"$. They store response activations before the layer $W$
into $X_"in" = [X_1, ..., X_n] in RR^(d_"in" times n)$. They find a low-rank
matrix $hat(W)$ where the Frobenius norm of the change to the output is
minimized: $ hat(W) = limits(arg min)_("rank" hat(W) <= r) norm(W X_"in" - hat(W) X_"in")^2_"F" $

= Methodology

#lorem(80)

= Data Sets

#lorem(80)

= Data Analysis

#lorem(80)

= Plan of Activities

#lorem(80)

// Uncomment this to include your bibliography:
// #add-bib-resource(read("custom.bib"))
// #print-acl-bibliography()
