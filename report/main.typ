
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

Wollschlager et al. find refusal in LLMs to be based on a conic region of multiple basis refusal vectors rather than one refusal direction. Their methods of Refusal Direction Optimization (RDO) for one vector and Refusal Cone Optimization (RCO) for multiple vectors perform gradient descent to converge on bases for refusal direction(s). Their results show refusal performance gains up to a four-dimensional conic region representing refusal on the Gemma-2 model. 

= Methodology

#lorem(80)

= Data Sets

#lorem(80)

= Data Analysis

#lorem(80)

= Plan of Activities

#lorem(80)

// Uncomment this to include your bibliography:
// #add-bib-resource(read("bibliography.bib"))
// #print-acl-bibliography()
