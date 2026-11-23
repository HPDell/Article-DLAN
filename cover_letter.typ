#set page(paper: "a4", margin: 2cm)
#set text(font: "Times New Roman", size: 11pt)
#set par(first-line-indent: 0em, justify: true, spacing: 0.85em)

#let metadata = (
  title: "Dimension-specific local attention for nonlinear regression in geographical and temporal spaces",
  journal: "International Journal of Geographical Information Science",
)

#align(right)[
  Binbin Lu \
  School of Remote Sensing and Information Engineering \
  Wuhan University, Wuhan 430079, China \
  #link("mailto:binbinlu@whu.edu.cn")[binbinlu\@whu.edu.cn]
]

Editorial Office \
#metadata.journal

#v(1em)

Dear Editor,

#set par(first-line-indent: 2em)

We are pleased to submit our manuscript, #emph(metadata.title), for consideration as a research article in #emph(metadata.journal).

Spatial dependence and spatial heterogeneity motivate local regression methods that account for geographical context when modelling relationships between covariates and responses.
Geographically weighted regression and its extensions provide a well-established framework, but their local linear formulations can restrict the representation of nonlinear relationships.
Existing nonlinear approaches are designed for specific types of spatial data and do not always distinguish the scales and contributions of individual geographical and temporal dimensions, causing several problems when applied to multidimensional data, like spatio-temporal and spatial flow data.
Our manuscript addresses these issues through a dimension-specific local attention network (DLAN), extending the dimension-specific locality of geographical and temporal density regression to nonlinear prediction.
DLAN learns separate scales and attention projections for each coordinate dimension.
It combines content similarity, directional geographical biases and a Gaussian product prior to aggregate information from neighbouring training observations.
The resulting local context modulates a nonlinear prediction network, allowing covariate–response relationships to vary with geographical and temporal conditions.
This construction accommodates planar, spatio-temporal and spatial flow data within a common modelling framework.

We evaluate DLAN through four case studies covering carbon dioxide emissions in China, urban mobility flows and housing prices in Wuhan, and the green view index in Shanghai.
Across ten training runs per case, DLAN achieves the lowest mean test RMSE among the models compared in the reported configurations.
Further analyses examine learned coordinate-specific scales, residual patterns and sensitivity to neighbourhood size and attention width, clarifying how the model performs across different geographical data structures.

We believe the manuscript will interest the journal's readership through its contribution to spatially explicit modelling and geographical information science.
By connecting dimension-specific geographical weighting with locally conditioned nonlinear regression, DLAN provides a common approach to prediction across heterogeneous coordinate spaces.
The environmental, urban flow, housing and street-level greenery applications illustrate its relevance to geographical analysis and GeoAI.

// Authors should verify the submission-status and author-approval declarations before submission.
We confirm that this manuscript is original, has not been published previously, and is not under consideration for publication elsewhere.
The word count including references is less than 8000 words.
All authors have approved the manuscript and its submission.
The authors declare no competing interests.
A private link to the data and code is provided in the manuscript for review.
The manuscript also discloses the use of generative AI and the authors' responsibility for its content.

Thank you for considering our manuscript.

#v(1em)

#block[
  Best regards,\
  Binbin Lu \
  Corresponding author, on behalf of all authors
]
