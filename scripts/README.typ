#import "@preview/cmarker:0.1.10"
#import "@preview/mitex:0.2.7"
#import "@preview/parize:0.2.1"
#import "@preview/itemize:0.3.1"
#import "@preview/codly:2.0.0" as codly

#set text(font: "Times New Roman")
#set par(first-line-indent: (amount: 1em, all: true))
#set heading(numbering: "1.1.")
#show heading: set block(below: 1em)
#show: itemize.default-enum-list.with(
  indent: 1em
)
#set table(
  align: (x, y) => if y == 0 {center + horizon} else {left},
  stroke: (x, y) => if y == 0 {(
    top: 1pt,
    bottom: 0.5pt
  )} else if y == 1 {(
    top: 0.5pt
  )} else {(
    top: 0pt,
    bottom: 1pt
  )}
)
#set table.header(repeat: true)
#show table.cell.where(y: 0): strong
#show raw.where(block: true): codly.new

#show: parize.par-indent.with(
  include-elem: (heading, math.equation, list, enum, terms),
  use-par-leading: false
)

#cmarker.render(
  read("README.md"),
  math: mitex.mitex,
  h1-level: 0
)

