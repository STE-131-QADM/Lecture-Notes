# Notations, definitions and other miscellaneous mathematical topics

<style>
.md-typeset a[href^="http://"],
.md-typeset a[href^="https://"] {
    color: #1976D2 !important;
}

.md-typeset a[href^="http://"]:hover,
.md-typeset a[href^="https://"]:hover {
    color: #0D47A1 !important;
}
</style>

## Linearity

A function \(f:\mathbb{R}^n\to\mathbb{R}\) is linear if

\[
f(\alpha x+\beta y)
=
\alpha f(x)+\beta f(y)
\]

for all \(x,y\in\mathbb{R}^n\) and all \(\alpha,\beta\in\mathbb{R}\). This is equivalent to requiring that a function is additive and homogenous, *i.e.* satisfying following properties:

**Additivity**

\[
f(x+y)=f(x)+f(y)
\]

**Homogeneity of degree one**

\[
f(\alpha x)=\alpha f(x).
\]

