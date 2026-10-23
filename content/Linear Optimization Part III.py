import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Optimization Part III: Duality

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
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path
    from urllib.request import urlretrieve
    import importlib
    import importlib.util
    import sys

    # Determine whether the local plotting module is available
    local_paths = [
        Path("utils/plot_lp_2d.py"),
        Path("content/utils/plot_lp_2d.py"),
    ]

    local_path = next(
        (p for p in local_paths if p.is_file()),
        None,
    )

    if local_path is not None:
        # Local notebook or marimo-book build
        # No static import for marimo's dependency scanner to detect
        parent_dir = local_path.parent.parent.resolve()
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        plot_lp_2d = importlib.import_module(
            "utils.plot_lp_2d"
        ).plot_lp_2d

    else:
        # Molab: download the source file from GitHub
        url = (
            "https://raw.githubusercontent.com/"
            "STE-131-QADM/Lecture-Notes/"
            "main/content/utils/plot_lp_2d.py"
        )

        module_path = Path("/tmp/ste131_plot_lp_2d.py")
        urlretrieve(url, module_path)

        spec = importlib.util.spec_from_file_location(
            "ste131_plotting",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        plot_lp_2d = module.plot_lp_2d
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pyomo.environ as pyo
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    import highspy
    from matplotlib.patches import Circle, Polygon

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have now already seen the dual parameters in action in the food network example. This example already showed some ways that the solution of an optimisation problem is connected to the value of the dual parameter.

    This part of the linear optimization lectures will be all about that. We will be revolving around one main question: **how valuable would additional amount of our scarce resource be?**

    ## Lagrangian Function

    We are already familiar with the inequality form of a LP, that turned out to be general enough for it to be sufficient to represent any LP. Let's now consider a problem of form

    \[
        \begin{aligned}
            &\min_x c^\top x \\
            \text{s.t. }& \\
            & Ax\leq b.
        \end{aligned}
    \]

    This formulation requires that our constraint is satisfied exactly *i.e.* the formulation requires that \(Ax\leq b\) holds without any exceptions. What would it mean for the constraint to be enforced only approximately?

    One approach to this problem, that turns out to be important in the analysis and solving optimization problems is to add the constraint into the objective function with a penalisation. We do the following tric, called *relaxation* of the constraint

    \[
        \begin{aligned}
            \mathcal{L}(x,\lambda)&=c^\top x+\sum_{i=1}^m \lambda_i(a_ix-b_i)\\
            &=c^\top x+\lambda^\top (Ax-b)
        \end{aligned}
    \]

    where \(a_i\) denotes the \(t^\text{th}\) row of \(A\in\mathbb{R}^{m\times n}\) and \(\lambda\geq 0\) is a parameter of the Lagrangian function .

    ???- question "Lagrangian as a function of \(\lambda\)"
        Consider what happens to the problem when \(\lambda\) is increased or decreased.

        **Draw the \(\mathcal L \) as a function of \(\lambda\) for constant \(x\).**

        **How does this change for \(Ax=b\) or \(Ax\neq b\).**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Additional Materials

    + If you are interested in learning more about mathematical optimization, the most used book on the intermediate level is [Boyd, S. and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge: Cambridge University Press.](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) It is an exellent book for understanding the mathematics, but it goes far beyond the topics of this course. Body and Vandenberghe write about the audience as follows:

    !!! quote "Boyd, S. and Vandenberghe, L. *Convex Optimization*"
        This book is meant for the researcher, scientist, or engineer who uses mathematical optimization, or more generally, computational mathematics.

    + A good treatment from the operations research perspective is [Hillier, F.S. (1980). *Introduction to operation research* Frederick S. Hillier, Gerald J. Lieberman.](https://www.semanticscholar.org/paper/Introduction-to-operation-research-Frederick-S.-J.-Hillier/0f8a045d6d7b6e165ac806872d858fb0b83c7e78?utm_source=direct_link)

    + Good treatment of linear optimization topics can be found from [Fabricio Oliveira (2024) *Linear optimisation notes*.](https://github.com/gamma-opt/linopt-notes) These notes go through the theory of linear optimisation and mixed integer linear programming in detail and also introduces the main algorithms that are used to solve such problems.

    + If you are interested in optimisation applied to a specific field, and the field happens to be electricity, a good treatment of all topics in electricity markets and transmission can be found from [Papavasiliou, A. (2024). *Optimization Models in Electricity Markets*. Cambridge: Cambridge University Press.](https://www.cambridge.org/highereducation/books/optimization-models-in-electricity-markets/0D2D36891FB5EB6AAC3A4EFC78A8F1D3?utm_campaign=shareaholic&utm_medium=copy_link&utm_source=bookmark)

    + Similarly for forest management [Buongiorno, J., & Gilless, K. J. (2003). *Decision methods for forest resource management* Joseph Buongiorno, J. Keith Gilless. Academic Press.](https://helka.helsinki.fi/permalink/358UOH_INST/1r0ctnh/alma9917166953506253)

    + And [Kangas, A., Kurttila, M., Hujala, T., Eyvindson, K. J., & Kangas, J. (2015). *Decision support in Forest Management: Second Edition.* (2 ed.) (Managing Forest Ecosystems; No. 30). Springer.](https://link.springer.com/book/10.1007/978-3-319-23522-6)
    """)
    return


if __name__ == "__main__":
    app.run()
