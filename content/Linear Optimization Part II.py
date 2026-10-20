import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Optimization Part II: General Formulation

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

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the previous part, we introduced linear optimization through examples involving one and two decision variables. However, the same mathematical principles apply regardless of the number of variables. In fact, every finite-dimensional linear optimization problem with finitely many linear constraints can be expressed in a common mathematical form.

    From a computational perspective, the number of decision variables and constraints, as well as the structure of the problem, can significantly affect how difficult it is to find an optimal solution. Nevertheless, these differences do not change the general mathematical formulation.

    ## General Formulation

    Recall that in a linear optimization problem, the objective function and the left-hand sides of the constraints are linear in the decision variables. We can express such problems compactly using vectors and matrices.

    Let

    - \(x\in\mathbb{R}^n\) denote the vector of \(n\) decision variables,
    - \(c\in\mathbb{R}^n\) denote the vector of objective-function coefficients,
    - \(A\in\mathbb{R}^{m\times n}\) denote the matrix of constraint coefficients,
    - \(b\in\mathbb{R}^m\) denote the vector of right-hand sides of the \(m\) constraints.

    Then, every linear optimization problem can be written, after suitable transformations, in the form

    \[
    \begin{aligned}
    \min_{x\in\mathbb{R}^n}\quad & c^\top x\\
    \text{subject to}\quad & Ax\leq b.
    \end{aligned}
    \]

    Here, the inequality \(Ax\leq b\) is interpreted componentwise: each row of \(A\) defines a linear inequality constraint.

    Although the formulation contains only minimization and constraints of the form \(\leq\), this does not restrict its generality. Maximizing an objective function is equivalent to minimizing its negative

    $$
        \max_x f(x)=-\left(\min_x -f(x)\right).
    $$

    Inequalities of the form \(\geq\) can be reversed by multiplication by \(-1\), and equality constraints can be expressed as pairs of inequalities. Constant terms in the objective function can also be omitted without changing the optimal decisions.

    Consequently, the formulation above is sufficiently general to represent all linear optimization problems, even when their original formulations look quite different.

    ???- question "Problem formulation ☠️"
        Consider the electricity-import optimization problem introduced in the previous part.

        **Express the problem in the general form**

        $$
        \begin{aligned}
            \min_{x\in\mathbb{R}^n}\quad & c^\top x\\
            \text{subject to}\quad & Ax\leq b.
        \end{aligned}
        $$

        In particular, determine the decision-variable vector \(x\), the objective-function coefficient vector \(c\), the constraint matrix \(A\), and the constraint vector \(b\).

        Remember that all the constraints must be expressed using inequalities of the form \(\leq\). Include the nonnegativity constraints in your formulation.

        **Verify that your formulation is equivalent to the original problem.**

    ???- question "Check linearity ☠️☠️"
        Using the definition of linearity introduced in the previous part, consider the function

        $$
        f(x)=c^\top x,
        $$

        where \(c\in\mathbb{R}^n\) is a fixed vector.

        **Show that \(f\) is a linear function.**

    ???- question "Relationship between minimization and maximization ☠️☠️☠️"
        Consider a real-valued function \(f\) defined on a nonempty feasible set \(\mathcal F\), and assume that \(f\) attains its maximum on $\mathcal F$.

        **Prove that**

        $$
        \max_{x\in\mathcal F} f(x)
        =
        -\min_{x\in\mathcal F}\bigl(-f(x)\bigr).
        $$

        Does this relationship require \(f\) to be a linear function?

    !!! info "Representation of linear functionals in $\mathbb{R}^n$"
        It turns out that *every* linear function \(f:\mathbb{R}^n\to\mathbb{R}\) can be represented as

        $$
        f(x)=c^\top x
        $$

        for a unique vector \(c\in\mathbb{R}^n\).

        To see why, recall that any vector \(x\in\mathbb{R}^n\) can be written as

        $$
        x=\sum_{i=1}^n x_i e_i,
        $$

        where \(e_i\) are the standard basis vectors of \(\mathbb{R}^n\) *i.e.* vectors where the $i^{\text{th}}$ element is one and every other element is zero. By linearity,

        $$
        f(x)=\sum_{i=1}^n x_i f(e_i).
        $$

        Thus, defining \(c_i=f(e_i)\) gives the desired representation.

        This elementary result from linear algebra is also a finite-dimensional special case of the more general [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem) from functional analysis.

    ## Feasible Set and Convexity

    The feasible set of a linear problem is always of the form

    $$
        \mathcal{F}=\{x\in\mathbb{R}^n\mid Ax\leq b\}
    $$

    by the previous section. It turns out that the feasible set satisfies a mathematical property that consequently turns out to be one of the most important properties in mathematical optimization. The property is called convexity.

    !!! definition "test"
        test
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
