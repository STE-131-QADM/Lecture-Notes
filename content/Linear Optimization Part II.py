import marimo

__generated_with = "0.25.2"
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
        color: #0D47A1 !important;<style>
    .md-typeset a[href^="http://"],
    .md-typeset a[href^="https://"] {
        color: #1976D2 !important;
    }

    .md-typeset a[href^="http://"]:hover,
    .md-typeset a[href^="https://"]:hover {
        color: #0D47A1 !important;
    }
    </style>
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
    return (plot_lp_2d,)


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

    return Circle, Polygon, mo, np, plt, pyo


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

    Then, every linear optimization problem can be written, after suitable transformations, in the form inequality form.

    !!! definition "Inequality form of the LP"

        The inequality form of an LP is:

        \[
        \begin{aligned}
        \min_{x\in\mathbb{R}^n}\quad & c^\top x\\
        \text{subject to}\quad & Ax\leq b.
        \end{aligned}
        \]

        Every LP can be expressed in this form.

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

        where \(e_i\) are the standard basis vectors of \(\mathbb{R}^n\), *i.e.* vectors whose \(i^{\text{th}}\) component is one and all other components are zero. By linearity,

        $$
        f(x)=\sum_{i=1}^n x_i f(e_i).
        $$

        Thus, defining \(c_i=f(e_i)\) gives the desired representation.

        This elementary result from linear algebra is also a finite-dimensional special case of the more general [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem) from functional analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feasible Set and Convexity

    The feasible set of a linear problem is always of the form

    $$
        \mathcal{F}=\{x\in\mathbb{R}^n\mid Ax\leq b\}
    $$

    by the previous sectins argument used for the matrix inequality. It turns out that the feasible set satisfies a mathematical property that consequently turns out to be one of the most important properties in mathematical optimization. The property is called convexity.

    Convex sets can be described as *sets where every two points inside the set can be connected by straight line so that the line is fully contained inside the set.* Convexity can be understood easiest by picture, as it is very much a geometric concept.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lambda_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.1,
        value=0.0,
        show_value=True,
        label="λ"
    )
    lambda_slider
    return (lambda_slider,)


@app.cell(hide_code=True)
def _(Circle, Polygon, lambda_slider, np, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    set_color = "#A9D8C1"
    edge_color = "#247A57"
    line_color = "#2563EB"
    invalid_color = "#DC2626"

    ax = axes[0]

    circle = Circle(
        (1.5, 1.5),
        radius=1.15,
        facecolor=set_color,
        edgecolor=edge_color,
        linewidth=2,
        alpha=0.8,
    )
    ax.add_patch(circle)

    x = np.array([0.95, 1.9])
    y = np.array([2.05, 1.1])
    z = 0.5 * (x + y)

    ax.plot(
        [x[0], y[0]],
        [x[1], y[1]],
        color=line_color,
        linewidth=2.5,
        zorder=3,
    )

    ax.scatter(
        [x[0], y[0]],
        [x[1], y[1]],
        color=line_color,
        s=50,
        zorder=5,
    )
    ax.scatter(
        [(1-lambda_slider.value)*x[0]+lambda_slider.value*y[0]],
        [(1-lambda_slider.value)*x[1]+lambda_slider.value*y[1]],
        color=line_color,
        s=50,
        zorder=5,
    )

    ax.text(x[0] - 0.2, x[1] + 0.12, r"$x$", fontsize=12)
    ax.text(y[0] + 0.1, y[1] - 0.12, r"$y$", fontsize=12)
    ax.set_title("Convex set", fontsize=15)

    ax = axes[1]

    vertices = np.array([
        [0.3, 0.3],
        [2.8, 0.3],
        [2.8, 1.1],
        [1.1, 1.1],
        [1.1, 2.8],
        [0.3, 2.8],
    ])

    polygon = Polygon(
        vertices,
        closed=True,
        facecolor=set_color,
        edgecolor=edge_color,
        linewidth=2,
        alpha=0.8,
    )
    ax.add_patch(polygon)

    x = np.array([0.7, 2.4])
    y = np.array([2.4, 0.7])
    z = 0.5 * (x + y)

    ax.plot(
        [x[0], y[0]],
        [x[1], y[1]],
        color=invalid_color,
        linewidth=2.5,
        linestyle="--",
        zorder=3,
    )

    ax.scatter(
        [x[0], y[0]],
        [x[1], y[1]],
        color=line_color,
        s=50,
        zorder=5,
    )
    ax.scatter(
        [(1-lambda_slider.value)*x[0]+lambda_slider.value*y[0]],
        [(1-lambda_slider.value)*x[1]+lambda_slider.value*y[1]],
        color=line_color,
        s=50,
        zorder=5,
    )

    ax.text(x[0] - 0.2, x[1] + 0.12, r"$x$", fontsize=12)
    ax.text(y[0] + 0.1, y[1] - 0.12, r"$y$", fontsize=12)

    ax.set_title("Nonconvex set", fontsize=15)

    for ax in axes:
        ax.set_xlim(0, 3.1)
        ax.set_ylim(0, 3.1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(w_pad=3)

    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    However, we also need a mathematical definition for the convexity to be able to determine whether a given set is convex.

    !!! definition "Convex set"

        Let \(X\subseteq\mathbb{R}^n\). The set \(X\) is called *convex* if, for every \(x,y\in X\) and every \(\lambda\in[0,1]\),

        \[
        \lambda x+(1-\lambda)y\in X.
        \]

    Now we can show that the feasible set of the inequality form LP satisfies this definition. This definition is also saying the exact thing that the above figure is showing. You can see that by shifting the lambda parameter in the figure.

    !!! proposition "Convexity of feasible set"

        The feasible set of inequality form LP $\mathcal{F}=\{x\in\mathbb{R}^n\mid Ax\leq b\}$ is a convex set.

    ???- proof "Proof"

        Let \(\lambda\in[0,1]\) and \(x,y\in\mathcal F\) be arbitrary. By the definition of the feasible set,

        \[
            Ax\leq b
            \qquad\text{and}\qquad
            Ay\leq b.
        \]

        Therefore,

        \[
            \begin{aligned}
                A(\lambda x+(1-\lambda)y)
                &=A(\lambda x)+A((1-\lambda)y)
                &&\text{(Additivity)}\\
                &=\lambda Ax+(1-\lambda)Ay
                &&\text{(Homogeneity)}\\
                &\leq\lambda b+(1-\lambda)b
                &&\text{(Since }Ax\leq b,\ Ay\leq b\text{)}\\
                &=b.
            \end{aligned}
        \]

        Thus,

        \[
            A(\lambda x+(1-\lambda)y)\leq b,
        \]

        which implies that

        \[
            \lambda x+(1-\lambda)y\in\mathcal F.
        \]

        Since this holds for arbitrary \(x,y\in\mathcal F\) and \(\lambda\in[0,1]\), the feasible set \(\mathcal F\) is convex.

    Convexity is a fundamental concept in mathematical optimization. One of the most important properties of convex optimization problems is that every *local optimum* is also a *global optimum*. This means that if we find a locally optimal solution, we can be certain that no other feasible solution has a better objective-function value.

    Convexity therefore provides particularly strong mathematical guarantees about optimality and is one of the main reasons why convex optimization problems are often easier to analyze and solve than general nonconvex optimization problems. We are not going to work very much with general convex problems, but as shown the linear problems are subset of convex problems and therefore also benefit from the results about globality.

    ## Local and Global Optimizers

    !!! definition "Local and global maximizers"
        Let \(f:\mathbb{R}^n\to\mathbb{R}\) be a function and let \(\mathcal F\subseteq\mathbb{R}^n\) be a nonempty feasible set. Consider a point \(x^*\in\mathcal F\).

        The point \(x^*\) is a local maximizer of \(f\) over \(\mathcal F\) if there exists \(\varepsilon>0\) such that

        \[
        f(x^*)\geq f(x)
        \]

        for all \(x\in\mathcal F\) satisfying \(\|x-x^*\|<\varepsilon\).

        The point \(x^*\) is a global maximizer of \(f\) over \(\mathcal F\) if

        \[
        f(x^*)\geq f(x)
        \qquad\text{for all }x\in\mathcal F.
        \]

        The corresponding definitions of local and global minimizers are obtained by reversing the inequalities.

    The local and global maximizers are also shown in the plot below for one particular function.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    # Function with a local and a global maximum
    t_optima = np.linspace(-4, 4, 2000)

    g_optima = (
        1.2 * np.exp(-(t_optima + 1.6)**2 / 0.5)
        + 2.2 * np.exp(-(t_optima - 1.4)**2 / 0.7)
        - 0.03 * t_optima**2
    )

    # Identify local maxima from the sampled values
    maxima_indices_optima = np.where(
        (g_optima[1:-1] > g_optima[:-2])
        & (g_optima[1:-1] > g_optima[2:])
    )[0] + 1

    # Identify the global and lower local maximum
    global_idx_optima = np.argmax(g_optima)

    local_idx_optima = min(
        maxima_indices_optima,
        key=lambda idx: g_optima[idx],
    )

    # Local and global maximizer coordinates
    t_local_optima = t_optima[local_idx_optima]
    g_local_optima = g_optima[local_idx_optima]

    t_global_optima = t_optima[global_idx_optima]
    g_global_optima = g_optima[global_idx_optima]

    # Choose an epsilon-neighborhood around the local maximizer
    epsilon_optima = 0.75
    left_eps_optima = t_local_optima - epsilon_optima
    right_eps_optima = t_local_optima + epsilon_optima

    # Create the figure
    fig_optima, ax_optima = plt.subplots(figsize=(9, 5))

    ax_optima.plot(
        t_optima,
        g_optima,
        color="tab:blue",
        linewidth=2,
    )

    # Shade epsilon-neighborhood around the local maximizer
    ax_optima.axvspan(
        left_eps_optima,
        right_eps_optima,
        alpha=0.075,
        color="tab:orange",
    )

    # Vertical dashed lines marking t* - epsilon and t* + epsilon
    ax_optima.axvline(
        left_eps_optima,
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        alpha=.5
    )
    ax_optima.axvline(
        right_eps_optima,
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        alpha=.5
    )

    # Horizontal dotted line at g(t*)
    ax_optima.axhline(
        g_local_optima,
        color="tab:orange",
        linestyle=":",
        linewidth=1.5,
        alpha=.5
    )

    # Local maximizer
    ax_optima.scatter(
        t_local_optima,
        g_local_optima,
        color="tab:orange",
        s=75,
        zorder=5,
        label="Local maximizer",
    )

    ax_optima.annotate(
        "Local maximizer",
        xy=(t_local_optima, g_local_optima),
        xytext=(-140, 35),
        textcoords="offset points",
        zorder=6,
        arrowprops=dict(arrowstyle="->"),
    )

    # Global maximizer
    ax_optima.scatter(
        t_global_optima,
        g_global_optima,
        color="tab:red",
        s=75,
        zorder=5,
        label="Global maximizer",
    )

    ax_optima.annotate(
        "Global maximizer",
        xy=(t_global_optima, g_global_optima),
        xytext=(25, -40),
        textcoords="offset points",
        zorder=6,
        arrowprops=dict(arrowstyle="->"),
    )

    # Label epsilon-neighborhood
    ax_optima.text(
        t_local_optima,
        ax_optima.get_ylim()[0] + 0.08 * (ax_optima.get_ylim()[1] - ax_optima.get_ylim()[0]),
        r"$(x^*-\varepsilon,\; x^*+\varepsilon)$",
        ha="center",
        color="tab:orange",
        fontsize=11,
    )

    ax_optima.text(
        t_local_optima - 1.9,
        g_local_optima - 0.35,
        r"$f(x)\leq f(x^*)$" "\n" r"for nearby $x$",
        color="tab:orange",
        fontsize=11,
        bbox=dict(boxstyle="round", alpha=0.1),
    )

    # Formatting
    ax_optima.set_xlabel(r"$x$")
    ax_optima.set_ylabel(r"$f(x)$")
    ax_optima.set_title("Local and global maximizers")
    ax_optima.grid(alpha=0.25)
    ax_optima.legend()

    fig_optima.tight_layout()

    fig_optima
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feasibility and Uniqueness

    We have already seen a problem that doesn't have a solution. In part I, the electricity example doen't have a solution when the transfer of energy from sweden was set to be 2100 MWh. In that case we get that \( 2100\text{ MWh}+1000\text{ MWh}=3100\text{ MWh}\ngeq 3200\text{MWh}\). In this case it was straight forward to see that the problem cannot have a solution.

    In a similar fashion, an optimization problem could have in some cases many solutions that all yield equal objective values. Let's consider a problem of form

    \[
        \begin{aligned}
            \max_x& \;x_1+x_2\\
            \text{s.t.}& \\
            & 5x_1+5x_2\leq 6 \\
            & x_1,x_2\geq 0
        \end{aligned}
    \]

    You can find the optimum of the problem by using the slider. Let $Z=x_1+x_2$ denote the objective function. You can shift the level of the objective using the slider and see that the optimum for the problem if attained at any point that belongs to a set \(\{x\in\mathbb{R}_+^2\mid x_1+x_2=6/5\}\), where \(\mathbb{R}_+^2\) is the first quadrant of the plane and more generally \(\mathbb{R}_+^n=\{x\in\mathbb{R}^n\mid x_i\geq 0 \text{ for all }0\leq i\leq n\}\).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    nu_slider = mo.ui.slider(
        start=0.0,
        stop=2.0,
        step=0.2,
        value=0.2,
        show_value=True,
        label="Objective function value Z"
    )
    nu_slider
    return (nu_slider,)


@app.cell(hide_code=True)
def _(np, nu_slider, plt):
    non_unique_fig, non_unique_ax = plt.subplots()

    nu_domain = np.linspace(0,6/5,100)

    non_unique_ax.plot(
        [0, 6/5],
        [6/5, 0]
    )

    non_unique_ax.fill_between(
        nu_domain,
        6/5-nu_domain,
        alpha=.2
    )

    non_unique_ax.set_xlim(0,7/5)
    non_unique_ax.set_ylim(0,7/5)

    non_unique_ax.plot(
        [0, nu_slider.value],
        [nu_slider.value, 0],
        linestyle="dashed",
        color= "green" if nu_slider.value==6/5 else "red"
    )

    non_unique_ax.set_title("LP problem with non-unique solution")
    non_unique_ax.set_xlabel(r"$x_1$")
    non_unique_ax.set_ylabel(r"$x_2$")

    non_unique_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's now take another look at the electricity model from the previous part. This time the optimization problem is formulated as an abstrct model in the inequality form. The abstract model is produced by the `electricity_model()` function below.
    """)
    return


@app.cell
def _(pyo):
    def electricity_model():
        # Initiate abstract Pyomo model
        model = pyo.AbstractModel()

        # Initiate problem dimensions, parameters, and variables
        model.N = pyo.Set()
        model.M = pyo.Set()
        model.c = pyo.Param(model.N)
        model.A = pyo.Param(model.M, model.N)
        model.b = pyo.Param(model.M)

        model.x = pyo.Var(model.N, within=pyo.NonNegativeReals)

        # Define the constraints based on matrix A, variable x and vector b 
        def con_rule(model, m):
            return sum(model.A[m,i]*model.x[i] for i in model.N) >= model.b[m]
        model.con = pyo.Constraint(model.M, rule=con_rule)

        # Define the objective function
        def obj(model):
            return sum(model.c[i]*model.x[i] for i in model.N)
        model.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

        return model

    return (electricity_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `create_parameters(A, b, c, M, N)` function is just a helper function that takes the parameters of an inequality form LP problem and turns those into a format that the Pyomo understands.
    """)
    return


@app.function
def create_parameters(A, b, c, M, N):
    # Helper function to format the data for Pyomo
    return {
        None: {
            "N": {None: list(range(N))},
            "M": {None: list(range(M))},
            "c": {i: float(c[i]) for i in range(N)},
            "A": {
                (m, i): float(A[m, i])
                for m in range(M)
                for i in range(N)
            },
            "b": {m: float(b[m]) for m in range(M)},
        }
    }


@app.cell
def _(np):
    # Write the problem formulation in the inequality form
    c = np.array([40., 60.])
    # Note that we need to multiply the greater than or equal
    # inequalities by -1, since we can only use less than or equal
    # in the Pyomo formulation
    A = np.array(
        [
            [1., 1.],
            [-1., 0.],
            [0., -1.]
        ]
    )
    # Set the constraint values
    b = np.array([3200., -3100., -1000.])
    # Get the problem dimensions M=3 (number of constraints) and N=2
    # (number of variables)
    M, N = np.shape(A)
    return A, M, N, b, c


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can solve the problem similarly as in the previous part, and the solver returns the same solution as expected.
    """)
    return


@app.cell
def _(A, M, N, b, c, electricity_model, pyo):
    # Define the model
    model = electricity_model()

    # Instantiate a concrete Pyomo model from the abstract definition
    instance = model.create_instance(
        create_parameters(
            A,
            b,
            c,
            M,
            N
        )
    )

    # Solve the model using HiGHS solver
    solver = pyo.SolverFactory("highs")
    results = solver.solve(instance)
    return model, results


@app.cell
def _(results):
    results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's now see what happens when we give the solver an infeasible version of the electricity problem.
    """)
    return


@app.cell
def _(A, M, N, c, model, np, pyo):
    # Reinstantiate the model with different constraints
    instance_infeasible = model.create_instance(
        create_parameters(
            A,
            np.array([3200., -2100., -1000.]),
            c,
            M,
            N
        )
    )

    solver_infeasible = pyo.SolverFactory("highs")
    # Solve the model using load_solutions=False not to throw NoFeasibleSolutionError
    results_infeasible = solver_infeasible.solve(instance_infeasible, load_solutions=False)
    return instance_infeasible, results_infeasible


@app.cell
def _(results_infeasible):
    results_infeasible.solver.termination_condition
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The solver would throw an error if we wouldn't use the `load_solution=False` flag. Now we are able to check the termination condition, which confirms the infesibility. We can also take a look at the problem in plane as before.
    """)
    return


@app.cell(hide_code=True)
def _(instance_infeasible, mo, plot_lp_2d):
    fig_if, ax_if = plot_lp_2d(
        instance_infeasible,
        xlim=(0, 3500),
        ylim=(0, 3500),
        var_labels={
            "x[0]": r"$x_S$",
            "x[1]": r"$x_E$",
        },
        title="Electricity LP withouth a solution"
    )

    mo.mpl.interactive(fig_if)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    The model is infeasible because the required import load is not possible due to insufficient transfer capacity. In the above plot we can see that the infeasible set is empty.

    Now, let's imagine that we get a new 700 MW transfer cable built between Finland and Estonia. At the same time the electricity price in Sweden rises to par with Estonia, so that both are now at 60 $/MWh. As a consequence we get a non-unique solution for our electricity optimization problem, analogously to the toy example previously. Below you can see the graphical solution to the electricity problem. Even though the optimizer chooses the optimum allocation to be 1500 MWh from Sweden and 1700 MWh from Estonia, we can clearly see that any point on the vertex would give us the same value and thus be a solution to the optimization problem.

    Problems in higher dimensions are of course not as easy to distinguish to have multiple solutions. We will see in the next lecture how we can still find out if the solution we have found is unique.
    """)
    return


@app.cell
def _(A, M, N, model, np, pyo):
    # Reinstantiate the model with different constraint and different objective
    instance_nonunique = model.create_instance(
        create_parameters(
            A,
            np.array([3200., -3100., -1700.]),
            np.array([60.,60.]),
            M,
            N
        )
    )

    solver_nonunique = pyo.SolverFactory("highs")
    results_nonunique = solver_nonunique.solve(instance_nonunique, load_solutions=True)
    return instance_nonunique, results_nonunique


@app.cell
def _(results_nonunique):
    results_nonunique
    return


@app.cell(hide_code=True)
def _(instance_nonunique, mo, plot_lp_2d):
    fig_nu, ax_nu = plot_lp_2d(
        instance_nonunique,
        xlim=(0, 3500),
        ylim=(0, 3500),
        var_labels={
            "x[0]": r"$x_S$",
            "x[1]": r"$x_E$",
        },
        title="Electricity LP-solution with new objective and constraints"
    )

    mo.mpl.interactive(fig_nu)
    return


if __name__ == "__main__":
    app.run()
