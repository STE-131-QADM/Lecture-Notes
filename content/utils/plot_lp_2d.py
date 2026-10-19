# Mostly AI generated function


import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyomo.repn.standard_repn import generate_standard_repn


def plot_lp_2d(
    model,
    xlim,
    ylim,
    resolution=500,
    show_solution=True,
    ax=None,
    var_labels=None,
    title=None
):
    """
    Visualize a two-variable linear Pyomo optimization problem.

    Parameters
    ----------
    model : Pyomo model
        Model containing exactly two unfixed decision variables.
    xlim, ylim : tuple
        Axis limits, e.g. (0, 3500), (0, 1200).
        These limits affect only the visualization.
    resolution : int
        Resolution of the feasibility grid.
    show_solution : bool
        Show variable values and the corresponding objective
        contour if a solution is available.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axis.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """

    # --------------------------------------------------
    # 1. Extract decision variables
    # --------------------------------------------------

    variables = [
        v
        for v in model.component_data_objects(
            pyo.Var, active=True
        )
        if not v.fixed
    ]

    if len(variables) != 2:
        raise ValueError(
            "The model must contain exactly two unfixed variables."
        )

    # --------------------------------------------------
    # 2. Extract linear expression coefficients
    # --------------------------------------------------

    def linear_coefficients(expr):
        rep = generate_standard_repn(
            expr, compute_values=True
        )

        if not rep.is_linear():
            raise ValueError(
                "Only linear expressions are supported."
            )

        coefficients = np.zeros(2)

        for variable, coefficient in zip(
            rep.linear_vars,
            rep.linear_coefs,
        ):
            for i, v in enumerate(variables):
                if variable is v:
                    coefficients[i] += float(
                        pyo.value(coefficient)
                    )
                    break
            else:
                raise ValueError(
                    "Unexpected variable in expression."
                )

        constant = float(
            pyo.value(rep.constant or 0)
        )

        return coefficients, constant

    # --------------------------------------------------
    # 3. Collect constraints
    # --------------------------------------------------

    constraints = []

    def add_constraint(a, b, sense, name):
        constraints.append(
            (np.asarray(a, dtype=float), float(b), sense, name)
        )

    # Variable bounds
    for i, variable in enumerate(variables):
        a = np.eye(2)[i]

        if variable.lb is not None:
            add_constraint(
                a,
                pyo.value(variable.lb),
                ">=",
                f"{variable.name} lower",
            )

        if variable.ub is not None:
            add_constraint(
                a,
                pyo.value(variable.ub),
                "<=",
                f"{variable.name} upper",
            )

    # Explicit model constraints
    for con in model.component_data_objects(
        pyo.Constraint, active=True
    ):
        a, constant = linear_coefficients(con.body)

        if con.equality:
            add_constraint(
                a,
                pyo.value(con.lower) - constant,
                "=",
                con.name,
            )
        else:
            if con.has_lb():
                add_constraint(
                    a,
                    pyo.value(con.lower) - constant,
                    ">=",
                    f"{con.name} lower",
                )

            if con.has_ub():
                add_constraint(
                    a,
                    pyo.value(con.upper) - constant,
                    "<=",
                    f"{con.name} upper",
                )

    # --------------------------------------------------
    # 4. Construct plotting grid
    # --------------------------------------------------

    x = np.linspace(*xlim, resolution)
    y = np.linspace(*ylim, resolution)

    X, Y = np.meshgrid(x, y)

    def satisfies(lhs, b, sense):
        tolerance = 1e-8 * (1 + abs(b))

        if sense == "<=":
            return lhs <= b + tolerance

        if sense == ">=":
            return lhs >= b - tolerance

        return np.abs(lhs - b) <= tolerance

    feasible = np.ones_like(X, dtype=bool)

    for a, b, sense, _ in constraints:
        if sense != "=":
            feasible &= satisfies(
                a[0] * X + a[1] * Y,
                b,
                sense,
            )

    # --------------------------------------------------
    # 5. Create Matplotlib figure
    # --------------------------------------------------

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # --------------------------------------------------
    # 6. Shade feasible region
    # --------------------------------------------------

    has_equalities = any(
        sense == "="
        for _, _, sense, _ in constraints
    )

    if not has_equalities:
        ax.contourf(
            X,
            Y,
            feasible.astype(float),
            levels=[0.5, 1.5],
            colors=["tab:green"],
            alpha=0.22,
        )

    # --------------------------------------------------
    # 7. Plot constraint boundaries
    # --------------------------------------------------

    for a, b, sense, name in constraints:

        if np.linalg.norm(a) < 1e-12:
            continue

        Z = a[0] * X + a[1] * Y - b

        if Z.min() <= 0 <= Z.max():
            ax.contour(
                X,
                Y,
                Z,
                levels=[0],
                colors=["tab:blue"],
                linestyles=(
                    "-" if sense == "=" else "--"
                ),
                linewidths=1.5,
            )

    # --------------------------------------------------
    # 8. Highlight feasible equality segments
    # --------------------------------------------------

    if has_equalities:
        for a, b, sense, _ in constraints:

            if sense != "=":
                continue

            if np.linalg.norm(a) < 1e-12:
                continue

            if abs(a[1]) > abs(a[0]):
                xp = np.linspace(
                    *xlim, resolution * 2
                )

                yp = (
                    b - a[0] * xp
                ) / a[1]

            else:
                yp = np.linspace(
                    *ylim, resolution * 2
                )

                xp = (
                    b - a[1] * yp
                ) / a[0]

            valid = (
                (xp >= xlim[0])
                & (xp <= xlim[1])
                & (yp >= ylim[0])
                & (yp <= ylim[1])
            )

            for aa, bb, ss, _ in constraints:
                valid &= satisfies(
                    aa[0] * xp + aa[1] * yp,
                    bb,
                    ss,
                )

            ax.plot(
                np.where(valid, xp, np.nan),
                np.where(valid, yp, np.nan),
                color="tab:green",
                linewidth=4,
                zorder=3,
            )

    # --------------------------------------------------
    # 9. Plot solution and objective contour
    # --------------------------------------------------

    values = [
        pyo.value(v, exception=False)
        for v in variables
    ]

    objectives = list(
        model.component_data_objects(
            pyo.Objective, active=True
        )
    )

    solution_plotted = False
    objective_plotted = False

    if (
        show_solution
        and all(v is not None for v in values)
    ):
        xstar, ystar = map(float, values)

        ax.scatter(
            xstar,
            ystar,
            color="tab:red",
            s=85,
            zorder=5,
        )

        ax.annotate(
            f"({xstar:,.1f}, {ystar:,.1f})",
            (xstar, ystar),
            xytext=(10, 10),
            textcoords="offset points",
        )

        solution_plotted = True

        if len(objectives) == 1:

            c, constant = linear_coefficients(
                objectives[0].expr
            )

            if np.linalg.norm(c) > 1e-12:

                zstar = (
                    c[0] * xstar
                    + c[1] * ystar
                )

                Z = (
                    c[0] * X
                    + c[1] * Y
                )

                if Z.min() <= zstar <= Z.max():

                    ax.contour(
                        X,
                        Y,
                        Z,
                        levels=[zstar],
                        colors=["black"],
                        linestyles=":",
                        linewidths=2,
                    )

                    objective_plotted = True

    # --------------------------------------------------
    # 10. Appearance and legend
    # --------------------------------------------------

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if var_labels is None:
        var_labels = {}

    xlabel = var_labels.get(variables[0].name, variables[0].name)
    ylabel = var_labels.get(variables[1].name, variables[1].name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if not title is None:
        ax.set_title(title)

    ax.grid(alpha=0.2)

    legend_elements = [
        Line2D(
            [0], [0],
            color="tab:blue",
            linestyle="--",
            linewidth=1.5,
            label="Constraint boundary",
        ),
        Patch(
            facecolor="tab:green",
            alpha=0.22,
            label="Feasible region",
        ),
    ]

    if objective_plotted:
        legend_elements.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle=":",
                linewidth=2,
                label="Objective contour",
            )
        )

    if solution_plotted:
        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                color="tab:red",
                linestyle="None",
                markersize=8,
                label="Solution",
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="best",
    )

    fig.tight_layout()

    return fig, ax