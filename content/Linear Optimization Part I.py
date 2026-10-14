# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "highspy==1.15.1",
#     "marimo>=0.25.1",
#     "matplotlib==3.11.2",
#     "numpy==2.5.4",
#     "pyomo==6.10.1",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Optimization: Part I
    """)
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

    return cm, colors, mo, np, plt, pyo


@app.cell(hide_code=True)
def _(mo):
    a = mo.ui.slider(0, 10, step=1, show_value=True, value=3)
    mo.md(f"Choose a value for $a$: {a}")
    return (a,)


@app.cell(hide_code=True)
def _(a, b, mo):
    mo.md(rf"""
    This slider modifies the objective function coefficient so that:

    \[
        f_0(x)=ax_1+bx_2= {a.value} x_1+ {b} x_2
    \]
    """)
    return


@app.cell
def _(np):
    b = 5
    F = np.array(
        [
            [1, 1/4],
            [1/4, 1],
            [1, 1]
        ]
    )
    c = np.array([3, 3, 4])
    return F, b, c


@app.cell(hide_code=True)
def _(cm, colors, np, plt):
    def plot_linear_problem(A,b,obj,domain=np.linspace(0,5,200), opt=None):
        fig, ax = plt.subplots()

        x = np.linspace(0, np.min(b / A[:, 0]), 500)

        for row in range(A.shape[0]):
            ax.plot(
                [0, b[row]/A[row,0]],
                [b[row]/A[row,1], 0],
                label=f"Constraint {row}"
            )

        y_max = np.min(
            (b[:, None] - A[:, 0, None] * x) / A[:, 1, None],
            axis=0
        )

        ax.fill_between(x, 0, y_max, alpha=0.15, label="Feasible set")

        ax.set_xlim(left=0,right=5)
        ax.set_ylim(bottom=0, top=5)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")

        if not domain is None:
            X, Y = np.meshgrid(domain,domain)
            Z = obj(X,Y)

            objective_contour = ax.contour(
                X, Y, 
                Z,
                levels=10,
                cmap="PiYG"
            )

            # Continuous color mapping for the colorbar
            norm = colors.Normalize(
                vmin=Z.min(),
                vmax=Z.max()
            )

            sm = cm.ScalarMappable(
                norm=norm,
                cmap="PiYG"
            )

            fig.colorbar(sm, ax=ax, label="Objective contours")

        if not opt is None:
            ax.scatter(opt[0], opt[1], c="red", marker="*", label="Optimum", zorder=4, s=75)

        ax.legend()

        return fig

    return (plot_linear_problem,)


@app.cell(hide_code=True)
def _(F, a, b, c, plot_linear_problem):
    plot_linear_problem(F,c,lambda x,y: a.value*x+b*y)
    return


@app.cell
def _(F, a, b, c, pyo):
    # Define the model object
    model = pyo.ConcreteModel()

    # Define variables
    model.x_1 = pyo.Var(within=pyo.NonNegativeReals)
    model.x_2 = pyo.Var(within=pyo.NonNegativeReals)

    # Define objective
    model.obj = pyo.Objective(
        expr=model.x_1*a.value + model.x_2*b,
        sense=pyo.maximize
    )

    # Define constraints
    model.con1 = pyo.Constraint(expr=F[0,0]*model.x_1 + F[0,1]*model.x_2 <= c[0])
    model.con2 = pyo.Constraint(expr=F[1,0]*model.x_1 + F[1,1]*model.x_2 <= c[1])
    model.con3 = pyo.Constraint(expr=F[2,0]*model.x_1 + F[2,1]*model.x_2 <= c[2])

    # Choose solver algorithm
    opt = pyo.SolverFactory('highs')
    # Solve the model
    results = opt.solve(model)
    # Check the validity of found solution
    pyo.assert_optimal_termination(results)
    return (model,)


@app.cell(hide_code=True)
def _(mo, model, pyo):
    mo.md(rf"""
    The optimal decision is

    \[
    x_1 = {pyo.value(model.x_1):.2f}, \qquad
    x_2 = {pyo.value(model.x_2):.2f}.
    \]

    The resulting objective value is

    \[
    z^* = ax_1+bx_2 = {pyo.value(model.obj):.2f}.
    \]
    """)
    return


@app.cell(hide_code=True)
def _(F, a, b, c, model, plot_linear_problem, pyo):
    plot_linear_problem(F,c,lambda x,y: a.value*x+b*y, opt=(pyo.value(model.x_1), pyo.value(model.x_2)))
    return


if __name__ == "__main__":
    app.run()
