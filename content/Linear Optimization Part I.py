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

    return mo, plt, pyo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Optimization: Part I

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

    This first part of the course introduces the basic ideas of mathematical optimization. In an optimization problem, we choose between a set of possible alternatives in order to maximize or minimize some quantity of interest. This quantity is described by an *objective function*. For example, a firm might seek to maximize profit, whereas a transportation company might seek to minimize operating costs or emissions.

    We denote the decision variables by \(x\) and the objective function by \(f\). In general, \(x\) may contain several decision variables, in which case we can think of \(x\) as a vector. An optimization problem can then be expressed informally as

    \[
    \text{choose }x\text{ so that }f(x)\text{ is as large or as small as possible}.
    \]

    You may already have encountered optimization problems in calculus, where conditions such as \(f'(x)=0\) are used to identify candidate points for minima or maxima of simple functions. In most practical decision problems, however, not every possible value of \(x\) is allowed. The decisions must also satisfy various restrictions, which we represent mathematically as *constraints*.

    ## Linearity

    We will first concentrate on optimization problems in which the objective function and the constraints are linear. A function \(f:\mathbb{R}^n\rightarrow\mathbb{R}\) is linear if

    \[
    f(\alpha x+\beta y)
    =
    \alpha f(x)+\beta f(y)
    \]

    for all \(x,y\in\mathbb{R}^n\) and all \(\alpha,\beta\in\mathbb{R}\).

    Equivalently, a linear function satisfies two important properties:

    \[
    f(x+y)=f(x)+f(y)
    \]

    and

    \[
    f(\alpha x)=\alpha f(x).
    \]

    These properties are called *additivity* and *homogeneity*, respectively. They also have natural interpretations in many applied models.

    ???- question "Equivalence of the definitions"

        Recall that a function \(f:\mathbb{R}^n\to\mathbb{R}\) is linear if

        $$
        f(\alpha x+\beta y)
        =
        \alpha f(x)+\beta f(y)
        $$

        for all \(x,y\in\mathbb{R}^n\) and \(\alpha,\beta\in\mathbb{R}\).

        **Show that \(f\) is linear if and only if it is both additive and homogeneous; that is,**

        $$
        f(x+y)=f(x)+f(y)
        $$

        **and**

        $$
        f(\alpha x)=\alpha f(x).
        $$

        Your proof should establish both directions of the equivalence.

        Note that this means that we could have given the defintion of linearity just as well by saying that "A function $f$ if linear if it is additive and homogenous."

    Consider, for example, an electricity producer selling electricity at a given market price during a particular time period. Let \(x\) denote the amount of electricity sold during this period, measured in MWh, and let \(p\) denote the price received for electricity, measured in €/MWh. If we assume that the producer can sell electricity at the same price \(p\) regardless of the amount sold, its revenue is

    \[
    R(x)=px.
    \]

    ???- question "Check the units"

        Consider the revenue function

        $$
        R(x) = px.
        $$

        Suppose that the electricity price $p$ is measured in €/MWh.

        **What must the unit of $x$ be so that the unit of $R(x)$ is €?**

        Verify your answer by multiplying the units.

    Suppose the producer has two generators that supply \(x\) and \(x'\) MWh. The total revenue is then

    \[
    R(x+x')
    =
    p(x+x')
    =
    px+px'
    =
    R(x)+R(x').
    \]

    Thus, the revenues generated by the two generators are additive.

    Similarly, doubling the amount of electricity sold doubles the revenue:

    \[
    R(2x)
    =
    p(2x)
    =
    2px
    =
    2R(x).
    \]

    More generally,

    \[
    R(\alpha x)=p(\alpha x)=\alpha px=\alpha R(x).
    \]

    The revenue function is therefore also homogeneous and we can conclude that it is linear in the amount of electricity sold.

    However, notice that the linearity of \(R(x)=px\) follows from an important modelling assumption: the price \(p\) is treated as constant and independent of the amount sold. Whether this is a reasonable assumption depends on the application. Mathematical optimization does not determine whether an assumption is realistic; rather, it determines the best decision conditional on the mathematical model we have chosen.

    The field of mathematical optimization is vast and extends far beyond linear models. Many problems involve nonlinear relationships, discrete decisions, uncertainty, or decisions made over time. During this course, we will encounter some of these more complicated models as well. Linear optimization nevertheless provides an important starting point because it introduces many of the central ideas that recur throughout mathematical optimization.

    ## Constrained problems

    An electricity producer may want to sell more electricity, but it cannot simply supply an unlimited amount of electricity to the grid. Depending on the generation technology, production may be constrained by the availability of wind or solar energy, or by the technical capacity of a nuclear, hydroelectric, or thermal power plant.

    Let \(G\) denote the set of available generators and let \(\bar{x}_g\) denote the maximum amount of electricity that generator \(g\in G\) can produce during the considered time period. The production of generator \(g\) must then satisfy

    \[
    x_g\leq \bar{x}_g.
    \]

    For wind and solar generation, \(\bar{x}_g\) should be interpreted as the generation capacity available under the prevailing weather conditions, rather than simply the installed capacity of the generator.

    Let us also assume that the producer cannot consume electricity from the grid or store electricity. Production must therefore be nonnegative, giving the constraint

    \[
    0\leq x_g\leq\bar{x}_g.
    \]

    The set of values from which a decision can be chosen after all constraints have been taken into account is called the *feasible set*. For the constraint above, the feasible set is

    \[
    \mathcal{F}_g
    =
    \left\{
    x_g\in\mathbb{R}
    \mid
    0\leq x_g\leq\bar{x}_g
    \right\}.
    \]

    In this case the constraint is just one dimensional, and the feasible set is an interval of the real line as the figure below shows.
    """)
    return


@app.cell(hide_code=True)
def _(plt):
    xbar = 10

    fig, ax = plt.subplots(figsize=(8, 2.2))

    # Feasible interval
    ax.plot([0, xbar], [0, 0], linewidth=6, solid_capstyle="butt")
    ax.scatter([0, xbar], [0, 0], s=70, zorder=3)

    # Labels
    ax.text(0, -0.13, r"$0$", ha="center", va="top", fontsize=12)
    ax.text(xbar, -0.13, r"$\bar{x}_g$", ha="center", va="top", fontsize=12)
    ax.text(
        xbar / 2,
        0.12,
        r"Feasible set $\mathcal{F}_g$",
        ha="center",
        va="bottom",
        fontsize=12,
    )

    ax.set_xlim(-1, xbar + 1)
    ax.set_ylim(-0.35, 0.35)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(r"Electricity production $x_g$")

    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

    ax.spines["bottom"].set_position(("data", 0))

    fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now express the optimization problem of an electricity producer seeking to maximize its revenue as

    \[
    x_g^*
    =
    \operatorname*{arg\,max}_{x_g\in\mathcal{F}_g}
    R(x_g).
    \]

    We will generally use a star, such as \(x_g^*\), to denote an optimal value of a decision variable. The operator \(\operatorname{arg\,max}\) returns the value, or values, of the decision variable that maximize the objective function.

    In contrast,

    \[
    \max_{x_g\in\mathcal{F}_g}R(x_g)
    \]

    denotes the maximum value attained by the objective function. Therefore,

    \[
    \max_{x_g\in\mathcal{F}_g}R(x_g)
    =
    R(x_g^*).
    \]

    For the revenue function

    \[
    R(x_g)=px_g,
    \]

    this optimization problem is particularly easy to solve. If we assume that

    \[
    p>0,
    \]

    then \(R(x_g)\) is strictly increasing in \(x_g\). Consequently, the producer maximizes its revenue by producing as much electricity as possible:

    \[
    x_g^*=\bar{x}_g.
    \]

    !!! note "Negative electricity prices"

        Electricity prices are not necessarily always positive. Negative prices can occasionally arise from errors in market bids, but they can also occur as a consequence of the normal operation of the electricity market.

        Electricity systems must remain balanced: electricity supplied to the system must continuously be matched by consumption, exports, storage, and other withdrawals from the system. At the same time, some generators are costly or technically difficult to reduce or shut down, while production from sources such as wind may be abundant during periods of low demand. In such situations, producers may be willing to pay for electricity to be consumed, resulting in a negative market price.

        Negative prices have occurred repeatedly in Finland. See, for example, the [Finnish electricity prices in 2023](https://laskurisi-fi.translate.goog/sahko/tilastot/2023/?_x_tr_sl=fi&_x_tr_tl=en&_x_tr_hl=en-US&_x_tr_pto=wapp).

    ???- question "Optimal solutions under different electricity prices"

        Consider the revenue-maximization problem

        $$
        \max_{0\leq x_g\leq \bar{x}_g} px_g.
        $$

        We have already considered the case \(p>0\), for which the optimal solution is

        $$
        x_g^*=\bar{x}_g.
        $$

        Now consider the two remaining price regimes:

        1. \(p=0\)
        2. \(p<0\)

        **Determine the optimal solution \(x_g^*\) in each case. Is the optimal solution unique?**

        Explain your answer by considering how the revenue function

        $$
        R(x_g)=px_g
        $$

        changes as \(x_g\) increases.

    ## Example: An optimization problem with two decision variables

    We rarely consider optimization problems in which there is only one variable we can control. Introducing multiple decision variables allows us to represent more interesting management problems, particularly situations in which we must make trade-offs between different alternatives.

    Let us continue with our discussion of electricity markets and consider a specific situation in the Finnish electricity system.

    ### Electricity adequacy in Finland during winter 2026–2027

    On 24 September 2026, the Finnish transmission system operator Fingrid published its assessment of electricity adequacy for the coming winter. According to Fingrid, on an extremely cold and windless winter day, electricity consumption in Finland could reach approximately **15,200 MW**, while domestic electricity production could supply approximately **12,000 MW**.

    This leaves a difference of

    \[
    15\,200-12\,000=3\,200\text{ MW}
    \]

    that would need to be covered by electricity imported from neighbouring countries, assuming that domestic production and consumption remain at these levels.

    Fingrid estimates in [electricity adequacy outlook for winter 2026–2027](https://www.fingrid.fi/en/news/news/2026/electricity-is-expected-to-be-sufficient-this-winter-provided-that-domestic-production-and-transmission-lines-operate-reliably/), that this electricity can be imported from Sweden and Estonia, provided that the transmission connections operate normally.

    ### Formulating the optimization problem

    Let us consider a simplified situation in which we must decide how much electricity to import from Sweden and Estonia to cover the difference between domestic production and consumption. We consider a one-hour period during which electricity production and consumption are assumed to remain constant. Thus, a power level of 3,200 MW corresponds to 3,200 MWh of electricity during this hour.

    We define two decision variables:

    \[
    \begin{aligned}
    x_S &= \text{electricity imported from Sweden [MWh]},\\
    x_E &= \text{electricity imported from Estonia [MWh]}.
    \end{aligned}
    \]

    We assume that electricity imports from each country have constant prices that are known when making our decision. For this example, let us assume the following prices:

    | Country | Import price |
    |---|---:|
    | Sweden | 40 €/MWh |
    | Estonia | 60 €/MWh |

    These prices are illustrative and do not represent actual electricity market prices for a particular hour. The objective is to minimize the total cost of importing electricity. The objective function is therefore

    \[
    C(x_S,x_E)=40x_S+60x_E.
    \]

    Unlike our earlier example, the objective function now depends on two decision variables. However, we cannot choose the import quantities arbitrarily. They must satisfy certain constraints.

    ### Electricity supply requirement

    First, the total electricity imported must be sufficient to cover the estimated difference between Finnish electricity consumption and domestic production. This gives the constraint

    \[
    x_S+x_E\geq3\,200.
    \]

    Notice that the direction of the inequality differs from the capacity constraint in the previous example. Here, the constraint establishes a *lower bound* rather than an upper bound.

    ### Transmission capacity constraints

    Electricity transmission connections have limited capacities. Therefore, electricity imports from each country cannot exceed the available transmission capacity. According to Fingrid, the total maximum import capacity from Sweden is approximately 3,100 MW, while the corresponding capacity from Estonia is approximately 1,000 MW.

    For our one-hour period, these capacities translate into the following maximum import quantities:

    | Country | Maximum available imports |
    |---|---:|
    | Sweden | 3,100 MWh |
    | Estonia | 1,000 MWh |

    The corresponding constraints are

    \[
    x_S\leq3\,100
    \]

    and

    \[
    x_E\leq1\,000.
    \]

    Finally, since we are considering imports rather than exports, both decision variables must be nonnegative:

    \[
    x_S,x_E\geq0.
    \]

    ### The complete optimization problem

    Combining the objective function and constraints, we obtain

    \[
    \begin{aligned}
    \min_{x_S,x_E}\quad
    &40x_S+60x_E\\
    \text{subject to}\quad
    &x_S+x_E\geq3\,200,\\
    &x_S\leq3\,100,\\
    &x_E\leq1\,000,\\
    &x_S,x_E\geq0.
    \end{aligned}
    \]

    We now have a linear optimization problem with two decision variables. Although this is a simplified description of electricity imports rather than a model of actual Nordic electricity-market clearing, it illustrates a fundamental management problem: how should a required quantity of a resource be obtained from different suppliers when their costs and available quantities differ? In a more realistic models there could be much more choices, the problem could span over a time period, or it could contain some uncertainties. Complicating the model can easily take the numer of decision variables to hundreds, thousands, or even higher.

    ### Finding the optimal solution

    Before solving this problem computationally, let us consider what we can conclude from the mathematical formulation. Both import prices are positive. Therefore, importing more electricity than necessary would only increase the total costs. Consequently, at the optimum,

    \[
    x_S+x_E=3\,200.
    \]

    We can write

    \[
    x_E=3\,200-x_S
    \]

    and substitute this into the objective function:

    \[
    \begin{aligned}
    C(x_S)
    &=40x_S+60(3\,200-x_S)\\
    &=192\,000-20x_S.
    \end{aligned}
    \]

    The objective function is strictly decreasing in \(x_S\). In other words, increasing the imports from Sweden reduces the total cost because Swedish electricity is cheaper than Estonian electricity in our example. Therefore, we should import as much electricity as possible from Sweden.

    Since

    \[
    x_S\leq3\,100,
    \]

    the optimal solution is

    \[
    x_S^*=3\,100,\qquad x_E^*=100.
    \]

    The corresponding minimum cost is

    \[
    \begin{aligned}
    C(x_S^*,x_E^*)
    &=40(3\,100)+60(100)\\
    &=130\,000\text{ €}.
    \end{aligned}
    \]

    Notice that at the optimum, the entire available import capacity from Sweden is used, while the capacity from Estonia is not.

    ???- question "Changes in transmission capacity"

        Suppose a technical disturbance reduces the maximum available import capacity from Sweden from 3,100 MWh to 2,600 MWh.

        **Determine the new optimal solution and the corresponding minimum cost.**

        How much does the minimum cost increase compared with the original problem?

        What happens if Swedish import capacity is reduced further to 2,100 MWh?

        **Does the optimization problem still have a feasible solution? Justify your answer mathematically.**

    !!! note "Modelling assumptions"

        The electricity demand, domestic production, and transmission-capacity figures are based on Fingrid's published information about the Finnish electricity system. The import prices, however, are illustrative.

        Our model assumes that the entire available transmission capacity can be used to import electricity at the stated prices. In reality, electricity availability and prices in neighbouring countries depend on market conditions, and the actual Nordic electricity market is more complicated.

        Thus, the solution is optimal **within the assumptions of our model**, not necessarily an optimal or achievable outcome in the actual electricity market.
    """)
    return


@app.cell
def _(pyo):
    model = pyo.ConcreteModel()

    # Decision variables (MWh)
    model.x_S = pyo.Var(bounds=(0, 3100))
    model.x_E = pyo.Var(bounds=(0, 1000))

    # Minimize electricity import costs (€)
    model.cost = pyo.Objective(
        expr=40 * model.x_S + 60 * model.x_E,
        sense=pyo.minimize,
    )

    # Electricity supply requirement (MWh)
    model.supply = pyo.Constraint(
        expr=model.x_S + model.x_E >= 3200
    )

    # Solve
    solver = pyo.SolverFactory("highs")
    results = solver.solve(model)
    pyo.assert_optimal_termination(results)
    return (model,)


@app.cell(hide_code=True)
def _(mo, model, pyo):
    mo.md(f"""
    **Optimal electricity imports**

    Sweden: {pyo.value(model.x_S):,.0f} MWh

    Estonia: {pyo.value(model.x_E):,.0f} MWh

    Minimum cost: {pyo.value(model.cost):,.0f} €
    """)
    return


@app.cell(hide_code=True)
def _(mo, model, plot_lp_2d):
    fig_lp, ax_lp = plot_lp_2d(
        model,
        xlim=(0, 3500),
        ylim=(0, 1200),
        var_labels={
            "x_S": r"$x_S$",
            "x_E": r"$x_E$",
        },
        title="Electricity LP-solution"
    )

    mo.mpl.interactive(fig_lp)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
