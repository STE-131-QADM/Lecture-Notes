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

    return mo, np, plt, pyo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Optimization Part III: Duality and Shadow Prices

    In Part I, we studied how Finland could cover a shortage of electricity by importing it from Sweden and Estonia. We found the least-cost combination of imports. In Part II, we used the same problem to investigate feasibility and the structure of linear optimization models.

    But many times we might be interested in questions that go beyond the cheapest import plan. For example:

    - How much would a larger transmission connection from Sweden save?
    - How expensive would it be to cover another MWh of electricity demand?
    - Which limits actually prevent the system from operating more cheaply?

    Of course, we could change the problem's numbers and solve it again. Duality provides a more systematic way to understand how an optimal solution responds to changes in requirements and resource limits. The resulting values are called *shadow prices*.

    This chapter focuses on interpreting those values for management decisions. We will also learn how the primal and dual linear programs are related. A more mathematical derivation using the Lagrangian is provided as a note at the end.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Returning to the electricity-import problem

    Recall the example from Part I and II. Fingrid's assessment of electricity adequacy for winter 2026–2027 motivated a simplified scenario in which Finnish electricity consumption could exceed domestic production by 3 200 MW. We modelled imports over one hour, so this corresponds to a requirement of 3 200 MWh.

    We considered only imports from Sweden and Estonia, subject to the following assumptions:

    | | Sweden | Estonia |
    |:--|--:|--:|
    | Import cost | 40 €/MWh | 60 €/MWh |
    | Maximum available imports in one hour | 3 100 MWh | 1 000 MWh |

    The prices are hypothetical. The capacity and electricity-adequacy figures were presented in Part I, based on
    [Fingrid's winter 2026–2027 assessment](https://www.fingrid.fi/en/news/news/2026/electricity-is-expected-to-be-sufficient-this-winter-provided-that-domestic-production-and-transmission-lines-operate-reliably/).

    Let \(x_S\) be imports from Sweden and \(x_E\) imports from Estonia, both measured in MWh. The optimization problem is

    \[
    \begin{aligned}
    \min_{x_S,x_E}\quad &40x_S+60x_E\\
    \text{subject to}\quad
        &x_S+x_E\geq 3\,200 &&\text{(import requirement)},\\
        &x_S\leq3\,100 &&\text{(Swedish connection)},\\
        &x_E\leq1\,000 &&\text{(Estonian connection)},\\
        &x_S,x_E\geq0.
    \end{aligned}
    \]

    In the language of duality, this is the primal problem: it determines the quantities we should choose.

    Because Swedish imports cost less, we first use all 3 100 MWh of Swedish capacity. Estonia supplies the remaining 100 MWh. Thus,

    \[
      x_S^*=3\,100,\qquad x_E^*=100,
    \]

    and the minimum total cost is

    \[
      C^*=40(3\,100)+60(100)=130\,000\text{ €}.
    \]

    Notice an important difference: all Swedish import capacity is used, but 900 MWh of Estonian import capacity remains unused. We called these binding and nonbinding constraints in Part I.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Value of additional transmission

    Suppose the connection from Sweden could deliver 100 MWh more during the hour. Its limit would increase from 3 100 MWh to 3 200 MWh.

    Sweden could then cover the entire import requirement:

    \[
       x_S^*=3\,200,\qquad x_E^*=0.
    \]

    The new minimum cost would be

    \[
      40(3\,200)=128\,000\text{ €}.
    \]

    Compared with the original cost of €130 000, this would save €2 000:

    \[
       \frac{128\,000-130\,000}{3\,200-3\,100}
          =-20\text{ €/MWh}.
    \]

    So, near the original solution, one additional MWh of Swedish import capacity reduces minimum cost by €20. That makes sense: it lets us replace one MWh of €60 electricity from Estonia with €40 electricity from Sweden. The savings are \(60-40=20\) € per MWh.

    Now consider Estonia instead. If its capacity increased from 1 000 to 1 100 MWh, our cheapest import plan would stay exactly the same. The extra capacity would have no immediate value in this model.

    The effect of relaxing a constraint depends on whether that constraint actually limits the optimal decision.

    !!! definition "Shadow price"
        A shadow price measures how the optimal objective value changes when the right-hand side of a constraint changes by a small amount, while the other model parameters remain fixed.

        Here the objective is a cost. Increasing Swedish capacity changes minimum cost at a rate of −20 €/MWh.

    The sign and the units matter. Because the objective is measured in euros and a transmission limit in MWh, the shadow price is measured in €/MWh.

    ???- question "Value of a constraint ☠️"
        Suppose the Swedish connection could carry an additional 20 MWh during the hour, with everything else unchanged.

        Predict the new minimum import cost without resolving the entire optimization problem. Would the same reasoning work for 20 additional MWh of Estonian connection capacity?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The value of Swedish connection capacity

    The value of additional transmission capacity cannot remain €20/MWh forever. Once Sweden can supply all 3 200 MWh, there is no further need to import electricity from Estonia in our simplified model.

    Use the slider to change the maximum quantity that can be imported from Sweden. The demand requirement (3 200 MWh), Estonian capacity (1 000 MWh), and the two import prices remain unchanged.

    The first plot shows the chosen imports. The second shows the minimum achievable cost at different Swedish capacity limits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    swedish_capacity_slider = mo.ui.slider(
        start=2300,
        stop=3500,
        step=100,
        value=3100,
        show_value=True,
        label="Available import capacity from Sweden (MWh)",
    )
    swedish_capacity_slider
    return (swedish_capacity_slider,)


@app.cell(hide_code=True)
def _(current_imports, np, plt):
    _fig, (_ax_mix, _ax_cost) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    from matplotlib.patches import Rectangle

    _fig, (_ax_mix, _ax_cost) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # Import quantities
    _s = current_imports["sweden"]
    _e = current_imports["estonia"]

    # Bar dimensions
    _y = 0
    _h = 0.45

    # Sweden: blue background and yellow cross
    _ax_mix.barh(_y, _s, height=_h, color="#006AA7", label="Sweden")

    _ax_mix.add_patch(Rectangle(
        (0, _y - _h * 0.1),
        _s, _h * 0.2,
        color="#FECC00",
    ))

    _ax_mix.add_patch(Rectangle(
        (_s * 0.35, _y - _h / 2),
        _s * 0.10, _h,
        color="#FECC00",
    ))

    # Estonia: blue, black and white horizontal stripes
    for i, color in enumerate(["#4891D9", "#000000", "#FFFFFF"]):
        _ax_mix.add_patch(Rectangle(
            (_s, _y + _h / 2 - (i + 1) * _h / 3),
            _e, _h / 3,
            facecolor=color,
            edgecolor="none",
        ))

    # Outline the Estonian segment
    _ax_mix.add_patch(Rectangle(
        (_s, _y - _h / 2),
        _e, _h,
        fill=False,
        edgecolor="gray",
        linewidth=0.8,
    ))

    _ax_mix.set_xlim(0, 3500)
    _ax_mix.set_ylim(-0.65, 0.65)
    _ax_mix.set_yticks([0])
    _ax_mix.set_yticklabels(["Imports"])
    _ax_mix.set_xlabel("Electricity imported (MWh)")
    _ax_mix.set_title("Least-cost import plan")
    _ax_mix.grid(axis="x", alpha=0.2)
    _ax_mix.set_axisbelow(True)

    # Right panel stays unchanged
    _caps = np.linspace(2200, 3500, 261)
    _sweden = np.minimum(_caps, 3200)
    _estonia = 3200 - _sweden
    _costs = 40 * _sweden + 60 * _estonia

    _ax_cost.plot(_caps, _costs, linewidth=2.2, label="Minimum import cost")
    _ax_cost.scatter(
        [current_imports["swedish_capacity"]],
        [current_imports["cost"]],
        s=75, zorder=4, label="Selected capacity",
    )
    _ax_cost.axvline(
        3200, linestyle="--", alpha=0.6,
        label="Sweden covers all imports",
    )
    _ax_cost.set(
        xlim=(2200, 3500),
        xlabel="Swedish import capacity (MWh)",
        ylabel="Minimum total cost (€)",
        title="Cost of the optimal solution",
    )
    _ax_cost.grid(alpha=0.2)
    _ax_cost.legend(fontsize=8)

    _fig.tight_layout()
    _fig

    _caps = np.linspace(2200, 3500, 261)
    _sweden = np.minimum(_caps, 3200)
    _estonia = 3200 - _sweden
    _costs = 40 * _sweden + 60 * _estonia
    _ax_cost.plot(_caps, _costs, linewidth=2.2, label="Minimum import cost")
    _ax_cost.scatter(
        [current_imports["swedish_capacity"]], [current_imports["cost"]],
        s=75, zorder=4, label="Selected capacity",
    )
    _ax_cost.axvline(3200, linestyle="--", alpha=0.6,
                     label="Sweden covers all imports")
    _ax_cost.set(
        xlim=(2200, 3500),
        xlabel="Swedish import capacity (MWh)",
        ylabel="Minimum total cost (€)",
        title="Cost of the optimal solution",
    )
    _ax_cost.grid(alpha=0.2)
    _ax_cost.legend(fontsize=8)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(current_imports, mo):
    _capacity = current_imports["swedish_capacity"]
    _swedish_marginal_saving = (
        20 if _capacity < 3200 else 0
    )
    _explanation = (
        "At exactly 3,200 MWh, the cost curve has a **corner**. "
        "The effect of adding capacity (zero) differs from the effect "
        "of removing capacity (€20/MWh). A dual solution need not assign "
        "a unique shadow price at this point."
        if _capacity == 3200
        else (
            "Extra Swedish capacity replaces more expensive Estonian "
            "imports, until Swedish capacity reaches 3,200 MWh."
            if _capacity < 3200
            else "Sweden can already cover the full requirement, so "
                 "further Swedish capacity cannot reduce the modelled cost."
        )
    )
    mo.md(f"""
    **Optimal imports at the selected Swedish capacity**

    | Result | Value |
    |:--|--:|
    | Imports from Sweden | {current_imports['sweden']:,.0f} MWh |
    | Imports from Estonia | {current_imports['estonia']:,.0f} MWh |
    | **Minimum total cost** | **€{current_imports['cost']:,.0f}** |
    | Savings from increasing Swedish capacity by one MWh (to the right) | €{_swedish_marginal_saving}/MWh |

    {_explanation}
    """)
    return


@app.cell
def _(pyo):
    def solve_imports(swedish_capacity, demand=3200.0, estonian_capacity=1000.0):
        """Solve both the primal import LP and its explicit dual with HiGHS."""
        solver = pyo.SolverFactory("highs")
        if not solver.available():
            raise RuntimeError("HiGHS is unavailable; install highspy and pyomo.")

        primal = pyo.ConcreteModel()
        primal.sweden = pyo.Var(domain=pyo.NonNegativeReals)
        primal.estonia = pyo.Var(domain=pyo.NonNegativeReals)
        primal.cost = pyo.Objective(
            expr=40 * primal.sweden + 60 * primal.estonia,
            sense=pyo.minimize,
        )
        primal.requirement = pyo.Constraint(
            expr=primal.sweden + primal.estonia >= demand,
        )
        primal.sweden_limit = pyo.Constraint(
            expr=primal.sweden <= swedish_capacity,
        )
        primal.estonia_limit = pyo.Constraint(
            expr=primal.estonia <= estonian_capacity,
        )
        pyo.assert_optimal_termination(solver.solve(primal))

        # Nonnegative dual variables represent the demand cost and the
        # cost-saving magnitudes of relaxing the two capacity limits.
        dual = pyo.ConcreteModel()
        dual.requirement_value = pyo.Var(domain=pyo.NonNegativeReals)
        dual.sweden_value = pyo.Var(domain=pyo.NonNegativeReals)
        dual.estonia_value = pyo.Var(domain=pyo.NonNegativeReals)
        dual.objective = pyo.Objective(
            expr=(
                demand * dual.requirement_value
                - swedish_capacity * dual.sweden_value
                - estonian_capacity * dual.estonia_value
            ),
            sense=pyo.maximize,
        )
        dual.sweden_condition = pyo.Constraint(
            expr=dual.requirement_value - dual.sweden_value <= 40,
        )
        dual.estonia_condition = pyo.Constraint(
            expr=dual.requirement_value - dual.estonia_value <= 60,
        )
        pyo.assert_optimal_termination(solver.solve(dual))

        cost = pyo.value(primal.cost)
        dual_bound = pyo.value(dual.objective)
        if abs(cost - dual_bound) > 1e-5:
            raise RuntimeError("Unexpected primal–dual objective mismatch.")

        return {
            "sweden": pyo.value(primal.sweden),
            "estonia": pyo.value(primal.estonia),
            "cost": cost,
            "swedish_capacity": float(swedish_capacity),
            "demand_value": pyo.value(dual.requirement_value),
            "swedish_capacity_value": pyo.value(dual.sweden_value),
            "estonian_capacity_value": pyo.value(dual.estonia_value),
            "dual_bound": dual_bound,
        }

    return (solve_imports,)


@app.cell(hide_code=True)
def _(solve_imports, swedish_capacity_slider):
    current_imports = solve_imports(float(swedish_capacity_slider.value))
    return (current_imports,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The value of meeting additional electricity demand

    So far, we have changed the capacity of a connection. What if instead the required quantity of imports increases?

    Return to the original limits: Sweden can supply at most 3 100 MWh, and Estonia at most 1 000 MWh. Suppose the import requirement rises from 3 200 to 3 300 MWh.

    Swedish imports remain at 3 100 MWh, so Estonia must supply 200 MWh. The new minimum cost becomes

    \[
      40(3\,100)+60(200)=136\,000\text{ €}.
    \]

    Thus, one additional MWh of required imports increases the cost by \(60\) € near the original solution:

    \[
       \frac{136\,000-130\,000}{3\,300-3\,200}
          =\boxed{60\text{ €/MWh}}.
    \]

    This is the shadow price of the import-requirement constraint. It differs from the €20/MWh value of Swedish transmission capacity. An extra MWh of demand must be purchased from Estonia for €60, whereas an extra MWh of Swedish capacity merely replaces one MWh of expensive imports with a cheaper one.

    At the original optimum, the three constraints have the following interpretations:

    | Constraint | Status | Marginal change in minimum cost |
    |:--|:--|:--|
    | Required imports: at least 3 200 MWh | Binding | +60 € per additional required MWh |
    | Sweden: at most 3 100 MWh | Binding | −20 € per additional available MWh |
    | Estonia: at most 1 000 MWh | Nonbinding | 0 € per additional available MWh |

    A constraint is binding when it holds with equality at the optimum. Otherwise it has slack. In a linear program, a
    nonbinding inequality constraint has a zero associated dual value. A binding constraint, however, can sometimes also have a zero value.

    ???- question "Exercise 4 — Demand and capacity are not the same ☠️"
        Suppose the import requirement increases by 10 MWh, with capacities unchanged. Predict the increase in the minimum cost.

        Now suppose that Swedish import capacity instead increases by 10 MWh, while the requirement stays at 3 200 MWh. How much would that save? Explain why the answers differ.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The dual problem

    We can now describe the electricity problem from two viewpoints:

    - The primal problem asks which import quantities minimize cost.
    - The dual problem asks which values can be assigned to the import requirement and transmission capacities while remaining consistent with the costs of the available import options.

    Introduce one nonnegative dual variable for each original constraint:

    | Dual variable | Meaning | Units |
    |:--|:--|:--|
    | \(\lambda\) | Additional cost of requiring one more MWh of imports | €/MWh |
    | \(\mu_S\) | Cost-saving value of one more MWh of Swedish capacity | €/MWh |
    | \(\mu_E\) | Cost-saving value of one more MWh of Estonian capacity | €/MWh |

    The dual linear program is

    \[
    \begin{aligned}
    \max_{\lambda,\mu_S,\mu_E}\quad
       &3\,200\lambda-3\,100\mu_S-1\,000\mu_E\\
    \text{subject to}\quad
       &\lambda-\mu_S\leq40 &&\text{(Sweden)},\\
       &\lambda-\mu_E\leq60 &&\text{(Estonia)},\\
       &\lambda,\mu_S,\mu_E\geq0.
    \end{aligned}
    \]

    The objective is built from each constraint's right-hand side multiplied by its resource value. Capacity terms have negative signs because making an upper capacity limit less restrictive can only maintain or reduce the minimum cost.

    The two dual constraints have a straightforward interpretation. For example, \(\lambda-\mu_S\leq40\) means that the marginal value attributed to an MWh of imports, after accounting for scarce Swedish transmission capacity, cannot be greater than the €40 cost of the Swedish import option. The Estonian option gives an analogous
    condition with €60.

    Unlike the primal, the dual is a maximization problem. Its purpose is to find the strongest lower bound on the minimum import cost. We will see below why this is useful.

    At the original capacity limits, an optimal dual solution is

    \[
       \lambda^*=60,\qquad \mu_S^*=20,\qquad \mu_E^*=0.
    \]

    These are exactly the marginal values we calculated by changing the primal's right-hand sides.

    ???- question "Read the dual constraints ☠️☠️"
        Substitute \(\lambda=60\), \(\mu_S=20\), and \(\mu_E=0\) into the two dual constraints. Are they both satisfied?

        In your own words, explain why \(\mu_E=0\) is consistent with using only 100 MWh of the available 1 000 MWh of Estonian transmission capacity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Weak and strong duality

    The primal chooses imports and arrives at the minimum cost

    \[
       40(3\,100)+60(100)=\boxed{130\,000}.
    \]

    The dual chooses resource values. At its optimum, its objective is

    \[
       3\,200(60)-3\,100(20)-1\,000(0)
            =\boxed{130\,000}.
    \]

    So the two optimization problems have the same optimal objective value, even though their decision variables have very different meanings.

    ### Weak duality: establishing a lower bound

    In a cost-minimization problem, every feasible dual solution provides a lower bound on the cost of every feasible primal
    solution. This is called weak duality.

    For example, if we have found a feasible dual solution with value €125 000, we know that the optimal import cost cannot be below €125 000. Such a bound is useful even before we know the best possible import plan.

    ### Strong duality: optimal values coincide

    Strong duality says that when a linear program has a feasible solution and a finite optimum, its dual also has an optimum, and the two optimal objective values are equal.

    That is exactly what happens above: the cost of our best import plan and the best dual lower bound are both €130,000. This equality can also help us verify that an apparently good decision is truly
    optimal.

    ???- question "Exercise 6 — Verify optimality ☠️☠️"
        A colleague proposes the import plan \(x_S=3\,100\) MWh and \(x_E=100\) MWh. You already have the
        feasible dual solution \(\lambda=60,\ \mu_S=20,\ \mu_E=0\).

        Show that the objective values match. Why does this establish that the proposed import plan is optimal?

    ???- info "Deriving lower bounds"
        Let \(x_S,x_E\) be any feasible import plan and
        \(\lambda,\mu_S,\mu_E\) any feasible dual solution.
        From the dual constraints,

        \[
          (\lambda-\mu_S)x_S+(\lambda-\mu_E)x_E
            \leq 40x_S+60x_E.
        \]

        Using the original import requirement and capacity limits, together with nonnegative dual variables, gives

        \[
        \begin{aligned}
        40x_S+60x_E
        &\geq \lambda(x_S+x_E)-\mu_Sx_S-\mu_Ex_E\\
        &\geq 3\,200\lambda-3\,100\mu_S-1\,000\mu_E.
        \end{aligned}
        \]

        Thus every feasible dual objective value is at most the cost of every feasible import plan. This proves weak duality for our particular model. No calculus is required.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deriving the dual for a general linear program

    The electricity example shows what a dual problem looks like, but how can we construct the dual of any linear program? Rather than memorizing rules, we can derive the formulation from one simple idea: find a lower bound on the cost of every feasible decision.

    ### Step 1: Start with a general primal problem

    Consider a cost-minimization problem with nonnegative decision variables, written using the matrix notation from Part II:

    !!! definition "Primal linear program"
        \[
        \begin{aligned}
        (P)\qquad \min_{x\geq0}\quad &c^\top x\\
        \text{subject to}\quad &Ax\geq b.
        \end{aligned}
        \]

    Here, \(x\) contains the decisions, \(c\) contains their unit costs, and each row of \(Ax\geq b\) describes one
    requirement. Suppose there are \(n\) decision variables and \(m\) constraints, so that \(A\) has \(m\) rows and \(n\) columns.

    Part II introduced another inequality form, with constraints \(Ax\leq b\), and allowed unrestricted decision variables.
    We use \(Ax\geq b\) and \(x\geq0\) here because it makes the derivation especially transparent for a purchasing problem. This is a standard form, not a restriction on what LPs can represent: an inequality can be reversed by multiplying by \(-1\), and an unrestricted variable can be written as the difference of two nonnegative variables. We return to the Part II form in the optional Lagrangian note at the end.

    ### Step 2: Put a nonnegative weight on every requirement

    Give constraint \(i\) a weight \(y_i\geq0\). There is one weight for each of the \(m\) constraints. Because every constraint must be satisfied, multiplying by nonnegative weights and adding the results gives

    \[
       y^\top Ax\geq y^\top b.
    \]

    We can read the right-hand side, \(b^\top y\), as a proposed valuation of all the requirements. The key question is: under what conditions can we be sure that this amount is no greater than the cost of any feasible plan?

    ### Step 3: Require the weighted activities to cost no more than their actual prices

    Notice that

    \[
       y^\top Ax=(A^\top y)^\top x.
    \]

    The vector \(A^\top y\) has one entry for each decision variable. Its \(j\)-th entry tells us how much value the weights assign to one unit of activity \(j\). If we require

    \[
       A^\top y\leq c,
    \]

    then the assigned value of every activity is at most its actual unit cost. Since the activities satisfy \(x\geq0\), we can multiply these inequalities by the activity levels and add them:

    \[
       (A^\top y)^\top x\leq c^\top x.
    \]

    Combining this with Step 2 produces the important result

    \[
       \underbrace{b^\top y}_{\text{lower bound}}
       \ \leq\ y^\top Ax\ \leq\
       \underbrace{c^\top x}_{\text{cost of a feasible plan}}.
    \]

    This inequality holds for every feasible primal decision \(x\) and every nonnegative \(y\) satisfying \(A^\top y\leq c\). We have therefore found a systematic way to construct lower bounds on the minimum cost. This is also a derivation of weak duality for the general formulation.

    ???- question "Why must the weights be feasible? ☠️☠️"
        In Step 2 we required \(y\geq0\), and in Step 3 we required \(A^\top y\leq c\).

        1. Why would multiplying a \(\geq\) constraint by a negative weight make the argument in Step 2 invalid?
        2. Why do we need \(x\geq0\) to conclude that \((A^\top y)^\top x\leq c^\top x\)?

    ### Step 4: Choose the strongest lower bound

    All feasible weights give valid lower bounds, but some bounds may be more informative than others. We therefore maximize the lower bound \(b^\top y\) while maintaining the conditions that made it valid.

    !!! definition "Dual linear program"
        The dual of the primal problem above is

        \[
        \begin{aligned}
        (D)\qquad \max_{y\geq0}\quad &b^\top y\\
        \text{subject to}\quad &A^\top y\leq c.
        \end{aligned}
        \]

    This is a second LP. It has \(m\) decision variables one for each primal constraint and \(n\) inequalities one for each primal variable. If the primal has a finite optimal value, strong duality tells us that the best lower bound equals the minimum cost itself.

    | Primal problem | Dual problem |
    |:--|:--|
    | Choose activity levels \(x\) | Choose constraint weights \(y\) |
    | Minimize \(c^\top x\) | Maximize \(b^\top y\) |
    | Requirements \(Ax\geq b\) | Value restrictions \(A^\top y\leq c\) |
    | \(n\) variables, \(m\) constraints | \(m\) variables, \(n\) constraints |

    ### Step 5: Recover the electricity dual

    The same general derivation applies to our electricity problem. First rewrite the upper capacity limits as lower inequalities:

    \[
       -x_S\geq-3\,100,\qquad -x_E\geq-1\,000.
    \]

    The vectors and matrix in the standard form become

    \[
       x=\begin{pmatrix}x_S\\x_E\end{pmatrix},\quad
       c=\begin{pmatrix}40\\60\end{pmatrix},\quad
       A=\begin{pmatrix}
          1&1\\-1&0\\0&-1
       \end{pmatrix},\quad
       b=\begin{pmatrix}3\,200\\-3\,100\\-1\,000\end{pmatrix}.
    \]

    There are three constraints, so introduce three nonnegative weights \(y=(\lambda,\mu_S,\mu_E)^\top\). The general dual objective becomes

    \[
       b^\top y=3\,200\lambda-3\,100\mu_S-1\,000\mu_E,
    \]

    and \(A^\top y\leq c\) becomes

    \[
       \lambda-\mu_S\leq40,\qquad
       \lambda-\mu_E\leq60.
    \]

    These are exactly the objective and constraints of the electricity dual from Section 5. Notice that the capacity values have negative coefficients in \(b\), because we reversed the original inequalities. Hence \(\mu_S=20\) represents a saving of €20 per additional MWh of Swedish capacity, while the change in minimum cost is \(-20\) €/MWh.

    ???- question "Construct another dual ☠️☠️"
        Consider a simpler electricity-import problem:

        \[
        \begin{aligned}
        \min_{x_S,x_E\geq0}\quad &30x_S+50x_E\\
        \text{subject to}\quad &x_S+x_E\geq100,\\
                               &x_S\leq70.
        \end{aligned}
        \]

        First rewrite the Swedish capacity constraint in \(\geq\) form. Then identify \(A\), \(b\), and \(c\), and use \(A^\top y\leq c\) to construct the dual. What do the two dual variables represent? Try to identify their optimal values without using a solver.

    !!! note "Other LP forms"
        The formula above is for the particular standard form \(\min\{c^\top x:Ax\geq b,\ x\geq0\}\). Different sign or variable restrictions lead to corresponding changes in the dual. For example, an unrestricted primal variable produces an equality dual constraint, rather than a \(\leq\) constraint. The optional note at the end derives the exact inequality form used in Part II.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using shadow prices in management decisions

    Shadow prices are most useful when we interpret them alongside the decision problem rather than as isolated numbers.

    Suppose the Finnish electricity system could temporarily obtain another 50 MWh of Swedish import capacity for the modelled hour. At the original optimum, this reduces the total cost by

    \[
        50\text{ MWh}\times20\text{ €/MWh}=1\,000\text{ €}.
    \]

    If the additional capacity cost €600 to arrange for that hour, it would yield an immediate net saving of €400 under the model's assumptions. However, we should not automatically extend this calculation to an entire year: electricity demand, import prices, and connection availability change from hour to hour.

    Shadow prices are local. At a Swedish capacity of 3 200 MWh, increasing the limit further saves nothing because Sweden can already supply the entire requirement. At this point the cost function has a corner. Reducing Swedish capacity by one MWh raises cost by €20, while increasing it by one MWh has no effect. The two one-sided changes are different, and an optimal dual solution need not have a unique shadow price there.

    Remember also that the shadow prices describe this simplified optimization model. They are not observed market prices or
    necessarily what a transmission operator should charge. The model omits many relevant features of electricity systems, including other production sources, simultaneous exports, losses, congestion within countries, and the responses of market participants.

    ???- question "A capacity investment decision ☠️☠️"
        An operator offers 50 additional MWh of Swedish import capacity during the hour for a fixed payment of €1 200. Using the original optimization model, would purchasing the additional capacity reduce the total expenditure including that payment? Explain your calculation.

        What further information would you need before recommending a long-term transmission investment?
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have now already seen the dual parameters in action in the food network example. This example already showed some ways that the solution of an optimisation problem is connected to the value of the dual parameter.

    This part of the linear optimization lectures will be all about that. We will be revolving around one main question: **how valuable would additional amount of our scarce resource be?**

    Let's consider the food network example. For example in that a reasonable question you could have as a person responsible for the logistics could be, how much would expanding one of the logistics hubs reduce the cost of the logistics network. This question could of course be analysed by doing for example some *ad hoc* scenario work with the model or other kinds of approaches.

    However, it turns out that there is a pricipled way to approach the question. To understand the approach, that yields a very practial residue *shadow price*, we need some insights into the theory of optimisation.

    ## Lagrangian Function

    We are already familiar with the inequality form of a LP, that turned out to be general enough for it to be sufficient to represent any LP. Let's now consider a *primal problem* of form

    \[
        \begin{aligned}
            p^*=\min_x \;&c^\top x \\
            \text{s.t. }& \\
            Ax&\leq b \\
            x&\geq 0.
        \end{aligned}
    \]

    This formulation requires that our constraint is satisfied exactly *i.e.* the formulation requires that \(Ax\leq b\) holds without any exceptions. What would it mean for the constraint to be enforced only approximately? Note also that the nonnegativity of \(x\) can be expressed as a less than or equal constraint so it doesn't cause any problems.

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

    The relaxation of constraints is very useful and widely used technique. We can understand it as augmenting the objective function. In the original problem each constraint needs to hold exactly. Optimizing Lagrangian function with respect to \(x\) is an unconstrained problem and it provides lower bound for the original problem. Since \(\lambda\) is nonnegative \(\lambda^\top(Ax-b)\leq 0\) so from that follows that with any feasible \(x\), the following inequality holds \(c^\top x\geq c^\top x+\lambda^\top(Ax-b)\).

    Let's now think about the following function

    \[
        \begin{aligned}
            \theta(\lambda)&=\min_{x\geq 0}{c^\top x+\lambda^\top(Ax-b)} \\
            &=\min_{x\geq 0}{\mathcal{L}(x,\lambda)}.
        \end{aligned}
    \]

    It is called the *Lagrange dual function*.

    !!! note "Definition of $\theta$"
        The previous statement is slightly incorrect. Actually the dual is defined as

        \[
            \theta(\lambda)=\inf_{x\geq 0} \mathcal{L}(x,\lambda)
        \]

        where \(\inf\) denotes the [infimum](https://en.wikipedia.org/wiki/Infimum_and_supremum#Formal_definition). This is the correct definition, because it is possible for the Lagrange dual to take values \(\infty\) and \(-\infty\). However, to make the definition slightly more familiar and easier we can suffice with the \(\min\) for now.

    ### Example: Lagrange dual function

    Now we know that the Lagrangian dual function produces a lower bound for our primal problem. Let's consider the following problem

    \[
        \begin{aligned}
            p^*=\min_{x,y\geq0}\; & 2x+3y \\
            \text{s.t.}& \\
             x + y &\geq 4\\
             x &\geq 0 \\
             y &\geq 0.
        \end{aligned}
    \]

    This problem implies the following dual function.

    \[
        \begin{aligned}
            \theta(\lambda)&=\min_{x,y\geq0}\;2x+3y+\lambda(4-x-y) \\
            &=\min_{x,y\geq0}\;(2-\lambda)x+(3-\lambda)y+4\lambda
        \end{aligned}
    \]

    We can solve the primal problem easily by noticing the the constraint is symmetric and the objective gives less weight for the \(x\). Therefore we must have \(x=4\) and \(y=0\), yielding objective value of \(p^*=8\). From the dual function we see that for the function to be bounded we must have \(\lambda\leq2\) and it is already required that \(\lambda\geq0\). The minimum inside the dual function is therefore achieved by setting \(x=y=0\). Therefore the dual function is

    \[
        \theta(\lambda)=\begin{cases}
            4\lambda & \text{for }0\leq\lambda\leq2 \\
            -\infty & \text{otherwise}.
        \end{cases}
    \]

    Both the primal problem and the Lagrange dual function are showed below in the figure.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    # --------------------------------------------------
    # Problem:
    # min 2x + 3y
    # s.t. x + y >= 4, x >= 0, y >= 0
    #
    # Lagrangian:
    # L(x,y,lambda) = (2-lambda)x + (3-lambda)y + 4lambda
    # --------------------------------------------------

    z_star = 8.0
    lambda_star = 2.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ==================================================
    # LEFT: PRIMAL PROBLEM
    # ==================================================

    ax = axes[0]

    x = np.linspace(0, 7, 400)
    y = np.linspace(0, 7, 400)
    X, Y = np.meshgrid(x, y)

    # Objective function
    Z = 2 * X + 3 * Y

    # Feasible region: x + y >= 4
    feasible = (X + Y >= 4)
    ax.contourf(
        X, Y, feasible.astype(float),
        levels=[0.5, 1.5],
        alpha=0.15
    )

    # Constraint boundary x + y = 4
    ax.plot(
        x, 4 - x,
        linewidth=2,
        label=r"$x+y=4$"
    )

    # Contours of objective 2x + 3y
    levels = [6, 8, 10, 12, 14, 16, 18]
    cs = ax.contour(X, Y, Z, levels=levels, linewidths=1.5)

    # Put contour values directly on the contour lines
    ax.clabel(cs, inline=True, fontsize=9, fmt=r"$z=%d$")

    # Optimal point
    ax.scatter(
        [4], [0],
        s=150,
        zorder=5,
        marker="*",
        color="red",
        label=r"Optimal point $(4,0)$"
    )

    ax.annotate(
        r"$z^*=8$",
        xy=tuple(np.array([4,0])+0.1*(np.array([4.6, 0.8])-np.array([4,0]))),
        xytext=(4.6, 0.8),
        arrowprops=dict(arrowstyle="->"),
        fontsize=11
    )

    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Primal problem")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    # ==================================================
    # RIGHT: LAGRANGIAN DUAL FUNCTION
    # ==================================================

    ax = axes[1]

    lam = np.linspace(0, 3.5, 400)
    q = np.where(lam <= 2, 4 * lam, np.nan)

    ax.plot(
        lam, q,
        linewidth=3,
        label=r"$\theta(\lambda)=4\lambda \;\; (0\leq \lambda \leq 2)$"
    )

    # Primal optimum
    ax.axhline(
        z_star,
        linestyle="--",
        linewidth=2,
        label=r"$p^*=8$"
    )

    # Dual optimum
    ax.scatter(
        [lambda_star], [z_star],
        s=150,
        zorder=5,
        marker="*",
        color="red",
    )

    ax.annotate(
        r"$\lambda^*=2,\; \theta(\lambda^*)=8$",
        xy=tuple(np.array([2,8])+0.05*(np.array([0.8,9.3])-np.array([2,8]))),
        # xy=(2, 8),
        xytext=(0.8, 9.3),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10
    )

    # Region where q(lambda) = -infinity
    ax.axvspan(
        2, 3.5,
        alpha=0.12,
        label=r"$\theta(\lambda)=-\infty$"
    )

    ax.text(
        2.7, 3.2,
        r"$\theta(\lambda)=-\infty$",
        ha="center",
        fontsize=10
    )

    ax.set_xlim(0, 3.5)
    ax.set_ylim(-1, 11)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\theta(\lambda)$")
    ax.set_title("Lagrangian dual")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Primal and Lagrangian Dual", fontsize=15)
    plt.tight_layout()

    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we know that the Lagrange dual function gives us *a* lower bound. What if we would like to find the highest lower bound? Since all values of \(\theta(\lambda)\) are less than \(p^*\), what would happen if we try to maximize the dual function \(d^*=\max_{\lambda\geq0}\theta(\lambda)\)? You probably already guess (and see in the figure), that we can establish a link between the maximum of the dual and primal problem. Since we already know that \(d^*\leq p^*\), which is called weak duality, the best we can hope is that \(d^*=p^*\). We call the case where dual and primal problems agree strong duality. Since the solutions for the minimization of the primal and maximization of the dual yield the same value, the optimization problem depicted above satisfies strong duality.

    ### Consequences of optimising dual function

    The results from the example hold more generally. Let's consider a primal problem

    \[
        \begin{aligned}
            (P): p^*=\min_{x}\;& c^\top x \\
            &\text{s.t.} \\
            Ax&\leq b\\
            x\geq 0.
        \end{aligned}
    \]

    This has the Lagrangian dual function as discussed. We can further manipulate the dual as

    \[
        \begin{aligned}
            \theta(\lambda) &= \min_{x\geq 0}\{c^\top x+\lambda^\top(Ax-b)\} \\
            &= \min_{x\geq 0}\{c^\top x+\lambda^\top Ax-\lambda^\top b)\} \\
            &= -\lambda^\top b+\min_{x\geq 0}\{(c^\top+\lambda^\top A)x\}
        \end{aligned}
    \]

    Let's think what we have here. Clearly the \(\lambda^\top b\) term is constant in the minimization, since the \(x\) doesn't enter it. In the latter term, if \(-c^\top\leq\lambda^\top A\), then \(\min_{x\geq 0}\{(c^\top+\lambda^\top A)x\}\geq0\) so the minimum must be \(0\) for the \(x\) value of \(0\). Therefore the value of the dual function is

    \[
        \theta(\lambda)=\begin{cases}
            -\lambda^\top b & \text{for }0\leq c^\top+\lambda^\top A \\
            -\infty & \text{otherwise.}
        \end{cases}
    \]

    Maximizing the lower bound yields therefore

    \[
        \begin{aligned}
            (D): \max_{\lambda\geq0}& -\lambda^\top b \\
            \text{s.t.}& \\
            -c^\top&\leq\lambda^\top A
        \end{aligned}
    \]

    which is exactly the dual problem. It is just an another LP problem.

    We now have rough understanding what the dual problem is in relation to the primal problem, but what can we actually achieve with it? Next we will see that it is possible to categorise the solutions of the primal problem by computing the dual problem value.

    !!! proposition "Consequence of weak duality"
        Assume a LP problem satisfies weak duality *i.e.* \(\lambda^\top b\leq c^\top x\).

        1. If the optimal value of the primal problem is \(-\infty\), then the dual problem must be infeasible.
        2. If the optimal value of the dual problem is \(\infty\), then the primal problem must be infeasible.
        3. Let \(x\) and \(\lambda\) be feasible for the primal and dual problems. Assume \(\lambda^\top b=c^\top x\), then \(x\) is optimal to the primal and \(\lambda\) to dual problem.

    ???- proof "Proof"
        1. Assume that the primal problem has the optimum at \(-\infty\) and the dual has a feasible solution. Then by the weak duality holds \(\lambda^\top b\leq c^\top x\=-\infty) contradicting the feasibility.
        2. Same proof, just change the roles of primal and dual.
        3. Let \(\hat{x}\) be another feasible solution for the primal problem. Then by assumption \(c^\top\hat{x}\geq\lambda^\top b=c^\top x\), so \(x\) is optimal. Similar argument holds for the dual problem.

    This result has very practical consequences. If we know that the
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ???- note "Mathematical treatment of duality"
        This note derives the dual for a general LP, rather than starting with the particular electricity example. It is optional; the main text's derivation using weighted constraints already explains how to construct and interpret the dual.

        Begin with

        \[
        \begin{aligned}
        \min_{x\geq0}\quad &c^\top x\\
        \text{subject to}\quad &Ax\geq b.
        \end{aligned}
        \]

        Rewrite the constraints as \(b-Ax\leq0\), multiply each by a nonnegative parameter \(y_i\), and add the result to the objective. This gives the Lagrangian

        \[
        \begin{aligned}
        \mathcal L(x,y)
        &=c^\top x+y^\top(b-Ax)\\
        &=b^\top y+(c-A^\top y)^\top x,
        \qquad y\geq0.
        \end{aligned}
        \]

        For a feasible \(x\), we have \(b-Ax\leq0\). Thus \(\mathcal L(x,y)\leq c^\top x\): adding the weighted constraints produces a lower bound on the cost.

        To obtain a bound that depends only on \(y\), minimize the Lagrangian over \(x\geq0\):

        \[
        \theta(y)
        =\inf_{x\geq0}\mathcal L(x,y)
        =b^\top y+\inf_{x\geq0}(c-A^\top y)^\top x.
        \]

        If every component of \(c-A^\top y\) is nonnegative, the last term has minimum zero (take \(x=0\)). Otherwise, increasing an activity with a negative coefficient without bound sends the Lagrangian to \(-\infty\). Therefore

        \[
            \theta(y)=
            \begin{cases}
            b^\top y,&A^\top y\leq c,\\
            -\infty,&\text{otherwise}.
            \end{cases}
            \qquad y\geq0.
        \]

        Maximizing \(\theta(y)\) gives precisely

        \[
            \begin{aligned}
                \max_{y\geq0}\quad &b^\top y\\
                \text{subject to}\quad &A^\top y\leq c.
            \end{aligned}
        \]

        The inequality form from Part II. Earlier we also considered the more general representation

        \[
            \min_{x\in\mathbb R^n}c^\top x
            \quad\text{subject to}\quad Ax\leq b,
        \]

        where the decision variables have no sign restrictions. For \(\lambda\geq0\), the Lagrangian is

        \[
        \mathcal L(x,\lambda)
        =c^\top x+\lambda^\top(Ax-b)
        =-b^\top\lambda+(c+A^\top\lambda)^\top x.
        \]

        Because \(x\) is unrestricted, minimizing a nonzero linear coefficient over all real values gives \(-\infty\). A finite lower bound is possible only when \(c+A^\top\lambda=0\). The dual is therefore

        \[
            \begin{aligned}
                \max_{\lambda\geq0}\quad &-b^\top\lambda\\
                \text{subject to}\quad &A^\top\lambda=-c.
            \end{aligned}
        \]

        If we instead require \(x\geq0\) in this last primal, the equality changes to \(A^\top\lambda\geq-c\). This is why it matters to check both the direction of the inequalities and the sign restrictions on the decision variables when writing down a dual problem.

        The electricity model is a concrete application of the first derivation. Its dual variables have the useful interpretation of the marginal cost of required imports and the saving from relaxing transmission-capacity limits.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
