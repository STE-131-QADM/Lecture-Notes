import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mixed Integer Linear Programming

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


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pyomo.environ as pyo
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    import highspy
    from matplotlib.patches import Circle, Polygon, Rectangle

    return mo, np, plt, pyo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this chapter we are going to add one, initially relatively innocent looking, constraint to our optimisation model. The new constraint will reduce a variable to take values inside the set \(\mathbb{Z}=\{\ldots,-2,-1,0,1,2,\ldots\}\). This model has important special cases where the set is \(\mathbb N\) or \(\{0,1\}\).

    This solves one of our very practical problem. For example, let's think about the food network problem once more. In That example we assumed that we can choose the amount of food we transported from each place in arbitrary precision (or at least machine precision). This may be very sensible assumption. For example, if we are transporting grains, the precision is more than enough for this to be good approximation.

    However, now imagine that the farm already does some kind of processing for the food and packages it on boxes and those boxes are further bundled to for example 150 kg units. All of a sudden it can make quite a big difference whether we are able to add additional 149.5 kg or 150.5 kg to our truck. In this case using continuous variables for the decision can lead to insufficient approximation of the real process and therefore to very inefficient solutions.

    ## Difference between LP and MILP solutions

    We can easily cook up an example where the difference between LP and MILP solutions can be made arbitrarily big. Let's take the following model

    \[
        \begin{aligned}
            \min_{x,y}\quad & x+My\\
            \text{s.t.}\quad
            &x\geq 1,\\
            &x\leq My,\\
            &y\in\{0,1\}.
        \end{aligned}
    \]

    Now the LP chooses \(x^*=1\) and \(y^*=1/M\). The MILP must choose \(x^*=1\) and \(y^*=1\).

    Consequently,
    \[
    V_{\mathrm{LP}}(M)=2
    \qquad
    V_{\mathrm{MILP}}(M)=M+1
    \]

    Thus absolute integrality gap is

    \[
    V_{\mathrm{MILP}}-V_{\mathrm{LP}}=M-1,
    \]

    which is unbounded when we start to increase \(M\).

    Below you can see the Pyomo version of the model.
    """)
    return


@app.cell
def _(pyo):
    def solve_model(M, relax=False):
        model = pyo.ConcreteModel()

        model.x = pyo.Var(domain=pyo.NonNegativeReals)

        model.y = pyo.Var(
            domain=pyo.UnitInterval if relax else pyo.Binary
        )

        model.demand = pyo.Constraint(expr=model.x >= 1)
        model.capacity = pyo.Constraint(
            expr=model.x <= M * model.y
        )

        model.obj = pyo.Objective(
            expr=model.x + M * model.y,
            sense=pyo.minimize
        )

        solver = pyo.SolverFactory("highs")
        results = solver.solve(model)
        pyo.assert_optimal_termination(results)

        return (
            pyo.value(model.x),
            pyo.value(model.y),
            pyo.value(model.obj)
        )

    return (solve_model,)


@app.cell
def _(np, solve_model):
    # Compare optimal values as M increases
    M_values = np.geomspace(2, 1000, 30)

    lp_values = np.array([
        solve_model(M, relax=True)[2] for M in M_values
    ])

    milp_values = np.array([
        solve_model(M, relax=False)[2] for M in M_values
    ])
    return M_values, lp_values, milp_values


@app.cell(hide_code=True)
def _(M_values, lp_values, milp_values, plt):
    gap_fig, gap_ax = plt.subplots(figsize=(6, 4))

    gap_ax.plot(M_values, lp_values, label="LP")
    gap_ax.plot(M_values, milp_values, label="MILP")
    gap_ax.set_xscale("log")
    gap_ax.set_yscale("log")
    gap_ax.set_xlabel("Facility capacity M")
    gap_ax.set_ylabel("Optimal cost")
    gap_ax.set_title("LP vs. MILP")
    gap_ax.legend()
    gap_ax.grid(alpha=0.2)

    gap_fig
    return


@app.cell(hide_code=True)
def _(np, plt, solve_model):
    # Choose a fixed M for the geometric visualization
    M = 10

    # Optimal solutions from Pyomo
    x_lp, y_lp, z_lp = solve_model(M, relax=True)
    x_milp, y_milp, z_milp = solve_model(M, relax=False)

    # Grid for objective contours
    x = np.linspace(0, M + 1, 400)
    y = np.linspace(0, 1.2, 400)
    X, Y = np.meshgrid(x, y)

    # Objective function
    Z = X + M * Y

    fig_feasible, ax = plt.subplots(figsize=(8, 5.5))

    # LP feasible region: triangle
    vertices = np.array([
        [1, 1 / M],
        [1, 1],
        [M, 1]
    ])

    ax.fill(
        vertices[:, 0],
        vertices[:, 1],
        alpha=0.2,
        label="LP feasible region"
    )

    # Constraint boundaries
    ax.axvline(
        1,
        linestyle="--",
        linewidth=1.5,
        label=r"$x \geq 1$"
    )

    ax.plot(
        x,
        x / M,
        linestyle="--",
        linewidth=1.5,
        label=r"$x \leq My$"
    )

    ax.axhline(
        1,
        linestyle="--",
        linewidth=1.5,
        label=r"$y \leq 1$"
    )

    # MILP feasible set (y = 1, 1 <= x <= M)
    ax.plot(
        [1, M],
        [1, 1],
        linewidth=4,
        color="tab:red",
        label="MILP feasible set"
    )

    # Objective contours: x + My = constant
    levels = np.sort(np.unique(np.r_[
        np.linspace(z_lp, z_milp + M, 8),
        z_milp
    ]))

    contours = ax.contour(
        X, Y, Z,
        levels=levels,
        colors="gray",
        alpha=0.65,
        linewidths=1
    )

    ax.clabel(contours, inline=True, fontsize=9, fmt="%.1f")

    # Optimal LP solution
    ax.scatter(
        x_lp, y_lp,
        s=110,
        marker="o",
        color="tab:blue",
        zorder=5,
        label=f"LP optimum: {z_lp:.1f}"
    )

    # Optimal MILP solution
    ax.scatter(
        x_milp, y_milp,
        s=150,
        marker="*",
        color="tab:red",
        zorder=6,
        label=f"MILP optimum: {z_milp:.1f}"
    )

    ax.set_xlim(0, M + 1)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Treatment quantity x")
    ax.set_ylabel("Facility construction y")
    ax.set_title(f"LP relaxation vs. MILP (M = {M})")
    ax.grid(alpha=0.15)
    ax.legend(loc="center right", fontsize=9)

    fig_feasible.tight_layout()
    fig_feasible
    return


@app.cell(hide_code=True)
def _(mo, np):
    mo.md(rf"""
    At this point things start to look relatively bleak. All of a sudden the model we were using seems to produce very bad results in a very important category of problems.

    If you really think about it, you will quickly notice that many (even most?) useful models include variables that cannot plausibly be assumed to take continuous values.

    And after all, aren't we always playing around with discrete quantities while using computers? With floating-point numbers, there is always some distance between consecutive representable values.

    For example, near 1.0, this spacing is characterized by [machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon), which on this computer is \(\varepsilon_{{\text{{MACH}}}} =\) {np.finfo(float).eps}. You can find this information by using `np.finfo(float).eps`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MILP problems

    Fortunately, it turns out that machine precision is usually not a problem in practice, and we have good algorithms at our
    disposal to solve problems involving integer values.

    MILP also unlocks many different modeling techniques. For example, we can implement conditional constraints. Conditional constraints can be used for example to encode affine relationships fo form \(\alpha+\beta x\) where \(\alpha\) and \(\beta\) are parameters and \(x\) is a variable. Note that even though we many times use the word linear to describe this relationship, it is actually not. It is affine since for a function \(f(x)=\alpha+\beta x\) with \(\alpha\neq0\) hold following relationship

    \[
        \begin{aligned}
            f(x+y)&=\alpha+\beta x+\beta y \\
            &\neq 2\alpha+\beta x+\beta y \\
            &=f(x)+f(y)
        \end{aligned}
    \]

    Therefore, to determine for example which production facilities \(i\) with fixed costs \(c^f_i\) and variable costs \(c^v_i\) for quantity \(q_i\) we need to compute the costs that are of form

    \[
        C_i(q_i)=\begin{cases}
            c^f_i+c^v_i q_i, & \text{for }q_i>0 \\
            0, & \text{for }q_i=0
        \end{cases}
    \]

    By using boolean or indicator variables taking a value of 0 or 1 we can encode this relationsip. Consider the following MILP

    \[
        \begin{split}\begin{array}{ll}
        \min& {c^f}^T z + {c^v}^T q\\
        \mbox{s.t.}& q_i \leq M_i z_i, \quad i=1,\dots,n\\
                   & q\geq 0,\\
                   & z\in \{0,1\}^n
        \end{array}\end{split}
    \]

    where \(c^f\) and \(c^v\) are the vectors of fixed and variable costs and \(M_i\) is a valid upper bound for the amount produced by facility \(i\). In a full model, production, demand, or other operational constraints would also be included.

    It turns out that expanding the model to contain integer values gives us much more flexibility than what it initially seems. It is possible to encode logical operations with these variables as well. Let's consider a situation where want to conditinally constrain our problem with some inequality \(a^\top x\leq b\), but on some events we don't want this constrain to apply. We can use the following scheme let \(z\) denote the even so that when \(z=1\) we should enforce the constraint. Then by using the following constraints

    \[
        a^\top x\leq b+(1-z)M \\
        z\in \{0,1\}

    \]

    and by setting \(M\) sufficiently high we can achieve this. When \(z=1\) the constraint reduces to \(a^\top x\leq b\) and otherwise it becomes irrelevant if \(M\) is set large enough. This method is so widely used that it even has a name. It is called, very imaginatively, a [Big M method](https://en.wikipedia.org/wiki/Big_M_method).

    We could craft more this kind of examples and for example express the operators like OR, AND and NOT using MILP. The decision version of integer linear programming is [nondeterministic polynomial-time (NP)-complete](https://en.wikipedia.org/wiki/NP-completeness). In practice, it is one reason why we must be careful about computational effort. The associated optimization problems are in general [NP-hard](https://en.wikipedia.org/wiki/NP-hardness). This means in practice that it may not always be possible to solve the optimization problem you write down. Fortunately the modern solvers are very good at crunching this kinds of problems and in many cases it is feasible to solve many of the practical problems.

    What should you take from all of this? Probably not that you are required to get a PhD in theoretical computer science to use these methods (at least I don't 🤷). Instead, even though we are not going to concern ourselves too much with the solver algorithms, it is good to be aware of some of the limitations and possibilities of this kind of models. This being said, you are not expected to learn the deep theory behind this during this course, but you should be able to apply these tools to decision problems. And if you after all would be interested in some of the more theoretical aspects, there is also this added bonus of the possibility to earn million dollars just by solving one [small problem](https://www.claymath.org/millennium/p-vs-np/) in the field.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Application: Electricity procurement when decisions have fixed costs

    In Linear Optimization Part I and II, we investigated a one-hour shortage of 3 200 MWh of electricity in Finland. Our simplified least-cost LP purchased 3 100 MWh from Sweden at 40 €/MWh and 100 MWh from Estonia at 60 €/MWh, for a total of 130 000 €. We will now return to exactly this decision.

    A linear program implicitly assumes that the operator can import an arbitrarily small amount from either supplier and pay only the variable electricity price. That may be a good approximation for ordinary spot-market transactions. However, some procurement and operational arrangements have fixed costs or minimum commitments. For example, activating a
    stand-by power plant may require a start-up cost, and a special procurement contract may have an administrative or capacity-reservation fee.

    Suppose that the operator can also activate domestic flexible generation for this one-hour period. Consider the following illustrative scenario:

    | Source | Available quantity (MWh) | Variable cost (€/MWh) | Activation cost (€) |
    |:--|--:|--:|--:|
    | Sweden (existing import access) | 3 100 | 40 | 0 |
    | Estonia (optional contract) | 1 000 | 60 | 8 000 |
    | Domestic flexible generation | 700 | 85 | 5 000 |

    The capacity figures for Sweden and Estonia are carried over from the introductory example while the rest are made up for this example. We are modeling an operator's procurement plan, not Nordic market clearing.

    The decision is no longer just how much to buy. It is also which arrangements to activate. We will progressively add contractual rules and security requirements to see how they affect the optimal plan.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1: Activation decisions and fixed costs

    Define continuous purchased or generated quantities \(x_S,x_E,x_R\geq0\) and binary activation variables
    \(y_E,y_R\in\{0,1\}\). Sweden is already available, so it needs no activation variable in this example.

    The objective is to minimize both energy and activation costs:

    \[
      \min\; 40x_S+60x_E+85x_R+8000y_E+5000y_R.
    \]

    Supply and capacity constraints become

    \[
    \begin{aligned}
    x_S+x_E+x_R &= 3200,\\
    0\leq x_S &\leq 3100,\\
    0\leq x_E &\leq 1000y_E,\\
    0\leq x_R &\leq 700y_R,\\
    y_E,y_R &\in\{0,1\}.
    \end{aligned}
    \]

    The inequalities \(x_E\leq1000y_E\) and \(x_R\leq700y_R\) are activation constraints. If the contract is not activated, the corresponding quantity must equal zero. Conversely, using even a small positive quantity forces the binary variable to one, so the fixed cost is paid.

    We have used the physical capacity bounds (1 000 and 700 MWh) as the Big-M coefficients, rather than an arbitrary large number. The demand is expressed as an equality because we assume that surplus energy cannot be stored or sold in this simplified one-hour decision.

    ???- question "Which option would you activate? ☠️"
        If the operator purchases 3 100 MWh from Sweden, the remaining shortage is just 100 MWh. Compute the total cost of meeting it from (i) Estonia and (ii) domestic flexible generation, including the activation costs. Which supplier do you expect to be chosen?
    """)
    return


@app.cell
def _(pyo):
    def build_electricity_case(
        sweden_capacity=3100,
        fee_E=8000,
        fee_R=5000,
        minimum_batch=0,
        security_rule=False,
        # single_agreement=False,
        relax=False,
        big_m_factor=1.0,
    ):
        if not 0 <= minimum_batch <= 700:
            raise ValueError("minimum_batch must lie between 0 and 700 MWh")
        if not 0 <= sweden_capacity <= 3100:
            raise ValueError("sweden_capacity must lie between 0 and 3100 MWh")
        if big_m_factor < 1:
            raise ValueError("big_m_factor must be at least 1")

        m = pyo.ConcreteModel()
        binary = pyo.UnitInterval if relax else pyo.Binary

        ### STEP 1 ###

        # Energy decisions in MWh
        m.x_S = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, sweden_capacity))
        m.x_E = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1000))
        m.x_R = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 700))

        # Whether to activate Estonia and domestic flexible generation
        m.y_E = pyo.Var(domain=binary)
        m.y_R = pyo.Var(domain=binary)

        m.balance = pyo.Constraint(expr=m.x_S + m.x_E + m.x_R == 3200)

        # Big-M
        m.activate_E = pyo.Constraint(
            expr=m.x_E <= 1000 * big_m_factor * m.y_E
        )
        m.activate_R = pyo.Constraint(
            expr=m.x_R <= 700 * big_m_factor * m.y_R
        )

        ### STEP 2 ###
    
        if minimum_batch > 0:
            # If on, a contract must be used for at least 'minimum_batch' MWh.
            m.minimum_E = pyo.Constraint(expr=m.x_E >= minimum_batch * m.y_E)
            m.minimum_R = pyo.Constraint(expr=m.x_R >= minimum_batch * m.y_R)

        # if single_agreement:
        #     # Either Estonia or domestic reserve can be activated, not both.
        #     m.single_agreement = pyo.Constraint(expr=m.y_E + m.y_R <= 1)

        ### STEP 3 ###
    
        if security_rule and sweden_capacity > 2800:
            # If Swedish imports exceed 2 800 MWh, domestic reserve must
            # produce at least 150 MWh. The binary is a logical switch.
            m.high_sweden = pyo.Var(domain=binary)
            m.high_sweden_link = pyo.Constraint(
                expr=m.x_S <= 2800 + (sweden_capacity - 2800) * m.high_sweden
            )
            m.required_reserve = pyo.Constraint(
                expr=m.x_R >= 150 * m.high_sweden
            )

        m.energy_cost = pyo.Expression(
            expr=40 * m.x_S + 60 * m.x_E + 85 * m.x_R
        )
        m.activation_cost = pyo.Expression(
            expr=fee_E * m.y_E + fee_R * m.y_R
        )
        m.total_cost = pyo.Expression(expr=m.energy_cost + m.activation_cost)
        m.objective = pyo.Objective(expr=m.total_cost, sense=pyo.minimize)
        return m

    def solve_electricity_case(**kwargs):
        m = build_electricity_case(**kwargs)
        solver = pyo.SolverFactory("highs")
        if not solver.available():
            raise RuntimeError("HiGHS is unavailable; install 'highspy' and 'pyomo'.")
        results = solver.solve(m)
        pyo.assert_optimal_termination(results)
        return m

    return (solve_electricity_case,)


@app.cell
def _(solve_electricity_case):
    # All these scenarios use the same quantities and costs.
    electricity_original_lp = solve_electricity_case(
        fee_E=0, fee_R=0, relax=True
    )
    electricity_fixed = solve_electricity_case()
    electricity_fixed_lp_relax = solve_electricity_case(relax=True)
    electricity_minimum = solve_electricity_case(minimum_batch=200)
    electricity_security = solve_electricity_case(
        minimum_batch=200, security_rule=True
    )
    return (
        electricity_fixed,
        electricity_fixed_lp_relax,
        electricity_minimum,
        electricity_original_lp,
        electricity_security,
    )


@app.cell(hide_code=True)
def _(
    electricity_fixed,
    electricity_fixed_lp_relax,
    electricity_minimum,
    electricity_original_lp,
    electricity_security,
    mo,
    pyo,
):
    _plans = [
        ("Original continuous-cost LP", electricity_original_lp),
        ("Fixed fees: MILP", electricity_fixed),
        ("Fixed fees: LP relaxation", electricity_fixed_lp_relax),
        ("Fixed fees + minimum lots", electricity_minimum),
        ("Minimum lots + security rule", electricity_security),
    ]
    _rows = "\n    ".join(
        f"| {name} | {pyo.value(m.x_S):,.0f} | {pyo.value(m.x_E):,.0f} | "
        f"{pyo.value(m.x_R):,.0f} | {pyo.value(m.total_cost):,.0f} |"
        for name, m in _plans
    )
    mo.md(f"""
    ### Five versions of the procurement decision

    | Model | Sweden (MWh) | Estonia (MWh) | Reserve (MWh) | Total cost (€) |
    |:--|--:|--:|--:|--:|
    {_rows}

    The LP relaxation of the fixed-fee problem allows the activation variables
    to take fractional values. This can substantially understate the actual
    cost of an indivisible commitment. In contrast, the MILP chooses a plan
    that respects whole activation decisions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2: Minimum purchases and semi-continuous variables

    Imagine that the Estonian contract requires an order of at least 200 MWh, and that the domestic reserve generator cannot operate below 200 MWh in this one-hour period. These are invented operational conditions to illustrate indivisible commitments.

    A quantity \(x\) that is either zero or lies between a positive lower bound \(L\) and an upper bound \(U\) is called semi-continuous:

    \[
       x\in\{0\}\cup[L,U].
    \]

    With a binary \(y\), the exact MILP representation is

    \[
       Ly\leq x\leq Uy,\qquad y\in\{0,1\}.
    \]

    Thus, our new constraints are

    \[
    200y_E\leq x_E\leq1000y_E,\qquad
    200y_R\leq x_R\leq700y_R.
    \]

    Compare the last three integer rows of the table above. In the fixed-fee model the domestic generator is slightly cheaper for the last 100 MWh. Once we require a minimum activation level of 200 MWh, it may be necessary to buy less from the cheapest supplier (Sweden) in order to meet the contractual quantities. The optimal arrangement can therefore switch even though the energy prices have not changed.

    ???- question "Rounding the LP solution not enough? ☠️☠️"
        Suppose the LP relaxation gives \(y_E=0.1\), \(x_E=100\). Why does simply rounding \(y_E\) to one not necessarily produce an optimal MILP solution? What happens if the 200 MWh minimum purchase is enforced? Try to think of an example where rounding goes badly wrong.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Explore the value of a procurement contract

    The following slider changes only the fixed fee for activating the Estonian contract. The two models shown below differ only in whether the minimum 200 MWh commitment is enforced. Try to identify at which activation fee the optimal procurement plan switches to domestic reserve.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    electricity_fee_slider = mo.ui.slider(
        start=0, stop=15000, step=500, value=8000, show_value=True,
        label="Fixed cost of the Estonian contract (€)"
    )
    electricity_fee_slider
    return (electricity_fee_slider,)


@app.cell
def _(electricity_fee_slider, solve_electricity_case):
    electricity_no_min_by_fee = solve_electricity_case(
        fee_E=electricity_fee_slider.value
    )
    electricity_with_min_by_fee = solve_electricity_case(
        fee_E=electricity_fee_slider.value, minimum_batch=200
    )
    return electricity_no_min_by_fee, electricity_with_min_by_fee


@app.cell(hide_code=True)
def _(
    electricity_fee_slider,
    electricity_no_min_by_fee,
    electricity_with_min_by_fee,
    mo,
    pyo,
):
    def _description(m):
        _name = "Estonia" if pyo.value(m.y_E) > 0.5 else "domestic reserve"
        return (
            f"{_name}; Sweden {pyo.value(m.x_S):,.0f} MWh; "
            f"Estonia {pyo.value(m.x_E):,.0f} MWh; "
            f"reserve {pyo.value(m.x_R):,.0f} MWh; "
            f"cost {pyo.value(m.total_cost):,.0f} €"
        )

    mo.md(f"""
    **Estonian activation fee:** {electricity_fee_slider.value:,.0f} €

    **Without minimum quantity:** {_description(electricity_no_min_by_fee)}

    **With 200 MWh minimum quantity:** {_description(electricity_with_min_by_fee)}

    An activation fee is a lumpy cost: at the point where a contract is
    first used, the total cost jumps by the fixed payment. This is why
    marginal reasoning based on €/MWh alone can give the wrong decision.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3: Conditional operating rules

    Suppose the system operator introduces an security rule:

    > If imports from Sweden exceed 2 800 MWh during the hour, the domestic reserve must actually produce at least 150 MWh.

    This is a conditional constraint. Let \(h\in\{0,1\}\) be a binary indicator that permits Swedish imports above the threshold. We can write

    \[
    \begin{aligned}
    x_S &\leq 2800+300h,\\
    x_R &\geq 150h,\\
    h &\in\{0,1\}.
    \end{aligned}
    \]

    If \(x_S>2800\), the first inequality forces \(h=1\) and then the second inequality requires reserve production. If \(x_S\leq2800\), the optimizer can choose \(h=0\) and there is no additional requirement. Importantly, we only need a one-way implication. The model does not claim that \(h=1\) if and only if imports exceed 2 800 MWh.

    The number 300 is not arbitrary: it is the difference between the maximum Swedish imports (3 100 MWh) and the security threshold (2 800 MWh). This is another example of using the smallest valid Big-M rather than a huge coefficient.

    ???- question "Check the logic ☠️☠️"
        Can the solution have \(x_S=3000\) and \(h=0\)? Can it have \(x_S=2700\) and \(h=1\)? Explain which direction of the logical implication the model enforces.

    ???- question "Model a choice between contracts ☠️"
        Suppose company policy permits activating at most one of the two contingency arrangements. Express that rule using \(y_E\) and \(y_R\). How would the rule change if the manager must activate exactly one or at least one of them?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 4: The value of transmission capacity is no longer smooth

    In Linear Optimization Part III, we discussed shadow prices and how they can represent the marginal value of relaxing a capacity constraint in an LP. Once activation costs and minimum quantities enter the model, that interpretation needs more care.

    We can study the optimal total cost as the available Swedish import capacity decreases. Below we solve a sequence of different models, keeping demand and the other supply capacities constant.

    - **Continuous-cost LP:** no fixed fees and no minimum commitments.
    - **MILP:** fixed fees and 200 MWh minimum quantities.
    - **MILP with security rule:** same MILP plus the conditional
      security requirement above.

    An LP's optimal value as a function of the right-hand-side parameter is continuous and piecewise linear on a feasible interval. A MILP's optimal value need not be continuous: activating a new unit or switching contracts may cause a discontinuous change in cost. Consequently, an LP shadow price is not generally a globally valid marginal price
    for an integer optimization problem. Comparing complete scenarios is often a better management tool.
    """)
    return


@app.cell
def _(np, solve_electricity_case):
    electricity_capacities = np.arange(2200, 3101, 50)
    electricity_lp_cost_curve = np.array([
        float(solve_electricity_case(
            sweden_capacity=int(cap), fee_E=0, fee_R=0, relax=True
        ).total_cost())
        for cap in electricity_capacities
    ])
    electricity_milp_cost_curve = np.array([
        float(solve_electricity_case(
            sweden_capacity=int(cap), minimum_batch=200
        ).total_cost())
        for cap in electricity_capacities
    ])
    electricity_secure_cost_curve = np.array([
        float(solve_electricity_case(
            sweden_capacity=int(cap), minimum_batch=200, security_rule=True
        ).total_cost())
        for cap in electricity_capacities
    ])
    return (
        electricity_capacities,
        electricity_lp_cost_curve,
        electricity_milp_cost_curve,
        electricity_secure_cost_curve,
    )


@app.cell(hide_code=True)
def _(
    electricity_capacities,
    electricity_lp_cost_curve,
    electricity_milp_cost_curve,
    electricity_secure_cost_curve,
    plt,
):
    _fig, _ax = plt.subplots(figsize=(8, 4.6))
    _ax.plot(electricity_capacities, electricity_lp_cost_curve,
             marker="o", markersize=3.5, label="Continuous-cost LP")
    _ax.plot(electricity_capacities, electricity_milp_cost_curve,
             marker="s", markersize=3.5, label="MILP: minimum lots")
    _ax.plot(electricity_capacities, electricity_secure_cost_curve,
             marker="^", markersize=3.5, label="MILP: security rule")
    _ax.set_xlabel("Available Swedish import capacity (MWh)")
    _ax.set_ylabel("Minimum total procurement cost (€)")
    _ax.set_title("Cost respond to a capacity shock")
    _ax.grid(alpha=0.2)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Application: Choosing between electricity contract tariffs

    Binary variables are also useful when a decision must be chosen from a finite menu of alternatives. Suppose the operator must procure electricity from Estonia but can select one of two mutually exclusive illustrative contracts:

    | Contract | Fixed fee (€) | Unit price (€/MWh) | Minimum quantity (MWh) |
    |:--|--:|--:|--:|
    | Standard | 8,000 | 60 | 0 |
    | Volume discount | 15,000 | 50 | 500 |

    The maximum import quantity remains 1 000 MWh in either contract. Let \(y_t\) indicate the chosen contract and \(q_t\) its quantity. Variables \(c_t\) and \(F_t\) denote the variable and fixed costs respectively.  Rather than trying to multiply two decision variables to calculate a cost, allocate a separate continuous quantity variable to each contract option:

    \[
    \begin{aligned}
      \min\quad &40x_S+\sum_{t\in T}(c_tq_t+F_ty_t)\\
      \text{s.t.}\quad &x_S+\sum_{t\in T}q_t=3200,\\
      &0\leq x_S\leq\bar x_S,\\
      &L_ty_t\leq q_t\leq1000y_t,\quad t\in T,\\
      &\sum_{t\in T}y_t=1,\qquad y_t\in\{0,1\}.
    \end{aligned}
    \]

    The equation \(\sum_t y_t=1\) is an exactly-one constraint.

    At modest quantities, the lower activation fee may make the standard contract preferable. When purchases are large, the 10 €/MWh discount can compensate for the higher fixed fee.
    """)
    return


@app.cell
def _(pyo):
    def solve_electricity_tariffs(sweden_capacity):
        tariff_unit = {"Standard": 60, "Volume discount": 50}
        tariff_fee = {"Standard": 8000, "Volume discount": 15000}
        tariff_minimum = {"Standard": 0, "Volume discount": 500}

        m = pyo.ConcreteModel()
        m.T = pyo.Set(initialize=list(tariff_unit))
        m.x_S = pyo.Var(bounds=(0, sweden_capacity))
        m.q = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.y = pyo.Var(m.T, domain=pyo.Binary)

        m.balance = pyo.Constraint(
            expr=m.x_S + sum(m.q[t] for t in m.T) == 3200
        )
        m.choose_one = pyo.Constraint(
            expr=sum(m.y[t] for t in m.T) == 1
        )
        m.upper = pyo.Constraint(
            m.T, rule=lambda m, t: m.q[t] <= 1000 * m.y[t]
        )
        m.lower = pyo.Constraint(
            m.T, rule=lambda m, t: m.q[t] >= tariff_minimum[t] * m.y[t]
        )
        m.total_cost = pyo.Expression(
            expr=40 * m.x_S + sum(
                tariff_unit[t] * m.q[t] + tariff_fee[t] * m.y[t]
                for t in m.T
            )
        )
        m.objective = pyo.Objective(expr=m.total_cost, sense=pyo.minimize)
        results = pyo.SolverFactory("highs").solve(m)
        pyo.assert_optimal_termination(results)
        return m

    electricity_tariff_2600 = solve_electricity_tariffs(2600)
    electricity_tariff_2400 = solve_electricity_tariffs(2400)
    return electricity_tariff_2400, electricity_tariff_2600


@app.cell(hide_code=True)
def _(electricity_tariff_2400, electricity_tariff_2600, mo, pyo):
    def _chosen_tariff(m):
        return next(t for t in m.T if pyo.value(m.y[t]) > 0.5)

    mo.md(f"""
    | Swedish import capacity | Required Estonian imports | Chosen contract | Cost (€) |
    |:--|--:|:--|--:|
    | 2,600 MWh | {sum(pyo.value(electricity_tariff_2600.q[t]) for t in electricity_tariff_2600.T):,.0f} MWh | {_chosen_tariff(electricity_tariff_2600)} | {pyo.value(electricity_tariff_2600.total_cost):,.0f} |
    | 2,400 MWh | {sum(pyo.value(electricity_tariff_2400.q[t]) for t in electricity_tariff_2400.T):,.0f} MWh | {_chosen_tariff(electricity_tariff_2400)} | {pyo.value(electricity_tariff_2400.total_cost):,.0f} |

    ???- question "When does the discount pay off? ☠️☠️"
        Ignoring the 500 MWh eligibility threshold for a moment,
        derive the purchase quantity at which the standard and
        discounted contract have the same cost.
        Then check whether the minimum quantity allows using the
        discounted tariff at that break-even quantity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modeling toolbox: other useful MILP constructions

    Many apparently nonlinear decision rules can be expressed using binary variables and linear constraints. The following patterns from the [MOSEK mixed-integer modeling chapter](https://docs.mosek.com/modeling-cookbook/mio.html) are particularly common in management applications:

    | Decision rule | MILP representation | Example |
    |:--|:--|:--|
    | Activate a facility before using it | \(0\leq x\leq Uy\) | Warehouse opening |
    | Operate at zero or above a minimum | \(Ly\leq x\leq Uy\) | Generator minimum output |
    | A must imply B | \(y_A\leq y_B\) | A project requires a permit |
    | At most \(k\) selections | \(\sum_i y_i\leq k\) | Limited number of contracts |
    | Exactly \(k\) selections | \(\sum_i y_i=k\) | Technology or tariff choice |
    | Enforce \(a^Tx\leq b\) when \(y=1\) | \(a^Tx\leq b+M(1-y)\) | Conditional environmental standard |

    ???- question "Selecting a piecewise tariff ☠️☠️☠️"
        Suppose a supplier offers two price bands: the first 400 MWh costs 65 €/MWh and any additional energy (up to 1,000 MWh) costs 45 €/MWh. The lower price only applies to units beyond 400 MWh. Construct a linear model using two nonnegative quantity variables. Why does minimizing a positive total cost not by itself guarantee that the first segment is filled before the cheaper second segment? Add a binary variable to impose that ordering.
    """)
    return


if __name__ == "__main__":
    app.run()
