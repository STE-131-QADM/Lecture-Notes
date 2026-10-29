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
    In this chapter we introduce mixed-integer linear programming (MILP). As in linear programming (LP), the objective and constraints are linear in the decision variables. The difference is that some variables must take integer values. For example, a variable may represent a number of trucks, the number of production batches, or a yes/no decision.

    In mathematical notation, an MILP can contain continuous variables \(x\in\mathbb{R}^n\) and integer variables \(y\in\mathbb{Z}^m\). Binary variables \(y_i\in\{0,1\}\) are a particularly important special case. A problem with only integer decision variables is often called an *integer linear program* (ILP).

    Think back to our food distribution network. Modeling the shipment of a bulk product, such as grain, with continuous quantities is often reasonable. But suppose a processed product can only be shipped in packages of 150 kg. If \(n\) denotes the number of packages, then the shipped quantity in kilograms is \(x=150n\), where \(n\in\mathbb{Z}_{\geq 0}\). A plan involving 149.5 kg or 150.5 kg may no longer be feasible.

    Integer variables do much more than count indivisible items. They also let us represent whether a plant is activated, which supplier is selected, or when a conditional operating rule must be enforced.

    ## Difference between LP and MILP solutions

    We can construct a simple example where the difference between an LP relaxation and an MILP becomes arbitrarily large. Let \(M\geq1\) and consider

    \[
        \begin{aligned}
            \min_{x,y}\quad & x+My\\
            \text{s.t.}\quad
            &x\geq 1,\\
            &x\leq My,\\
            &x\geq0,\quad y\in\{0,1\}.
        \end{aligned}
    \]

    Its LP relaxation replaces \(y\in\{0,1\}\) with \(0\leq y\leq1\). The relaxation chooses \(x^*=1\) and \(y^*=1/M\), while the MILP must choose \(x^*=1\) and \(y^*=1\).

    Consequently,
    \[
    V_{\mathrm{LP}}(M)=2
    \qquad
    V_{\mathrm{MILP}}(M)=M+1
    \]

    The absolute integrality gap in this example is therefore

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
def _(mo):
    mo.md(rf"""
    The LP relaxation gives a lower bound on the minimum cost, but a fractional activation decision such as \(y=0.1\) may have no practical meaning. This is why replacing an MILP by an LP can produce a plan that is impossible to implement, even if its numerical solution looks reasonable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MILP problems

    The need for binary variables is not limited to decisions that count whole items. Many management problems involve fixed costs: a facility incurs an additional cost when it operates at all, regardless of how little it produces.

    \[
        C_i(q_i)=\begin{cases}
            c^f_i+c^v_i q_i, & q_i>0,\\
            0, & q_i=0.
        \end{cases}
    \]

    When \(c_i^f>0\), this is a discontinuous, nonconvex cost function. Introduce a binary activation variable \(z_i\in\{0,1\}\) and a valid upper production bound \(U_i\). We can include fixed and variable costs in an MILP as

    \[
        \begin{aligned}
            \min_{q,z}\quad &\sum_{i=1}^n(c_i^f z_i+c_i^v q_i)\\
            \text{s.t.}\quad &0\leq q_i\leq U_i z_i,\quad i=1,\ldots,n,\\
            &z_i\in\{0,1\},\quad i=1,\ldots,n,\\
            &\text{other demand and production constraints.}
        \end{aligned}
    \]

    When \(q_i>0\), the inequality forces \(z_i=1\). If \(q_i=0\), the constraint also permits \(z_i=1\); however, with strictly positive fixed costs and no other incentive to activate a facility, minimization drives it to \(z_i=0\). Thus the modeled costs agree with the intended fixed-charge structure at the optimum. The bound \(U_i\) should come from the physical or contractual limits of the actual problem.

    ### Conditional constraints and Big-M

    Binary variables can also encode logical conditions. Suppose we require \(a^\top x\leq b\) when a binary variable \(z\) is one, but do not want to impose the inequality when \(z=0\). We can write

    \[
        a^\top x\leq b+M(1-z),\qquad z\in\{0,1\}.
    \]

    If \(z=1\), the original inequality is enforced. If \(z=0\), the constraint must become redundant. This requires a *valid* value of \(M\): for example, a known upper bound on \(a^\top x-b\) over the original variable bounds. A larger number is not automatically better. An excessively large \(M\) weakens the LP relaxation and may cause numerical difficulties. This construction is often called a [Big M method](https://en.wikipedia.org/wiki/Big_M_method).

    Similar techniques can represent logical operations such as AND, OR, and NOT, or restrictions on how many projects can be chosen. We will use several of these in the electricity procurement example below.

    !!! note "Computational difficulty"
        The decision version of general integer linear programming is [NP-complete](https://en.wikipedia.org/wiki/NP-completeness), while general integer linear optimization is [NP-hard](https://en.wikipedia.org/wiki/NP-hardness). These are worst-case complexity results, not claims that every practical MILP is impossible to solve. Modern MILP solvers handle many real-world instances effectively, but an exact optimum cannot always be proven within the available computing time.

    You are not expected to learn the underlying solver algorithms in detail. The aim is to formulate discrete management decisions correctly, solve them using appropriate tools, and understand the limitations of the resulting plans. If you after all would be interested in some of the more theoretical aspects, there is also this added bonus of the possibility to earn million dollars just by solving one [small problem](https://www.claymath.org/millennium/p-vs-np/) in the field.
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
        single_agreement=False,
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
        
        if single_agreement:
            # Either Estonia or domestic reserve can be activated, not both.
            m.single_agreement = pyo.Constraint(expr=m.y_E + m.y_R <= 1)

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
    ### How do we know whether an MILP solution is optimal?

    In Linear Optimization Part III, we used duality to obtain bounds on an LP's optimal value. LP relaxations also give us a way to judge an MILP solution, even when an integer optimizer has not yet proven optimality.

    For a minimization problem, let \(L\) be a valid lower bound, for example the optimum of its LP relaxation. Let \(U\) be the objective value of a feasible integer solution found by the solver. Then

    \[
        L\leq V^*_{\mathrm{MILP}}\leq U.
    \]

    The LP relaxation changes the constraint for the decision variable to take values in \(\{0,1\}\) to taking values from \([0,1]\). Clearly \(\{0,1\}\subseteq [0,1]\) and as we are not modifying the feasible set otherwise we have the following relationship \(\mathcal{F}_{\{0,1\}}\subseteq\mathcal{F}_{[0,1]}\) for the entire feasible sets. Now the result about nested feasible sets that was established in Food Network cont'd holds here and we can know for sure that the LP relaxation always achieves at least as low minimum as the MILP for minimization problem and vice versa for maximization.

    The difference \(U-L\) bounds how much the feasible solution could be improved. A commonly used relative measure is \(\frac{U-L}{\max(1,|U|)}\), although solvers can differ in their precise definition of the relative MIP gap. Solvers update both bounds during the search and can stop when the gap is sufficiently small, even without proving the exact optimum.

    For the fixed-fee electricity case, compare the LP relaxation with the optimal MILP solution already computed above:
    """)
    return


@app.cell(hide_code=True)
def _(electricity_fixed, electricity_fixed_lp_relax, mo, pyo):
    _lower_bound = pyo.value(electricity_fixed_lp_relax.total_cost)
    _integer_cost = pyo.value(electricity_fixed.total_cost)
    _difference = _integer_cost - _lower_bound
    _relative_difference = 100 * _difference / max(1, abs(_integer_cost))

    mo.md(f"""
    | Quantity | Cost |
    |:--|--:|
    | LP relaxation lower bound | {_lower_bound:,.0f} € |
    | Optimal feasible integer solution | {_integer_cost:,.0f} € |
    | Difference between the two bounds | {_difference:,.0f} € ({_relative_difference:.2f}% of integer cost) |

    The MILP optimum has been proved here by the solver. If we only knew these
    two bounds, the difference would quantify our remaining uncertainty about
    the true optimum. It is not the solver's final optimality gap after
    optimality has already been proved.
    """)
    return


@app.cell
def _(pyo, solve_electricity_case):
    big_m_comparison = []
    for _factor in (1, 10, 100):
        _relaxed = solve_electricity_case(relax=True, big_m_factor=_factor)
        _integer = solve_electricity_case(big_m_factor=_factor)
        big_m_comparison.append({
            "factor": _factor,
            "relaxed_cost": pyo.value(_relaxed.total_cost),
            "integer_cost": pyo.value(_integer.total_cost),
            "fractional_E": pyo.value(_relaxed.y_E),
        })
    return (big_m_comparison,)


@app.cell(hide_code=True)
def _(big_m_comparison, mo):
    _rows = "\n".join(
        f"    | {item['factor']} | {item['fractional_E']:.3f} | "
        f"{item['relaxed_cost']:,.0f} | {item['integer_cost']:,.0f} |"
        for item in big_m_comparison
    )
    mo.md(rf"""
    | Big-M multiplier | LP value of \(y_E\) | LP relaxation cost (€) | MILP cost (€) |
    |--:|--:|--:|--:|
    {_rows}

    A loose activation inequality can make the LP relaxation look
    artificially cheap even though the feasible integer solutions have not
    changed. Therefore it is preferable to use tight, justified bounds
    rather than an arbitrary huge coefficient.

    ???- question "LP bound getting worse ☠️☠️"
        With a fixed purchase of 100 MWh from Estonia, determine the
        smallest LP-relaxation value of \(y_E\) allowed by
        \(x_E\leq1000k y_E\), for \(k=1,10,100\).
        What happens to the fraction of the fixed fee paid in the LP relaxation?
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
        Suppose the LP relaxation gives \(y_E=0.1\), \(x_E=100\). Why does simply rounding \(y_E\) to one not necessarily produce an optimal MILP solution? What happens if the 200 MWh minimum purchase is enforced? Try to think of another example where rounding goes badly wrong.
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
    ### Step 4: Discrete decisions and the value of transmission capacity

    In Linear Optimization Part III, we discussed shadow prices and how they can represent the marginal value of relaxing a capacity constraint in an LP. Once activation costs and minimum quantities enter the model, that interpretation needs more care.

    We can study the optimal total cost as the available Swedish import capacity decreases. Below we solve a sequence of different models, keeping demand and the other supply capacities constant.

    - **Continuous-cost LP:** no fixed fees and no minimum commitments.
    - **MILP:** fixed fees and 200 MWh minimum quantities.
    - **MILP with security rule:** same MILP plus the conditional
      security requirement above.

    An LP's optimal value as a function of a right-hand-side parameter is typically piecewise linear and continuous on its feasible interval so the slope can change, but there is no jump. For an MILP, the optimal value can also be discontinuous, because a small change in capacity may force another complete contract or generator to be activated. Consequently, an LP shadow price is not generally a globally valid marginal price for an integer optimization problem. Comparing complete scenarios is often a better management tool.

    The experiment below includes capacities of both 2 200 and 2 199 MWh, which differ by just 1 MWh. Watch what happens when an additional arrangement must be activated to cover the shortage.
    """)
    return


@app.cell
def _(np, solve_electricity_case):
    # Include both sides of a 1 MWh capacity change at the threshold.
    electricity_capacities = np.array(sorted(set(
        np.arange(1900, 3101, 50).tolist() + [2199]
    )))
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
             label="Continuous-cost LP", linewidth=2)

    # Do not join points across the 2,199-to-2,200 capacity jump:
    # a straight line there would misleadingly imply intermediate feasible costs.
    _left = electricity_capacities <= 2199
    _right = electricity_capacities >= 2200
    for _series, _label, _marker in [
        (electricity_milp_cost_curve, "MILP: minimum lots", "s"),
        (electricity_secure_cost_curve, "MILP: security rule", "^"),
    ]:
        _line, = _ax.plot(
            electricity_capacities[_left], _series[_left],
            marker=_marker, markersize=3.5, label=_label,
        )
        _ax.plot(
            electricity_capacities[_right], _series[_right],
            marker=_marker, markersize=3.5, color=_line.get_color(),
        )

    _ax.axvline(2200, linewidth=1, linestyle="--", alpha=0.5)
    _ax.set_xlabel("Available Swedish import capacity (MWh)")
    _ax.set_ylabel("Minimum total procurement cost (€)")
    _ax.set_title("Procurement cost under a capacity shock")
    _ax.grid(alpha=0.2)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(pyo, solve_electricity_case):
    electricity_threshold_scenarios = []
    for _capacity in (2200, 2199):
        _m = solve_electricity_case(
            sweden_capacity=_capacity, minimum_batch=200
        )
        electricity_threshold_scenarios.append({
            "capacity": _capacity,
            "sweden": pyo.value(_m.x_S),
            "estonia": pyo.value(_m.x_E),
            "reserve": pyo.value(_m.x_R),
            "cost": pyo.value(_m.total_cost),
        })
    return (electricity_threshold_scenarios,)


@app.cell(hide_code=True)
def _(electricity_threshold_scenarios, mo):
    _rows = "\n".join(
        f"    | {r['capacity']:,.0f} | {r['sweden']:,.0f} | "
        f"{r['estonia']:,.0f} | {r['reserve']:,.0f} | {r['cost']:,.0f} |"
        for r in electricity_threshold_scenarios
    )
    _difference = (
        electricity_threshold_scenarios[1]["cost"]
        - electricity_threshold_scenarios[0]["cost"]
    )
    mo.md(f"""
    | Swedish capacity (MWh) | Sweden (MWh) | Estonia (MWh) | Reserve (MWh) | Cost (€) |
    |--:|--:|--:|--:|--:|
    {_rows}

    Decreasing Swedish capacity by only 1 MWh increases the minimum cost
    by {_difference:,.0f} €. With 2,199 MWh of Swedish capacity, neither
    Estonia nor domestic reserve can cover the remaining 1,001 MWh alone.
    The operator must activate both contracts, illustrating a discontinuity
    that has no direct LP shadow-price analogue.

    ???- question "A marginal capacity change with a large effect ☠️☠️"
        Explain why the cost change is much greater than the variable cost
        of 1 MWh. Which previously unused fixed cost is now unavoidable?
        What would happen if minimum purchase quantities were removed?
    """)
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
        Suppose a supplier offers an incremental two-band price schedule: the first 400 MWh costs 65 €/MWh and the next 600 MWh costs 45 €/MWh (maximum purchase: 1,000 MWh). The cheaper price applies only to units above the first 400 MWh, unlike the mutually exclusive contracts considered earlier.

        The operator must purchase exactly \(D\) MWh, with \(0\leq D\leq1000\). Let \(x\) and \(y\) be the purchased quantities in the first and second bands.

        (a) Formulate the continuous LP with \(x+y=D\), \(0\leq x\leq400\), and \(0\leq y\leq600\). Why does minimizing the positive cost \(65x+45y\) not force the first band to be filled before the second?

        (b) Introduce a binary variable \(z\) and constraints \(y\leq600z\) and \(x\geq400z\) to impose the ordering. Explain how they work for \(z=0\) and \(z=1\).

        (c) Compare the two model solutions for \(D=500\) MWh. Which formulation represents the actual tariff?
    """)
    return


if __name__ == "__main__":
    app.run()
