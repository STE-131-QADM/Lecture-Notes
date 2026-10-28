import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Food Network Optimization

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
    import pandas as pd
    import networkx as nx

    return mo, np, nx, pd, plt, pyo


@app.cell(hide_code=True)
def _(arc_data, pd, pyo):
    def shipment_table(model):
        rows = []

        for (i, j), properties in arc_data.items():
            tonnes = pyo.value(model.x[i, j])

            if tonnes > 1e-7:
                rows.append({
                    "From": i,
                    "To": j,
                    "Flow (t/week)": round(tonnes, 3),
                    "Cost (€)": round(properties["cost"] * tonnes, 2),
                    "Emissions (kg CO₂e)": round(
                        properties["emissions"] * tonnes, 2
                    ),
                })

        return pd.DataFrame(rows)

    return (shipment_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this study, we are going to develop a network optimization model for food networks. To begin, please take a look at the [Bortolini et al. (2016)](https://doi.org/10.1016/j.jfoodeng.2015.11.014). This exercise builds on that paper by using similar, but very much simplified model, where we are optimizing on the cost of the network. Additionally, we can set a CO<sub>2</sub>e limit for the logistics network and do the optimization conditional on that.

    ## Background

    Our problem is the following. A food distributor purchases fresh vegetables from farms, transports them to distribution centers, and delivers them to supermarkets. The distributor needs to decide how many tonnes of vegetables to transport along each possible route. Farms cannot supply unlimited quantities, distribution centers have limited throughput, and every supermarket's demand must be met.

    The simplest business objective is to minimize the total cost of transportation. However, the cheapest set of routes need not be the one with the lowest greenhouse gas emissions. This raises a second question: How much would a reduction in transportation emissions cost?

    ## Related research

    As mentioned, this exercise is inspired by [Bortolini et al. (2016)](https://doi.org/10.1016/j.jfoodeng.2015.11.014), *“Fresh food sustainable distribution: Cost, delivery time and carbon footprint three-objective optimization,”*

    Their model addresses multiple foods, distribution routes, costs, delivery time, carbon emissions, and quality loss. Our first version is deliberately smaller: a single homogeneous product, one week, and only two objectives considered in separate runs (cost and emissions). The numerical values below are all synthetic, not data from the publication.

    ## Distribution network

    The distributor has three farms (**F1–F3**), two distribution centers (**H1–H2**) and three supermarkets (**R1–R3**).

    Every farm can ship to either distribution center. Every distribution center can serve every supermarket. Direct farm-to-supermarket deliveries are excluded in this introductory example.

    - Farm supplies and hub capacities are upper bounds (tonnes/week).
    - Supermarket demands must be satisfied exactly (tonnes/week).
    - A hub is a transshipment node on weekly level: incoming and outgoing tonnes are equal.
    - Each route has a variable cost (€/tonne) and an emissions intensity
      (kg CO₂e/tonne). There are no fixed route-opening costs.
    - All delivered produce is assumed usable; travel time, spoilage, and
      inventory are not included yet.

    Let's first generate the data.
    """)
    return


@app.cell
def _():
    hubs = ("H1", "H2")
    retailers = ("R1", "R2", "R3")
    farms = ("F1", "F2", "F3")

    supply = {"F1": 70, "F2": 60, "F3": 80}
    hub_capacity = {"H1": 100, "H2": 100}
    demand = {"R1": 50, "R2": 65, "R3": 55}

    arc_data = {
        ("F1", "H1"): {
            "cost": 4, "emissions": 9
        },
        ("F1", "H2"): {
            "cost": 9, "emissions": 4
        },
        ("F2", "H1"): {
            "cost": 7, "emissions": 6
        },
        ("F2", "H2"): {
            "cost": 3, "emissions": 8
        },
        ("F3", "H1"): {
            "cost": 6, "emissions": 4
        },
        ("F3", "H2"): {
            "cost": 5, "emissions": 7
        },
        ("H1", "R1"): {
            "cost": 4, "emissions": 8
        },
        ("H1", "R2"): {
            "cost": 7, "emissions": 5
        },
        ("H1", "R3"): {
            "cost": 9, "emissions": 3
        },
        ("H2", "R1"): {
            "cost": 8, "emissions": 3
        },
        ("H2", "R2"): {
            "cost": 4, "emissions": 8
        },
        ("H2", "R3"): {
            "cost": 3, "emissions": 9
        },
    }

    return arc_data, demand, farms, hub_capacity, hubs, retailers, supply


@app.cell(hide_code=True)
def _(demand, hub_capacity, mo, pd, supply):
    _nodes = pd.DataFrame(
        [{"Node": n, "Type": "Farm", "Supply / capacity / demand (t/week)": v}
         for n, v in supply.items()]
        + [{"Node": n, "Type": "Distribution center", "Supply / capacity / demand (t/week)": v}
           for n, v in hub_capacity.items()]
        + [{"Node": n, "Type": "Retailer", "Supply / capacity / demand (t/week)": v}
           for n, v in demand.items()]
    )
    mo.vstack([
        mo.md("### Network data: capacities and demands"),
        _nodes])
    return


@app.cell(hide_code=True)
def _(arc_data, mo, pd):
    _routes = pd.DataFrame(
        [{"From": a, "To": b, "Cost (€/t)": p["cost"],
          "Emissions (kg CO₂e/t)": p["emissions"]}
         for (a, b), p in arc_data.items()]
    )

    mo.vstack([
        mo.md("### Route data: unit transportation cost and emissions"),
        _routes
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can also draw the distribution network. This is shown below.
    """)
    return


@app.cell(hide_code=True)
def _(arc_data, demand, farms, hub_capacity, hubs, nx, plt, retailers, supply):
    # Construct directed network
    G = nx.DiGraph()

    G.add_nodes_from(farms, layer="farm")
    G.add_nodes_from(hubs, layer="hub")
    G.add_nodes_from(retailers, layer="retailer")

    for (origin, destination), properties in arc_data.items():
        G.add_edge(origin, destination, **properties)

    # Fix positions to create three vertical layers
    pos = {
        "F1": (0, 2),
        "F2": (0, 1),
        "F3": (0, 0),
        "H1": (2, 1.5),
        "H2": (2, 0.5),
        "R1": (4, 2),
        "R2": (4, 1),
        "R3": (4, 0),
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        node_size=2400,
        width=1.6,
        edge_color="#94A3B8",
    )

    # Draw nodes by category
    for nodes, color in [
        (farms, "#86C5A3"),
        (hubs, "#F9CB80"),
        (retailers, "#93C5FD"),
    ]:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(nodes),
            node_color=color,
            node_size=2400,
            edgecolors="#334155",
            linewidths=1.5,
            ax=ax,
        )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=12,
        font_weight="bold",
        ax=ax,
    )

    # Display capacities, supplies, and demands under nodes
    node_details = {
        **{f: f"Supply: {supply[f]} t" for f in farms},
        **{h: f"Capacity: {hub_capacity[h]} t" for h in hubs},
        **{r: f"Demand: {demand[r]} t" for r in retailers},
    }

    for node, (x, y) in pos.items():
        ax.text(
            x,
            y - 0.27,
            node_details[node],
            ha="center",
            va="top",
            fontsize=9,
            color="#334155",
        )

    # Layer headings
    for x, heading in [
        (0, "Farms"),
        (2, "Distribution hubs"),
        (4, "Retailers"),
    ]:
        ax.text(
            x, 2.65, heading,
            ha="center",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_xlim(-0.65, 4.65)
    ax.set_ylim(-0.65, 2.9)
    ax.axis("off")
    ax.set_title(
        "Fresh Food Distribution Network",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mathematical formulation

    Let \(F\) be the farms, \(H\) the distribution centers, and \(R\) the retailers. Let \(A\) be the set of available directed transport routes (arcs). For each \((i,j)\in A\), define

    \[
    x_{ij} = \text{tonnes transported from node }i\text{ to node }j.
    \]

    Parameters are the farm supply \(s_f\), hub throughput capacity \(q_h\), retailer demand \(d_r\), route unit cost \(c_{ij}\), and route emissions intensity \(e_{ij}\).

    Note also how quickly the dimension of the problem starts to grow when we are moving towards more realistic models. In this model we already have Farms$\times$Distribution hubs$\times$Retailers$=3\times2\times3=18$ decision variables, and thus 18 dimensional problem. We are already long past the point where we could visualise the problem.

    ### Objective: minimize total transportation cost

    \[
    \min_{x}\; C(x)=\sum_{(i,j)\in A} c_{ij}x_{ij}.
    \]

    ### Constraints

    **Farm supply**: shipments cannot exceed available production.

    \[
    \sum_{h\in H} x_{fh}\le s_f \qquad \forall f\in F.
    \]

    **Flow conservation**: nothing is created or lost at a hub.

    \[
    \sum_{f\in F}x_{fh}=\sum_{r\in R}x_{hr}
    \qquad \forall h\in H.
    \]

    **Distribution center capacity**: hub throughput is limited.

    \[
    \sum_{f\in F}x_{fh}\le q_h \qquad \forall h\in H.
    \]

    **Demand satisfaction**: retailers receive the required quantities.

    \[
    \sum_{h\in H} x_{hr}=d_r \qquad \forall r\in R.
    \]

    **Nonnegativity:** \(x_{ij}\ge 0\) for all \((i,j)\in A\).

    This is a linear program: the objective and every constraint are linear in the decision variables. There are no binary decisions or nonlinear relationships.

    ## Implementation

    Let's now build a function that defines the LP problem and another that solves the problem. All of the constraints that were listed for the distribution network need to be encoded in the as Python functions.
    """)
    return


@app.cell
def _(arc_data, demand, farms, hub_capacity, hubs, pyo, retailers, supply):
    def build_model(carbon_budget=None, objective="cost"):
        """Construct a network LP

        objective: 'cost' or 'emissions'.
        carbon_budget: maximum kg CO2e/week, or None for no carbon cap.
        """
        if objective not in ("cost", "emissions"):
            raise ValueError("objective must be 'cost' or 'emissions'")

        model = pyo.ConcreteModel()
        model.F = pyo.Set(initialize=farms)
        model.H = pyo.Set(initialize=hubs)
        model.R = pyo.Set(initialize=retailers)
        model.A = pyo.Set(dimen=2, initialize=list(arc_data))

        # x[i, j] is tonnes shipped on route i -> j.
        model.x = pyo.Var(model.A, domain=pyo.NonNegativeReals)

        # Maximum what the farm can produce summed over hubs
        def _supply_rule(m, f):
            return sum(m.x[f, h] for h in m.H) <= supply[f]

        # In- and outflows of the hubs need to match
        def _flow_balance_rule(m, h):
            return sum(m.x[f, h] for f in m.F) == sum(m.x[h, r] for r in m.R)

        # Hubs can only store certain amount
        def _hub_capacity_rule(m, h):
            return sum(m.x[f, h] for f in m.F) <= hub_capacity[h]

        # Demand of retailers must be met exactly
        def _demand_rule(m, r):
            return sum(m.x[h, r] for h in m.H) == demand[r]

        model.supply_limit = pyo.Constraint(model.F, rule=_supply_rule)
        model.flow_balance = pyo.Constraint(model.H, rule=_flow_balance_rule)
        model.hub_limit = pyo.Constraint(model.H, rule=_hub_capacity_rule)
        model.meet_demand = pyo.Constraint(model.R, rule=_demand_rule)

        # Precompute costs and emissions
        model.total_cost = pyo.Expression(
            expr=sum(arc_data[i, j]["cost"] * model.x[i, j] for i, j in model.A)
        )
        model.total_emissions = pyo.Expression(
            expr=sum(arc_data[i, j]["emissions"] * model.x[i, j]
                     for i, j in model.A)
        )

        # Optional sustainability constraint
        if carbon_budget is not None:
            model.emissions_limit = pyo.Constraint(
                expr=model.total_emissions <= carbon_budget
            )

        target = model.total_cost if objective == "cost" else model.total_emissions
        model.objective = pyo.Objective(expr=target, sense=pyo.minimize)
        return model

    def solve_model(carbon_budget=None, objective="cost"):
        model = build_model(carbon_budget=carbon_budget, objective=objective)
        solver = pyo.SolverFactory("highs")
        if not solver.available():
            raise RuntimeError("HiGHS is unavailable. Install 'highspy' and 'pyomo'.")
        results = solver.solve(model)
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise RuntimeError(
                f"Optimization did not return an optimal solution: "
                f"{results.solver.termination_condition}"
            )
        return model

    return (solve_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Least cost and emission solutions

    First, solve the model without an emissions constraint. Compute the resulting transportation cost and the emissions associated with the optimal routes. Then solve a second LP that minimizes emissions alone: that gives the lowest achievable emissions for this network and demand.
    """)
    return


@app.cell
def _(pyo, solve_model):
    cost_model = solve_model(objective="cost")
    min_emissions_model = solve_model(objective="emissions")

    baseline_cost = pyo.value(cost_model.total_cost)
    baseline_emissions = pyo.value(cost_model.total_emissions)
    min_emissions = pyo.value(min_emissions_model.total_emissions)
    min_emissions_cost = pyo.value(min_emissions_model.total_cost)
    return (
        baseline_cost,
        baseline_emissions,
        cost_model,
        min_emissions,
        min_emissions_cost,
    )


@app.cell(hide_code=True)
def _(
    baseline_cost,
    baseline_emissions,
    min_emissions,
    min_emissions_cost,
    mo,
):
    mo.md(rf"""
    | Objective | Transportation cost | Emissions |
    |---|---:|---:|
    | Minimum cost | {baseline_cost:,.2f}€ | {baseline_emissions:,.1f} kg CO₂e |
    | Minimum emissions | {min_emissions_cost:,.2f}€ | {min_emissions:,.1f} kg CO₂e |

    The cost-minimizing and emissions-minimizing routings are generally different. The first row is the baseline against which we will evaluate emissions-reduction policies.
    """)
    return


@app.cell(hide_code=True)
def _(cost_model, mo, shipment_table):
    mo.vstack([
        mo.md("### Routes used in the minimum-cost solution"),
        shipment_table(cost_model),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimization with an emission budget

    The research paper studies a multi-objective trade-off. Here we use a
    simpler ε-constraint approach: continue minimizing costs but require
    the transportation network to stay within a carbon budget \(\overline E\).

    \[
        \begin{aligned}
            \min_x\quad & C(x)\\
            \text{subject to}\quad & E(x)=\sum_{(i,j)\in A}e_{ij}x_{ij}
                                 \leq \overline E,\\
                                  & \text{all original network constraints.}
        \end{aligned}
    \]

    Because the emissions coefficients are constants, this extra constraint
    is linear. Move the slider below to tighten the carbon budget.

    - **0%:** no reduction relative to the least-cost plan.
    - **100%:** require the minimum emissions achievable in this network.
    - Intermediate values: require the corresponding fraction of the maximum
      *feasible reduction* (not a percentage reduction from the baseline total).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    reduction_slider = mo.ui.slider(
        start=0,
        stop=100,
        step=5,
        value=0,
        show_value=True,
        label="Required share of the maximum feasible emissions reduction (%)",
    )

    reduction_slider
    return (reduction_slider,)


@app.cell
def _(baseline_emissions, min_emissions, pyo, reduction_slider, solve_model):
    carbon_budget = (
        baseline_emissions
        - (reduction_slider.value / 100.0)
        * (baseline_emissions - min_emissions)
    )

    sustainable_model = solve_model(
        carbon_budget=carbon_budget,
        objective="cost",
    )

    sustainable_cost = pyo.value(sustainable_model.total_cost)
    sustainable_emissions = pyo.value(sustainable_model.total_emissions)
    return (
        carbon_budget,
        sustainable_cost,
        sustainable_emissions,
        sustainable_model,
    )


@app.cell(hide_code=True)
def _(
    baseline_cost,
    baseline_emissions,
    carbon_budget,
    mo,
    reduction_slider,
    sustainable_cost,
    sustainable_emissions,
):
    _cost_change = 100 * (sustainable_cost / baseline_cost - 1)
    _emission_change = 100 * (1 - sustainable_emissions / baseline_emissions)
    mo.md(f"""
    ### Solution with a carbon budget

    **Requested share of feasible emissions reduction:** {reduction_slider.value}%  
    **Emissions budget:** {carbon_budget:,.1f} kg CO₂e per week

    | Metric | Least-cost baseline | With emissions budget |
    |---|---:|---:|
    | Transport cost (€ / week) | {baseline_cost:,.2f} | **{sustainable_cost:,.2f}** |
    | Emissions (kg CO₂e / week) | {baseline_emissions:,.1f} | **{sustainable_emissions:,.1f}** |

    **Change relative to baseline:** cost {_cost_change:+.1f}%;
    emissions {_emission_change:.1f}% lower.

    Notice that the constraint is a limit and emissions may be below the
    budget if a cheaper plan happens to emit less than the stated maximum.
    """)
    return


@app.cell(hide_code=True)
def _(mo, shipment_table, sustainable_model):
    mo.vstack([
        mo.md("### Routes used with the selected emissions budget"),
        shipment_table(sustainable_model),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pareto Frontier

    We can now find the best possible cost and emission levels for each level of emissions we are willing to tolerate. The combination of different points in cost-emission space that minimize the cost for the emission level form a Pareto frontier. Pareto frontier is the frontier on which we are not able to improve cost conditional on the level of emissions or vice versa. In other words, it is impossible to move from the blue line towards the origo. It is of course always possible to do worse. It is also important to remember that when we say *it is impossible to do better* we are already making a ton of modeling assumptions. Those are given in the programmed optimization problem, but in reality, the biggest changes to the underlying problem can be made by changing the structure of the problem entirely. This could also be reflected in our analysis. We could for example compute some alternative model where we change the structure of the problem for some part.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, pyo, solve_model):
    # Solve the two extreme optimization problems
    pareto_min_cost = solve_model(objective="cost")
    pareto_min_emissions = solve_model(objective="emissions")

    # Obtain the emissions levels at the two extreme solutions
    E_cost = pyo.value(pareto_min_cost.total_emissions)
    E_min = pyo.value(pareto_min_emissions.total_emissions)

    # Solve the cost minimization problem under different
    # emissions limits (epsilon-constraint method)
    pareto_points = []

    for cb in np.linspace(E_min, E_cost, 51):
        solution = solve_model(
            carbon_budget=float(cb),
            objective="cost",
        )

        pareto_points.append({
            "emissions": pyo.value(solution.total_emissions),
            "cost": pyo.value(solution.total_cost),
        })

    # Extract the coordinates of the frontier
    pareto_emissions = [
        point["emissions"] for point in pareto_points
    ]
    pareto_costs = [
        point["cost"] for point in pareto_points
    ]

    # Plot
    pareto_fig, pareto_ax = plt.subplots(figsize=(9, 5.5))

    pareto_ax.plot(
        pareto_emissions,
        pareto_costs,
        color="#2563EB",
        linewidth=2.5,
        label="Pareto frontier",
    )

    # Highlight the extreme solutions
    pareto_ax.scatter(
        [E_cost, E_min],
        [
            pyo.value(pareto_min_cost.total_cost),
            pyo.value(pareto_min_emissions.total_cost),
        ],
        color=["#16A34A", "#DC2626"],
        s=85,
        zorder=5,
    )

    pareto_ax.annotate(
        "Minimum cost",
        (E_cost, pyo.value(pareto_min_cost.total_cost)),
        xytext=(-50, 19),
        textcoords="offset points",
    )

    pareto_ax.annotate(
        "Minimum emissions",
        (E_min, pyo.value(pareto_min_emissions.total_cost)),
        xytext=(30, -10),
        textcoords="offset points",
    )

    pareto_ax.set_xlabel("Emissions (kg CO₂e/week)")
    pareto_ax.set_ylabel("Transportation cost (€/week)")
    pareto_ax.set_title("Cost–Emissions Pareto Frontier")

    pareto_ax.grid(alpha=0.25)
    pareto_ax.legend()
    pareto_fig.tight_layout()

    pareto_fig
    return pareto_costs, pareto_emissions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Pareto Frontier and Marginal Abatement Costs

    We can construct the Pareto frontier between transportation costs and emissions using the ε-constraint method. The idea is to minimize transportation costs while imposing an upper bound on the total emissions of the network. Mathematically, we solve

    \[
    \begin{aligned}
    V(\varepsilon)
    =\min_x \quad & C(x)\\
    \text{subject to}\quad
    & E(x)\leq \varepsilon,\\
    & x\in\mathcal F,
    \end{aligned}
    \]

    where \(C(x)\) denotes the total transportation cost, \(E(x)\) denotes the total emissions, and \(\mathcal F\) is the feasible set of the transportation network. The parameter \(\varepsilon\) represents the maximum amount of emissions allowed. The \(V(\epsilon)\) is called a value function. Note that the value function is not same as the objective function. Indeed it is not even linear. The value function is shown as the pareto frontier in the above plot. While the objective function gives the value we achieve by using any particular variable values, the value function gives the *optimal* value of the objective function for any given constraint value.

    By varying the emissions budget over

    \[
    \varepsilon\in[1420,2745],
    \]

    we obtain the optimal cost \(V(\varepsilon)\) for different levels of allowed emissions. Plotting these optimal costs against the corresponding emissions levels gives us the Pareto frontier.

    An interesting property of this optimization problem is that the slope of the Pareto frontier has an economic interpretation. In particular, when the value function is differentiable and the appropriate regularity conditions hold, the optimal Lagrange multiplier \(\lambda^*\) associated with the emissions constraint satisfies

    \[
    \frac{\text{d}V(\varepsilon)}{\text{d}\varepsilon}=-\lambda^*.
    \]

    Thus, the magnitude of the slope measures the marginal cost of reducing emissions, expressed in euros per kilogram of CO₂e avoided. This is also called the marginal abatement cost (MAC).

    In the next chapter, we will study Lagrange multipliers and examine why this relationship holds and how it can be used to interpret optimization results.

    From the Pareto frontier above, we observe that allowing higher emissions reduces the minimum transportation cost. However, the additional cost savings generally become smaller as the emissions budget increases. Equivalently, reducing emissions becomes increasingly expensive as we move toward the minimum-emissions solution.

    ### Elasticity

    We can examine the relationship between transportation costs and emissions further by calculating their elasticity.

    Elasticity is a widely used concept in economics that measures how sensitive one quantity is to changes in another. For example, the price elasticity of demand is defined approximately as

    \[
    E_d=\frac{\Delta Q/Q}{\Delta P/P},
    \]

    where \(Q\) denotes the quantity demanded and \(P\) denotes the price.

    Elasticity measures the percentage change in quantity associated with a 1% change in price. For example, a price elasticity of demand of \(E_d=-1.5\) means that a 1% increase in price is associated with approximately a 1.5% decrease in quantity demanded.

    We can apply the same concept to our transportation problem. Let us define the cost elasticity with respect to emissions as

    \[
    E_{C,E}
    =
    \frac{\Delta C/C}{\Delta E/E}.
    \]

    Here, \(C\) denotes transportation costs and \(E\) denotes emissions. The elasticity measures the percentage change in the optimal transportation cost associated with a 1% change in emissions.

    Because reducing emissions increases transportation costs along the Pareto frontier, the elasticity is negative. For example, if

    \[
    E_{C,E}=-0.8,
    \]

    then a 1% reduction in emissions is associated with approximately a 0.8% increase in transportation costs.

    Consequently, the quantity \(-E_{C,E}\) provides a positive measure of the relative cost of emissions reduction at different emissions levels.

    For a differentiable cost function, we can express this elasticity as

    \[
    E_{C,E}
    =
    \frac{\text{d}C}{\text{d}E}\frac{E}{C}.
    \]

    In practice, we approximate the derivative using consecutive points on the Pareto frontier.

    The figure below illustrates how the cost elasticity varies along the frontier. Its magnitude tells us how expensive further emissions reductions are in relative terms. A higher value of \(-E_{C,E}\) indicates that achieving an additional percentage reduction in emissions requires a larger percentage increase in transportation costs.

    Alternatively, the reciprocal elasticity \(1/E_{C,E}\) measures the percentage change in emissions associated with a 1% change in transportation costs.

    ### Marginal Abatement Cost

    Another useful quantity is the marginal abatement cost (MAC), which measures the additional transportation cost associated with reducing emissions by one unit.

    It is defined as

    \[
    \mathrm{MAC}(E)
    =
    -\frac{\text{d}C}{\text{d}E}.
    \]

    Using the definition of elasticity, we can also write

    \[
    \mathrm{MAC}(E)
    =
    -\frac{C}{E}E_{C,E}.
    \]

    In our numerical calculations, we approximate the marginal abatement cost using

    \[
    \mathrm{MAC}(E)
    \approx
    -\frac{\Delta C}{\Delta E}.
    \]

    The distinction between elasticity and marginal abatement cost is their interpretation and units:

    - **Elasticity** is dimensionless and measures relative changes in costs and emissions.
    - **Marginal abatement cost** is measured in €/kg CO₂e and expresses the monetary cost of reducing emissions.

    Elasticity is particularly useful when comparing relative trade-offs across different models or situations, whereas marginal abatement cost allows us to express the environmental trade-off directly in monetary terms.

    From the figure, we can see that at an emissions level of approximately 2,000 kg CO₂e/week, the marginal abatement cost is around €0.90/kg CO₂e. In other words, reducing weekly emissions by an additional kilogram at this point increases the optimal weekly transportation cost by approximately €0.90.

    As we move toward lower emissions levels, the marginal abatement cost generally increases. This illustrates an important economic principle: achieving increasingly ambitious emissions reductions tends to become progressively more expensive. This is the law of diminishing returns in action, and something we already saw in the pareto frontier plot.
    """)
    return


@app.cell(hide_code=True)
def _(np, pareto_costs, pareto_emissions, plt):
    E = np.asarray(pareto_emissions)
    C = np.asarray(pareto_costs)

    # Midpoints between consecutive Pareto points
    E_mid = (E[:-1] + E[1:]) / 2
    C_mid = (C[:-1] + C[1:]) / 2

    # Elasticity of cost with respect to emissions
    elasticity = (np.diff(C) / np.diff(E)) * (E_mid / C_mid)

    # Marginal abatement cost
    mac = -np.diff(C) / np.diff(E)

    elasticity_fig, ax_el = plt.subplots(figsize=(9, 5))

    # Elasticity
    ax_el.plot(
        E_mid,
        -elasticity,
        marker="o",
        linewidth=1.8,
        color="#2563EB",
        label=r"Elasticity ($-E_{C,E}$)"
    )

    # Marginal abatement cost
    ax_el.step(
        E_mid,
        mac,
        where="mid",
        linestyle="--",
        linewidth=1.8,
        color="#DC2626",
        label="Marginal abatement cost (€/kg CO₂e)"
    )

    ax_el.set_xlabel("Emissions (kg CO₂e/week)")
    ax_el.set_ylabel("Elasticity / Marginal abatement cost")
    ax_el.set_title("Elasticity and Marginal Abatement Cost")

    ax_el.axhline(0, color="black", linewidth=0.8)
    ax_el.grid(alpha=0.25)
    ax_el.legend(loc="upper right")

    elasticity_fig.tight_layout()
    elasticity_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    Bortolini, M., Faccio, M., Ferrari, E., Gamberi, M., & Pilati, F.
    (2016). Fresh food sustainable distribution: Cost, delivery time
    and carbon footprint three-objective optimization.
    *Journal of Food Engineering, 174*, 56–67.
    https://doi.org/10.1016/j.jfoodeng.2015.11.014
    """)
    return


if __name__ == "__main__":
    app.run()
