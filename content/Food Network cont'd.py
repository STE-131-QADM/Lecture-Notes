import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Food Network: Route Disruptions and Expected Costs
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
def _(mo):
    mo.md(r"""
    In the previous food-network example, we assumed that every transportation route was available at a known cost and emissions intensity. We now relax this assumption: a planned route may be blocked, requiring a more expensive or more emissions-intensive detour.

    Our goal is to investigate how uncertain transportation conditions affect optimal routing and the value of scarce resources. We will compare the original (certain) model with an *expected-value model*, examine their cost–emissions trade-offs, and use dual variables to interpret resource constraints.

    We will see that incorporating uncertainty does not necessarily require nonlinear optimization. Under the assumptions made here, the problem remains a linear program.

    ## Network and route data

    We retain the same network as in the previous exercise: three farms (F1–F3), two distribution hubs (H1–H2), and three retailers (R1–R3). Farms have limited weekly supply, hubs have throughput capacities, and retailers have fixed weekly demand. The network structure and these constraints remain unchanged.

    What changes is our description of each transportation arc. Besides its normal cost and emissions intensity, an arc now has a possible alternative-route outcome and an estimated probability that the alternative will be needed. All numerical values are illustrative.
    """)
    return


@app.cell(hide_code=True)
def _():
    hubs = ("H1", "H2")
    retailers = ("R1", "R2", "R3")
    farms = ("F1", "F2", "F3")

    supply = {"F1": 70, "F2": 60, "F3": 80}
    hub_capacity = {"H1": 100, "H2": 100}
    demand = {"R1": 50, "R2": 65, "R3": 55}

    arc_data = {
        ("F1", "H1"): {
            "cost": 4, "emissions": 9,
            "cost_hat": 5, "emissions_hat": 9, "p": 0.25
        },
        ("F1", "H2"): {
            "cost": 9, "emissions": 4,
            "cost_hat": 10, "emissions_hat": 5, "p": 0.13
        },
        ("F2", "H1"): {
            "cost": 7, "emissions": 6,
            "cost_hat": 9, "emissions_hat": 7, "p": 0.37
        },
        ("F2", "H2"): {
            "cost": 3, "emissions": 8,
            "cost_hat": 16, "emissions_hat": 12, "p": 0.40
            # Changed cost_hat: 6 -> 16
        },
        ("F3", "H1"): {
            "cost": 6, "emissions": 4,
            "cost_hat": 6, "emissions_hat": 7, "p": 0.28
        },
        ("F3", "H2"): {
            "cost": 5, "emissions": 7,
            "cost_hat": 6, "emissions_hat": 8, "p": 0.46
        },
        ("H1", "R1"): {
            "cost": 4, "emissions": 8,
            "cost_hat": 7, "emissions_hat": 12, "p": 0.46
        },
        ("H1", "R2"): {
            "cost": 7, "emissions": 5,
            "cost_hat": 7, "emissions_hat": 8, "p": 0.27
        },
        ("H1", "R3"): {
            "cost": 9, "emissions": 3,
            "cost_hat": 9, "emissions_hat": 14, "p": 0.47
            # Changed emissions_hat: 10 -> 14
        },
        ("H2", "R1"): {
            "cost": 8, "emissions": 3,
            "cost_hat": 11, "emissions_hat": 5, "p": 0.09
        },
        ("H2", "R2"): {
            "cost": 4, "emissions": 8,
            "cost_hat": 5, "emissions_hat": 10, "p": 0.03
        },
        ("H2", "R3"): {
            "cost": 3, "emissions": 9,
            "cost_hat": 18, "emissions_hat": 11, "p": 0.24
            # Changed cost_hat: 4 -> 18
        },
    }

    return arc_data, demand, farms, hub_capacity, hubs, retailers, supply


@app.cell(hide_code=True)
def _(arc_data, demand, farms, hub_capacity, hubs, pyo, retailers, supply):
    def build_certain_model(carbon_budget=None, objective="cost"):
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
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

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

    def solve_certain_model(carbon_budget=None, objective="cost"):
        model = build_certain_model(carbon_budget=carbon_budget, objective=objective)
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

    return (solve_certain_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's consider the food network example. The data describing the model can be seen below.
    """)
    return


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
          "Cost alt. (€/t)": p["cost_hat"],
          "Emissions (kg CO₂e/t)": p["emissions"], 
          "Emissions alt. (kg CO₂e/t)": p["emissions_hat"],
         "Probability of failure": p["p"]}
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
    ## Modeling uncertain routes

    Suppose that a road on the planned route from node \(i\) to node \(j\) may be blocked. With probability \(p_{ij}\), a shipment must take an alternative route and with probability \(1-p_{ij}\), it uses the original route.

    Let \(c_{ij}\) and \(e_{ij}\) denote the normal route's cost (€/tonne) and emissions intensity (kg CO₂e/tonne). The corresponding values for the alternative route are \(\hat c_{ij}\) and \(\hat e_{ij}\). We assume these coefficients and probabilities are known estimates when the transportation plan is chosen.

    The expected unit transportation cost and expected unit emissions are therefore

    \[
    \begin{aligned}
    \bar c_{ij}&=(1-p_{ij})c_{ij}+p_{ij}\hat c_{ij},\\
    \bar e_{ij}&=(1-p_{ij})e_{ij}+p_{ij}\hat e_{ij}.
    \end{aligned}
    \]

    For example, the normal route F2 → H2 costs €3 per tonne, whereas its detour costs €16 per tonne. Because the estimated probability of needing the detour is 0.40, its expected unit cost is

    \[
    \bar c_{\mathrm{F2,H2}}=0.60(3)+0.40(16)=8.20\text{ €/tonne}.
    \]

    The corresponding expected emissions intensity is \(0.60(8)+0.40(12)=9.6\) kg CO₂e/tonne. This example illustrates why a route that looks attractive under normal conditions may become less attractive once disruptions are taken into account.

    ### Why is the new objective still linear?

    The values \(p_{ij}\), \(c_{ij}\), and \(\hat c_{ij}\) are parameters, not decision variables. Consequently, \(\bar c_{ij}\) is a fixed coefficient. For any shipment plans \(x,y\) and scalars \(\alpha,\beta\), the expected-cost function satisfies

    \[
    \begin{aligned}
    \bar C(\alpha x+\beta y)
    &=\sum_{(i,j)\in A}\bar c_{ij}(\alpha x_{ij}+\beta y_{ij})\\
    &=\alpha\sum_{(i,j)\in A}\bar c_{ij}x_{ij}
    +\beta\sum_{(i,j)\in A}\bar c_{ij}y_{ij}\\
    &=\alpha\bar C(x)+\beta\bar C(y).
    \end{aligned}
    \]

    Thus, taking expectations changes the objective coefficients but does not change its linearity. The expected-emissions expression is linear for the same reason.

    ## Reformulating the optimization problem

    As before, let \(F\) denote the farms, \(H\) the distribution hubs, \(R\) the retailers, and \(A\) the set of available directed arcs. Our decision variable is

    \[
    x_{ij}=\text{tonnes planned to be shipped along arc }(i,j)\in A.
    \]

    The other parameters are farm supply \(s_f\), hub throughput capacity \(q_h\), and retailer demand \(d_r\). The expected-cost minimization problem is

    \[
    \min_{x\in\mathcal F}\;\bar C(x)
    =\min_{x\in\mathcal F}\sum_{(i,j)\in A}\bar c_{ij}x_{ij},
    \]

    where \(\mathcal F\) is defined by the original network constraints:

    **Farm supply:**

    \[
    \sum_{h\in H}x_{fh}\le s_f\qquad\forall f\in F.
    \]

    **Flow conservation at distribution hubs:**

    \[
    \sum_{f\in F}x_{fh}=\sum_{r\in R}x_{hr}
    \qquad\forall h\in H.
    \]

    **Hub throughput capacity:**

    \[
    \sum_{f\in F}x_{fh}\le q_h\qquad\forall h\in H.
    \]

    **Retailer demand satisfaction:**

    \[
    \sum_{h\in H}x_{hr}=d_r\qquad\forall r\in R.
    \]

    All shipments must also be nonnegative: \(x_{ij}\ge0\) for every \((i,j)\in A\).

    We can alternatively minimize expected emissions, \(\bar E(x)=\sum_{(i,j)\in A}\bar e_{ij}x_{ij}\), or minimize expected costs subject to an expected-emissions budget:

    \[
    \bar E(x)\le\varepsilon.
    \]

    ### What does the model assume?

    The shipments \(x_{ij}\) are planned before we learn whether a particular route is blocked. If a disruption occurs, we assume that the corresponding alternative is available and can carry the planned shipment. Its consequences are represented by the altered unit cost and emissions; the detour is not modeled as a new network arc with its own capacity constraints.

    This is an expected-value formulation rather than a two-stage stochastic program: there are no explicit decisions that can be revised after observing disruptions. Moreover, \(\mathbb E[E(x)]\le\varepsilon\) is a restriction on average emissions, not a guarantee that realized emissions will always remain below \(\varepsilon\). We do not need to assume that disruptions on different arcs are independent to calculate these expectations, because expectation is linear.

    ## Implementation

    The implementation mirrors the original Pyomo model. Its supply, capacity, flow-conservation, and demand constraints are unchanged. The two expressions `model.total_cost` and `model.total_emissions` now use the expected unit coefficients \(\bar c_{ij}\) and \(\bar e_{ij}\). An optional emissions constraint uses expected emissions, not realized emissions.
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
        # Instead of writing the dual as a second model
        # we can use the functionality provided by Pyomo
        # to create the dual model
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

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
        # Now we compute the expected cost
        model.total_cost = pyo.Expression(
            expr=sum(
                (arc_data[i, j]["p"]*arc_data[i, j]["cost_hat"] 
                 + (1-arc_data[i, j]["p"])*arc_data[i, j]["cost"]) 
                * model.x[i, j] 
                for i, j in model.A)
        )
        # Similarly we compute the expected emissions
        model.total_emissions = pyo.Expression(
            expr=sum((arc_data[i, j]["p"]*arc_data[i, j]["emissions_hat"]
                     + (1-arc_data[i, j]["p"])*arc_data[i, j]["emissions"]) 
                     * model.x[i, j] 
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

    return build_model, solve_model


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
        min_emissions_model,
    )


@app.cell(hide_code=True)
def _(pyo, solve_certain_model):
    cost_model_certain = solve_certain_model(objective="cost")
    min_emissions_model_certain = solve_certain_model(objective="emissions")

    baseline_cost_certain = pyo.value(cost_model_certain.total_cost)
    baseline_emissions_certain = pyo.value(cost_model_certain.total_emissions)
    min_emissions_certain = pyo.value(min_emissions_model_certain.total_emissions)
    min_emissions_cost_certain = pyo.value(min_emissions_model_certain.total_cost)
    return (
        baseline_cost_certain,
        baseline_emissions_certain,
        cost_model_certain,
        min_emissions_certain,
        min_emissions_cost_certain,
        min_emissions_model_certain,
    )


@app.cell(hide_code=True)
def _(
    baseline_cost,
    baseline_cost_certain,
    baseline_emissions,
    baseline_emissions_certain,
    min_emissions,
    min_emissions_certain,
    min_emissions_cost,
    min_emissions_cost_certain,
    mo,
):
    mo.md(rf"""
    ## Comparing optimal outcomes

    We first solve each model twice: once to minimize transportation costs and once to minimize emissions. These objectives generally require different shipment plans. The table reports the cost and emissions achieved by the optimizer under each objective, not two independently minimized quantities within the same column.

    | Quantity | Minimum-cost solution | Minimum-emissions solution |
    |---|---:|---:|
    | Expected transportation cost (uncertain model) | {baseline_cost:,.2f} € | {min_emissions_cost:,.2f} € |
    | Expected emissions (uncertain model) | {baseline_emissions:,.1f} kg CO₂e | {min_emissions:,.1f} kg CO₂e |
    | Transportation cost (original model) | {baseline_cost_certain:,.2f} € | {min_emissions_cost_certain:,.2f} € |
    | Emissions (original model) | {baseline_emissions_certain:,.1f} kg CO₂e | {min_emissions_certain:,.1f} kg CO₂e |

    The upper two rows describe expected outcomes under route disruptions. The lower two rows describe outcomes under the original, disruption-free assumptions. For example, the minimum-emissions column does not mean that transportation costs were also minimized but instead only that the reported costs are those of the emissions-minimizing solution.

    In this dataset, each detour is at least as expensive and at least as emissions-intensive as its corresponding normal route. For any fixed feasible shipment plan, its expected cost and expected emissions are therefore no smaller than its cost and emissions under normal conditions. This helps explain why the trade-off deteriorates when disruptions are introduced. It is a property of our illustrative coefficients, however, not a general theorem about uncertainty: an alternative route could sometimes be cheaper or cleaner.

    ### Comparing the Pareto frontiers

    We can go beyond the two extreme solutions using the ε-constraint method. For an emissions budget \(\varepsilon\), the uncertain model solves

    \[
    \begin{{aligned}}
    V_u(\varepsilon)=\min_x\quad&\bar C(x)\\
    \text{{subject to}}\quad&\bar E(x)\le\varepsilon,\\
    &x\in\mathcal F.
    \end{{aligned}}
    \]

    The original model has the same structure, but uses \(C(x)\) and \(E(x)\) instead of their expected values. Varying \(\varepsilon\) between each model's minimum achievable emissions and the emissions of its minimum-cost solution traces the respective Pareto frontiers.

    A point on either frontier represents a cost-minimizing plan for the specified emissions limit. Moving toward lower emissions typically requires higher costs. Keep in mind that the two curves use different accounting assumptions: one measures normal-route costs and emissions, while the other measures expected values under disruptions. We are comparing two descriptions of the transportation environment, not two plans evaluated under the same realized disruptions.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, pyo, solve_certain_model, solve_model):
    # Solve the two extreme optimization problems
    pareto_min_cost = solve_model(objective="cost")
    pareto_min_emissions = solve_model(objective="emissions")

    pareto_min_cost_certain = solve_certain_model(objective="cost")
    pareto_min_emissions_certain = solve_certain_model(objective="emissions")

    # Obtain the emissions levels at the two extreme solutions
    E_cost = pyo.value(pareto_min_cost.total_emissions)
    E_min = pyo.value(pareto_min_emissions.total_emissions)

    E_cost_certain = pyo.value(pareto_min_cost_certain.total_emissions)
    E_min_certain = pyo.value(pareto_min_emissions_certain.total_emissions)

    # Solve the cost minimization problem under different
    # emissions limits (epsilon-constraint method)
    pareto_points = []
    pareto_points_certain = []

    for cb in np.linspace(E_min, E_cost, 51):
        solution = solve_model(
            carbon_budget=float(cb),
            objective="cost",
        )
        pareto_points.append({
            "emissions": pyo.value(solution.total_emissions),
            "cost": pyo.value(solution.total_cost),
        })


    for cb in np.linspace(E_min_certain, E_cost_certain, 51):
        solution_certain = solve_certain_model(
            carbon_budget=float(cb),
            objective="cost",
        )

        pareto_points_certain.append({
            "emissions": pyo.value(solution_certain.total_emissions),
            "cost": pyo.value(solution_certain.total_cost),
        })

    # Extract the coordinates of the frontier
    pareto_emissions = [
        point["emissions"] for point in pareto_points
    ]
    pareto_emissions_certain = [
        point["emissions"] for point in pareto_points_certain
    ]
    pareto_costs = [
        point["cost"] for point in pareto_points
    ]
    pareto_costs_certain = [
        point["cost"] for point in pareto_points_certain
    ]

    # Plot
    pareto_fig, pareto_ax = plt.subplots(figsize=(9, 5.5))

    pareto_ax.plot(
        pareto_emissions,
        pareto_costs,
        color="navy",
        linewidth=2.5,
        label="Uncertain pareto frontier",
    )
    pareto_ax.plot(
        pareto_emissions_certain,
        pareto_costs_certain,
        color="lightblue",
        linewidth=2.5,
        label="Certain pareto frontier",
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

    pareto_ax.scatter(
        [E_cost_certain, E_min_certain],
        [
            pyo.value(pareto_min_cost_certain.total_cost),
            pyo.value(pareto_min_emissions_certain.total_cost),
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How does the optimal routing change?

    The four network diagrams show the shipment quantities chosen by the optimizer. The top row minimizes cost, and the bottom row minimizes emissions. The original model is shown on the left and the expected-value model on the right.

    Thicker edges carry more tonnes per week, with the same thickness scale in all four panels. Compare, for example, how shipments are allocated between F2 and F3 and how the distribution hubs serve R2 and R3. A change in optimal routing can occur even though the feasible network has not changed, because the model now assigns different effective costs and emissions to its arcs.
    """)
    return


@app.cell(hide_code=True)
def _(
    arc_data,
    cost_model,
    cost_model_certain,
    farms,
    hubs,
    min_emissions_model,
    min_emissions_model_certain,
    nx,
    plt,
    pyo,
    retailers,
):
    # Network structure
    comparison_network = nx.DiGraph()
    comparison_network.add_edges_from(arc_data.keys())

    comparison_positions = {
        "F1": (0, 2),
        "F2": (0, 1),
        "F3": (0, 0),
        "H1": (2, 1.5),
        "H2": (2, 0.5),
        "R1": (4, 2),
        "R2": (4, 1),
        "R3": (4, 0),
    }

    comparison_models = [
        (cost_model_certain, "Minimum cost — certain"),
        (cost_model, "Minimum cost — uncertain"),
        (min_emissions_model_certain, "Minimum emissions — certain"),
        (min_emissions_model, "Minimum emissions — uncertain"),
    ]

    # Common flow scale across all four solutions
    comparison_max_flow = max(
        pyo.value(m.x[i, j])
        for m, _ in comparison_models
        for i, j in arc_data
    )

    comparison_fig, comparison_axes = plt.subplots(
        2, 2, figsize=(13, 10)
    )

    for ax, (model, title) in zip(
        comparison_axes.flat, comparison_models
    ):
        # Draw all available routes in grey
        nx.draw_networkx_edges(
            comparison_network,
            comparison_positions,
            ax=ax,
            edge_color="#CBD5E1",
            width=1,
            alpha=0.5,
            arrows=True,
            node_size=1600,
        )

        # Routes carrying positive flow
        active_arcs = [
            (i, j)
            for i, j in arc_data
            if pyo.value(model.x[i, j]) > 1e-6
        ]

        widths = [
            1 + 6 * pyo.value(model.x[i, j]) / comparison_max_flow
            for i, j in active_arcs
        ]

        color = "#2563EB" if "certain" in title and "uncertain" not in title else "#DC2626"

        nx.draw_networkx_edges(
            comparison_network,
            comparison_positions,
            ax=ax,
            edgelist=active_arcs,
            width=widths,
            edge_color=color,
            arrows=True,
            arrowsize=16,
            node_size=1600,
        )

        # Draw nodes by category
        for nodes, node_color in [
            (farms, "#86C5A3"),
            (hubs, "#F9CB80"),
            (retailers, "#93C5FD"),
        ]:
            nx.draw_networkx_nodes(
                comparison_network,
                comparison_positions,
                nodelist=list(nodes),
                node_color=node_color,
                node_size=1600,
                edgecolors="#334155",
                ax=ax,
            )

        nx.draw_networkx_labels(
            comparison_network,
            comparison_positions,
            font_size=11,
            font_weight="bold",
            ax=ax,
        )

        ax.set_title(title)
        ax.axis("off")

    comparison_fig.suptitle(
        "Effect of Uncertainty on Optimal Transportation Flows",
        fontsize=16,
    )

    comparison_fig.tight_layout()
    comparison_fig

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How valuable is it to account for uncertainty?

    We have compared optimal solutions from two models. A different, practically important question is whether a logistics planner should change an existing transportation plan after learning that disruptions are possible.

    Let \(x_c^*\) be a minimum-cost plan computed using the original coefficients and \(x_u^*\) a minimum-expected-cost plan:

    \[
    \begin{aligned}
    x_c^*&\in\operatorname*{arg\,min}_{x\in\mathcal F}\;C(x),\\
    x_u^*&\in\operatorname*{arg\,min}_{x\in\mathcal F}\;\bar C(x).
    \end{aligned}
    \]

    The bar chart below compares three evaluations:

    1. **Original plan, normal conditions:** \(C(x_c^*)\). This is the cost of the original routing plan under the assumptions used to design it.
    2. **Original plan, expected conditions:** \(\bar C(x_c^*)\). We keep the original shipment decisions fixed, but evaluate them using the new expected-cost coefficients.
    3. **Reoptimized plan, expected conditions:** \(\bar C(x_u^*)\). We allow the shipment decisions to change and minimize expected cost.

    The gap between the first and second bars is the expected additional cost of disruptions if the original plan is  retained. The gap between the second and third bars isolates the benefit of adjusting the routing decisions to account for those disruptions:

    \[
    \underbrace{\bar C(x_c^*)-\bar C(x_u^*)}_{\text{Expected savings from reoptimization}}\ge0.
    \]

    The inequality follows because the two plans satisfy the same network constraints and \(x_u^*\) minimizes expected costs over that feasible set. If the original routing also minimizes expected costs, the difference is zero. Introducing uncertainty does not necessarily change an optimal decision.


    !!! note "Optimization over nested feasible sets"

        Consider two nonempty sets \(A\subseteq B\) and a function \(f:B\rightarrow\mathbb{R}\). Assume that the maximum and minimum of \(f\) exist on both sets. Then

        \[
        \max_{x\in A}f(x)\leq\max_{x\in B}f(x)
        \]

        and

        \[
        \min_{x\in A}f(x)\geq\min_{x\in B}f(x).
        \]

        In other words, enlarging the feasible set cannot decrease the maximum or increase the minimum of an optimization problem.

        To see why, consider the maximization problem. Let \(x_A^*\in A\) be a maximizer of \(f\) over \(A\), so that

        \[
        f(x_A^*)=\max_{x\in A}f(x).
        \]

        Since \(A\subseteq B\), we also have \(x_A^*\in B\). By the definition of the maximum over \(B\),

        \[
        f(x_A^*)\leq\max_{x\in B}f(x).
        \]

        Combining these two expressions gives

        \[
        \max_{x\in A}f(x)\leq\max_{x\in B}f(x).
        \]

        The argument for minimization is analogous. If \(x_A^*\) minimizes \(f\) over \(A\), then \(x_A^*\in B\), and hence

        \[
        \min_{x\in B}f(x)
        \leq f(x_A^*)
        =\min_{x\in A}f(x).
        \]

        Notice that the inequalities need not be strict. Enlarging the feasible set does not necessarily improve the optimal objective value.

    This quantity is also interesting from a point of view that it would be possible to measure it in some situations. This would allow you to put a quite accurate value on the improved model, and the solutions it can generate.
    """)
    return


@app.cell(hide_code=True)
def _(
    arc_data,
    baseline_cost,
    baseline_cost_certain,
    cost_model_certain,
    plt,
    pyo,
):
    # Expected cost and emissions of the original optimal plan
    original_plan_expected_cost = sum(
        (
            (1 - data["p"]) * data["cost"]
            + data["p"] * data["cost_hat"]
        ) * pyo.value(cost_model_certain.x[i, j])
        for (i, j), data in arc_data.items()
    )

    original_plan_expected_emissions = sum(
        (
            (1 - data["p"]) * data["emissions"]
            + data["p"] * data["emissions_hat"]
        ) * pyo.value(cost_model_certain.x[i, j])
        for (i, j), data in arc_data.items()
    )

    # Benefit of optimizing with expected costs
    uncertainty_savings = (
        original_plan_expected_cost - baseline_cost
    )

    # Compare three cases
    cost_comparison_fig, cost_comparison_ax = plt.subplots(
        figsize=(9, 5)
    )

    cost_cases = [
        "Original plan\noriginal conditions",
        "Original plan\nexpected conditions",
        "Reoptimized plan\nexpected conditions",
    ]

    cost_values = [
        baseline_cost_certain,
        original_plan_expected_cost,
        baseline_cost,
    ]

    bars = cost_comparison_ax.bar(
        cost_cases,
        cost_values,
        color=["#93C5FD", "#F59E0B", "#2563EB"],
        width=0.6,
    )

    cost_comparison_ax.bar_label(
        bars,
        fmt="%.1f €",
        padding=4,
    )

    cost_comparison_ax.set_ylabel("Transportation cost (€/week)")
    cost_comparison_ax.set_title(
        "Cost of Uncertainty and Benefit of Reoptimization"
    )

    cost_comparison_ax.set_ylim(
        0, max(cost_values) * 1.15
    )
    cost_comparison_ax.grid(axis="y", alpha=0.25)

    cost_comparison_fig.tight_layout()
    cost_comparison_fig

    return original_plan_expected_cost, uncertainty_savings


@app.cell(hide_code=True)
def _(
    baseline_cost,
    baseline_cost_certain,
    mo,
    original_plan_expected_cost,
    uncertainty_savings,
):
    mo.md(rf"""
    With the current illustrative data, the original cost-minimizing plan costs €{baseline_cost_certain:,.2f}/week under normal conditions. When evaluated using expected disruption costs, the same routing plan costs € {original_plan_expected_cost:,.2f}/week. Reoptimizing the shipments reduces that expected cost to € {baseline_cost:,.2f}/week. A saving of €{uncertainty_savings:,.2f}/week ({100 * uncertainty_savings / original_plan_expected_cost:.1f}% relative to keeping the original plan).

    This comparison separates the adverse change in operating conditions from the beneficial change in the decision. It does not describe the realized cost of any one week. Actual disruptions may differ from their expected frequencies.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shadow prices: which resources are worth expanding?

    Until now, we have focused on what the optimal solution is. Dual variables help answer another management question: How much could we reduce optimal transportation costs by relaxing a resource constraint?

    Both models contain upper bounds on farm supply and distribution-hub throughput. Consider a generic capacity constraint \(g(x)\le b\). If \(V(b)\) denotes the minimum cost when this right-hand side is \(b\), then, at a point where the value function is differentiable, the corresponding solver dual multiplier \(\mu\) satisfies

    \[
        \frac{dV(b)}{db}=\mu.
    \]

    For a capacity upper bound in our cost-minimization problem, this derivative is nonpositive: allowing more capacity cannot increase the minimum achievable cost. We therefore plot \(-\mu\), the marginal cost saving from one additional tonne of weekly capacity. For example, a value of €2 per additional tonne means that a small increase in that capacity reduces optimal weekly transportation costs by about €2 per additional tonne of throughput.

    Pyomo's `model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)` tells the solver interface to import the dual multipliers of the solved constraints. It does not build a new dual optimization model. In the bar chart below, the multipliers come from the two minimum-cost problems without an emissions constraint.

    The original model assigns positive marginal values to extra supply at F2 and extra throughput at H2. Under the expected-value coefficients, the valuable expansions shift toward supply at F1 and especially F3. This is economically plausible since disruption risks change which routes are attractive and, consequently, which resources limit a cheap routing plan.

    Two cautions are important. First, a capacity constraint can be binding but have a zero shadow price. Using all available capacity does not automatically make expansion valuable. Second, a high shadow price does not by itself justify investing in capacity. The marginal savings must be compared with the cost of expansion, and a local shadow price need not remain constant for a large change in capacity.
    """)
    return


@app.cell(hide_code=True)
def _(cost_model, cost_model_certain, farms, hubs, np, plt):
    # Helper for retrieving the marginal value of relaxing
    # a <= constraint in a minimization problem
    def capacity_value(model, constraint):
        dual = model.dual.get(constraint)

        if dual is None:
            raise ValueError(
                f"No dual value available for {constraint.name}"
            )

        return -float(dual)


    resources = (
        [f"Farm {f}" for f in farms]
        + [f"Hub {h}" for h in hubs]
    )

    certain_prices = np.array(
        [
            capacity_value(
                cost_model_certain,
                cost_model_certain.supply_limit[f]
            )
            for f in farms
        ]
        + [
            capacity_value(
                cost_model_certain,
                cost_model_certain.hub_limit[h]
            )
            for h in hubs
        ]
    )

    uncertain_prices = np.array(
        [
            capacity_value(
                cost_model,
                cost_model.supply_limit[f]
            )
            for f in farms
        ]
        + [
            capacity_value(
                cost_model,
                cost_model.hub_limit[h]
            )
            for h in hubs
        ]
    )

    y = np.arange(len(resources))

    shadow_fig, shadow_ax = plt.subplots(figsize=(9, 5))

    shadow_ax.barh(
        y - 0.2,
        certain_prices,
        height=0.4,
        color="lightblue",
        label="Original model"
    )

    shadow_ax.barh(
        y + 0.2,
        uncertain_prices,
        height=0.4,
        color="navy",
        label="Uncertain model"
    )

    shadow_ax.set_yticks(y, resources)
    shadow_ax.invert_yaxis()

    shadow_ax.set_xlabel(
        "Marginal cost saving "
        "(€/additional tonne of weekly capacity)"
    )
    shadow_ax.set_title("Shadow Prices of Network Capacities")
    shadow_ax.grid(axis="x", alpha=0.25)
    shadow_ax.legend()

    shadow_fig.tight_layout()
    shadow_fig

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Marginal abatement costs from dual variables

    We can apply the same sensitivity principle to the expected-emissions budget. Define the uncertain model's value function by

    \[
    \begin{aligned}
    V_u(\varepsilon)=\min_x\quad&\bar C(x)\\
    \text{subject to}\quad&\bar E(x)=\sum_{(i,j)\in A}\bar e_{ij}x_{ij}
    \le\varepsilon,\\
    &x\in\mathcal F.
    \end{aligned}
    \]

    Tightening the budget means decreasing \(\varepsilon\). Such a change generally makes the minimum expected cost increase. Using a nonnegative Lagrange multiplier \(\lambda^*\) for the constraint \(\bar E(x)-\varepsilon\le0\), we have, where the value function is differentiable,

    \[
    \frac{dV_u(\varepsilon)}{d\varepsilon}=-\lambda^*(\varepsilon).
    \]

    Thus the marginal abatement cost that tells the additional minimum transportation cost associated with avoiding another kilogram of expected CO₂e emissions is

    \[
        \mathrm{MAC}(\varepsilon)
        =-\frac{dV_u(\varepsilon)}{d\varepsilon}
        =\lambda^*(\varepsilon)\ge0.
    \]

    For this minimization problem, Pyomo/HiGHS returns a nonpositive dual value for the `<=` emissions constraint. The plot therefore uses `-model.dual[model.emissions_limit]` to display a nonnegative marginal abatement cost in €/kg CO₂e. We repeat the calculation for the original model and compare the resulting curves over the emissions-budget range feasible for both.

    Because the problem is linear, the value function is piecewise linear, and its marginal abatement cost is typically piecewise constant. At a breakpoint, the value function may not be differentiable and several optimal dual multipliers may be possible. Therefore, the step changes in the graph are meaningful features of LP sensitivity analysis, rather than numerical artifacts.

    The two MAC curves are not measured under identical assumptions. One uses normal-route costs and emissions, while the other uses expected costs and emissions. A common numerical emissions budget therefore represents a different modeled operating environment in each problem.
    """)
    return


@app.cell(hide_code=True)
def _(
    baseline_emissions,
    baseline_emissions_certain,
    min_emissions,
    min_emissions_certain,
    np,
    plt,
    solve_certain_model,
    solve_model,
):
    # Common feasible range of emissions budgets
    budget_min = max(
        min_emissions,
        min_emissions_certain
    )

    budget_max = min(
        baseline_emissions,
        baseline_emissions_certain
    )

    if budget_min >= budget_max:
        raise ValueError("No overlapping emissions-budget range.")

    # Interior budgets avoid extreme-endpoint degeneracy
    budgets = np.linspace(
        budget_min, budget_max, 52
    )[1:-1]

    mac_uncertain = []
    mac_certain = []

    for budget in budgets:

        uncertain_solution = solve_model(
            carbon_budget=float(budget),
            objective="cost"
        )

        certain_solution = solve_certain_model(
            carbon_budget=float(budget),
            objective="cost"
        )

        mac_uncertain.append(
            -uncertain_solution.dual[
                uncertain_solution.emissions_limit
            ]
        )

        mac_certain.append(
            -certain_solution.dual[
                certain_solution.emissions_limit
            ]
        )

    mac_fig, mac_ax = plt.subplots(figsize=(9, 5))

    mac_ax.step(
        budgets,
        mac_certain,
        where="mid",
        color="lightblue",
        linewidth=2,
        label="Original model"
    )

    mac_ax.step(
        budgets,
        mac_uncertain,
        where="mid",
        color="navy",
        linewidth=2,
        label="Uncertain model"
    )

    mac_ax.set_xlabel("Emissions budget (kg CO₂e/week)")
    mac_ax.set_ylabel("Marginal abatement cost (€/kg CO₂e)")
    mac_ax.set_title("Marginal Abatement Costs from Dual Variables")

    mac_ax.grid(alpha=0.25)
    mac_ax.legend()

    mac_fig.tight_layout()
    mac_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sensitivity analysis: expanding hub H2

    Shadow prices are local quantities and they describe the effect of a small relaxation around the current solution. To investigate how long that interpretation remains valid, we can vary the throughput capacity of H2 and resolve the LP at each capacity level.

    Let \(q\) denote the weekly capacity of H2 and \(V(q)\) the corresponding minimum expected transportation cost, with all other model parameters fixed. The upper panel below plots \(V(q)\); the lower panel plots the marginal cost saving from an additional tonne of H2 capacity:

    \[
    S(q)=-\frac{dV(q)}{dq}\ge0
    \]

    wherever the derivative exists. The dashed vertical line marks the original capacity of 100 tonnes/week.

    The figure makes an important distinction visible. At the current capacity, adding more H2 throughput may have zero marginal value, even though reducing its capacity below that level would increase expected costs. The value function can have a kink at such a threshold, so the left- and right-hand slopes may differ and the dual multiplier at the kink need not be unique.

    This is why a shadow price should not be interpreted as a fixed value that applies to arbitrarily large capacity investments: after a change in capacity, a different set of constraints may determine the optimal solution.
    """)
    return


@app.cell(hide_code=True)
def _(build_model, hub_capacity, np, plt, pyo):

    hub = "H2"

    capacities = np.linspace(75, 130, 56)

    capacity_costs = []
    capacity_duals = []

    solver = pyo.SolverFactory("highs")

    for capacity in capacities:

        # Create the uncertain model
        m = build_model(objective="cost")

        # Replace the capacity constraint for H2
        m.hub_limit[hub].set_value(
            sum(m.x[f, hub] for f in m.F)
            <= float(capacity)
        )

        results = solver.solve(m)
        pyo.assert_optimal_termination(results)

        capacity_costs.append(
            pyo.value(m.total_cost)
        )

        capacity_duals.append(
            -m.dual[m.hub_limit[hub]]
        )

    capacity_fig, (ax_cost, ax_dual) = plt.subplots(
        2, 1,
        figsize=(9, 7),
        sharex=True
    )

    # Optimal transportation cost
    ax_cost.plot(
        capacities,
        capacity_costs,
        color="navy",
        linewidth=2
    )

    ax_cost.set_ylabel("Optimal expected cost (€/week)")
    ax_cost.set_title("Value of Expanding Hub H2")
    ax_cost.grid(alpha=0.25)

    # Shadow price of capacity
    ax_dual.step(
        capacities,
        capacity_duals,
        where="mid",
        color="#DC2626",
        linewidth=2
    )

    ax_dual.set_xlabel("Hub H2 capacity (tonnes/week)")
    ax_dual.set_ylabel(
        "Marginal cost saving (€/additional tonne)"
    )
    ax_dual.grid(alpha=0.25)

    # Mark the current capacity
    for ax2 in (ax_cost, ax_dual):
        ax2.axvline(
            hub_capacity[hub],
            linestyle="--",
            color="gray",
            alpha=0.7,
            label="Current capacity"
        )

    ax_cost.legend()

    capacity_fig.tight_layout()
    capacity_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ???- question "Changing probability ☠️"
        Vary the disruption probability \(p_{\mathrm{F2,H2}}\). At what probability does the expected-cost-minimizing routing change? How does the optimal objective value behave near that threshold?
    """)
    return


if __name__ == "__main__":
    app.run()
