import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inventory Models

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

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Inventory models describe how the quantity of some stock evolves over time. In the simplest case, the stock could be a product that a firm needs to replenish periodically as customers purchase it. We can describe this process using a *balance equation*:

    \[
        I_{t+1}=I_t+Q_t-D_t.
    \]

    Here, \(I_t\) is the inventory at the beginning of period \(t\), \(Q_t\) is the quantity received during that period, and \(D_t\) is the quantity demanded during the period. The equation states that the inventory remaining for the next period equals the current inventory plus deliveries minus demand. In this discrete-time equation, \(Q_t\) and \(D_t\) are quantities per period, not rates.

    We assume for now that there are no inventory losses or other inflows and outflows. The same idea can later be extended to more complicated inventories, including stocks of natural resources.

    Our models will build on this basic description of how inventory changes over time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Economic order quantity

    The economic order quantity (EOQ) model can be derived from the inventory balance equation together with assumptions about demand, replenishment, and inventory costs. We will consider it in continuous time, which makes the derivation particularly convenient. The model uses the following variables and parameters:

    | Symbol | Description | Unit | Assumptions |
    |---|---|---|---|
    | $D$ | Demand rate | Units/year | Demand is constant, known, and continuous over time. $D>0$. |
    | $Q$ | Order quantity | Units/order | Every order contains the same quantity $Q>0$. This is the decision variable. |
    | $I(t)$ | Inventory level at time $t$ | Units | Inventory decreases at rate $D$ between deliveries. Shortages are not allowed, so $I(t)\geq 0$. |
    | $K$ | Fixed ordering cost | €/order | Each order incurs the same fixed cost $K>0$, independent of the order quantity. |
    | $h$ | Inventory holding cost | €/(unit·year) | Holding costs are constant and proportional to the inventory level and the time held. $h>0$. |
    | $T$ | Length of an ordering cycle | Years | Each cycle has the same length, which we will show is $T=Q/D$. |

    We also make the following assumptions, which are reflected in the formulation of the model:

    | Assumption | Description |
    |---|---|
    | Instantaneous replenishment | The entire order quantity $Q$ arrives at once, rather than gradually. |
    | No shortages | Demand is always satisfied, and inventory cannot become negative. |
    | No quantity discounts | The purchase price per unit does not depend on the order quantity. |
    | Unlimited planning horizon | The inventory cycle repeats indefinitely under identical conditions. We minimize long-run average annual costs. |
    | No capacity constraints | There are no restrictions on storage capacity or maximum order quantity. |

    It goes without saying that these assumptions restrict our model more than is realistic to assume in most scenarios. However, this formulation gives a nice closed form solution that can be used as a benchmark for more complex models.

    ### Inventory dynamics

    How do we move from the discrete-time balance equation to a continuous-time model? Suppose that demand occurs at the constant rate \(D\), and consider a short time interval \(\Delta t\) during which no delivery arrives. Inventory falls by \(D\Delta t\), so

    \[
        I(t+\Delta t)-I(t)=-D\Delta t.
    \]

    Dividing by \(\Delta t\) and taking the limit as \(\Delta t\) approaches zero gives the differential equation

    \[
        \frac{\mathrm{d}I(t)}{\mathrm{d}t}=-D.
    \]

    This equation holds *between deliveries*. When a new order arrives, inventory instead increases instantaneously by \(Q\).

    Integrating the differential equation gives

    \[
        \begin{aligned}
            \int\frac{\text{d}I(t)}{\text{d}t}\text{d}t &= I(t) \\
            &= \int-Ddt \\
            &= -Dt+C_0
        \end{aligned}
    \]

    where \(C_0\) is an integration constant. Let \(t=0\) be the moment immediately after receiving an order. We assume that inventory was zero just before the delivery, so the initial condition is \(I(0)=Q\). Therefore, \(C_0=Q\), and

    \[
        I(t)=Q-Dt,\qquad 0\leq t\leq T.
    \]

    To find when inventory reaches zero, we solve

    \[
        \begin{aligned}
            I(T)&=Q-DT=0,\\
            T&=\frac{Q}{D}.
        \end{aligned}
    \]

    Thus, each ordering cycle lasts \(Q/D\) years. The same process then repeats: inventory jumps up to \(Q\) when the next delivery arrives and decreases at the constant demand rate until the following delivery.
    """)
    return


@app.cell(hide_code=True)
def _(plt):
    # Illustrative inventory trajectory: three identical ordering cycles
    demand_rate = 1200  # Units/year
    order_size = 240    # Units/order
    cycle_length = order_size / demand_rate  # Years

    fig_cycle, ax_cycle = plt.subplots(figsize=(9, 4.5))

    for cycle in range(3):
        start = cycle * cycle_length
        end = (cycle + 1) * cycle_length
        ax_cycle.plot([start, end], [order_size, 0], color="tab:blue", linewidth=2.5)
        if cycle < 2:
            # Inventory arrives instantaneously at the start of the next cycle.
            ax_cycle.plot([end, end], [0, order_size], color="tab:blue", linewidth=2.5)

    ax_cycle.axhline(
        order_size / 2,
        color="tab:green",
        linestyle="--",
        linewidth=1.8,
        label=r"Average inventory $\overline{I}=Q/2$",
    )

    ax_cycle.set_xticks(
        [i * cycle_length for i in range(4)],
        [r"$0$", r"$T$", r"$2T$", r"$3T$"],
    )
    ax_cycle.set_yticks([0, order_size / 2, order_size],
                  [r"$0$", r"$Q/2$", r"$Q$"])
    ax_cycle.set_xlim(0, 3 * cycle_length + 0.02)
    ax_cycle.set_ylim(0, 1.15 * order_size)
    ax_cycle.set_xlabel("Time")
    ax_cycle.set_ylabel("Inventory (units)")
    ax_cycle.set_title("Inventory over three ordering cycles")
    ax_cycle.grid(alpha=0.2)
    ax_cycle.legend(loc="upper right")

    fig_cycle.tight_layout()
    fig_cycle
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Average inventory and ordering frequency

    We now want to calculate the cost of keeping products in inventory. Since holding costs accumulate over time, we first need to know the *average inventory level* during an ordering cycle.

    As the figure shows, inventory decreases linearly from \(Q\) to zero, so we would expect the average to be \(Q/2\). We can verify this by integrating the inventory level over a cycle and dividing by the cycle length:

    \[
        \begin{aligned}
            \overline{I}
            &=\frac{1}{T}\int_0^T I(t)\,\mathrm{d}t\\
            &=\frac{1}{T}\int_0^T(Q-Dt)\,\mathrm{d}t\\
            &=\frac{1}{T}\left[Qt-\frac{Dt^2}{2}\right]_0^T\\
            &=\frac{1}{T}\left(QT-\frac{DT^2}{2}\right)\\
            &=Q-\frac{DT}{2}.
        \end{aligned}
    \]

    Substituting \(T=Q/D\) gives

    \[
        \overline{I}=Q-\frac{D}{2}\frac{Q}{D}=\frac{Q}{2}.
    \]

    We also need to determine how frequently orders are placed, because each order incurs a fixed cost. Since one cycle lasts \(T\) years, the long-run number of orders per year is

    \[
        N(Q)=\frac{1}{T}=\frac{D}{Q}.
    \]

    ### Constructing the cost function

    The relevant annual cost consists of two components: fixed ordering costs and inventory holding costs.

    The annual ordering cost is the cost of one order multiplied by the number of orders per year:

    \[
        C_K(Q)=KN(Q)=K\frac{D}{Q}.
    \]

    The annual holding cost equals the average inventory multiplied by the holding cost per unit per year:

    \[
        C_h(Q)=h\overline{I}=h\frac{Q}{2}.
    \]

    Adding these components gives the relevant annual cost function

    \[
        C(Q)=K\frac{D}{Q}+h\frac{Q}{2}.
    \]

    Why are purchasing costs not included? If the purchase price is \(p\) euros per unit, the total annual cost would be

    \[
        C_{\mathrm{total}}(Q)=pD+K\frac{D}{Q}+h\frac{Q}{2}.
    \]

    Since annual demand \(D\) and the unit price \(p\) are constant, \(pD\) does not depend on the decision variable \(Q\). Omitting this constant changes the value of the objective function but not the optimal order quantity.

    We have now constructed the EOQ optimization problem. Holding costs increase with the order quantity, whereas ordering costs decrease. Choosing the optimal order size requires us to balance these two costs. The figure below illustrates the trade-off and marks the optimal order quantity by \(Q^*\).
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):

    # EOQ model parameters
    D = 1200  # Annual demand (units/year)
    K = 50    # Fixed ordering cost (€/order)
    h = 2     # Annual holding cost (€/unit/year)

    # Optimal economic order quantity
    Q_star = np.sqrt(2 * K * D / h)

    # Order quantities for plotting
    Q = np.linspace(0.1 * Q_star, 3 * Q_star, 500)

    # Annual cost components
    ordering_cost = K * D / Q
    holding_cost = h * Q / 2
    total_cost = ordering_cost + holding_cost

    # Minimum annual cost
    C_star = K * D / Q_star + h * Q_star / 2

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(Q, ordering_cost, label=r"Ordering cost $C_K(Q)$", linewidth=2)
    ax.plot(Q, holding_cost, label="rHolding cost $C_h(Q)$", linewidth=2)
    ax.plot(Q, total_cost, label=r"Total cost $C(Q)$", linewidth=2.5)

    # Mark the optimal order quantity
    ax.scatter(Q_star, C_star, color="red", s=75, zorder=5, marker="*")
    ax.axvline(Q_star, color="red", linestyle="--", alpha=0.7)

    ax.annotate(
        rf"$Q^* = {Q_star:.1f}$",
        xy=(Q_star, C_star),
        xytext=(Q_star * 1.4, C_star * 1.4),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=12,
    )

    ax.set_xlabel("Order quantity Q (units)")
    ax.set_ylabel("Annual cost (€)")
    ax.set_title("Economic Order Quantity (EOQ) Model")

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Finding the optimal order quantity

    We can now formulate the EOQ cost-minimization problem as

    \[
        \min_{Q>0}\;C(Q)=\min_{Q>0}\left\{K\frac{D}{Q}+h\frac{Q}{2}\right\}.
    \]

    We restrict \(Q\) to be strictly positive because the ordering-cost term is not defined at \(Q=0\). To find a candidate optimum, we set the first derivative equal to zero:

    \[
        \begin{aligned}
            C'(Q)&=-\frac{KD}{Q^2}+\frac{h}{2}=0,\\
            Q^2&=\frac{2KD}{h},\\
            Q^*&=\sqrt{\frac{2KD}{h}}.
        \end{aligned}
    \]

    We can also check that this candidate is indeed the unique global minimum. The second derivative is

    \[
        C''(Q)=\frac{2KD}{Q^3}>0 \qquad\text{for all }Q>0.
    \]

    Therefore, \(C(Q)\) is strictly convex on its domain, and the stationary point \(Q^*\) is its unique global minimizer.

    ???- question "Optimal costs and sensitivity ☠️☠️"
        Verify by substituting \(Q^*\) into the cost components that the annual ordering cost and annual holding cost are equal at the optimum:

        \[
            C_K(Q^*)=C_h(Q^*).
        \]

        What happens to the optimal order quantity if the demand rate \(D\) doubles while \(K\) and \(h\) remain unchanged? Is \(Q^*\) doubled as well?

    !!! note "What EOQ does and does not optimize"
        The EOQ model assumes that the same order quantity is used in every cycle, and it optimizes only this quantity. In a general multiperiod inventory model, the replenishment quantities can instead be separate decision variables for each period. The balance equation still applies, but we no longer need to assume that every ordering cycle is identical.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Economic Production Lot Size Model

    In the EOQ model, we assumed that orders arrive instantaneously, meaning that the entire order quantity $Q$ becomes available at once. However, this assumption is not always realistic. For example, a manufacturing company might produce its own products gradually rather than ordering them from an external supplier.

    The Economic Production Lot Size (EPQ) model extends EOQ by allowing inventory to be replenished gradually through production. We retain the assumptions of the EOQ model, but introduce one additional parameter:

    | Symbol | Description | Unit | Assumptions |
    |---|---|---|---|
    | $P$ | Production rate | Units/year | Production occurs at a constant rate $P>D$. |

    Here, $D$ is still the constant demand rate, and $Q$ now represents the number of units produced during one production run.

    The important difference is that demand continues while production takes place. Consequently, inventory does not accumulate at rate $P$, but at the net rate $P-D$.

    We can express the inventory dynamics as

    $$
    \frac{dI(t)}{dt}=
    \begin{cases}
    P-D, & \text{during production},\\
    -D, & \text{when production stops}.
    \end{cases}
    $$

    Notice that we require $P>D$. Otherwise, production would not be sufficient to satisfy demand.

    ### Maximum and average inventory

    Suppose that production begins when inventory reaches zero. Since production takes place at the constant rate $P$, producing a lot of $Q$ units takes

    $$
    T_P=\frac{Q}{P}
    $$

    years.

    During this period, inventory accumulates at rate $P-D$. Therefore, the maximum inventory level is

    $$
    \begin{aligned}
    I_{\max}
    &=(P-D)T_P\\
    &=(P-D)\frac{Q}{P}\\
    &=Q\left(1-\frac{D}{P}\right).
    \end{aligned}
    $$

    This differs from the EOQ model, where the maximum inventory equals $Q$.

    Since inventory increases linearly during production and decreases linearly after production stops, the average inventory over a complete production cycle is

    $$
    \overline I
    =\frac{I_{\max}}{2}
    =\frac{Q}{2}\left(1-\frac{D}{P}\right).
    $$

    Because $P>D$, we have

    $$
    \overline I<\frac Q2.
    $$

    Thus, gradual replenishment reduces the average inventory level compared with instantaneous replenishment.

    ### The optimization problem

    We can now construct the cost function in the same way as in the EOQ model.

    Let $K$ denote the fixed cost of starting a production run and $h$ the annual holding cost per unit.

    Since each production run produces $Q$ units, the number of production runs per year remains $D/Q$. Therefore, the annual setup cost is

    $$
    C_K(Q)=K\frac{D}{Q}.
    $$

    However, the holding cost changes because average inventory is now lower:

    $$
    C_h(Q)=h\frac{Q}{2}\left(1-\frac DP\right).
    $$

    The total relevant annual cost is consequently

    $$
    C(Q)=K\frac{D}{Q}+h\frac{Q}{2}\left(1-\frac{D}{P}\right).
    $$

    The optimization problem becomes

    $$
    \min_{Q>0}\left\{
    K\frac{D}{Q}
    +h\frac{Q}{2}\left(1-\frac{D}{P}\right)
    \right\}.
    $$

    Differentiating the cost function with respect to $Q$ gives

    $$
    C'(Q)=-\frac{KD}{Q^2}+\frac h2\left(1-\frac DP\right).
    $$

    Setting the derivative equal to zero and solving for $Q$ gives the optimal production lot size:

    $$
    Q_{\mathrm{EPQ}}^*
    =\sqrt{\frac{2KD}{h\left(1-\frac{D}{P}\right)}}.
    $$

    Compare this with the EOQ solution:

    $$
    Q_{\mathrm{EOQ}}^*=\sqrt{\frac{2KD}{h}}.
    $$

    Since $0<1-D/P<1$, it follows that

    $$
    Q_{\mathrm{EPQ}}^*>Q_{\mathrm{EOQ}}^*.
    $$

    The optimal production lot size is larger than the optimal order quantity in the EOQ model.

    This follows from the fact that gradual production reduces average inventory and, consequently, holding costs. Because holding costs are lower for any given lot size, it becomes economically optimal to produce larger batches and incur setup costs less frequently.

    An interesting limiting case occurs when the production rate becomes very large:

    $$
    \lim_{P\to\infty}Q_{\mathrm{EPQ}}^*
    =\sqrt{\frac{2KD}{h}}
    =Q_{\mathrm{EOQ}}^*.
    $$

    Thus, the EOQ model can be understood as a limiting case of the EPQ model in which replenishment becomes instantaneous.

    ???- question "Economic Production Lot Size ☠️☠️"

        Suppose that a manufacturing company produces goods at twice the rate at which they are demanded, so that $P=2D$.

        1. Determine the maximum inventory level during a production cycle as a function of $Q$.
        2. Determine the average inventory level during a production cycle.
        3. Show that the optimal production lot size satisfies

            $$
            Q_{\mathrm{EPQ}}^*=\sqrt{2}\,Q_{\mathrm{EOQ}}^*.
            $$

        4. Explain why gradual production leads to a larger optimal production lot size than instantaneous replenishment.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):

    # EPQ model parameters
    _epq_D = 1200  # Annual demand (units/year)
    _epq_P = 2400  # Production rate (units/year), P > D
    _epq_K = 50    # Fixed setup cost (€)
    _epq_h = 2     # Holding cost (€/unit/year)

    # Optimal lot sizes
    _epq_Q_eoq = np.sqrt(2 * _epq_K * _epq_D / _epq_h)
    _epq_Q_star = np.sqrt(
        2 * _epq_K * _epq_D / (_epq_h * (1 - _epq_D / _epq_P))
    )

    # Create figure
    epq_comparison_fig, (_epq_ax1, _epq_ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5)
    )

    # --------------------------------------------------
    # 1. Inventory dynamics: EOQ vs EPQ
    # --------------------------------------------------

    # Same lot size for both models to isolate the
    # effect of gradual production.
    _epq_Q = _epq_Q_star

    _epq_T = _epq_Q / _epq_D
    _epq_T_prod = _epq_Q / _epq_P
    _epq_I_max = _epq_Q * (1 - _epq_D / _epq_P)

    for _epq_i in range(3):
        _epq_start = _epq_i * _epq_T

        # EOQ: instantaneous replenishment
        _epq_ax1.plot(
            [_epq_start, _epq_start + _epq_T],
            [_epq_Q, 0],
            color="tab:blue",
            linestyle="--",
            linewidth=2,
            label="EOQ" if _epq_i == 0 else None,
        )

        _epq_ax1.plot(
            [_epq_start, _epq_start],
            [0, _epq_Q],
            color="tab:blue",
            linestyle="--",
            linewidth=1.5,
        )

        # EPQ: gradual production followed by depletion
        _epq_ax1.plot(
            [
                _epq_start,
                _epq_start + _epq_T_prod,
                _epq_start + _epq_T,
            ],
            [0, _epq_I_max, 0],
            color="tab:green",
            linewidth=2.5,
            label="EPQ" if _epq_i == 0 else None,
        )

    # Average EPQ inventory
    _epq_ax1.axhline(
        _epq_I_max / 2,
        color="tab:green",
        linestyle=":",
        alpha=0.8,
        label=r"Average EPQ inventory",
    )

    _epq_ax1.set_xlabel("Time (years)")
    _epq_ax1.set_ylabel("Inventory level (units)")
    _epq_ax1.set_title("Inventory Dynamics: EOQ vs EPQ")
    _epq_ax1.grid(alpha=0.25)
    _epq_ax1.legend()

    # --------------------------------------------------
    # 2. Annual cost functions
    # --------------------------------------------------

    _epq_Q_values = np.linspace(
        0.35 * _epq_Q_star,
        2.5 * _epq_Q_star,
        500,
    )

    # EPQ cost components
    _epq_setup_cost = _epq_K * _epq_D / _epq_Q_values
    _epq_holding_cost = (
        _epq_h * _epq_Q_values / 2
        * (1 - _epq_D / _epq_P)
    )
    _epq_total_cost = _epq_setup_cost + _epq_holding_cost

    # EOQ total cost for comparison
    _epq_eoq_cost = (
        _epq_K * _epq_D / _epq_Q_values
        + _epq_h * _epq_Q_values / 2
    )

    # Minimum costs
    _epq_C_eoq = (
        _epq_K * _epq_D / _epq_Q_eoq
        + _epq_h * _epq_Q_eoq / 2
    )

    _epq_C_star = (
        _epq_K * _epq_D / _epq_Q_star
        + _epq_h * _epq_Q_star / 2
        * (1 - _epq_D / _epq_P)
    )

    # Cost curves
    _epq_ax2.plot(
        _epq_Q_values, _epq_setup_cost,
        label="Setup cost", linewidth=1.8,
    )

    _epq_ax2.plot(
        _epq_Q_values, _epq_holding_cost,
        label="EPQ holding cost", linewidth=1.8,
    )

    _epq_ax2.plot(
        _epq_Q_values, _epq_total_cost,
        label="EPQ total cost", linewidth=2.5,
    )

    _epq_ax2.plot(
        _epq_Q_values, _epq_eoq_cost,
        label="EOQ total cost",
        linewidth=2,
        linestyle="--",
        color="tab:gray",
    )

    # Mark optimal solutions
    _epq_ax2.scatter(
        _epq_Q_eoq, _epq_C_eoq,
        color="tab:gray", s=75, zorder=5,
    )

    _epq_ax2.scatter(
        _epq_Q_star, _epq_C_star,
        color="tab:red", s=85, zorder=5,
    )

    _epq_ax2.axvline(
        _epq_Q_eoq, color="tab:gray",
        linestyle=":", alpha=0.7,
    )

    _epq_ax2.axvline(
        _epq_Q_star, color="tab:red",
        linestyle=":", alpha=0.7,
    )

    # Annotations
    _epq_ax2.annotate(
        rf"$Q_{{EOQ}}^*={_epq_Q_eoq:.1f}$",
        xy=(_epq_Q_eoq, _epq_C_eoq),
        xytext=(25, 45),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
    )

    _epq_ax2.annotate(
        rf"$Q_{{EPQ}}^*={_epq_Q_star:.1f}$",
        xy=(_epq_Q_star, _epq_C_star),
        xytext=(45, 30),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
    )

    _epq_ax2.set_xlabel("Lot size Q (units)")
    _epq_ax2.set_ylabel("Annual cost (€)")
    _epq_ax2.set_title("EOQ and EPQ Cost Functions")
    _epq_ax2.grid(alpha=0.25)
    _epq_ax2.legend()

    epq_comparison_fig.tight_layout()
    epq_comparison_fig

    return


if __name__ == "__main__":
    app.run()
