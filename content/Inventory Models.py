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
    import scipy.optimize as opt

    return mo, np, opt, plt


@app.cell
def _(np):
    # set random seed for reproducability
    np.random.seed(123)
    return


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

    The economic order quantity (EOQ) model can be derived from the inventory balance equation together with assumptions about demand, replenishment, and inventory costs. We will consider it in continuous time, which makes the derivation particularly convenient by allowing the use of calculus. The model uses the following variables and parameters:

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

    It goes without saying that these assumptions restrict our model more than is realistic to assume in most scenarios. In this formulation we can acquire a nice closed form solution that can be used as a benchmark for more complex models.

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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    \lim_{P\to\infty}Q_{\mathrm{EPQ}}^*=\lim_{P\to\infty}\sqrt{\frac{2KD}{h\left(1-\frac{D}{P}\right)}}
    =\sqrt{\frac{2KD}{h}}
    =Q_{\mathrm{EOQ}}^*.
    $$

    Thus, the EOQ model can be understood as a limiting case of the EPQ model in which replenishment becomes instantaneous.

    ???- question "Economic Production Lot Size ☠️☠️"

        Suppose that a manufacturing company produces goods at twice the rate at which they are demanded, so that $P=2D$.

        1. Determine the maximum inventory level during a production cycle as a function of $Q$.
        2. Determine the average inventory level during a production cycle.
        3. Show that the optimal production lot size satisfies

            \[
            Q_{\mathrm{EPQ}}^*=\sqrt{2}\,Q_{\mathrm{EOQ}}^*.
            \]

        4. Explain why gradual production leads to a larger optimal production lot size than instantaneous replenishment.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):

    # Model parameters
    _epq_D = 1200
    _epq_P = 2400
    _epq_K = 50
    _epq_h = 2

    # Optimal lot sizes
    _epq_Q_eoq = np.sqrt(2 * _epq_K * _epq_D / _epq_h)
    _epq_Q_epq = np.sqrt(
        2 * _epq_K * _epq_D /
        (_epq_h * (1 - _epq_D / _epq_P))
    )

    # Inventory levels and cycle durations
    _epq_I_max = _epq_Q_epq * (1 - _epq_D / _epq_P)

    _epq_T_eoq = _epq_Q_eoq / _epq_D
    _epq_T_epq = _epq_Q_epq / _epq_D
    _epq_T_prod = _epq_Q_epq / _epq_P

    # Create figure
    epq_corrected_fig, (_epq_ax1, _epq_ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5)
    )

    # --------------------------------------------------
    # 1. Inventory trajectories at optimal lot sizes
    # --------------------------------------------------

    _epq_horizon = 3 * _epq_T_epq

    # EOQ: instantaneous replenishment
    for _epq_i in range(5):
        _epq_start = _epq_i * _epq_T_eoq
        _epq_end = min(_epq_start + _epq_T_eoq, _epq_horizon)

        if _epq_start >= _epq_horizon:
            break

        # Instantaneous delivery
        _epq_ax1.plot(
            [_epq_start, _epq_start],
            [0, _epq_Q_eoq],
            color="tab:blue",
            linestyle="--",
            linewidth=1.8,
        )

        # Inventory depletion
        _epq_ax1.plot(
            [_epq_start, _epq_end],
            [_epq_Q_eoq,
             _epq_Q_eoq - _epq_D * (_epq_end - _epq_start)],
            color="tab:blue",
            linestyle="--",
            linewidth=2,
            label=(
                rf"EOQ: $Q^*={_epq_Q_eoq:.1f}$"
                if _epq_i == 0 else None
            ),
        )

    # EPQ: gradual production and depletion
    for _epq_i in range(3):
        _epq_start = _epq_i * _epq_T_epq

        _epq_ax1.plot(
            [
                _epq_start,
                _epq_start + _epq_T_prod,
                _epq_start + _epq_T_epq,
            ],
            [0, _epq_I_max, 0],
            color="tab:green",
            linewidth=2.5,
            label=(
                rf"EPQ: $Q^*={_epq_Q_epq:.1f}$"
                if _epq_i == 0 else None
            ),
        )

    _epq_ax1.set_xlim(0, _epq_horizon)
    _epq_ax1.set_ylim(bottom=0)
    _epq_ax1.set_xlabel("Time (years)")
    _epq_ax1.set_ylabel("Inventory level (units)")
    _epq_ax1.set_title("Inventory at Optimal Lot Sizes")
    _epq_ax1.grid(alpha=0.25)
    _epq_ax1.legend()

    # --------------------------------------------------
    # 2. Annual cost comparison
    # --------------------------------------------------

    _epq_Q_values = np.linspace(
        0.4 * _epq_Q_eoq,
        2.2 * _epq_Q_epq,
        500,
    )

    # EOQ and EPQ total costs
    _epq_C_eoq = (
        _epq_K * _epq_D / _epq_Q_values
        + _epq_h * _epq_Q_values / 2
    )

    _epq_C_epq = (
        _epq_K * _epq_D / _epq_Q_values
        + _epq_h * _epq_Q_values / 2
        * (1 - _epq_D / _epq_P)
    )

    # Optimal costs
    _epq_C_eoq_star = (
        _epq_K * _epq_D / _epq_Q_eoq
        + _epq_h * _epq_Q_eoq / 2
    )

    _epq_C_epq_star = (
        _epq_K * _epq_D / _epq_Q_epq
        + _epq_h * _epq_Q_epq / 2
        * (1 - _epq_D / _epq_P)
    )

    # Cost curves
    _epq_ax2.plot(
        _epq_Q_values, _epq_C_eoq,
        color="tab:blue",
        linestyle="--",
        linewidth=2,
        label="EOQ total cost",
    )

    _epq_ax2.plot(
        _epq_Q_values, _epq_C_epq,
        color="tab:green",
        linewidth=2.5,
        label="EPQ total cost",
    )

    # Mark the two optimal lot sizes
    for _epq_q, _epq_c, _epq_color, _epq_label in [
        (_epq_Q_eoq, _epq_C_eoq_star, "tab:blue", "EOQ"),
        (_epq_Q_epq, _epq_C_epq_star, "tab:green", "EPQ"),
    ]:
        _epq_ax2.scatter(
            _epq_q, _epq_c,
            color=_epq_color, s=85, zorder=5,
        )

        _epq_ax2.axvline(
            _epq_q,
            color=_epq_color,
            linestyle=":",
            alpha=0.7,
        )

        _epq_ax2.annotate(
            rf"{_epq_label}: $Q^*={_epq_q:.1f}$",
            xy=(_epq_q, _epq_c),
            xytext=(20, 35),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=_epq_color),
            color=_epq_color,
        )

    _epq_ax2.set_xlabel("Lot size Q (units)")
    _epq_ax2.set_ylabel("Annual cost (€)")
    _epq_ax2.set_title("Optimal Lot Size: EOQ vs EPQ")
    _epq_ax2.grid(alpha=0.25)
    _epq_ax2.legend()

    epq_corrected_fig.tight_layout()
    epq_corrected_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Uncertain demand simulation

    In the EOQ and EPQ we had known constant demands. However, in many practical applications, demand is uncertain and may vary considerably between periods. This uncertainty can make it difficult to determine when and how much inventory should be replenished.

    We can extend our inventory models by allowing demand to be stochastic. Instead of assuming a constant demand rate, we assume that demand in each period follows a Poisson distribution:

    $$
    D_t\sim\operatorname{Poisson}(\lambda),
    $$

    where $\lambda$ denotes the expected demand per period. We assume that demand is independent between periods.

    We will use a simulation model to investigate how different ordering policies perform under uncertain demand. The model allows us to specify different ordering rules, delivery lead times, and economic costs. We can then compare these policies by simulating inventory dynamics over multiple periods.

    ### Model parameters and variables

    In addition to the variables introduced in the EOQ model, we introduce the following parameters and variables:

    | Symbol | Description | Unit | Assumptions |
    |---|---|---|---|
    | $D_t$ | Demand during period $t$ | Units | Independent Poisson random variables with expected value $\lambda$. |
    | $\lambda$ | Expected demand per period | Units/period | Constant and known, with $\lambda\geq 0$. |
    | $I_0$ | Initial inventory | Units | Known and nonnegative. |
    | $I_t$ | Inventory at the beginning of period $t$ | Units | Inventory cannot be negative. |
    | $q_t$ | Quantity ordered during period $t$ | Units | Determined by the ordering policy. |
    | $R$ | Reorder threshold | Units | An order is placed when the inventory position is at or below $R$ under the fixed-lot policy. |
    | $\ell$ | Delivery lead time | Periods | Known, constant, and a nonnegative integer. |
    | $A_t$ | Quantity received during period $t$ | Units | Depends on previously placed orders and the lead time. |
    | $L_t$ | Lost sales during period $t$ | Units | Demand that cannot be satisfied is lost rather than backordered. |
    | $c$ | Unit purchasing cost | €/unit | Constant for all orders. |
    | $p$ | Lost-sales penalty | €/unit | Constant cost for each unit of unsatisfied demand. |
    | $T$ | Simulation horizon | Periods | Number of periods simulated. |
    | $N$ | Number of simulations | Simulations | Independent demand realizations used to estimate expected costs. |

    The fixed ordering cost $K$ and holding cost $h$ retain their earlier interpretations, but $h$ is now measured in euros per unit per simulation period. Here, $T$ denotes the number of discrete simulation periods, rather than the ordering-cycle duration used in the EOQ derivation.

    ### Inventory dynamics

    The model operates in discrete time. At the beginning of each period, previously ordered products scheduled for delivery arrive. The ordering policy then determines whether a new order should be placed.

    If the delivery lead time is zero, the new order arrives immediately. Otherwise, it is scheduled to arrive in a future period. Demand is then realized, and the remaining inventory is recorded.

    Let $A_t$ denote all deliveries received during period $t$, including any immediately delivered orders. The inventory balance equation becomes

    $$
    I_{t+1}=I_t+A_t-S_t,
    $$

    where $S_t$ denotes the quantity of demand actually satisfied:

    $$
    S_t=\min\{D_t,I_t+A_t\}.
    $$

    If demand exceeds available inventory, the difference is recorded as lost sales:

    $$
    L_t=D_t-S_t.
    $$

    Consequently, the inventory balance can also be written as

    $$
    I_{t+1}=I_t+A_t-D_t+L_t.
    $$

    Unlike the deterministic EOQ model, inventory now follows a stochastic trajectory because demand varies between periods.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ordering policies

    An ordering policy determines how much to order based on the information available when the decision is made.

    For example, a simple fixed-lot policy orders $Q$ units whenever the inventory position falls to or below a threshold $R$:

    $$
    q_t=
    \begin{cases}
    Q, & IP_t\leq R,\\
    0, & IP_t>R,
    \end{cases}
    $$

    where $IP_t$ is the inventory position, defined as the inventory currently available after scheduled deliveries plus the quantity of outstanding orders.

    The simplest policy uses $R=0$, meaning that a new order is placed only when the inventory position reaches zero. More sophisticated policies can use a positive reorder threshold or information from previous demand observations.

    One could also develope much more complicated and interesting policies for orders. For example, there could be some much more sopisticated demand forecast that would determine the order quantity. However, in our model it would be pointless to make this too complicated as we know the nature of demand. Thus we would need to also expand the simulation to see the benefit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Economic costs

    We evaluate the performance of an ordering policy using four cost components: fixed ordering costs, purchasing costs, holding costs, and the costs associated with lost sales.

    The cost incurred during period $t$ is

    $$
    C_t
    =
    K\mathbf{1}_{\{q_t>0\}}
    +cq_t
    +hI_{t+1}
    +pL_t,
    $$

    where $\mathbf{1}_{\{q_t>0\}}$ equals one if an order is placed and zero otherwise.

    Purchasing and fixed ordering costs are incurred when orders are placed, even if delivery occurs after the simulation ends. Holding costs are approximated using inventory remaining at the end of each period. The model assigns no remaining value to inventory at the end of the simulation horizon; these conventions can affect comparisons over short horizons.

    Our objective is to find an ordering policy that minimizes the expected total cost:

    $$
    \min_{\pi}\;
    \mathbb E_\pi\left[
    \sum_{t=0}^{T-1}C_t
    \right],
    $$

    where $\pi$ denotes the ordering policy.

    Because the costs depend on random demand, we can approximate their expected value using Monte Carlo simulation. If we simulate $N$ independent demand trajectories, the expected total cost can be estimated as

    $$
    \widehat C(\pi)
    =
    \frac{1}{N}
    \sum_{j=1}^{N}
    \sum_{t=0}^{T-1}C_{t,j}(\pi).
    $$

    We can then use numerical optimization to find the parameters of an ordering policy that minimize this estimated cost. However, the objective is not linear due to the temporal structure of the problem and order policy.

    An important advantage of simulation is that we can investigate inventory policies even when deriving an analytical solution is difficult. However, the results depend on our assumptions about demand, costs, and the ordering process.

    ## Implementation

    The following Python class defines the simulation model. We specify the cost parameters, delivery lead time, and an ordering policy. The ordering policy is a function that receives a dictionary called `state`, containing the information supplied by the model when the ordering decision is made. It returns the quantity to order. We can change the policy without rewriting the simulation.

    For simplicity, the model assumes that even unmet demand is observed. In practice, demand is often only partly observable during stockouts, because we may only see completed sales.
    """)
    return


@app.cell
def _(np, plt):
    class InventorySimulator:
        """Periodic-review inventory simulation with interchangeable ordering rules.

        Each period: receive scheduled deliveries, choose an order,
        receive orders in `lead_time`, then observe demand.
        Unmet demand is lost, not backordered.

        Costs:
          setup_cost: fixed amount per nonzero order (€/order)
          unit_cost: purchase cost (€/unit ordered)
          holding_cost: inventory holding cost (€/unit/period)
          lost_sale_cost: cost per unit of unsatisfied demand (€/unit)

        Costs are charged when an order is placed, at each period end
        for remaining inventory, and when demand is lost.
        """

        def __init__(self, policy, lead_time=0, *, setup_cost=0.0,
                     unit_cost=0.0, holding_cost=0.0, lost_sale_cost=0.0):
            if not callable(policy):
                raise TypeError("policy must be callable")
            if not isinstance(lead_time, (int, np.integer)) or lead_time < 0:
                raise ValueError("lead_time must be a nonnegative integer")
            for name, value in {
                "setup_cost": setup_cost,
                "unit_cost": unit_cost,
                "holding_cost": holding_cost,
                "lost_sale_cost": lost_sale_cost,
            }.items():
                if not np.isscalar(value) or not np.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and nonnegative")

            self.policy = policy
            self.lead_time = int(lead_time)
            self.setup_cost = float(setup_cost)
            self.unit_cost = float(unit_cost)
            self.holding_cost = float(holding_cost)
            self.lost_sale_cost = float(lost_sale_cost)

        @staticmethod
        def _as_matrix(demand, T, N):
            arr = np.asarray(demand, dtype=float)
            if arr.ndim == 1 and N == 1:
                arr = arr[:, None]
            if arr.shape != (T, N) or not np.all(np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError("demand must be nonnegative with shape (T, N)")
            return arr

        def simulate(self, I0, T, *, lam=50, N=1, demand=None, seed=None,
                     policy=None):
            """Return (T,N) dimensional simulation array

            inventory and cumulative cost has T+1 rows, other arrays have T rows
            policy can override the default for this simulation only
            """
            if not isinstance(T, (int, np.integer)) or T < 1:
                raise ValueError("T must be a positive integer")
            if not isinstance(N, (int, np.integer)) or N < 1:
                raise ValueError("N must be a positive integer")
            if not np.isfinite(I0) or I0 < 0 or not np.isfinite(lam) or lam < 0:
                raise ValueError("I0 and lam must be finite and nonnegative")
            active_policy = self.policy if policy is None else policy
            if not callable(active_policy):
                raise TypeError("policy must be callable")

            demands = (np.random.default_rng(seed).poisson(lam, size=(T, N))
                       if demand is None else self._as_matrix(demand, T, N))

            names = ("orders", "receipts", "sold", "lost_sales", "on_order",
                     "setup_cost", "purchase_cost", "holding_cost", "lost_sale_cost")
            result = {name: np.zeros((T, N), dtype=float) for name in names}
            result["inventory"] = np.zeros((T + 1, N), dtype=float)
            result["inventory"][0, :] = I0
            result["demand"] = demands

            for j in range(N):
                stock = float(I0)
                # Accounting for the arrival of orders
                pipeline = {}

                for t in range(T):
                    # Add arrived orders to stock, if none then add 0
                    arriving = pipeline.pop(t, 0.0)
                    stock += arriving
                    # Received orders in period t
                    result["receipts"][t, j] = arriving
                    # Sum of orders at period t
                    outstanding = sum(pipeline.values())

                    state = {
                        "t": t,
                        "on_hand": stock,
                        "on_order": outstanding,
                        "inventory_position": stock + outstanding,
                        "last_demand": None if t == 0 else float(demands[t - 1, j]),
                        "demand_history": demands[:t, j].copy(),
                        "order_history": result["orders"][:t, j].copy(),
                        "lost_sales_history": result["lost_sales"][:t, j].copy(),
                        "lead_time": self.lead_time,
                        "expected_demand": lam,
                    }
                    # Compute the order quantity according to the policy
                    quantity = active_policy(state)
                    if not np.isscalar(quantity) or not np.isfinite(quantity) or quantity < 0:
                        raise ValueError("policy must return a finite nonnegative quantity")
                    quantity = float(quantity)
                    result["orders"][t, j] = quantity

                    # If lead time is assumed to zero i.e. immediate delivery add order to stock
                    if self.lead_time == 0:
                        stock += quantity
                        result["receipts"][t, j] += quantity
                    # If lead time is positive add to pipeline
                    elif quantity > 0:
                        due = t + self.lead_time
                        pipeline[due] = pipeline.get(due, 0.0) + quantity

                    # Compute sold, unsold and stock items
                    units_sold = min(stock, float(demands[t, j]))
                    units_lost = float(demands[t, j]) - units_sold
                    stock -= units_sold

                    result["sold"][t, j] = units_sold
                    result["lost_sales"][t, j] = units_lost
                    result["inventory"][t + 1, j] = stock
                    result["on_order"][t, j] = sum(pipeline.values())
                    result["setup_cost"][t, j] = self.setup_cost * (quantity > 0)
                    result["purchase_cost"][t, j] = self.unit_cost * quantity
                    result["holding_cost"][t, j] = self.holding_cost * stock
                    result["lost_sale_cost"][t, j] = self.lost_sale_cost * units_lost

            result["total_cost"] = sum(result[k] for k in
                ("setup_cost", "purchase_cost", "holding_cost", "lost_sale_cost"))
            result["cumulative_cost"] = np.vstack((
                np.zeros((1, N)), np.cumsum(result["total_cost"], axis=0)))

            if N == 1:
                return {name: arr[:, 0] for name, arr in result.items()}
            return result

        @staticmethod
        def cost(results, *, per_period=False):
            """Return mean simulated cost

            By default, each path's costs are summed across all T periods
            If per_period=True, divide by T for an average cost per period
            """
            costs = np.asarray(results["total_cost"], dtype=float)
            if costs.ndim not in (1, 2) or costs.shape[0] == 0:
                raise ValueError("total_cost must have shape (T,) or (T, N)")
            value = costs.sum(axis=0).mean()
            if per_period:
                value /= costs.shape[0]
            return float(value)

        @staticmethod
        def summarize(results):
            """Average performance across simulated paths."""
            def matrix(key):
                a = np.asarray(results[key])
                return a[:, None] if a.ndim == 1 else a

            demands = matrix("demand")
            sold = matrix("sold")
            orders = matrix("orders")
            inv = matrix("inventory")
            total_dem = demands.sum()
            out = {
                "mean_total_cost": float(matrix("total_cost").sum(axis=0).mean()),
                "mean_cost_per_period": float(matrix("total_cost").sum(axis=0).mean() / len(demands)),
                "mean_end_inventory": float(inv[1:].mean()),
                "mean_orders_per_path": float((orders > 0).sum(axis=0).mean()),
                "fill_rate": float(sold.sum() / total_dem) if total_dem > 0 else 1.0,
                "mean_lost_sales_per_path": float(matrix("lost_sales").sum(axis=0).mean()),
            }
            for key in ("setup_cost", "purchase_cost", "holding_cost", "lost_sale_cost"):
                out["mean_" + key] = float(matrix(key).sum(axis=0).mean())
            return out

        def plot(self, results, *, max_paths=10, show_flows=True,
                 show_costs=True, figsize=(11, 8)):
            """Plot inventory, optional flows, and cost components."""
            def matrix(key):
                a = np.asarray(results[key])
                return a[:, None] if a.ndim == 1 else a

            inv = matrix("inventory")
            T = inv.shape[0] - 1
            N = inv.shape[1]
            nrows = 1 + int(show_flows) + int(show_costs)
            fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True,
                                     squeeze=False)
            axes = axes[:, 0]
            ax = axes[0]
            time = np.arange(T + 1)
            if N == 1:
                ax.step(time, inv[:, 0], where="post", linewidth=2, label="Inventory")
            else:
                for j in range(min(N, max_paths)):
                    ax.step(time, inv[:, j], where="post", color="tab:blue",
                            alpha=0.3, linewidth=1,
                            label="Individual paths" if j == 0 else None)
                ax.plot(time, inv.mean(axis=1), color="tab:red", linewidth=2.5,
                        label="Mean inventory")
            ax.set_ylabel("Inventory (units)")
            ax.set_title("Inventory simulation")
            ax.grid(alpha=0.25)
            ax.legend()

            periods = np.arange(T) + 0.5
            next_ax = 1
            if show_flows:
                ax = axes[next_ax]
                next_ax += 1
                for label, key in (("Demand", "demand"), ("Orders", "orders"),
                                   ("Deliveries", "receipts"),
                                   ("Lost sales", "lost_sales")):
                    a = matrix(key)
                    y = a[:, 0] if N == 1 else a.mean(axis=1)
                    ax.plot(periods, y, linewidth=1.5, label=label)
                ax.set_ylabel("Units per period" if N == 1 else "Mean units / period")
                ax.grid(alpha=0.25)
                ax.legend(ncol=4)

            if show_costs:
                ax = axes[next_ax]
                for label, key in (("Setup", "setup_cost"),
                                   ("Purchasing", "purchase_cost"),
                                   ("Holding", "holding_cost"),
                                   ("Lost sales", "lost_sale_cost"),
                                   ("Total", "total_cost")):
                    a = matrix(key)
                    y = a[:, 0] if N == 1 else a.mean(axis=1)
                    ax.plot(periods, y, linewidth=2 if key == "total_cost" else 1.2,
                            label=label)
                ax.set_ylabel("€/period" if N == 1 else "Mean €/period")
                ax.grid(alpha=0.25)
                ax.legend(ncol=5)

            axes[-1].set_xlabel("Time period")
            axes[0].set_xlim(0, T)
            fig.tight_layout()
            return fig


    return (InventorySimulator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us define a simple ordering policy. The `fixed_lot_policy` function creates the threshold rule described above. The first argument, \(Q\), is the order quantity and the optional `reorder_point` argument is the threshold \(R\).
    """)
    return


@app.function
def fixed_lot_policy(Q, reorder_point=0):
    """Order Q units when inventory is at or below the threshold."""
    if Q <= 0:
        raise ValueError("Q must be positive")

    def rule(state):
        return Q if state["inventory_position"] <= reorder_point else 0.0

    return rule


@app.cell(hide_code=True)
def _(baseline_results, mo, np):
    _mean_sales = np.asarray(baseline_results["sold"]).sum(axis=0).mean()
    _mean_lost = np.asarray(baseline_results["lost_sales"]).sum(axis=0).mean()
    mo.md(rf"""
    ### Comparing simulation with the classical EOQ benchmark

    Let us simulate a simple inventory system for 52 weeks, using an expected weekly demand of $\lambda=50$ units, an ordering cost of €50 per order, and a holding cost of €1.25 per unit per week. We begin with an order quantity of 63 units and an initial inventory of 63 units. The classical EOQ formula gives

    \[
        Q_{{\mathrm{{EOQ}}}}^*=\sqrt{{\frac{{2(50)(50)}}{{1.25}}}}\approx63.25.
    \]

    However, our simulation differs from EOQ in two important ways. First, demand is random and inventory is reviewed only at the beginning of each week, whereas classical EOQ replenishes inventory as soon as it reaches zero. Second, we initially set the lost-sales penalty to zero, meaning that demand left unsatisfied is not counted as a cost.

    In the classical EOQ model, all annual demand is satisfied, giving $50\cdot52=2600$ units sold. With the simple periodic-review policy $Q=63$ and $R=0$, our simulation sells on average {_mean_sales:.1f} units and loses {_mean_lost:.1f} units of demand over 52 weeks.

    We should therefore not interpret a lower simulated cost as evidence that the stochastic policy is more efficient: it may simply be serving fewer customers.
    """)
    return


@app.cell
def _(InventorySimulator):
    # Order 63 units whenever inventory position is zero.
    baseline_policy = fixed_lot_policy(Q=63, reorder_point=0)
    baseline_simulator = InventorySimulator(
        baseline_policy,
        lead_time=0,
        setup_cost=50,
        holding_cost=1.25,
        lost_sale_cost=0,
    )
    return (baseline_simulator,)


@app.cell
def _(baseline_simulator):
    baseline_results = baseline_simulator.simulate(
        I0=63, T=52, lam=50, N=500, seed=123
    )
    return (baseline_results,)


@app.cell
def _(InventorySimulator):
    # Create a policy with 150 units order threshold
    threshold_policy = fixed_lot_policy(63)
    # Pass the policy for our simulation class
    EOQ = InventorySimulator(threshold_policy, setup_cost=50, holding_cost=1.25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The simulation's mean ordering and holding cost over 52 weeks, excluding any penalty for unmet demand, is:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the simulation model, our total cost with order size of 63 items is on average:
    """)
    return


@app.cell
def _(baseline_results, baseline_simulator):
    baseline_simulator.cost(baseline_results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can compare this with the classical EOQ long-run cost benchmark, calculated from the same weekly demand and holding cost. We multiply weekly costs by 52 to obtain annual costs.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    print(f"""
    EOQ optimum: {np.sqrt((2*50*50)/1.25):.1f}
    EOQ annual order cost: {(52*50*50/np.sqrt((2*50*50)/1.25)):.1f}
    EOQ annual holding cost: {(52*1.25*np.sqrt((2*50*50)/1.25)/2):.1f}
    EOQ annual total cost: {(52*50*50/np.sqrt((2*50*50)/1.25)+52*1.25*np.sqrt((2*50*50)/1.25)/2):.1f}
    """)
    return


@app.cell(hide_code=True)
def _(baseline_results, baseline_simulator, mo, np):
    _baseline_stats = baseline_simulator.summarize(baseline_results)
    _eoq_q = np.sqrt(2 * 50 * 50 / 1.25)
    _eoq_orders = 52 * 50 / _eoq_q
    _baseline_orders = _baseline_stats["mean_orders_per_path"]
    _baseline_avg_stock = _baseline_stats["mean_end_inventory"]
    _order_cost_difference = 50 * (_eoq_orders - _baseline_orders)
    mo.md(rf"""
    The classical EOQ model predicts approximately

    \[
        N_{{\mathrm{{EOQ}}}}=\frac{{2600}}{{63.25}}\approx41.1
    \]

    orders per year, whereas the simulation places an average of {_baseline_orders:.1f} orders. This accounts for approximately €{_order_cost_difference:.1f} of the difference in annual ordering costs.

    Classical EOQ also predicts an average inventory of

    \[
        \overline I=\frac{{Q_{{\mathrm{{EOQ}}}}^*}}{{2}}\approx31.6
    \]

    units. The simulation has a mean end-of-week inventory of {_baseline_avg_stock:.1f} units, which reduces its reported holding costs. Note that end-of-week inventory is not the same measure as the continuous-time average inventory used in EOQ.

    Both differences arise largely because the simulation allows stockouts without penalizing them. The comparison also depends on the initial stock (63 units), the finite 52-week horizon, and the weekly review schedule. These effects need to be taken into account when interpreting the results.
    """)
    return


@app.cell
def _(baseline_results, baseline_simulator):
    baseline_simulator.plot(baseline_results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optimizing the order quantity

    We can use our stochastic inventory simulation to determine the order quantity that minimizes the estimated expected total cost. For a fixed reorder threshold $R$, in this case this is zero, our optimization problem becomes

    $$
    \min_{Q\in\mathbb{Z}_{>0}}\;\mathbb{E}[C(Q;R)],
    $$

    where $C(Q;R)$ denotes the total cost of the inventory system over the simulation horizon.

    Since the expected cost may be difficult to calculate analytically, we approximate it using $N$ independent simulations:

    $$
    \min_{Q\in\mathbb{Z}_{>0}}\;
    \widehat{C}(Q;R)
    =
    \min_{Q\in\mathbb{Z}_{>0}}\;
    \frac{1}{N}\sum_{j=1}^{N}C_j(Q;R).
    $$

    Here, $C_j(Q;R)$ is the total cost obtained from simulation $j$.

    We can solve this problem numerically by defining an objective function that runs the inventory simulation for a given order quantity $Q$ and returns the average simulated cost. A numerical optimizer can then search for the order quantity that minimizes this cost.

    To make the comparison between different order quantities more reliable, we use the same randomly generated demand realizations for each candidate solution.

    We can compare the numerical solution with the classical EOQ quantity of approximately $63$ units. However, the initial optimization includes no lost-sales penalty. Therefore, a low-cost solution may achieve its lower cost by not serving much of the demand. We should examine the fill rate in addition to the cost.

    Below, `quantity_objective` defines the simulated objective function for SciPy. We keep the initial stock and the demand scenarios fixed so that the order quantity is the only changing input.
    """)
    return


@app.cell
def _(InventorySimulator, np, opt):
    # Simulator with economic costs
    quantity_simulator = InventorySimulator(
        policy=fixed_lot_policy(Q=63, reorder_point=0),
        lead_time=0,
        setup_cost=50,
        unit_cost=0,
        holding_cost=1.25,
        lost_sale_cost=0
    )

    # Generate demand once, so every candidate policy
    # is evaluated against the same random scenarios.
    quantity_demand = np.random.default_rng(123).poisson(
        lam=50,
        size=(52, 500),
    )

    # Objective function
    def quantity_objective(x):

        _Q = int(x[0])

        _results = quantity_simulator.simulate(
            I0=63,
            T=52,
            N=500,
            demand=quantity_demand,
            policy=fixed_lot_policy(
                Q=_Q
            ),
        )

        return quantity_simulator.cost(
            _results,
            per_period=True,
        )

    # Optimize over integer order quantities
    quantity_optimum = opt.differential_evolution(
        quantity_objective,
        bounds=[(5, 150)],
        integrality=[True],
        seed=123,
    )
    assert quantity_optimum.success
    print("Optimal Q:", quantity_optimum.x[0])
    print("Expected cost per period:", quantity_optimum.fun)
    return quantity_demand, quantity_optimum, quantity_simulator


@app.cell(hide_code=True)
def _(np, plt, quantity_demand, quantity_optimum, quantity_simulator):
    # Estimated cost at a range of order quantities, with the same demand paths.
    _quantity_grid = np.arange(5, 151, 5)
    _quantity_costs = []
    for _quantity_candidate in _quantity_grid:
        _quantity_trial = quantity_simulator.simulate(
            I0=63, T=52, N=500, demand=quantity_demand,
            policy=fixed_lot_policy(Q=int(_quantity_candidate)),
        )
        _quantity_costs.append(
            quantity_simulator.cost(_quantity_trial, per_period=True)
        )

    quantity_cost_fig, _quantity_ax = plt.subplots(figsize=(9, 4.5))
    _quantity_ax.plot(_quantity_grid, _quantity_costs, marker="o", markersize=3,
                      label="Simulated mean cost")
    _quantity_ax.axvline(np.sqrt(2 * 50 * 50 / 1.25), color="tab:gray",
                         linestyle="--", label="Classical EOQ optimum")
    _quantity_ax.scatter(quantity_optimum.x[0], quantity_optimum.fun,
                         color="tab:red", zorder=5, label="Simulated optimum")
    _quantity_ax.set_xlabel("Order quantity Q (units)")
    _quantity_ax.set_ylabel("Mean simulated cost (€/week)")
    _quantity_ax.set_title("Order quantity under weekly stochastic demand")
    _quantity_ax.legend()
    _quantity_ax.grid(alpha=0.25)
    quantity_cost_fig.tight_layout()
    quantity_cost_fig
    return


@app.cell(hide_code=True)
def _(mo, quantity_optimum):
    mo.md(rf"""
    The optimizer returns an order quantity of {int(quantity_optimum.x[0])} units for these demand scenarios and parameter bounds. This minimizes the estimated cost of the model we specified, not the classical EOQ cost. Because lost sales have no penalty here, the solution may allow substantial unmet demand.

    We can inspect the cost components, the fill rate, and the inventory trajectories for the resulting policy.
    """)
    return


@app.cell
def _(quantity_demand, quantity_optimum, quantity_simulator):
    quantity_result = quantity_simulator.simulate(
        I0=63,
        T=52,
        N=500,
        demand=quantity_demand,
        policy=fixed_lot_policy(
            Q=int(quantity_optimum.x[0])
        ),
    )
    return (quantity_result,)


@app.cell
def _(quantity_result, quantity_simulator):
    for key,value in quantity_simulator.summarize(quantity_result).items():
        print(key, value)
    return


@app.cell
def _(quantity_result, quantity_simulator):
    quantity_simulator.plot(quantity_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optimizing quantity and threshold

    There is no reason to stop here with the optimization. In our policy we have also defined the inventory threshold after which the order should be made as a variable. To make the optimization slightly more interesting let's also define a cost for missing sales.

    The missing sale cost is a bit unsatisfactory quantity. It is clearly not equal to the selling price or profit of the product. If we would set it that level we would be essentially assuming that the sale is lost for good. Instead we would need to somehow estimate what losing that particular sales action costs us. The customer may be coming back later, but this is not certain.

    This is further complicated by the fact that scarcity can be also used as a sales tactic, *e.g.* luxury goods. In this case the assumption is that losing some sales is actually beneficial.

    We will however set the `lost_sale_cost` to 1.5€ per item that would be demanded but cant be sold due to shortage.

    Now we will simultaneously optimize both variables, quantity and order threshold. This leads to following optimization problem

    $$
        \min_{Q,R\in\mathbb{Z}_{>0}}\;
        \widehat{C}(Q,R)
        =
        \min_{Q,R\in\mathbb{Z}_{>0}}\;
        \frac{1}{N}\sum_{j=1}^{N}C_j(Q,R).
    $$

    Implementation for this follows the implementation of only quantity optimization closely. Instead of having only the quantity as an optimization parameter in the objective, we instead now also include the threshold \(R\).
    """)
    return


@app.cell
def _(InventorySimulator, np, opt):
    # Simulator with economic costs
    threshold_simulator = InventorySimulator(
        policy=fixed_lot_policy(Q=63, reorder_point=0),
        lead_time=0,
        setup_cost=50,
        unit_cost=0,
        holding_cost=1.25,
        lost_sale_cost=1.5
    )

    # Generate demand once, so every candidate policy
    # is evaluated against the same random scenarios.
    shared_demand = np.random.default_rng(123).poisson(
        lam=50,
        size=(52, 500),
    )

    # Objective function
    def inventory_objective(x):

        _Q, _R = map(int, x)

        _results = threshold_simulator.simulate(
            I0=63,
            T=52,
            N=500,
            demand=shared_demand,
            policy=fixed_lot_policy(
                Q=_Q,
                reorder_point=_R,
            ),
        )

        return threshold_simulator.cost(
            _results,
            per_period=True,
        )

    # Optimize
    inventory_optimum = opt.differential_evolution(
        inventory_objective,
        bounds=[(5, 500), (0, 300)],
        integrality=[True, True],
        seed=123,
    )
    assert inventory_optimum.success
    print("Optimal Q:", inventory_optimum.x[0])
    print("Optimal R:", inventory_optimum.x[1])
    print("Expected cost per period:", inventory_optimum.fun)
    return inventory_optimum, shared_demand, threshold_simulator


@app.cell
def _(shared_demand, threshold_simulator):
    threshold_result = threshold_simulator.simulate(
            I0=63,
            T=52,
            N=500,
            demand=shared_demand,
            policy=fixed_lot_policy(
                Q=49,
                reorder_point=10,
            ),
        )
    return (threshold_result,)


@app.cell(hide_code=True)
def _(inventory_optimum, mo, threshold_result, threshold_simulator):
    _threshold_summary = threshold_simulator.summarize(threshold_result)
    mo.md(rf"""
    For these simulated scenarios, SciPy proposes $Q={int(inventory_optimum.x[0])}$ units and a reorder threshold $R={int(inventory_optimum.x[1])}$ units. The estimated cost is €{_threshold_summary["mean_cost_per_period"]:.2f} per week, and the fill rate is {_threshold_summary["fill_rate"]:.1%}.

    This is an estimated optimum for the chosen cost parameters, the finite 52-week horizon, the specified bounds, and the fixed-lot policy class. `differential_evolution` does not guarantee a global optimum, and the result should ideally be checked on fresh demand scenarios. The value is not directly comparable to classical EOQ because the simulation permits lost sales and uses end-of-period holding costs.
    """)
    return


@app.cell
def _(threshold_result, threshold_simulator):
    for k, v in threshold_simulator.summarize(
        threshold_result
    ).items():
        print(k, v)
    return


@app.cell
def _(threshold_result, threshold_simulator):
    threshold_simulator.plot(
        threshold_result
    )
    return


@app.cell(hide_code=True)
def _(np, plt, shared_demand, threshold_simulator):

    # Grid of order quantities and reorder thresholds
    _surface_Q_values = np.arange(10, 131, 6)
    _surface_R_values = np.arange(0, 101, 5)

    # Use the same demand realizations for every policy.
    # Use 100 paths to keep plotting reasonably fast.
    _surface_demand = shared_demand[:, :100]
    _surface_T, _surface_N = _surface_demand.shape

    # Evaluate the cost function on the grid
    _surface_costs = np.zeros((
        len(_surface_R_values),
        len(_surface_Q_values),
    ))

    for _surface_i, _surface_R in enumerate(_surface_R_values):
        for _surface_j, _surface_Q in enumerate(_surface_Q_values):

            _surface_results = threshold_simulator.simulate(
                I0=63,
                T=_surface_T,
                N=_surface_N,
                demand=_surface_demand,
                policy=fixed_lot_policy(
                    Q=int(_surface_Q),
                    reorder_point=int(_surface_R),
                ),
            )

            _surface_costs[_surface_i, _surface_j] = (
                threshold_simulator.cost(
                    _surface_results,
                    per_period=True,
                )
            )

    # Meshgrid for plotting
    _surface_Q_grid, _surface_R_grid = np.meshgrid(
        _surface_Q_values,
        _surface_R_values,
    )

    # Find the best point evaluated on the grid
    _surface_best_idx = np.unravel_index(
        np.argmin(_surface_costs),
        _surface_costs.shape,
    )

    _surface_best_R = _surface_R_values[_surface_best_idx[0]]
    _surface_best_Q = _surface_Q_values[_surface_best_idx[1]]
    _surface_best_cost = _surface_costs[_surface_best_idx]

    # Create figure
    inventory_cost_surface_fig = plt.figure(figsize=(15, 6))

    # --------------------------------------------------
    # 1. Three-dimensional cost surface
    # --------------------------------------------------

    _surface_ax1 = inventory_cost_surface_fig.add_subplot(
        1, 2, 1, projection="3d",
        computed_zorder=False,
    )

    _surface_ax1.plot_surface(
        _surface_Q_grid,
        _surface_R_grid,
        _surface_costs,
        cmap="viridis",
        alpha=0.9,
        edgecolor="none",
    )

    _surface_ax1.scatter(
        _surface_best_Q,
        _surface_best_R,
        _surface_best_cost,
        color="red",
        edgecolor="white",
        s=70,
        marker="*",
        depthshade=False,
        label="Best grid point",
        zorder=10
    )

    _surface_ax1.set_xlabel("Order quantity Q")
    _surface_ax1.set_ylabel("Reorder threshold R")
    _surface_ax1.set_zlabel("Expected cost (€/week)")
    _surface_ax1.set_title("Simulated Inventory Cost Surface")
    _surface_ax1.view_init(elev=25, azim=-100)
    _surface_ax1.legend(loc="upper right")

    # --------------------------------------------------
    # 2. Contour plot
    # --------------------------------------------------

    _surface_ax2 = inventory_cost_surface_fig.add_subplot(
        1, 2, 2
    )

    _surface_contour = _surface_ax2.contourf(
        _surface_Q_grid,
        _surface_R_grid,
        _surface_costs,
        levels=25,
        cmap="viridis",
    )

    _surface_ax2.scatter(
        _surface_best_Q,
        _surface_best_R,
        color="red",
        marker="*",
        s=180,
        edgecolor="white",
        linewidth=0.8,
        label=(
            f"Best grid point: "
            f"Q={_surface_best_Q}, R={_surface_best_R}"
        ),
    )

    _surface_ax2.set_xlabel("Order quantity Q (units)")
    _surface_ax2.set_ylabel("Reorder threshold R (units)")
    _surface_ax2.set_title("Cost Contours")
    _surface_ax2.legend(loc="upper right")

    inventory_cost_surface_fig.colorbar(
        _surface_contour,
        ax=_surface_ax2,
        label="Estimated expected cost (€/week)",
    )

    inventory_cost_surface_fig.tight_layout()
    inventory_cost_surface_fig

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The figures illustrate how the estimated expected inventory cost changes with the order quantity $Q$ and the reorder threshold $R$. The left figure shows the three-dimensional cost surface, while the right figure presents the same information as a contour plot.

    The red marker indicates the lowest cost among the evaluated combinations of $Q$ and $R$. Notice that the cost surface is not necessarily smooth or convex, since small changes in the decision variables can affect the timing and number of orders. This makes finding the optimal ordering policy more challenging than in the classical EOQ model.
    """)
    return


if __name__ == "__main__":
    app.run()
