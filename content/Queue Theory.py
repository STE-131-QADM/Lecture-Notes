import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Queue Theory

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

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the previous chapters, we considered inventory models in which demand could be uncertain. Queueing theory addresses a related question: what happens when customers or jobs arrive randomly, but the system has limited capacity to serve them?

    For example, trucks arriving at a timber-processing facility may have to wait if its unloading station is occupied. Even when a station can serve more trucks per hour than arrive *on average*, queues can form because arrivals and service times are uncertain. Here, a *customer* can be a person, a vehicle, or a job waiting to be processed.

    In this chapter we are using *Kendall notation* for the queue processes. This means that we describe the queue by \(I/F/S/N\) where each letter denotes

    + \(I\): Arrival distribution
    + \(F\): Service distribution
    + \(S\): Number of servers
    + \(N\): Number of customers allowed in a system

    This notation could be futher expanded for more complicated queues. In our models \(I\) will take value of \(M\) denoting Poisson processes. \(F\) will take values of \(M\) or \(G\) where \(G\) denotes general processes. \(S\) and \(N\) are numbers. If the number of customers allowed in a system is not limited, it will be dropped from the notation.

    ## The $M/M/1$ queue

    We begin with one of the simplest queueing models, the $M/M/1$ queue. The first $M$ indicates Markovian (Poisson-process) arrivals, the second $M$ indicates exponentially distributed service times, and $1$ indicates a single server.

    We make the following assumptions:

    1. Customers arrive according to a Poisson process with constant rate $\lambda$. If $A(t)$ is the number of arrivals by time $t$, the number arriving during an interval of length $h$ has distribution \(A(t+h)-A(t)\sim \operatorname{Poisson}(\lambda h)\). Arrivals in disjoint intervals are independent. Equivalently, the times between successive arrivals are independent and exponentially distributed with rate $\lambda$.
    2. Service times are independent and exponentially distributed with rate $\mu$. Their expected duration is $1/\mu$. Service times are also independent of arrivals.
    3. There is one server and an unlimited waiting area. Arriving customers join the queue if the server is busy, and nobody leaves without service.
    4. Customers are served in first-come, first-served order. The rates $\lambda$ and $\mu$ do not change over time.

    The two rates must use consistent time units. For example, $\lambda=8$ trucks/hour and $\mu=10$ trucks/hour correspond to a mean time between arrivals of $1/8$ hour and a mean service time of $1/10$ hour.

    ### The number of customers as a Markov chain

    Let $N(t)$ denote the number of customers in the system at time $t$, including both customers waiting and the customer being served. We say that the system is in state $n$ when $N(t)=n$.

    Recall that the exponential distribution has the memoryless property so the remaining time until an event does not depend on how much time has already elapsed. Together with the independence assumptions, this means that the future evolution of $N(t)$ depends only on the current number of customers, not on the history of the system. We do not need to know how long the current customer has already been receiving service.

    This is the Markov property. Hence, $N(t)$ is a continuous-time Markov chain on the state space $\{0,1,2,\ldots\}$. It is also called a birth–death process, since arrivals increase the state by one and service completions decrease it by one.

    - From any state $n\geq0$, an arrival changes the state to $n+1$ at rate $\lambda$.
    - From any state $n\geq1$, a service completion changes the state to $n-1$ at rate $\mu$.
    - When $n=0$, the server is idle, so a service completion is not possible.

    The arrows in the diagram below are labeled by *transition rates*, not probabilities. For a short interval of length $\Delta t$, the probability of an arrival is approximately $\lambda\Delta t$, while the probability of a service completion, when the server is busy, is approximately $\mu\Delta t$.
    """)
    return


@app.cell(hide_code=True)
def _(plt):
    # Transition-rate diagram. All variable names are local to this cell.
    _ctmc_states = 5
    _ctmc_spacing = 2.6
    _ctmc_x = [_ctmc_spacing * n for n in range(_ctmc_states)]
    _ctmc_fig, _ctmc_ax = plt.subplots(figsize=(11.5, 3.0))

    for _ctmc_n, _ctmc_xn in enumerate(_ctmc_x):
        _ctmc_circle = plt.Circle(
            (_ctmc_xn, 0), 0.35,
            facecolor="#f8fafc", edgecolor="#334155",
            linewidth=1.6, zorder=5,
        )
        _ctmc_ax.add_patch(_ctmc_circle)
        _ctmc_ax.text(
            _ctmc_xn, 0, str(_ctmc_n),
            ha="center", va="center", fontsize=13, zorder=6,
        )

    for _ctmc_i in range(_ctmc_states - 1):
        _ctmc_left, _ctmc_right = _ctmc_x[_ctmc_i:_ctmc_i + 2]
        _ctmc_mid = (_ctmc_left + _ctmc_right) / 2

        _ctmc_ax.annotate(
            "", xy=(_ctmc_right - 0.42, 0.15),
            xytext=(_ctmc_left + 0.42, 0.15),
            arrowprops=dict(
                arrowstyle="-|>", color="#2563eb",
                lw=1.9, mutation_scale=15,
            ),
        )
        _ctmc_ax.annotate(
            "", xy=(_ctmc_left + 0.42, -0.15),
            xytext=(_ctmc_right - 0.42, -0.15),
            arrowprops=dict(
                arrowstyle="-|>", color="#d97706",
                lw=1.9, mutation_scale=15,
            ),
        )
        _ctmc_ax.text(
            _ctmc_mid, 0.42, r"$\lambda$",
            color="#2563eb", fontsize=13,
            ha="center", va="bottom",
        )
        _ctmc_ax.text(
            _ctmc_mid, -0.42, r"$\mu$",
            color="#d97706", fontsize=13,
            ha="center", va="top",
        )

    _ctmc_ax.text(
        _ctmc_x[-1] + 0.9, 0, r"$\cdots$",
        ha="center", va="center", fontsize=24,
    )
    _ctmc_ax.set_xlim(-0.65, _ctmc_x[-1] + 1.7)
    _ctmc_ax.set_ylim(-0.95, 0.95)
    _ctmc_ax.set_aspect("equal", adjustable="box")
    _ctmc_ax.axis("off")
    _ctmc_ax.set_title(r"State transitions of an $M/M/1$ queue", pad=12)
    _ctmc_fig.tight_layout()
    _ctmc_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stationary probabilities and balance equations

    We are interested in the long-run probability of finding exactly $n$ customers in the system. If the state probabilities converge to a stationary distribution, we denote them by

    $$
    P_n=\lim_{t\to\infty}\mathbb{P}(N(t)=n),\qquad n=0,1,2,\ldots.
    $$

    We will first find a possible form for these probabilities, and then determine when a stationary distribution actually exists.

    Consider state $0$. The system can enter this state only when service is completed in state $1$. When the system is in state $1$, service completions occur at rate $\mu$, and the system spends a fraction $P_1$ of the long run in that state. Thus, the long-run average rate of transitions from state $1$ to state $0$ is $\mu P_1$.

    Similarly, the long-run average rate of transitions out of state $0$ is $\lambda P_0$. In a stationary system, the average rate of transitions *into* a state must equal the average rate *out of* it. Therefore,

    $$
    \lambda P_0=\mu P_1.
    $$

    State $1$ can be entered through an arrival in state $0$ or a service completion in state $2$. It can be left through an arrival or a service completion. The same balance argument gives

    $$
    \lambda P_0+\mu P_2=(\lambda+\mu)P_1.
    $$

    More generally, for each state $n\geq1$,

    $$
    \lambda P_{n-1}+\mu P_{n+1}=(\lambda+\mu)P_n.
    $$

    These are called the *stationary balance equations*. Written as a system, they have a repeated pattern:

    $$
    \begin{array}{rcrcrcrcrcl}
    -\lambda P_0 &+&\mu P_1&&&&&&&=&0\\
    \lambda P_0 &-&(\lambda+\mu)P_1 &+&\mu P_2&&&&&=&0\\
    &&\lambda P_1 &-&(\lambda+\mu)P_2 &+&\mu P_3&&&=&0\\
    &&&&\lambda P_2 &-&(\lambda+\mu)P_3 &+&\mu P_4&=&0\\
    &&&&\ddots&&\ddots&&\ddots&&\vdots
    \end{array}
    $$

    Although this system has infinitely many equations, it is possible to find a simple expression for all the unknown probabilities.

    From the first balance equation,

    $$
    P_1=\frac{\lambda}{\mu}P_0.
    $$

    Substituting into the second equation,

    $$
    \begin{aligned}
    \lambda P_0+\mu P_2
    &=(\lambda+\mu)\frac{\lambda}{\mu}P_0\\
    &=\frac{\lambda^2}{\mu}P_0+\lambda P_0,\\[4pt]
    P_2&=\left(\frac{\lambda}{\mu}\right)^2P_0.
    \end{aligned}
    $$

    By continuing in the same way, we obtain

    $$
    P_n=\left(\frac{\lambda}{\mu}\right)^nP_0,\qquad n\geq0.
    $$

    !!! note "An alternative way to see the pattern"
        In the long run, crossings from the states $\{0,\ldots,n\}$ to the states $\{n+1,n+2,\ldots\}$ must balance crossings in the opposite direction. Only two transitions cross this boundary, so $\lambda P_n=\mu P_{n+1}$. Repeatedly applying $P_{n+1}=(\lambda/\mu)P_n$ gives the same expression.

    Let us denote the ratio of the arrival and service rates by

    $$
    \varrho=\frac{\lambda}{\mu}.
    $$

    The quantity $\varrho$ is the *utilization* of the server when the system is stable: it is the long-run proportion of time the server is busy.

    All stationary probabilities must add up to one. Hence,

    $$
    \begin{aligned}
    1&=\sum_{n=0}^{\infty}P_n\\
     &=P_0\sum_{n=0}^{\infty}\varrho^n\\
     &=\frac{P_0}{1-\varrho}.
    \end{aligned}
    $$

    The geometric series converges only if $\varrho<1$, or equivalently $\lambda<\mu$. This is the *stability condition* of the $M/M/1$ queue. When it holds,

    $$
    P_0=1-\varrho,\qquad
    P_n=(1-\varrho)\varrho^n,\qquad n\geq0.
    $$

    This is a geometric distribution on $\{0,1,2,\ldots\}$ with parameter $1-\varrho$. Notice that $P_0$ is also the long-run probability that the server is idle.

    If $\lambda\geq\mu$, no stationary probability distribution exists: customers arrive at least as quickly as the single server can handle them on average. At $\lambda=\mu$ the system is critically loaded; for $\lambda>\mu$ the backlog tends to grow without bound.

    ???- question "Why does the general expression hold? ☠️☠️"
        Starting from $P_1=(\lambda/\mu)P_0$, use the balance equations to derive $P_2$ and $P_3$. Then explain why the relationship $P_{n+1}=\varrho P_n$ holds for every $n\geq0$ in stationarity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Properties of the $M/M/1$ queue

    The stationary distribution allows us to calculate four important performance measures:

    - $L$: expected number of customers in the system (including service).
    - $L_Q$: expected number waiting in the queue (excluding service).
    - $W$: expected total time a customer spends in the system.
    - $W_Q$: expected time spent waiting *before* service.

    The following expressions assume $\lambda<\mu$ and that the system is in its stationary regime. We first calculate $L$, using the expected value of the geometric distribution:

    $$
    \begin{aligned}
    L&=\mathbb{E}[N(t)]\\
     &=\sum_{n=0}^{\infty}nP_n\\
     &=\sum_{n=0}^{\infty}n(1-\varrho)\varrho^n\\
     &=(1-\varrho)\frac{\varrho}{(1-\varrho)^2}\\
     &=\frac{\varrho}{1-\varrho}
      =\frac{\lambda}{\mu-\lambda}.
    \end{aligned}
    $$

    When there are $n\geq1$ customers in the system, one is being served and the other $n-1$ are waiting. If the system is empty, nobody is waiting. The number waiting is therefore $\max\{0,N(t)-1\}$, and its expected value is

    $$
    \begin{aligned}
    L_Q&=\mathbb{E}[\max\{0,N(t)-1\}]\\
       &=\sum_{n=1}^{\infty}(n-1)(1-\varrho)\varrho^n\\
       &=(1-\varrho)\sum_{n=0}^{\infty}n\varrho^{n+1}\\
       &=(1-\varrho)\varrho\sum_{n=0}^{\infty}n\varrho^n\\
       &=\frac{\varrho^2}{1-\varrho}
        =\frac{\lambda^2}{\mu(\mu-\lambda)}.
    \end{aligned}
    $$

    We can now derive the corresponding times from *Little's law*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    !!! theorem "Little's law (Little, J. D. C.)"
        For a stable system in the long run, the average number of customers in the system $L$, the average time spent in the system $W$, and the effective throughput (arrival) rate $\lambda_a$ satisfy

        $$
        L=\lambda_a W.
        $$

        The same reasoning applies just to the waiting queue: $L_Q=\lambda_a W_Q$. Little's law is much more general than the $M/M/1$ model; it does not require Poisson arrivals or exponential service times.


    ???- proof "Proof (sketch)"
        Consider a queueing system observed over a time interval $[0,T]$. Let $N(t)$ denote the number of customers in the system at time $t$.

        The total time accumulated by all customers in the system during this interval is

        $$
        H(T)=\int_0^T N(t)\,dt.
        $$

        For example, if three customers are in the system for two minutes, they accumulate a total of six customer-minutes.

        We can also calculate the total time from the perspective of individual customers. Let $A(T)$ denote the number of customers arriving during the observation period, and let $W_i$ denote the total time customer $i$ spends in the system.

        Then, approximately,

        $$
        H(T)\approx\sum_{i=1}^{A(T)}W_i.
        $$

        The approximation arises because some customers may already be present at time $0$, while others may still be in the system at time $T$.

        Dividing both sides by $T$, we obtain

        $$
        \begin{aligned}
        \frac{1}{T}\int_0^T N(t)\,dt
        &\approx \frac{1}{T}\sum_{i=1}^{A(T)}W_i\\
        &=\frac{A(T)}{T}
        \left(\frac{1}{A(T)}\sum_{i=1}^{A(T)}W_i\right).
        \end{aligned}
        $$

        Now consider what happens as $T\to\infty$. Under suitable stability and finite-mean assumptions, the contributions from customers at the boundaries become negligible. Furthermore,

        - The left-hand side converges to $L$, the long-run average number of customers in the system.
        - The term $A(T)/T$ converges to $\lambda_a$, the long-run effective arrival rate.
        - The term inside the parentheses converges to $W$, the average time a customer spends in the system.

        Consequently, taking the limit gives

        $$
        L=\lambda_a W.
        $$

        The same argument applies when considering only the customers waiting in the queue. Therefore, we also obtain

        $$
        L_Q=\lambda_a W_Q.
        $$

        Notice that this argument does not require arrivals to follow a Poisson process or service times to be exponentially distributed. Little's law holds much more generally, provided the required long-run averages exist and the appropriate stability conditions are satisfied.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the $M/M/1$ model with unlimited waiting space, every arrival is eventually served, and the effective throughput rate is $\lambda_a=\lambda$. Thus,

    $$
    \begin{aligned}
    W&=\frac{L}{\lambda}
      =\frac{\varrho}{\lambda(1-\varrho)}
      =\frac{1}{\mu-\lambda},\\[5pt]
    W_Q&=\frac{L_Q}{\lambda}
      =\frac{\varrho^2}{\lambda(1-\varrho)}
      =\frac{\lambda}{\mu(\mu-\lambda)}.
    \end{aligned}
    $$

    We can check both results using linearity of expectation. The total time a customer spends in the system equals the time waiting plus the service time. Since the expected service time is $1/\mu$,

    $$
    W=W_Q+\frac{1}{\mu}.
    $$

    Similarly, the total number of customers equals the number waiting plus the number receiving service. As there is one server, the number receiving service is either zero or one. In stationarity, the server is busy with probability

    $$
    \mathbb{P}(N(t)>0)=1-P_0=\varrho.
    $$

    Therefore,

    $$
    L=L_Q+\varrho.
    $$

    These relationships provide useful consistency checks for the queueing formulas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpreting utilization and congestion

    The formulas above show that the average waiting time depends not only on the arrival rate, but also on how close the arrival rate is to the service capacity. In particular,

    $$
    W=\frac{1}{\mu-\lambda}
    $$

    grows rapidly as $\lambda$ approaches $\mu$ from below. A facility can therefore have a service rate higher than its average arrival rate and still experience substantial queues.

    The following example uses hypothetical data for a timber-unloading station. Keep the service rate fixed at $\mu=10$ trucks/hour and change the average arrival rate. Observe how congestion increases as the server approaches full utilization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    queue_mm1_arrival_slider = mo.ui.slider(
        start=0.5,
        stop=9.5,
        step=0.5,
        value=8.0,
        show_value=True,
        label="Average arrival rate λ (trucks/hour)",
    )
    queue_mm1_arrival_slider
    return (queue_mm1_arrival_slider,)


@app.cell(hide_code=True)
def _(mo, queue_mm1_arrival_slider):
    _qperf_mu = 10.0
    _qperf_lambda = queue_mm1_arrival_slider.value

    _qperf_rho = _qperf_lambda / _qperf_mu
    _qperf_L = _qperf_rho / (1 - _qperf_rho)
    _qperf_Lq = _qperf_rho**2 / (1 - _qperf_rho)
    _qperf_W = 60 / (_qperf_mu - _qperf_lambda)
    _qperf_Wq = (
        60 * _qperf_lambda
        / (_qperf_mu * (_qperf_mu - _qperf_lambda))
    )

    mo.md(
        rf"""
        For $\lambda = {_qperf_lambda:.1f}$ trucks/hour and
        $\mu = {_qperf_mu:.0f}$ trucks/hour:

        | Quantity | Stationary expected value |
        |---|---:|
        | Utilization $\varrho$ | {_qperf_rho:.0%} |
        | Customers in the system $L$ | {_qperf_L:.2f} |
        | Customers waiting $L_Q$ | {_qperf_Lq:.2f} |
        | Total time $W$ | {_qperf_W:.1f} min |
        | Queue waiting time $W_Q$ | {_qperf_Wq:.1f} min |
        """
    )
    return


@app.cell(hide_code=True)
def _(np, plt, queue_mm1_arrival_slider):

    _qperf_mu = 10.0
    _qperf_lambda = queue_mm1_arrival_slider.value

    _qperf_lambdas = np.linspace(0, 9.7, 350)

    _qperf_total_minutes = 60 / (
        _qperf_mu - _qperf_lambdas
    )

    _qperf_queue_minutes = (
        60 * _qperf_lambdas
        / (_qperf_mu * (_qperf_mu - _qperf_lambdas))
    )

    # Waiting times at the selected arrival rate
    _qperf_W = 60 / (_qperf_mu - _qperf_lambda)
    _qperf_Wq = (
        60 * _qperf_lambda
        / (_qperf_mu * (_qperf_mu - _qperf_lambda))
    )

    _qperf_fig, _qperf_ax = plt.subplots(figsize=(8, 4.4))

    # Expected total time
    _qperf_ax.plot(
        _qperf_lambdas,
        _qperf_total_minutes,
        label=r"Total time $W$",
        lw=2.2,
        color="#2563eb",
    )

    # Expected waiting time in the queue
    _qperf_ax.plot(
        _qperf_lambdas,
        _qperf_queue_minutes,
        label=r"Waiting time $W_Q$",
        lw=2.2,
        color="#d97706",
    )

    # Highlight the selected arrival rate
    _qperf_ax.scatter(
        [_qperf_lambda, _qperf_lambda],
        [_qperf_W, _qperf_Wq],
        color=["#2563eb", "#d97706"],
        s=50,
        zorder=5,
    )

    _qperf_ax.axvline(
        _qperf_lambda,
        color="#64748b",
        ls="--",
        lw=1,
    )

    # Labels and formatting
    _qperf_ax.set(
        xlabel=r"Arrival rate $\lambda$ (trucks/hour)",
        ylabel="Expected time (minutes)",
        title="Waiting grows nonlinearly as utilization increases",
        xlim=(0, 9.7),
        ylim=(0, 200),
    )

    _qperf_ax.grid(alpha=0.2)
    _qperf_ax.legend()
    _qperf_fig.tight_layout()

    _qperf_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For example, increasing arrivals from $8$ to $9$ trucks/hour increases utilization from $80\%$ to $90\%$, but expected total time doubles from $30$ minutes to $60$ minutes. Thus, maximizing equipment utilization is not necessarily the same as minimizing the total cost of operating a facility.

    The stationary distribution also lets us calculate the probability of substantial congestion. For any integer $n\geq0$,

    $$
    \begin{aligned}
    \mathbb P(N(t)\geq n)
    &=\sum_{j=n}^{\infty}P_j\\
    &=(1-\varrho)\sum_{j=n}^{\infty}\varrho^j\\
    &=\varrho^n.
    \end{aligned}
    $$

    For example, if $\varrho=0.8$, the stationary probability of at least five trucks in the system (one being served and at least four waiting) is $0.8^5\approx0.328$.

    ???- question "Expected queue size and congestion probability ☠️"
        Suppose $\lambda=6$ trucks/hour and $\mu=10$ trucks/hour. Calculate $L_Q$, $W_Q$ in minutes, and the stationary probability that at least four trucks are in the system. Interpret each result.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limited space \(M/M/1/K\)

    In some sense the most obvious way to make the model more realistic in many circumstances would be to ditch the assumption about infinite space to queue. This can be done by assuing that only \(K\) customers can be in the queue at the same time.
    """)
    return


@app.cell(hide_code=True)
def _(plt):
    # Compact transition-rate diagram for an M/M/1/K queue.
    # This shows only the first few states and the final two states K-1 and K.

    _ctmc_K = 8   # maximum number of customers in the system
    _ctmc_fig, _ctmc_ax = plt.subplots(figsize=(12.5, 3.2))

    if _ctmc_K < 4:
        raise ValueError("Use K >= 4 for this compact visualization.")

    _ctmc_spacing = 2.2

    # x-locations of the visible states
    _ctmc_x0 = 0.0
    _ctmc_x1 = _ctmc_x0 + _ctmc_spacing
    _ctmc_x2 = _ctmc_x1 + _ctmc_spacing
    _ctmc_xK1 = _ctmc_x2 + 2.9 * _ctmc_spacing
    _ctmc_xK = _ctmc_xK1 + _ctmc_spacing

    _ctmc_positions = [
        (_ctmc_x0, "0"),
        (_ctmc_x1, "1"),
        (_ctmc_x2, "2"),
        (_ctmc_xK1, r"$K-1$"),
        (_ctmc_xK, r"$K$"),
    ]

    # Draw visible states
    for _ctmc_x, _ctmc_label in _ctmc_positions:
        _ctmc_circle = plt.Circle(
            (_ctmc_x, 0),
            0.35,
            facecolor="#f8fafc",
            edgecolor="#334155",
            linewidth=1.6,
            zorder=5,
        )
        _ctmc_ax.add_patch(_ctmc_circle)
        _ctmc_ax.text(
            _ctmc_x,
            0,
            _ctmc_label,
            ha="center",
            va="center",
            fontsize=13,
            zorder=6,
        )

    # Function for drawing a pair of arrows with lambda and mu labels
    def _ctmc_draw_pair(_ctmc_left, _ctmc_right):
        _ctmc_mid = (_ctmc_left + _ctmc_right) / 2

        # Arrival arrow
        _ctmc_ax.annotate(
            "",
            xy=(_ctmc_right - 0.42, 0.15),
            xytext=(_ctmc_left + 0.42, 0.15),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#2563eb",
                lw=1.9,
                mutation_scale=15,
            ),
        )

        # Service arrow
        _ctmc_ax.annotate(
            "",
            xy=(_ctmc_left + 0.42, -0.15),
            xytext=(_ctmc_right - 0.42, -0.15),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#d97706",
                lw=1.9,
                mutation_scale=15,
            ),
        )

        _ctmc_ax.text(
            _ctmc_mid,
            0.42,
            r"$\lambda$",
            color="#2563eb",
            fontsize=13,
            ha="center",
            va="bottom",
        )
        _ctmc_ax.text(
            _ctmc_mid,
            -0.42,
            r"$\mu$",
            color="#d97706",
            fontsize=13,
            ha="center",
            va="top",
        )

    # Draw the visible neighboring transitions
    _ctmc_draw_pair(_ctmc_x0, _ctmc_x1)
    _ctmc_draw_pair(_ctmc_x1, _ctmc_x2)
    _ctmc_draw_pair(_ctmc_xK1, _ctmc_xK)

    # Ellipsis in the middle
    _ctmc_ellipsis_x = (_ctmc_x2 + _ctmc_xK1) / 2
    _ctmc_ax.text(
        _ctmc_ellipsis_x,
        0,
        r"$\cdots$",
        ha="center",
        va="center",
        fontsize=24,
        color="#475569",
    )

    # Show that arrivals are blocked when the system is full
    _ctmc_ax.annotate(
        "",
        xy=(_ctmc_xK + 1.05, 0.15),
        xytext=(_ctmc_xK + 0.42, 0.15),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#94a3b8",
            lw=1.6,
            mutation_scale=15,
        ),
    )
    _ctmc_ax.text(
        _ctmc_xK + 0.75,
        0.42,
        r"blocked $\lambda$",
        color="#94a3b8",
        fontsize=12,
        ha="center",
        va="bottom",
    )

    # Optional note that K is the full-capacity state
    _ctmc_ax.text(
        _ctmc_xK,
        0.78,
        "full",
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
    )

    # Formatting
    _ctmc_ax.set_xlim(-0.7, _ctmc_xK + 1.5)
    _ctmc_ax.set_ylim(-0.95, 1.0)
    _ctmc_ax.set_aspect("equal", adjustable="box")
    _ctmc_ax.axis("off")
    _ctmc_ax.set_title(
        r"State transitions of an $M/M/1/K$ queue",
        pad=12,
    )

    _ctmc_fig.tight_layout()
    _ctmc_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use the same approach as for the infinite-capacity $M/M/1$ queue to solve for the stationary probabilities. Recall that $K$ denotes the maximum number of customers in the system, including the customer being served. When the system contains $K$ customers, new arrivals are rejected.

    The balance equations no longer continue indefinitely but instead stop at state $K$. We can write them as

    $$
    \begin{aligned}
    \lambda P_0 &= \mu P_1,\\
    \lambda P_0+\mu P_2 &= (\lambda+\mu)P_1,\\
    \lambda P_1+\mu P_3 &= (\lambda+\mu)P_2,\\
    &\;\;\vdots\\
    \lambda P_{K-2}+\mu P_K &= (\lambda+\mu)P_{K-1},\\
    \lambda P_{K-1} &= \mu P_K.
    \end{aligned}
    $$

    Notice that the last equation differs from the equations for the intermediate states. Since the system cannot contain more than $K$ customers, there is no transition from state $K$ to state $K+1$.

    Following the same reasoning as in the infinite-capacity model, the balance equations give

    $$
    P_n=\left(\frac{\lambda}{\mu}\right)^nP_0
    =\varrho^nP_0,
    \qquad n=0,\ldots,K,
    $$

    where $\varrho=\lambda/\mu$.

    We can now use the normalization condition to determine $P_0$. Since the probabilities of all possible states must sum to one,

    $$
    \begin{aligned}
    \sum_{n=0}^K P_n &= 1\\
    P_0\sum_{n=0}^K\varrho^n &= 1\\
    P_0 &= \left(\sum_{n=0}^K\varrho^n\right)^{-1}.
    \end{aligned}
    $$

    Unlike the infinite-capacity model, we do not require $\varrho<1$ for the sum to converge, since we are now considering a finite number of states.

    Using the formula for a finite geometric series, we obtain

    $$
    P_0=
    \begin{cases}
    \displaystyle\frac{1}{K+1}, & \text{if }\varrho=1,\\[6pt]
    \displaystyle\frac{1-\varrho}{1-\varrho^{K+1}}, & \text{if }\varrho\neq1.
    \end{cases}
    $$

    Consequently, the stationary probabilities are

    $$
    P_n=
    \begin{cases}
    \displaystyle\frac{1}{K+1}, & \text{if }\varrho=1,\\[6pt]
    \displaystyle\frac{1-\varrho}{1-\varrho^{K+1}}\varrho^n,
    & \text{if }\varrho\neq1,
    \end{cases}
    \qquad n=0,\ldots,K.
    $$

    Notice that, unlike the infinite-capacity $M/M/1$ queue, the finite-capacity model has a stationary distribution even when $\lambda\geq\mu$. When arrivals occur faster than customers can be served, the system will simply spend more time near its maximum capacity, and a larger proportion of arriving customers will be rejected.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Properties of the $M/M/1/K$ queue

    We can now calculate the same performance measures as for the infinite-capacity model.

    First, consider the expected number of customers in the system. For $\varrho\neq1$, we obtain

    $$
    \begin{aligned}
    L
    &= \sum_{k=0}^K kP_k\\
    &= \sum_{k=0}^K k\varrho^k
       \frac{1-\varrho}{1-\varrho^{K+1}}\\
    &= \frac{1-\varrho}{1-\varrho^{K+1}}
       \sum_{k=0}^K k\varrho^k\\
    &= \frac{1-\varrho}{1-\varrho^{K+1}}
       \frac{\varrho(1-\varrho^{K+1})
       -(K+1)\varrho^{K+1}(1-\varrho)}
       {(1-\varrho)^2}\\
    &= \frac{\varrho(1-\varrho^{K+1})
       -(K+1)\varrho^{K+1}(1-\varrho)}
       {(1-\varrho^{K+1})(1-\varrho)}\\
    &= \frac{\varrho}{1-\varrho}
       -\frac{(K+1)\varrho^{K+1}}{1-\varrho^{K+1}}.
    \end{aligned}
    $$

    Here, we used the finite geometric sum identity

    $$
    \sum_{k=0}^K k\varrho^k
    =
    \frac{\varrho(1-\varrho^{K+1})
    -(K+1)\varrho^{K+1}(1-\varrho)}
    {(1-\varrho)^2}.
    $$

    In the special case $\varrho=1$, the stationary probabilities are uniform, so

    $$
    \begin{aligned}
    L
    &=\sum_{k=0}^K kP_k\\
    &=\frac{1}{K+1}\sum_{k=0}^K k\\
    &=\frac{1}{K+1}\frac{K(K+1)}{2}\\
    &=\frac{K}{2}.
    \end{aligned}
    $$

    We can also recover the infinite-capacity $M/M/1$ model as a limiting case of the finite-capacity model. When $0\leq\varrho<1$, taking $K\to\infty$ gives

    $$
    \begin{aligned}
    \lim_{K\to\infty}L(K)
    &=\lim_{K\to\infty}
    \left(
    \frac{\varrho}{1-\varrho}
    -\frac{(K+1)\varrho^{K+1}}{1-\varrho^{K+1}}
    \right)\\
    &=\frac{\varrho}{1-\varrho}.
    \end{aligned}
    $$

    This is exactly the result obtained earlier for the infinite-capacity queue.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    !!! note "Geometric sum"
        We can compute the geometric sum by noticing that

        \[
            \begin{aligned}
                \frac{d}{d\varrho}\left(\sum_{k=0}^K \varrho^k\right) &=
                    \sum_{k=0}^K \frac{d\varrho^k}{d\varrho} \\
                    &= \sum_{k=0}^K k\varrho^{k-1}
            \end{aligned}
        \]

        and

        \[
            \begin{aligned}
                \frac{d}{d\varrho}\left(\sum_{k=0}^K \varrho^k\right) &=
                    \frac{d}{d\varrho}\left(\frac{1-\varrho^{K+1}}{1-\varrho} \right) \\
                &= \frac{1-\varrho^{K+1}-(K+1)\varrho^K(1-\varrho)}{(1-\varrho)^2}.
            \end{aligned}
        \]

        Therefore, the original sum can be written as

        \[
            \begin{aligned}
                \sum_{k=0}^K k\varrho^k &= \varrho\sum_{k=0}^K k\varrho^{k-1} \\
                &= \frac{\varrho(1-\varrho^{K+1})-(K+1)\varrho^{K+1}(1-\varrho)}{(1-\varrho)^2}.
            \end{aligned}
        \]

        For more information about geometric sums as well as the general formulations see the [Wikipedia article](https://en.wikipedia.org/wiki/Geometric_series).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation of finite capacity

    Another observation we can make is that, when $0<\varrho<1$,

    $$
    \frac{(K+1)\varrho^{K+1}}{1-\varrho^{K+1}}>0.
    $$

    Consequently,

    $$
    L_{M/M/1}>L_{M/M/1/K}.
    $$

    Thus, limiting the capacity of the system reduces the expected number of customers in the system. However, this does not of course mean that the service process has become more efficient. The difference arises because customers arriving when the system is full are not allowed to enter.

    This also affects how we calculate the expected time customers spend in the system. Recall Little's law,

    $$
    L=\lambda_a W,
    $$

    where $\lambda_a$ is the effective arrival rate of customers entering the system.

    In the infinite-capacity $M/M/1$ queue, every arriving customer enters the system, so $\lambda_a=\lambda$. However, in the finite-capacity model, some customers are rejected.

    The stationary probability that the system is full is $P_K$. Since arrivals follow a Poisson process, the probability that an arriving customer finds the system full is also $P_K$. Therefore, the effective arrival rate is

    $$
    \lambda_a=\lambda(1-P_K).
    $$

    This means that, on average, only a fraction $1-P_K$ of the arriving customers can enter the system.

    Consequently, the expected time in the system for admitted customers is

    $$
    W=\frac{L}{\lambda(1-P_K)}.
    $$

    Similarly, the expected number of customers waiting in the queue is

    $$
    L_Q=L-(1-P_0).
    $$

    This follows because there is at most one customer receiving service, and the probability that the server is busy is $1-P_0$.

    Applying Little's law to the waiting queue then gives

    $$
    W_Q=\frac{L_Q}{\lambda(1-P_K)}.
    $$

    We can also verify that

    $$
    W=W_Q+\frac{1}{\mu},
    $$

    just as in the infinite-capacity model. The total time spent in the system is the sum of waiting time and service time, and the expected service time remains $1/\mu$.

    Notice that reducing the capacity of the system does not make the server work any faster. Instead, it changes which customers are allowed to enter the system. In this model, the expected waiting time of admitted customers is reduced by limiting the number of customers that can accumulate in the queue.

    However, this does not mean that limiting queue capacity necessarily reduces congestion in reality. For example, suppose a timber-processing facility cannot admit additional trucks once its unloading area is full. The rejected trucks might wait outside the facility or travel to another processing location.

    Such behavior is not included in the standard $M/M/1/K$ model. In this model, rejected customers leave the system without receiving service. If they instead wait outside the facility, we would need another model that also accounts for this additional queue.

    Consequently, limiting queue capacity may reduce congestion inside a facility while shifting waiting costs elsewhere. In practical management problems, we may therefore need to consider both the waiting costs of admitted customers and the costs associated with rejecting customers.

    We could introduce various modifications to the $M/M/1$ model, such as limiting the capacity of the system or introducing additional servers. As long as arrivals follow a Poisson process and service times are exponentially distributed, many of these models can still be analyzed using similar methods to those we have already seen. The memoryless property allows us to describe the systems as continuous-time Markov chains and derive their stationary distributions using balance equations. The calculations may become more complicated, but closed-form solutions are available for many common models.

    We can also relax the assumptions about the arrival process or service-time distribution. This often makes the analysis more difficult, since the number of customers in the system may no longer be sufficient to describe its future evolution. In such cases, obtaining closed-form solutions may be difficult or impossible, and we may need to rely on numerical methods, approximations, or simulations.

    ???- question "Derive $L_Q$, $W$, and $W_Q$ ☠️☠️"
        Derive expressions for the expected number of customers waiting in the queue, $L_Q$, the expected total time in the system, $W$, and the expected waiting time, $W_Q$, for the $M/M/1/K$ model.

        Use the stationary probabilities and the expression for $L$ derived above.

        **Hint:** Recall that the number of customers in the system is the sum of the number waiting and the number receiving service. What is the probability that the server is busy?

        When applying Little's law, remember that the effective arrival rate is $\lambda_a=\lambda(1-P_K)$, rather than $\lambda$.

        Verify that your results satisfy

        $$
        W=W_Q+\frac{1}{\mu}.
        $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The $M/G/1$ queue

    So far, we have assumed that service times follow an exponential distribution. This assumption is convenient because the exponential distribution is memoryless, allowing us to describe the number of customers in the system using a continuous-time Markov chain.

    However, the exponential distribution may not always provide a realistic description of service times. For example, consider a facility where trucks arrive to unload materials. Some trucks may take only a few minutes to unload, while others may require considerably more time, depending on their cargo.

    We can relax the assumption about service times by considering an $M/G/1$ queue. In this model:

    - Customers arrive according to a Poisson process with rate $\lambda$.
    - Service times are independent and identically distributed according to some general distribution.
    - There is one server, customers are served in their order of arrival, and the waiting queue has unlimited capacity.

    The letter $G$ stands for *general*, meaning that we no longer need to specify a particular service-time distribution.

    Let $S$ denote the service time of a randomly selected customer. We define its expected value and variance as

    $$
    \mathbb{E}[S]=\frac{1}{\mu},
    \qquad
    \operatorname{Var}(S)=\sigma_S^2.
    $$

    Here, $\mu$ represents the average service rate, as before. The utilization of the server is

    $$
    \varrho=\lambda\mathbb{E}[S]=\frac{\lambda}{\mu}.
    $$

    For the system to have a stationary distribution, we require $\varrho<1$.

    Unlike the $M/M/1$ model, the number of customers in the system is generally not sufficient by itself to describe the future evolution of the queue as a Markov chain. This makes the analysis more complicated.

    Fortunately, we can still calculate the expected waiting time using the Pollaczek–Khinchine formula:

    $$
    W_Q=\frac{\lambda\mathbb{E}[S^2]}{2(1-\varrho)}.
    $$


    !!! theorem "Pollaczek–Khinchine formula"
        Consider an $M/G/1$ queue where customers arrive according to a Poisson process with rate $\lambda$ and service times are independent and identically distributed according to a general distribution $S$ with a finite second moment.

        Assume that customers are served in first-come, first-served order, the queue has unlimited capacity, and the stability condition

        $$
        \varrho=\lambda\mathbb{E}[S]<1
        $$

        holds.

        Then the expected time a customer spends waiting in the queue is

        $$
        W_Q=\frac{\lambda\mathbb{E}[S^2]}{2(1-\varrho)}.
        $$

        Since $\mathbb{E}[S^2]=\operatorname{Var}(S)+\mathbb{E}[S]^2$, we can equivalently write

        $$
        W_Q=
        \frac{\lambda\left(\operatorname{Var}(S)+\mathbb{E}[S]^2\right)}
        {2(1-\varrho)}.
        $$

    ???- proof "Proof (sketch)"
        Consider a customer arriving at the queue. If the server is busy, the customer must first wait for the service currently in progress to finish. The customer must also wait for all customers already in the queue to receive their service.

        Therefore, the expected waiting time consists of two parts:

        1. The expected remaining service time of the customer currently being served.
        2. The expected total service time of customers already waiting in the queue.

        Let $R$ denote the remaining service time of the customer currently being served, where $R=0$ if the server is idle.

        To calculate its expected value, consider a service that lasts $s$ units of time. During this service, the remaining service time decreases linearly from $s$ to zero. The area under this remaining-time curve is

        $$
        \int_0^s(s-t)\,dt=\frac{s^2}{2}.
        $$

        Since customers complete service at an average rate of $\lambda$ in a stable system, the long-run time average of the remaining service time is

        $$
        \mathbb{E}[R]=\frac{\lambda\mathbb{E}[S^2]}{2}.
        $$

        Here, periods when the server is idle contribute zero to the average.

        Because arrivals follow a Poisson process, an arriving customer observes the same average conditions as a randomly selected point in time. This property is known as *PASTA* (Poisson Arrivals See Time Averages).

        Consequently, an arriving customer finds, on average, $L_Q$ customers already waiting in the queue. Since their service times are independent with mean $\mathbb{E}[S]$, the expected total service time of these customers is $L_Q\mathbb{E}[S]$.

        Combining the two components of waiting time gives

        $$
        W_Q=\mathbb{E}[R]+L_Q\mathbb{E}[S].
        $$

        Substituting the expression for $\mathbb{E}[R]$, we obtain

        $$
        W_Q=
        \frac{\lambda\mathbb{E}[S^2]}{2}
        +L_Q\mathbb{E}[S].
        $$

        Recall from Little's law that $L_Q=\lambda W_Q$. Therefore,

        $$
        \begin{aligned}
        W_Q
        &=\frac{\lambda\mathbb{E}[S^2]}{2}
        +\lambda W_Q\mathbb{E}[S]\\
        W_Q-\lambda W_Q\mathbb{E}[S]
        &=\frac{\lambda\mathbb{E}[S^2]}{2}\\
        W_Q(1-\lambda\mathbb{E}[S])
        &=\frac{\lambda\mathbb{E}[S^2]}{2}\\
        W_Q(1-\varrho)
        &=\frac{\lambda\mathbb{E}[S^2]}{2}\\
        W_Q
        &=\frac{\lambda\mathbb{E}[S^2]}{2(1-\varrho)}.
        \end{aligned}
        $$

        This is the Pollaczek–Khinchine formula.

        Notice that the second moment $\mathbb{E}[S^2]$ appears because longer service times contribute disproportionately more to the remaining service time observed by arriving customers. This explains why greater variability in service times increases waiting times, even when the average service time remains unchanged.

    Here, $\mathbb{E}[S^2]$ is the second moment of the service-time distribution. Recall that the variance satisfies

    $$
    \operatorname{Var}(S)
    =\mathbb{E}[S^2]-\mathbb{E}[S]^2.
    $$

    Consequently, we can also write the expected waiting time as

    $$
    W_Q=
    \frac{\lambda\left(\sigma_S^2+\mathbb{E}[S]^2\right)}
    {2(1-\varrho)}.
    $$

    This result shows that the expected waiting time depends on both the average service time and the variance of service times.

    In particular, if we keep the arrival rate and the average service time fixed, increasing the variance of service times increases the expected waiting time in the queue.

    This is an important observation: even if two facilities have the same average service capacity, their waiting times may differ substantially if one facility has more variable processing times.

    Once we have calculated $W_Q$, we can obtain the remaining performance measures using the same relationships as before:

    $$
    \begin{aligned}
    W &= W_Q+\mathbb{E}[S],\\
    L_Q &= \lambda W_Q,\\
    L &= \lambda W.
    \end{aligned}
    $$

    These expressions follow from the expected service time and Little's law, which also applies to the $M/G/1$ model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing different service-time distributions

    Suppose trucks arrive at a processing facility at an average rate of $\lambda=8$ trucks per hour. The average service rate is $\mu=10$ trucks per hour, so the average service time is six minutes.

    The server utilization is therefore

    $$
    \varrho=\frac{8}{10}=0.8.
    $$

    Consider three possibilities for the service-time distribution:

    1. **Constant service time:** Every truck requires exactly six minutes of service.
    2. **Exponential service time:** Service times follow an exponential distribution with a mean of six minutes.
    3. **Highly variable service time:** Service times still average six minutes, but their variance is three times the squared mean service time.

    To make these cases easier to compare, define the squared coefficient of variation

    $$
    c_S^2=
    \frac{\operatorname{Var}(S)}{\mathbb{E}[S]^2}.
    $$

    This quantity measures service-time variability relative to the squared mean service time.

    The Pollaczek–Khinchine formula can now be written as

    $$
    W_Q=
    \frac{\varrho}{1-\varrho}
    \frac{1+c_S^2}{2}\mathbb{E}[S].
    $$

    For our example, this gives the following results:

    | Service-time distribution | $c_S^2$ | $W_Q$ | $W$ |
    |---|---:|---:|---:|
    | Constant | 0 | 12 min | 18 min |
    | Exponential | 1 | 24 min | 30 min |
    | Highly variable | 3 | 48 min | 54 min |

    Notice that the arrival rate, average service time, and server utilization are exactly the same in all three cases.

    Nevertheless, the expected waiting time increases from 12 minutes to 48 minutes as service-time variability increases.

    This illustrates that reducing variability in the service process may be just as important as increasing its average capacity. For example, standardizing unloading procedures, preparing materials in advance, or separating unusually complicated tasks from routine ones could help reduce waiting times.

    Of course, whether these changes are economically worthwhile depends on their implementation costs.

    ???- question "How does service-time variability affect waiting times? ☠️☠️"
        Consider two processing facilities with the same arrival rate and average service time. Both have a utilization of $\varrho=0.8$.

        At the first facility, service times are constant, while at the second facility, service times follow an exponential distribution.

        1. Which facility has a larger expected waiting time?
        2. How much larger is the expected waiting time?
        3. What practical measures could a facility take to reduce variability in its service times?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Remarks about Little's law

    Little's law is a powerful result because it allows us to determine the expected number of customers in a system from their average time in the system, or vice versa, provided we know the effective arrival rate. It also allows us to estimate the effective arrival rate from observations of the number of customers and their waiting times. Importantly, Little's law applies to a wide range of queueing systems, not just the models considered here (Little, 1961).

    For example, suppose we observe a queueing system with an average of $L=5$ customers and an average time in the system of $W=10$ minutes. Using Little's law, we obtain

    $$
    \begin{aligned}
    \lambda_a
    &=\frac{L}{W}\\
    &=\frac{5\text{ customers}}{10\text{ min}}\\
    &=0.5\text{ customers/min}.
    \end{aligned}
    $$

    Thus, the system processes one customer every two minutes on average. If we also know the average service time $\mathbb{E}[S]$, we can determine the remaining quantities:

    $$
    W_Q=W-\mathbb{E}[S],
    \qquad
    L_Q=\lambda_a W_Q.
    $$

    Consequently, Little's law is not merely an abstract result from stochastic process theory. It is also a practical tool for analyzing queueing systems using quantities that can be measured directly from observed data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example: Queues at a forestry weighbridge

    We can now apply the formulas from the $M/M/1$ and $M/G/1$ sections to data reported in a real forestry study. [McNickle and Woollons (1990), *Analysis and simulation of a logging weighbridge installation*](https://www.scionresearch.com/__data/assets/pdf_file/0009/59958/NZJFS201111MCNICKLE.pdf), measured the arrival and processing times of trucks at a forestry weighbridge in New Zealand. Their study also used queueing formulas and simulation to evaluate congestion and the possible benefits of additional capacity.

    The authors reported the following pooled observations from two working days:

    | Quantity | Mean | Standard deviation |
    |---|---:|---:|
    | Time between truck arrivals | 1.68 min | 1.74 min |
    | Weighing time | 1.21 min | 0.73 min |
    | Time for a truck to move onto the scale | 0.33 min | 0.27 min |

    The interarrival times were reasonably consistent with an exponential distribution, which supports using a Poisson arrival process as an approximation. Recall that following random variable \(E\sim\text{Exp}(\lambda)\) has expected value \(1/\lambda\) and variance \(1/\lambda^2\). This gives us easy test to see if the times approximately satisfy the relationship.  The service process requires more care: moving onto the scale takes time mainly when another truck has just left. Thus, service times are not strictly independent of whether the system is busy.

    For a simplified model, we make two additional approximations:

    1. Every truck is assigned both the weighing time and the moving-up time, even if the weighbridge was previously empty. This overstates the delay associated with moving onto the scale. The original paper uses a similar construction to obtain an upper approximation for waiting time.
    2. We treat weighing time and moving-up time as independent when calculating the variance of their sum. This is a simplifying assumption, not a result established by the published summary statistics.

    Let $f$ be the fraction of the original moving-up time that remains after a hypothetical process improvement. For example, $f=1$ represents the original process in our approximation, and $f=0$ represents eliminating this movement delay entirely. This is a hypothetical intervention, not an observed outcome in the paper.

    From the observed means and standard deviations, our simplified model has

    $$
    \begin{aligned}
    \lambda &= \frac{1}{1.68}\quad \text{trucks/min},\\
    \mathbb{E}[S_f] &= 1.21+0.33f\quad \text{min},\\
    \operatorname{Var}(S_f) &= 0.73^2+(0.27f)^2\quad \text{min}^2.
    \end{aligned}
    $$

    The expected queue waiting time is calculated with the Pollaczek–Khinchine formula. Compare it with the $M/M/1$ prediction using the same mean service time, but assuming exponentially distributed service times.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    forest_movement_fraction_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=1.0,
        show_value=True,
        label="Fraction of the original moving-up time remaining",
    )
    forest_movement_fraction_slider
    return (forest_movement_fraction_slider,)


@app.cell(hide_code=True)
def _(forest_movement_fraction_slider, mo):
    _forest_f = forest_movement_fraction_slider.value
    _forest_lambda = 1 / 1.68  # trucks per minute
    _forest_mean_s = 1.21 + 0.33 * _forest_f
    _forest_var_s = 0.73**2 + (0.27 * _forest_f)**2
    _forest_rho = _forest_lambda * _forest_mean_s
    _forest_Wq_mg1 = _forest_lambda * (
        _forest_mean_s**2 + _forest_var_s
    ) / (2 * (1 - _forest_rho))
    _forest_Wq_mm1 = (
        _forest_lambda * _forest_mean_s**2 / (1 - _forest_rho)
    )
    _forest_Lq = _forest_lambda * _forest_Wq_mg1
    _forest_wait_hours = _forest_Lq * 12
    _forest_cv2 = _forest_var_s / _forest_mean_s**2
    _forest_Lq_mm1 = _forest_lambda * _forest_Wq_mm1
    _forest_wait_hours_mm1 = _forest_Lq_mm1 * 12

    mo.md(
        rf"""
        For $f={_forest_f:.2f}$, the arrival rate is
        $\lambda={60*_forest_lambda:.2f}$ trucks/hour.

        | Performance measure | $M/G/1$ approximation | $M/M/1$ assumption |
        |---|---:|---:|
        | Mean service time | {_forest_mean_s:.2f} min | {_forest_mean_s:.2f} min |
        | Server utilization $\varrho$ | {_forest_rho:.1%} | {_forest_rho:.1%} |
        | Squared service-time variation $c_S^2$ | {_forest_cv2:.2f} | 1.00 |
        | Queue waiting time $W_Q$ | {_forest_Wq_mg1:.2f} min | {_forest_Wq_mm1:.2f} min |
        | Number waiting $L_Q$ | {_forest_Lq:.2f} trucks | {_forest_Lq_mm1:.2f} |
        | Total truck-hours waiting in a 12-hour operating day | {_forest_wait_hours:.1f} h | {_forest_wait_hours_mm1:.2f} |

        The daily total is a steady-state *illustration* assuming 12 hours of operation
        with the same parameters, not a measured daily result from the study.
        """
    )
    return


@app.cell(hide_code=True)
def _(forest_movement_fraction_slider, np, plt):
    _forest_grid = np.linspace(0.0, 1.0, 201)
    _forest_lambda = 1 / 1.68  # trucks per minute
    _forest_mean = 1.21 + 0.33 * _forest_grid
    _forest_variance = 0.73**2 + (0.27 * _forest_grid)**2
    _forest_rho = _forest_lambda * _forest_mean

    _forest_mg1_wait = (
        _forest_lambda * (_forest_mean**2 + _forest_variance)
        / (2 * (1 - _forest_rho))
    )
    _forest_mm1_wait = (
        _forest_lambda * _forest_mean**2 / (1 - _forest_rho)
    )

    _forest_selected_f = forest_movement_fraction_slider.value
    _forest_selected_Wq_mg1 = np.interp(
        _forest_selected_f, _forest_grid, _forest_mg1_wait
    )
    _forest_selected_Wq_mm1 = np.interp(
        _forest_selected_f, _forest_grid, _forest_mm1_wait
    )
    _forest_fig, _forest_ax = plt.subplots(figsize=(8.2, 4.4))
    _forest_ax.plot(
        _forest_grid, _forest_mg1_wait,
        color="#2563eb", lw=2.3, label=r"$M/G/1$ (reported service-time variation)",
    )
    _forest_ax.plot(
        _forest_grid, _forest_mm1_wait,
        color="#d97706", lw=2.0, ls="--",
        label=r"$M/M/1$ (exponential service times)",
    )
    _forest_ax.scatter(
        [_forest_selected_f], [_forest_selected_Wq_mg1],
        s=75, color="#2563eb", zorder=5,
    )
    _forest_ax.scatter(
        [_forest_selected_f], [_forest_selected_Wq_mm1],
        s=75, color="#d97706", zorder=5,
    )
    _forest_ax.axvline(
        _forest_selected_f, color="#64748b", ls=":", lw=1.2
    )
    _forest_ax.set(
        xlabel="Fraction of moving-up time remaining",
        ylabel="Expected queue waiting time (minutes)",
        title="Forestry weighbridge: a hypothetical process improvement",
        xlim=(0, 1),
        ylim=(0, 19),
    )
    _forest_ax.grid(alpha=0.2)
    _forest_ax.legend()
    _forest_fig.tight_layout()
    _forest_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpreting the forestry case

    Under the simplified baseline ($f=1$), the $M/G/1$ formula predicts approximately $10.6$ minutes of queue waiting per truck. Using an exponential service-time distribution with the same mean would instead predict approximately $16.9$ minutes. This difference illustrates why service-time variability matters.

    The published paper reports a waiting-time estimate of $10.72$ minutes for its corresponding upper approximation. Our result is slightly different because the paper reports the input means and standard deviations in rounded form. Its simulation of the actual system, which represents the movement delays more accurately, gives approximately $8.05$ minutes of waiting per truck with one weighbridge.

    Reducing the time trucks spend moving onto the scale lowers the mean service time, which also lowers utilization. This is particularly valuable when utilization is already high. Less truck waiting could also reduce vehicle idling and associated emissions, but quantifying emissions would require additional information about fuel use, engine operation, and emission factors. Queue waiting time alone is not a measured emission quantity.

    ???- question "Exercise: improving the forestry weighbridge ☠️☠️"
        1. Use the reported means to calculate the arrival rate $\lambda$, average service rate $\mu$, and utilization $\varrho$ at $f=1$.
        2. Calculate the squared coefficient of variation $c_S^2$ of the service time and use the Pollaczek–Khinchine formula to obtain $W_Q$.
        3. Set $f=0.5$. How much does the expected queue waiting time change? How much does the expected number of waiting trucks change?
        4. Use Little's law to calculate the total truck-hours spent waiting during a hypothetical 12-hour operating day.
        5. Explain why reducing queue waiting time does not necessarily imply that all waiting-related emissions disappear.

    ### Optional extension: finite entrance capacity

    Suppose that, for safety reasons, the weighbridge site can admit only $K$ trucks at a time, including the truck being weighed. We can approximate this alternative as an $M/M/1/K$ queue using the original mean service time. Note that this uses an *exponential service-time assumption* and is a different approximation from the $M/G/1$ analysis above.

    In this model, arrivals that find all $K$ spaces occupied are rejected, rather than waiting outside the entrance. The stationary blocking probability is $P_K$ and the effective throughput is $\lambda_a=\lambda(1-P_K)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    forest_capacity_slider = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=6,
        show_value=True,
        label="Maximum trucks admitted to the weighbridge system (K)",
    )
    forest_capacity_slider
    return (forest_capacity_slider,)


@app.cell(hide_code=True)
def _(forest_capacity_slider, mo):
    _forest_K = forest_capacity_slider.value
    _forest_lambda = 1 / 1.68  # arrivals/minute
    _forest_mean_s = 1.21 + 0.33
    _forest_rho = _forest_lambda * _forest_mean_s
    _forest_normalizer = sum(_forest_rho**n for n in range(_forest_K + 1))
    _forest_probs = [
        _forest_rho**n / _forest_normalizer
        for n in range(_forest_K + 1)
    ]
    _forest_P0 = _forest_probs[0]
    _forest_PK = _forest_probs[-1]
    _forest_L = sum(n * p for n, p in enumerate(_forest_probs))
    _forest_Lq = _forest_L - (1 - _forest_P0)
    _forest_lambda_a = _forest_lambda * (1 - _forest_PK)
    _forest_Wq = _forest_Lq / _forest_lambda_a

    mo.md(
        rf"""
        For capacity $K={_forest_K}$, with the original arrival rate and an
        exponential service-time approximation:

        | Quantity | Stationary value |
        |---|---:|
        | Blocking probability $P_K$ | {_forest_PK:.1%} |
        | Effective admitted arrival rate $\lambda_a$ | {60*_forest_lambda_a:.2f} trucks/hour |
        | Expected customers in the system $L$ | {_forest_L:.2f} trucks |
        | Queue waiting time $W_Q$ for admitted trucks | {_forest_Wq:.2f} min |

        A smaller capacity reduces waiting for admitted trucks, but it also
        increases the fraction of arriving trucks that are rejected.
        """
    )
    return


@app.cell(hide_code=True)
def _(forest_capacity_slider, np, plt):
    _forest_K_values = np.arange(1, 21)
    _forest_lambda = 1 / 1.68
    _forest_rho = _forest_lambda * (1.21 + 0.33)
    _forest_block = []
    _forest_wait = []

    for _forest_K in _forest_K_values:
        _forest_weights = _forest_rho ** np.arange(_forest_K + 1)
        _forest_p = _forest_weights / _forest_weights.sum()
        _forest_L = np.sum(np.arange(_forest_K + 1) * _forest_p)
        _forest_Lq = _forest_L - (1 - _forest_p[0])
        _forest_accepted_rate = _forest_lambda * (1 - _forest_p[-1])
        _forest_block.append(_forest_p[-1])
        _forest_wait.append(_forest_Lq / _forest_accepted_rate)

    _forest_selected_K = forest_capacity_slider.value
    _forest_fig, (_forest_ax1, _forest_ax2) = plt.subplots(
        2, 1, figsize=(8.2, 5.7), sharex=True
    )
    _forest_ax1.plot(
        _forest_K_values, np.array(_forest_block) * 100,
        color="#d97706", lw=2.3,
    )
    _forest_ax2.plot(
        _forest_K_values, _forest_wait,
        color="#2563eb", lw=2.3,
    )
    for _forest_ax in (_forest_ax1, _forest_ax2):
        _forest_ax.axvline(
            _forest_selected_K, ls="--", color="#64748b", lw=1.2
        )
        _forest_ax.grid(alpha=0.2)
        _forest_ax.set_xlim(1, 20)

    _forest_ax2.set_xticks(_forest_K_values)

    _forest_ax1.set(
        ylabel="Blocked arrivals (%)",
        title="A finite queue trades waiting for rejected arrivals",
    )
    _forest_ax2.set(
        ylabel="Waiting for admitted trucks (min)",
        xlabel="Maximum number of trucks in the system (K)",
    )
    _forest_fig.tight_layout()
    _forest_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ???- question "Exercise: capacity is not the same as service speed ☠️☠️"
        1. At $K=6$, calculate the expected number of trucks rejected per operating hour.
        2. Increase $K$ to 12. What happens to the rejection rate and the waiting time of admitted trucks?
        3. Would limiting the number of trucks allowed inside the weighbridge site eliminate queues if rejected trucks simply waited outside? Explain which assumption of the $M/M/1/K$ model would no longer hold.
        4. Which practical response would you recommend: reducing processing time, smoothing the arrival pattern, or restricting site capacity? Explain what additional data would be needed for a defensible decision.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Additional material

    + McNickle & Woollons (1990) [*Analysis and simulation of a logging weighbridge installation*](https://www.scionresearch.com/__data/assets/pdf_file/0009/59958/NZJFS201111MCNICKLE.pdf), New Zealand Journal of Forestry Science, 20(1), 111–119
    + Aalto, M., Korpinen, O. J., Loukola, J., & Ranta, T. (2018). Achieving a smooth flow of fuel deliveries by truck to an urban biomass power plant in Helsinki, Finland – an agent-based simulation approach. International Journal of Forest Engineering, 29(1), 21–30. [https://doi.org/10.1080/14942119.2018.1403809](https://doi.org/10.1080/14942119.2018.1403809)
    + For more queuing theory related material visit [Myron Hlynka's Queueing Theory Page](https://web2.uwindsor.ca/math/hlynka/queue.html).

    ## References
    +  Little, J. D. C. (1961). *A Proof for the Queuing Formula: L = λW*. Operations Research. 9 (3): 383–387. [doi:10.1287/opre.9.3.383](https://pubsonline.informs.org/doi/10.1287/opre.9.3.383). JSTOR 167570.
    """)
    return


if __name__ == "__main__":
    app.run()
