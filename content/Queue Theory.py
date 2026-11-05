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

    !!! theorem "Little's law"
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

    The following example uses hypothetical data for a timber-unloading station. Keep the service rate fixed at $\mu=10$ trucks/hour and change the average arrival rate. Observe how congestion increases as the server approaches full utilization.*
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


if __name__ == "__main__":
    app.run()
