import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Probability and Dynamic Systems

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
    In this part of the course, we are going to formulate models of inventories and queues. We will begin with simple models that involve only a few parameters. Some inventory problems can even be understood without introducing uncertainty. These simpler models will serve as benchmarks when we move on to more realistic ones.

    To describe more realistic inventory and queueing problems, we will need some additional tools, particularly from probability theory. We will therefore review some basic concepts and introduce a few probability distributions that will be useful later in the course.

    ## Decisions under uncertainty

    We have already considered optimization problems with several possible decisions, such as choosing transport routes in a food-distribution network. Specifically in Food Netrwork cont'd we had uncertainty over the possible routes. In that case it was still possible to represent the uncertainty within the linear model. Linear models can do many things and certain types of uncertainties can be added to those. A different situation arises when something important is unknown at the time we make our decision. For example, we might have to order products before we know how many customers will want to buy them.


    ### Newsvendor problem

    Consider a shop that sells newspapers. You need to decide how many newspapers to order before you know tomorrow's demand. Based on past experience, you believe demand can be anywhere between 0 and 100 newspapers. For simplicity, we assume that all values in this interval are equally likely. In other words, demand follows a uniform distribution:

    \[
        D\sim\mathcal{U}(0,100).
    \]

    Strictly speaking, newspapers can only be sold in whole units, so actual demand is discrete. We are using a continuous uniform distribution as an approximation because it makes the calculations easier. In this example, the optimal quantity will turn out to be a whole number anyway.

    Each newspaper costs us \(c=2\) € to buy, and we can sell it for \(p=5\) €. Any unsold newspapers have no value the next day, and we cannot restock during the day. How should we choose the order quantity \(q\)?

    Once we decide on \(q\), we must pay \(cq\) regardless of how many customers arrive. The purchase cost is therefore known when we place the order. Revenue, on the other hand, depends on the uncertain demand \(D\). We cannot sell more than we have ordered, and we cannot sell more than customers want to buy. Consequently, the number of newspapers sold is \(\min\{q,D\}\), and the realized profit is

    \[
        \pi(q,D)=p\min\{q,D\}-cq.
    \]

    Since we do not know the realized demand when choosing \(q\), we will maximize expected profit:

    \[
        \mathbb{E}[\pi(q,D)]
        =p\,\mathbb{E}[\min\{q,D\}]-cq.
    \]

    We do not need to consider ordering more than 100 newspapers. Demand cannot exceed 100 in this model, so ordering more than that could only increase costs without increasing sales. We can therefore restrict our decision to \(0\leq q\leq100\).

    To calculate expected sales, we use the probability density function (pdf) of the uniform distribution:

    \[
        f_D(x)=
        \begin{cases}
            \frac{1}{100}, & 0\leq x\leq100,\\
            0, & \text{otherwise}.
        \end{cases}
    \]

    The expected number of newspapers sold is

    \[
        \mathbb{E}[\min\{q,D\}]
        =\int_0^{100}\min\{q,x\}\,f_D(x)\,dx.
    \]

    The minimum inside the integral requires a little care. If demand \(x\) is below the order quantity \(q\), we sell only \(x\) newspapers. If demand exceeds \(q\), we sell all \(q\) newspapers. This allows us to split the integral into two parts:

    \[
        \begin{aligned}
        \mathbb{E}[\min\{q,D\}]
        &=\frac{1}{100}\left(\int_0^q x\,dx+\int_q^{100}q\,dx\right)\\
        &=\frac{1}{100}\left(\frac{q^2}{2}+q(100-q)\right)\\
        &=q-\frac{q^2}{200}.
        \end{aligned}
    \]

    We can now write the optimization problem without an expectation or an integral:

    \[
        \begin{aligned}
        \max_{0\leq q\leq100}\;\mathbb{E}[\pi(q,D)]
        &=\max_{0\leq q\leq100}
            \left\{p\left(q-\frac{q^2}{200}\right)-cq\right\}\\
        &=\max_{0\leq q\leq100}
            \left\{3q-\frac{q^2}{40}\right\}.
        \end{aligned}
    \]

    The objective function is a quadratic polynomial rather than a linear function. Its graph is a downward-opening parabola, as we can see below.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    profit_fig, profit_ax = plt.subplots()

    q = np.linspace(0,100, 250)
    pi = lambda q: 5*(200*q-q**2)/200-2*q

    profit_ax.plot(
        q, pi(q)
    )
    profit_ax.scatter(
        [60], [pi(60)],
        s=75,
        marker="*",
        color="red",
        zorder=5
    )
    profit_ax.text(69, 89, r'$\mathbb{E}[\pi(60)]=90$', color="red")
    profit_ax.set_ylabel(r"$\mathbb{E}[\pi(q)]$ (€)")
    profit_ax.set_xlabel(r"$q$")
    profit_ax.set_title(r"Expected profit function $\mathbb{E}[\pi(q)]=\frac{p(200q-q^2)}{200}-cq$")
    profit_ax.set_xlim(0,100)
    profit_ax.set_ylim(bottom=0)

    profit_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can solve this optimization problem using the same idea we have used for other differentiable functions: find where the derivative is zero. For the expected profit function,

    \[
        \begin{aligned}
        \frac{d}{dq}\mathbb{E}[\pi(q,D)]
        &=3-\frac{q}{20}=0\\
        &\Longrightarrow\quad q^*=60.
        \end{aligned}
    \]

    Because the second derivative is negative,

    \[
        \frac{d^2}{dq^2}\mathbb{E}[\pi(q,D)]=-\frac{1}{20}<0,
    \]

    the function is strictly concave. Since \(q^*=60\) is also within the allowed interval, it gives the unique maximum. The optimal order quantity is therefore 60 newspapers. Notice that our continuous approximation has produced a whole-number solution.

    The corresponding expected profit is

    \[
        \mathbb{E}[\pi(60,D)]
        =3\times60-\frac{60^2}{40}
        =90\text{ €}.
    \]

    This is an expected profit, not the profit we will necessarily make tomorrow. For example, if demand is zero, we lose \(2\times60=120\) €. If demand is at least 60, we sell all our newspapers and earn \(5\times60-120=180\) €.

    This simple model illustrates why probability theory is useful for decision-making. Even though we do not know tomorrow's demand, we can use its probability distribution to compare different order quantities. In this continuous-demand example, taking the expected value leads to a nonlinear optimization problem. However, uncertainty does not automatically rule out linear programming: for example, a version of the newsvendor problem with finitely many demand scenarios can also be formulated as a linear program.

    Maximizing expected profit still does not tell us everything about the possible outcomes. How much might daily profits vary around the expected value? What is the probability of making a loss? To answer such questions, we need concepts such as variance, standard deviation, and probability distributions.

    ???- question "Uncertainty in our profit ☠️☠️☠️"
        In the previous example, we found that the optimal order quantity is \(q^*=60\) newspapers, yielding an expected profit of €90.

        However, the actual profit depends on the uncertain demand \(D\sim\mathcal U(0,100)\).

        Compute the variance and standard deviation of the profit when ordering 60 newspapers. What do these results tell us about the uncertainty associated with the optimal decision?

        *Hint:* Recall that the variance of a random variable \(X\) can be calculated using

        \[
            \operatorname{Var}(X)=\mathbb E[X^2]-\mathbb E[X]^2.
        \]

        You may find it useful to express the profit as

        \[
            \pi(60,D)=5\min\{60,D\}-120.
        \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Probability theory

    The previous example used a probability distribution and an expected value. We will now take a closer look at the concepts behind those calculations.

    ### Random variables

    A random experiment can have several possible outcomes. The set of all possible outcomes is called the sample space, denoted by \(\Omega\). An event is a collection of outcomes, that is, a subset of \(\Omega\).

    A random variable assigns a numerical value to each possible outcome. For example, the outcome of a day could include many details about customers and their purchases, while the variable \(D\) records only the number of newspapers customers want to buy.

    !!! definition "Random variable"
        A (real-valued) random variable \(X\) is a function

        \[
            X:\Omega\to\mathbb R,
        \]

        which assigns a real number \(X(\omega)\) to each outcome \(\omega\in\Omega\).

    For the continuous newsvendor model, one convenient choice is \(\Omega=[0,100]\) with \(D(\omega)=\omega\). In this representation, the sample space is \([0,100]\), the codomain of \(D\) is \(\mathbb R\), and the set of values actually taken by \(D\) is \([0,100]\). The last set is called the range of \(D\). These sets do not have to coincide in general.

    Probabilities are assigned to events, not directly to functions. Let \(\mathbb P\) denote the probability measure that assigns a probability to each event we consider.

    !!! definition "Probability of a random variable taking values in a set"
        Let \(X:\Omega\to\mathbb R\) be a random variable and let \(S\subseteq\mathbb R\) be a set for which the probability is defined. Then

        \[
            \mathbb P(X\in S)
            =\mathbb P\bigl(\{\omega\in\Omega\mid X(\omega)\in S\}\bigr).
        \]

    In other words, the probability that \(X\) belongs to \(S\) is the probability of all outcomes that would make \(X\) fall in that set. For the newspaper demand, for example,

    \[
        \mathbb P(50\leq D\leq100)=\frac{50}{100}=0.5.
    \]

    We can also ask about values outside the range of the random variable. Since \(D\sim\mathcal U(0,100)\), we have \(\mathbb P(150\leq D\leq200)=0\). Notice also that the probability of any single exact value, such as \(D=50\), is zero for a continuous uniform distribution, even though intervals can have positive probability.

    Although a random variable is a mathematical function, the value we observe depends on which outcome \(\omega\) occurs. The function itself is fixed. It is the outcome of the experiment that is uncertain. One might still wonder how the name *random variable* was chosen. I would guess quite randomly.

    ### Discrete random variables

    Two common types of random variables are discrete and continuous. We will focus on these types in this course, although not every probability distribution falls neatly into just one of these categories.

    A random variable is discrete if its values lie in a finite or countable set (with probability one). For example, such a set could be \(\{0,1\}\), \(\{1,2,3,4,5,6\}\), or even the set of all integers \(\mathbb Z\). What matters here is the set of values taken by the random variable, not whether the sample space itself is finite or countable.

    For a discrete random variable, we can describe the probabilities of individual values with a **probability mass function** (pmf).

    !!! definition "Probability mass function"
        The probability mass function of a discrete random variable \(X\) is

        \[
            p_X(a)=\mathbb P(X=a).
        \]

        If \(A\) denotes the finite or countable set of possible values of \(X\), then

        \[
            p_X(a)\geq0\qquad\text{and}\qquad
            \sum_{a\in A}p_X(a)=1.
        \]

    Another important concept is the **cumulative distribution function** (cdf). Unlike the pmf, the cdf is defined for both discrete and continuous random variables.

    !!! definition "Cumulative distribution function"
        The cumulative distribution function \(F_X\) of a real-valued random variable \(X\) is

        \[
            F_X(x)=\mathbb P(X\leq x),\qquad x\in\mathbb R.
        \]

        Every cdf has the following properties:

        1. \(0\leq F_X(x)\leq1\), and \(F_X\) is nondecreasing in \(x\).
        2. \(\lim_{x\to-\infty}F_X(x)=0\) and \(\lim_{x\to\infty}F_X(x)=1\).
        3. \(F_X\) is right-continuous: its value at a point agrees with the limit approached from the right.

    For a discrete random variable, we can compute its cdf by adding together the probabilities of all possible values up to \(x\):

    \[
        F_X(x)=\sum_{\substack{a\in A\\a\leq x}}p_X(a).
    \]

    Consider a fair six-sided die. Each outcome \(1,\ldots,6\) has probability \(1/6\). Thus, for example, \(F_X(3)=\mathbb P(X\leq3)=3/6=0.5\). The pmf and cdf are shown below. Notice that the cdf stays constant between consecutive integer outcomes and jumps at each possible result of the die roll.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    # Outcomes of a fair six-sided die
    die_outcomes = np.arange(1, 7)

    # Probability mass function and cumulative distribution
    die_pmf = np.full(6, 1 / 6)
    die_cdf = np.cumsum(die_pmf)

    die_fig, die_axes = plt.subplots(1, 2, figsize=(10, 4))

    # PMF
    die_axes[0].bar(die_outcomes, die_pmf, width=0.6)
    die_axes[0].set(
        title="Probability mass function (PMF)",
        xlabel="Outcome",
        ylabel="Probability",
        xticks=die_outcomes,
        ylim=(0, 0.2),
    )

    # CDF: right-continuous step function
    die_axes[1].step(
        np.arange(0, 8),
        np.r_[0, die_cdf, 1],
        where="post",
        linewidth=2,
        color="C0",
    )

    # Filled circles: actual CDF values
    die_axes[1].scatter(
        die_outcomes,
        die_cdf,
        s=55,
        color="C0",
        zorder=5,
    )

    # Open circles: left-hand limits at each jump
    die_axes[1].scatter(
        die_outcomes,
        np.r_[0, die_cdf[:-1]],
        s=55,
        facecolors=die_axes[1].get_facecolor(),
        edgecolors="C0",
        linewidths=1.8,
        zorder=6,
    )

    die_axes[1].set(
        title="Cumulative distribution function (CDF)",
        xlabel="Outcome",
        ylabel="Cumulative probability",
        xticks=np.arange(0, 8),
        xlim=(0, 7),
        ylim=(0, 1.05),
    )

    die_fig.tight_layout()
    die_fig

    return


if __name__ == "__main__":
    app.run()
