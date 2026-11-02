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

    Most of the content you are facing in this chapter will be something you have probably already met at the earlier stages of your education. Many of the probability theory concepts at least have been already covered during the Data Science Methodology course you have already taken in the first period. Thus the main function of this chapter is to remaind you of the basic concepts and make sure you have easy access to find the results when needed in the exercises. However, we will also cover some new topics. You should make sure you are relatively comfortable working with the basic definitions and can use the basic probabilistic tools and random variables to recognize what assumptions we need to make in our models to use these. As a test of your understanding you could use the first example. If you are confident with the derivation, understand how changing for example the distribution of the demand or some other assumption would affect and how you would solve the new model, you should have quite good understanding of the topics.

    If you don't feel comfortable working with the basic tools of probability theory you can use for example [*Introduction to Probability Models*](https://www.sciencedirect.com/book/monograph/9780124079489/introduction-to-probability-models) by Sheldon M. Ross to remaind yourself of the basic results. However, in this course, we will not be needing the knowledge of the whole book.

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

    We can check for functions concavity by computing its second derivative. Because the second derivative is negative,

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some important probability distributions that we will encounter in applied settings are the following.

    #### Bernoulli random variable

    A Bernoulli random variable \(B\) describes one of the simplest random experiments: an experiment with only two possible outcomes, which we call *success* and *failure*. We assume that success occurs with probability \(p\) and failure with probability \(1-p\).

    We represent success by \(B=1\) and failure by \(B=0\). The probability mass function (PMF) of \(B\) is therefore

    \[
    p_B(b)=
    \begin{cases}
    p, & \text{if } b=1,\\
    1-p, & \text{if } b=0,\\
    0, & \text{otherwise}.
    \end{cases}
    \]

    An example of a Bernoulli random variable is the outcome of a coin toss, where we define heads as success and tails as failure. For a fair coin, \(p=1/2\).
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):

    bern_p = 0.4
    bern_x = np.array([0, 1])
    bern_pmf = np.array([1 - bern_p, bern_p])
    bern_cdf = np.cumsum(bern_pmf)

    bern_fig, bern_axes = plt.subplots(1, 2, figsize=(10, 4))

    # PMF
    bern_axes[0].bar(bern_x, bern_pmf, width=0.5)
    bern_axes[0].set(
        title=rf"Bernoulli PMF ($p={bern_p}$)",
        xlabel="Outcome",
        ylabel="Probability",
        xticks=bern_x,
        xlim=(-0.5, 1.5),
        ylim=(0, 1.05),
    )

    # CDF
    bern_axes[1].step(
        [-1, 0, 1, 2],
        [0, bern_cdf[0], bern_cdf[1], 1],
        where="post",
        color="C0",
        linewidth=2,
    )

    # Filled circles: actual CDF values
    bern_axes[1].scatter(
        bern_x, bern_cdf,
        color="C0", s=55, zorder=5,
    )

    # Open circles: left-hand limits
    bern_axes[1].scatter(
        bern_x, [0, bern_cdf[0]],
        facecolors="white", edgecolors="C0",
        s=55, linewidths=1.8, zorder=6,
    )

    bern_axes[1].set(
        title=rf"Bernoulli CDF ($p={bern_p}$)",
        xlabel="Outcome",
        ylabel="Cumulative probability",
        xticks=bern_x,
        xlim=(-0.5, 1.5),
        ylim=(-0.05, 1.05),
    )

    bern_fig.tight_layout()
    bern_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Binomial random variable

    A binomial random variable describes a similar situation, but instead of one experiment, we now consider \(n\) independent experiments, each with the same probability \(p\) of success. The binomial random variable \(X\) counts how many successes occur in these \(n\) experiments.

    The probability of observing exactly \(k\) successes is

    \[
    p_X(k)=\binom{n}{k}p^k(1-p)^{n-k},
    \qquad k=0,1,\ldots,n.
    \]

    Each term has a straightforward interpretation. Let's start with the last term. Since there are \(n\) experiments and \(k\) successes, there must be \(n-k\) failures. The probability of failure in an individual experiment is \(1-p\), so \((1-p)^{n-k}\) gives the probability associated with these \(n-k\) failures. Similarly, \(p^k\) gives the probability associated with the \(k\) successes.

    Together, \(p^k(1-p)^{n-k}\) gives the probability of any particular sequence containing exactly \(k\) successes and \(n-k\) failures.

    However, there are several different sequences that can produce the same number of successes. The first term,

    \[
    \binom{n}{k}=\frac{n!}{k!(n-k)!},
    \]

    counts the number of ways we can choose which \(k\) of the \(n\) experiments are successes. This is called the number of [combinations](https://en.wikipedia.org/wiki/Combination).

    Multiplying the probability of one such sequence by the number of possible sequences gives the probability of observing exactly \(k\) successes.

    An example of a binomial random variable would again be coin tossing. However, instead of tossing the coin once, we consider a sequence of \(n\) independent coin tosses and count how many times the coin lands on heads.

    The next distributions will be especially important for us when we study queueing theory.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):

    binom_n = 10
    binom_p = 0.4
    binom_x = np.arange(0, binom_n + 1)

    # Binomial coefficients
    binom_coeff = np.array([
        np.prod(np.arange(binom_n - k + 1, binom_n + 1))
        / np.prod(np.arange(1, k + 1))
        for k in binom_x
    ])

    # PMF and CDF
    binom_pmf = (
        binom_coeff
        * binom_p**binom_x
        * (1 - binom_p)**(binom_n - binom_x)
    )
    binom_cdf = np.cumsum(binom_pmf)

    binom_fig, binom_axes = plt.subplots(1, 2, figsize=(10, 4))

    # PMF
    binom_axes[0].bar(binom_x, binom_pmf, width=0.7)
    binom_axes[0].set(
        title=rf"Binomial PMF ($n={binom_n},\ p={binom_p}$)",
        xlabel="Number of successes",
        ylabel="Probability",
        xticks=binom_x,
        ylim=(0, 0.3),
    )

    # CDF
    binom_axes[1].step(
        np.arange(-1, binom_n + 2),
        np.r_[0, binom_cdf, 1],
        where="post",
        color="C0",
        linewidth=2,
    )

    # Filled circles
    binom_axes[1].scatter(
        binom_x, binom_cdf,
        color="C0", s=40, zorder=5,
    )

    # Open circles
    binom_axes[1].scatter(
        binom_x, np.r_[0, binom_cdf[:-1]],
        facecolors="white", edgecolors="C0",
        s=40, linewidths=1.8, zorder=6,
    )

    binom_axes[1].set(
        title=rf"Binomial CDF ($n={binom_n},\ p={binom_p}$)",
        xlabel="Number of successes",
        ylabel="Cumulative probability",
        xticks=binom_x,
        xlim=(-0.5, binom_n + 0.5),
        ylim=(-0.05, 1.05),
    )

    binom_fig.tight_layout()
    binom_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Geometric random variable

    Consider again a sequence of independent experiments, each with probability \(p\) of success. Instead of fixing the number of experiments and counting the successes, we now ask how many experiments are needed until the first success occurs.

    A geometric random variable \(G\) describes the number of trials required to observe the first success.

    For the first success to occur on the \(n\)-th trial, the preceding \(n-1\) trials must all be failures, followed by a success on trial \(n\). The probability mass function is therefore

    \[
    p_G(n)=(1-p)^{n-1}p,
    \qquad n=1,2,\ldots.
    \]

    For example, suppose we repeatedly toss a fair coin until it lands on heads. The geometric distribution describes the number of tosses needed to observe the first head.

    Notice the difference between the binomial and geometric distributions. In the binomial setting, we fix the number of trials and count the successes. In the geometric setting, we fix the number of successes we are waiting for (one) and count how many trials are needed.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):

    geom_p = 0.3
    geom_x = np.arange(1, 21)

    # PMF and CDF
    geom_pmf = (1 - geom_p)**(geom_x - 1) * geom_p
    geom_cdf = 1 - (1 - geom_p)**geom_x

    geom_fig, geom_axes = plt.subplots(1, 2, figsize=(10, 4))

    # PMF
    geom_axes[0].bar(geom_x, geom_pmf, width=0.7)
    geom_axes[0].set(
        title=rf"Geometric PMF ($p={geom_p}$)",
        xlabel="Trial of first success",
        ylabel="Probability",
        xticks=np.arange(1, 21, 2),
        xlim=(0, 21),
        ylim=(0, 0.35),
    )

    # CDF
    geom_axes[1].step(
        np.r_[0, geom_x, 21],
        np.r_[0, geom_cdf, geom_cdf[-1]],
        where="post",
        color="C0",
        linewidth=2,
    )

    # Filled circles
    geom_axes[1].scatter(
        geom_x, geom_cdf,
        color="C0", s=35, zorder=5,
    )

    # Open circles
    geom_axes[1].scatter(
        geom_x, np.r_[0, geom_cdf[:-1]],
        facecolors="white", edgecolors="C0",
        s=35, linewidths=1.8, zorder=6,
    )

    geom_axes[1].set(
        title=rf"Geometric CDF ($p={geom_p}$)",
        xlabel="Trial of first success",
        ylabel="Cumulative probability",
        xticks=np.arange(1, 21, 2),
        xlim=(0, 21),
        ylim=(-0.05, 1.05),
    )

    geom_fig.tight_layout()
    geom_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Poisson random variable

    A Poisson random variable is commonly used to model the number of events occurring within a fixed period of time. For example, we might be interested in how many customers arrive at a shop during one hour or how many phone calls a customer service centre receives during one minute.

    Let \(N\) denote the number of events occurring during a particular time interval. If \(N\) follows a Poisson distribution with parameter \(\lambda>0\), its probability mass function is

    \[
    p_N(n)=e^{-\lambda}\frac{\lambda^n}{n!},
    \qquad n=0,1,2,\ldots.
    \]

    The parameter \(\lambda\) represents the expected number of events occurring during the considered interval.

    The Poisson distribution is closely related to a *Poisson process*, which describes how events occur over time. In a Poisson process, events occur independently across disjoint time intervals and at a constant average rate.

    If the average arrival rate is \(\lambda\) events per unit of time, the number of arrivals during an interval of length \(t\), denoted by \(N(t)\), follows

    \[
    N(t)\sim\operatorname{Poisson}(\lambda t).
    \]

    Poisson processes are particularly important in queueing theory. For example, we can use a Poisson process to model customers arriving at a service desk, where \(N(t)\) describes how many customers have arrived during the first \(t\) units of time.

    Later, we will use this type of model to examine how uncertain customer arrivals, together with the time required to serve customers, affect queue lengths and waiting times.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):

    pois_lambda = 4
    pois_max = 16
    pois_x = np.arange(0, pois_max + 1)

    # PMF computed recursively
    pois_pmf = np.exp(-pois_lambda) * np.cumprod(
        np.r_[1.0, pois_lambda / np.arange(1, pois_max + 1)]
    )
    pois_cdf = np.cumsum(pois_pmf)

    pois_fig, pois_axes = plt.subplots(1, 2, figsize=(10, 4))

    # PMF
    pois_axes[0].bar(pois_x, pois_pmf, width=0.7)
    pois_axes[0].set(
        title=rf"Poisson PMF ($\lambda={pois_lambda}$)",
        xlabel="Number of arrivals",
        ylabel="Probability",
        xticks=np.arange(0, pois_max + 1, 2),
        ylim=(0, 0.25),
    )

    # CDF
    pois_axes[1].step(
        np.r_[-1, pois_x, pois_max + 1],
        np.r_[0, pois_cdf, pois_cdf[-1]],
        where="post",
        color="C0",
        linewidth=2,
    )

    # Filled circles
    pois_axes[1].scatter(
        pois_x, pois_cdf,
        color="C0", s=40, zorder=5,
    )

    # Open circles
    pois_axes[1].scatter(
        pois_x, np.r_[0, pois_cdf[:-1]],
        facecolors="white", edgecolors="C0",
        s=40, linewidths=1.8, zorder=6,
    )

    pois_axes[1].set(
        title=rf"Poisson CDF ($\lambda={pois_lambda}$)",
        xlabel="Number of arrivals",
        ylabel="Cumulative probability",
        xticks=np.arange(0, pois_max + 1, 2),
        xlim=(-0.5, pois_max + 0.5),
        ylim=(-0.05, 1.05),
    )

    pois_fig.tight_layout()
    pois_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Continuous random variables

    So far, we have considered discrete random variables, which take values from a finite or countable set. However, many quantities we are interested in can take any value within some interval. For example, the time a customer spends waiting in a queue or the amount of rainfall during a day can be represented using continuous random variables.

    Unlike discrete random variables, which we described using probability mass functions, the continuous distributions considered here are described by **probability density functions** (pdf). We denote the probability density function of a random variable \(X\) by \(f_X(x)\).

    An important difference is that the value of the density function does not directly give us the probability of observing a particular outcome. Instead, probabilities are obtained by integrating the density function over an interval:

    \[
    \mathbb P(a\leq X\leq b)=\int_a^b f_X(x)\,dx.
    \]

    The probability density function must be nonnegative, and the total area under it must equal one:

    \[
    f_X(x)\geq0,\qquad \int_{-\infty}^{\infty}f_X(x)\,dx=1.
    \]

    Since a single point has zero width, the probability of observing any particular value is zero for a random variable with a probability density function:

    \[
    \mathbb P(X=a)=0.
    \]

    This does not mean that observing the value \(a\) is impossible. Rather, probabilities are assigned to intervals instead of individual points.

    Just as with discrete random variables, we can also define the cumulative distribution function (CDF). For a random variable with density \(f_X\), it can be calculated as

    \[
    F_X(x)=\mathbb P(X\leq x)=\int_{-\infty}^x f_X(t)\,dt.
    \]

    Let's now consider three continuous distributions that are particularly useful in applied settings.

    #### Uniform random variable

    We have already encountered the uniform distribution in the newsvendor problem. A continuous uniform random variable describes a situation where all intervals of the same length within some range are equally likely.

    Let \(X\sim\mathcal U(a,b)\), where \(a< b\). The probability density function is

    \[
    f_X(x)=
    \begin{cases}
    \frac{1}{b-a},&\text{if }a\leq x\leq b,\\
    0,&\text{otherwise}.
    \end{cases}
    \]

    The density is constant between \(a\) and \(b\), meaning that the probability of observing a value within an interval depends only on the length of that interval.

    For example, in the newsvendor problem, we assumed that demand was uniformly distributed between 0 and 100:

    \[
    D\sim\mathcal U(0,100).
    \]

    Consequently, the probability that demand falls between 20 and 40 is

    \[
    \mathbb P(20\leq D\leq40)
    =\int_{20}^{40}\frac1{100}\,dx
    =\frac{20}{100}=0.2.
    \]
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):

    unif_a = 0
    unif_b = 10
    unif_x = np.linspace(-2, 12, 600)

    # PDF and CDF
    unif_pdf = np.where(
        (unif_x >= unif_a) & (unif_x <= unif_b),
        1 / (unif_b - unif_a),
        0,
    )

    unif_cdf = np.clip(
        (unif_x - unif_a) / (unif_b - unif_a),
        0, 1,
    )

    unif_fig, unif_axes = plt.subplots(1, 2, figsize=(10, 4))

    # PDF
    unif_axes[0].plot(
        unif_x, unif_pdf,
        color="C0", linewidth=2,
    )
    unif_axes[0].set(
        title=rf"Uniform PDF ($a={unif_a},\ b={unif_b}$)",
        xlabel="x",
        ylabel="Probability density",
        xlim=(-2, 12),
        ylim=(0, 0.13),
    )

    # CDF
    unif_axes[1].plot(
        unif_x, unif_cdf,
        color="C0", linewidth=2,
    )
    unif_axes[1].set(
        title=rf"Uniform CDF ($a={unif_a},\ b={unif_b}$)",
        xlabel="x",
        ylabel="Cumulative probability",
        xlim=(-2, 12),
        ylim=(-0.05, 1.05),
    )

    unif_fig.tight_layout()
    unif_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Exponential random variable

    The exponential distribution is especially important in queueing theory because it can be used to describe the time between consecutive events in a Poisson process.

    Consider a service desk where customers arrive randomly at an average rate of \(\lambda\) customers per unit of time. We might be interested in how long we need to wait until the next customer arrives.

    If arrivals follow a Poisson process with rate \(\lambda\), this waiting time follows an exponential distribution.

    Let \(T\sim\operatorname{Exp}(\lambda)\), where \(\lambda>0\). The probability density function is

    \[
    f_T(t)=
    \begin{cases}
    \lambda e^{-\lambda t},&\text{if }t\geq0,\\
    0,&\text{otherwise}.
    \end{cases}
    \]

    Notice that the density is largest at \(t=0\) and decreases as \(t\) increases. This means that short waiting times are more likely than long waiting times.

    The cumulative distribution function is obtained by integrating the density:

    \[
    \begin{aligned}
    F_T(t)
    &=\mathbb P(T\leq t)\\
    &=\int_0^t\lambda e^{-\lambda x}\,dx\\
    &=1-e^{-\lambda t},\qquad t\geq0.
    \end{aligned}
    \]

    An interesting property of the exponential distribution is that it is *memoryless*. This means that the probability of having to wait an additional amount of time does not depend on how long we have already waited.

    Mathematically,

    \[
    \mathbb P(T>s+t\mid T>s)=\mathbb P(T>t).
    \]

    This is the continuous counterpart of the memoryless property of the geometric distribution introduced earlier.

    The Poisson and exponential distributions are therefore closely connected. The Poisson distribution describes how many events occur during a given interval, whereas the exponential distribution describes how much time passes between consecutive events.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):

    exp_lambda = 0.5
    exp_x = np.linspace(-1, 12, 600)

    # PDF and CDF
    exp_pdf = np.where(
        exp_x >= 0,
        exp_lambda * np.exp(-exp_lambda * exp_x),
        0,
    )

    exp_cdf = np.where(
        exp_x >= 0,
        1 - np.exp(-exp_lambda * exp_x),
        0,
    )

    exp_fig, exp_axes = plt.subplots(1, 2, figsize=(10, 4))

    # PDF
    exp_axes[0].plot(
        exp_x, exp_pdf,
        color="C0", linewidth=2,
    )
    exp_axes[0].set(
        title=rf"Exponential PDF ($\lambda={exp_lambda}$)",
        xlabel="Waiting time",
        ylabel="Probability density",
        xlim=(-1, 12),
        ylim=(0, 0.55),
    )

    # CDF
    exp_axes[1].plot(
        exp_x, exp_cdf,
        color="C0", linewidth=2,
    )
    exp_axes[1].set(
        title=rf"Exponential CDF ($\lambda={exp_lambda}$)",
        xlabel="Waiting time",
        ylabel="Cumulative probability",
        xlim=(-1, 12),
        ylim=(-0.05, 1.05),
    )

    exp_fig.tight_layout()
    exp_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Normal random variable

    The normal distribution is one of the most widely used probability distributions in statistics. It is particularly useful for describing quantities that tend to vary around some average value.

    For example, measurement errors, forecasting errors, and many other quantities encountered in statistical analysis can often be approximated using normal distributions.

    A normal random variable \(X\) is characterized by two parameters: its mean \(\mu\) and standard deviation \(\sigma>0\). We write

    \[
    X\sim\mathcal N(\mu,\sigma^2).
    \]

    Its probability density function is

    \[
    f_X(x)=
    \frac{1}{\sigma\sqrt{2\pi}}
    \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),
    \qquad x\in\mathbb R.
    \]

    Although this expression might look complicated, the interpretation of its parameters is straightforward.

    The mean \(\mu\) determines the centre of the distribution, whereas the standard deviation \(\sigma\) determines how widely the values are spread around the mean.

    The normal distribution is symmetric around \(\mu\). Values close to the mean have a higher probability density, while values further away have a lower density.

    For a normal random variable,

    \[
    \mathbb E[X]=\mu,\qquad
    \operatorname{Var}(X)=\sigma^2.
    \]

    A particularly useful property is that approximately 68% of the probability lies within one standard deviation of the mean, approximately 95% within two standard deviations, and approximately 99.7% within three standard deviations.

    For example, suppose the error in a daily electricity demand forecast is approximately normally distributed with mean zero and standard deviation 100 MWh:

    \[
    X\sim\mathcal N(0,100^2).
    \]

    In this case, approximately 95% of the forecasting errors would fall between \(-200\) MWh and \(200\) MWh.

    Unlike the uniform and exponential distributions, the normal distribution assigns positive probability to intervals throughout the entire real line. This is useful for quantities such as forecasting errors, which can be either positive or negative, but requires some care when modelling quantities that cannot be negative.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):

    norm_mu = 0
    norm_sigma = 1
    norm_x = np.linspace(-5, 5, 1000)

    # PDF
    norm_pdf = (
        1 / (norm_sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((norm_x - norm_mu) / norm_sigma)**2)
    )

    # CDF computed by numerical integration
    norm_cdf = np.r_[
        0,
        np.cumsum(
            0.5 * (norm_pdf[1:] + norm_pdf[:-1])
            * np.diff(norm_x)
        ),
    ]

    norm_fig, norm_axes = plt.subplots(1, 2, figsize=(10, 4))

    # PDF
    norm_axes[0].plot(
        norm_x, norm_pdf,
        color="C0", linewidth=2,
    )
    norm_axes[0].set(
        title=rf"Normal PDF ($\mu={norm_mu},\ \sigma={norm_sigma}$)",
        xlabel="x",
        ylabel="Probability density",
        xlim=(-5, 5),
        ylim=(0, 0.45),
    )

    # CDF
    norm_axes[1].plot(
        norm_x, norm_cdf,
        color="C0", linewidth=2,
    )
    norm_axes[1].set(
        title=rf"Normal CDF ($\mu={norm_mu},\ \sigma={norm_sigma}$)",
        xlabel="x",
        ylabel="Cumulative probability",
        xlim=(-5, 5),
        ylim=(-0.05, 1.05),
    )

    norm_fig.tight_layout()
    norm_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ???- question "Expected value and variance ☠️☠️"
        Derive the expected value and variance for each or some subset of the random variables. Especially important random variables are Geometric, Poisson, and Exponential. The level of this exercise depends very much which distributions you choose to do and thus can be anythin from ☠️ to ☠️☠️☠️. You can find the definitions for the expected value and variance below from "Expected values and higher moments".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Independence

    An important concept in probability theory is *independence*. Two events are independent if knowing whether one event occurred does not change the probability of the other event occurring.

    For example, consider tossing a fair coin twice. The result of the first toss does not affect the result of the second toss. Therefore, the events of observing heads on the first and second tosses are independent.

    We can express this property mathematically.

    !!! definition "Independence of events"

        Two events \(A\) and \(B\) are independent if

        \[
            \mathbb P(A\cap B)=\mathbb P(A)\mathbb P(B),
        \]

        where \(A\cap B\) denotes the event that both \(A\) and \(B\) occur.

    Let's consider another example. Suppose we roll a fair six-sided die and define two events:

    - \(A\): The outcome is even, so \(A=\{2,4,6\}\).
    - \(B\): The outcome is divisible by three, so \(B=\{3,6\}\).

    We have

    \[
        \mathbb P(A)=\frac{3}{6}=\frac12
        \qquad\text{and}\qquad
        \mathbb P(B)=\frac26=\frac13.
    \]

    The only outcome that belongs to both events is six. Therefore,

    \[
        \mathbb P(A\cap B)=\frac16.
    \]

    Since

    \[
        \mathbb P(A)\mathbb P(B)
        =\frac12\cdot\frac13
        =\frac16,
    \]

    the two events are independent, even though they are determined by the same die roll.

    We can also extend independence to random variables. Two random variables \(X\) and \(Y\) are independent if observing the value of one does not provide information about the probability distribution of the other.

    More formally, \(X\) and \(Y\) are independent if, for all real numbers \(x\) and \(y\),

    \[
        \mathbb P(X\leq x,Y\leq y)
        =
        \mathbb P(X\leq x)\mathbb P(Y\leq y).
    \]

    This concept is particularly important for the distributions we introduced earlier. For example, the binomial distribution assumes that the individual Bernoulli trials are independent. Similarly, the Poisson process assumes that the numbers of arrivals occurring in disjoint time intervals are independent.

    However, independence is not something we can automatically assume. In many real-world applications, events and random variables influence each other. For example, the number of customers arriving at a shop during consecutive hours might depend on the time of day or other common factors. Whether independence is a reasonable assumption depends on the process we are modelling.

    ### Conditional probabilities

    So far, we have mostly considered probabilities without taking any additional information into account. However, in many decision-making situations, we obtain information that changes how likely we consider different events to be.

    We already encountered an idea closely related to conditional probability in the newsvendor example. Recall that when we calculated the expected number of newspapers sold, we divided the integral into two parts:

    \[
        \mathbb E[\min\{q,D\}]
        =\frac1{100}\left(\int_0^q x\,dx+\int_q^{100}q\,dx\right).
    \]

    What we effectively did was consider two different situations: one where demand does not exceed the quantity ordered, and another where demand exceeds the quantity ordered.

    We can express this using *conditional expectations*:

    \[
    \begin{aligned}
    \mathbb E[\min\{q,D\}]
    &=\mathbb E[\min\{q,D\}\mid D\leq q]\,\mathbb P(D\leq q)\\
    &\quad+\mathbb E[\min\{q,D\}\mid D>q]\,\mathbb P(D>q).
    \end{aligned}
    \]

    Here, the two conditional expectations describe the expected number of newspapers sold in each situation. We then multiply these expectations by the probabilities of the corresponding situations to obtain the overall expected value.

    This relationship is an example of the *law of total expectation*. To understand it more carefully, we first need to introduce conditional probabilities.

    !!! definition "Conditional probability"

        Let \(A\) and \(B\) be two events, with \(\mathbb P(B)>0\). The conditional probability of \(A\) given that \(B\) has occurred is

        \[
            \mathbb P(A\mid B)
            =
            \frac{\mathbb P(A\cap B)}{\mathbb P(B)}.
        \]

        The notation \(\mathbb P(A\mid B)\) is read as "the probability of \(A\) given \(B\)."

    The intuition behind this definition is quite simple. Once we know that \(B\) has occurred, we only need to consider outcomes belonging to \(B\). Among these outcomes, we are interested in the proportion that also belongs to \(A\).

    Consider again rolling a fair six-sided die. Suppose we define

    \[
        A=\{2,4,6\}
    \]

    as the event of observing an even number, and

    \[
        B=\{4,5,6\}
    \]

    as the event of observing a number greater than three.

    Without any additional information, the probability of an even number is

    \[
        \mathbb P(A)=\frac12.
    \]

    However, suppose someone tells us that the die landed on a number greater than three. We now know that the only possible outcomes are \(4,5,6\), of which two are even.

    Therefore,

    \[
    \begin{aligned}
        \mathbb P(A\mid B)
        &=\frac{\mathbb P(A\cap B)}{\mathbb P(B)}\\
        &=\frac{2/6}{3/6}\\
        &=\frac23.
    \end{aligned}
    \]

    Notice that the probability of observing an even number has increased from \(1/2\) to \(2/3\) because of the additional information.

    We can also use conditional probabilities to understand independence. Recall that two events are independent if

    \[
        \mathbb P(A\cap B)=\mathbb P(A)\mathbb P(B).
    \]

    Substituting this into the definition of conditional probability gives

    \[
    \begin{aligned}
        \mathbb P(A\mid B)
        &=\frac{\mathbb P(A)\mathbb P(B)}{\mathbb P(B)}\\
        &=\mathbb P(A).
    \end{aligned}
    \]

    Thus, when \(A\) and \(B\) are independent, learning that \(B\) has occurred does not change the probability of \(A\). This agrees with our earlier intuition about independence.

    ### Bayes' theorem

    Conditional probabilities are especially useful when we are trying to make decisions based on incomplete information. We might, for example, observe some data and want to use this information to determine how likely a particular event or situation is.

    Bayes' theorem provides a way to update probabilities when new information becomes available.

    We can derive Bayes' theorem directly from the definition of conditional probability. Recall that

    \[
        \mathbb P(A\mid B)
        =
        \frac{\mathbb P(A\cap B)}{\mathbb P(B)}.
    \]

    Similarly, reversing the roles of \(A\) and \(B\) gives

    \[
        \mathbb P(B\mid A)
        =
        \frac{\mathbb P(A\cap B)}{\mathbb P(A)}.
    \]

    Rearranging the second equation and substituting it into the first gives Bayes' theorem.

    !!! definition "Bayes' theorem"

        Let \(A\) and \(B\) be events with positive probabilities. Then

        \[
            \mathbb P(A\mid B)
            =
            \frac{\mathbb P(B\mid A)\mathbb P(A)}
            {\mathbb P(B)}
        \]

    The theorem gives us a way to compute the probability of \(A\) after observing \(B\), provided that we know the probability of \(A\), the probability of observing \(B\) when \(A\) occurs, and the overall probability of \(B\).

    Let's consider an example related to demand forecasting.

    Suppose a shop is interested in whether tomorrow will be a high-demand day. Based on its historical data, the shop estimates that 20% of days have unusually high demand. Let \(A\) denote the event that tomorrow is a high-demand day. Therefore,

    \[
        \mathbb P(A)=0.2.
    \]

    The shop also uses a forecasting system that can issue a warning when it predicts unusually high demand. Let \(B\) denote the event that the system issues a warning.

    Suppose that the forecasting system has the following properties:

    - On high-demand days, it issues a warning 85% of the time: \(\mathbb P(B\mid A)=0.85\).
    - On days without high demand, it incorrectly issues a warning 10% of the time: \(\mathbb P(B\mid A^c)=0.10\).

    Here, \(A^c\) denotes the complement of \(A\), meaning that tomorrow is not a high-demand day.

    Now suppose the forecasting system issues a warning. How likely is it that tomorrow will actually be a high-demand day?

    We want to calculate

    \[
        \mathbb P(A\mid B).
    \]

    Using Bayes' theorem,

    \[
        \mathbb P(A\mid B)
        =
        \frac{\mathbb P(B\mid A)\mathbb P(A)}
        {\mathbb P(B)}.
    \]

    However, we do not yet know the overall probability that the system issues a warning, \(\mathbb P(B)\).

    A warning can occur in two mutually exclusive situations: either tomorrow is a high-demand day or it is not. We can therefore calculate the probability of a warning by adding the probabilities of these two possibilities:

    \[
    \begin{aligned}
        \mathbb P(B)
        &=\mathbb P(B\mid A)\mathbb P(A)\\
        &\quad+\mathbb P(B\mid A^c)\mathbb P(A^c)\\
        &=0.85\cdot0.2+0.10\cdot0.8\\
        &=0.25.
    \end{aligned}
    \]

    This is an example of the *law of total probability*. Note that this closly corresponds to the newswendor example.

    Now we can apply Bayes' theorem:

    \[
    \begin{aligned}
        \mathbb P(A\mid B)
        &=\frac{0.85\cdot0.2}{0.25}\\
        &=0.68.
    \end{aligned}
    \]

    Thus, after receiving a warning, the shop estimates that the probability of unusually high demand tomorrow is 68%, compared with the original probability of only 20%.

    This illustrates the central idea of Bayes' theorem. We begin with an initial probability, often called the *prior probability*. When we observe new information, we use Bayes' theorem to update this probability. The resulting probability is called the *posterior probability*.

    In our example, the prior probability of high demand was 20%, while the posterior probability after observing a warning was 68%.

    An important distinction is that \(\mathbb P(A\mid B)\) and \(\mathbb P(B\mid A)\) are generally not the same. In this example, the probability that the system issues a warning on a high-demand day is 85%, but the probability that tomorrow is a high-demand day given a warning is only 68%.

    This type of reasoning is particularly useful when we combine statistical inference with decision-making. In many applications, we first obtain information about an uncertain quantity and then need to determine what decision to make based on that information. Bayes' theorem provides one way to update our understanding of uncertainty before making such decisions.

    ### Law of total probability

    The next result will be very useful for evaluating many probabilities. It allows you to evaluate a probability of an event by splitting it to multiple mutually exclusive conditional probabilities. We have already seen this multiple times in action and it has helped us to compute otherwise hard or impossible to evaluate probabilities.

    !!! proposition "Law of total probability"

        Let \(A_1,A_2,\ldots,A_n\) be mutually exclusive events that together cover the entire sample space \(\Omega\). In other words,

        \[
            A_i\cap A_j=\emptyset\quad\text{for }i\neq j,
            \qquad
            \bigcup_{i=1}^n A_i=\Omega.
        \]

        Assume that \(\mathbb P(A_i)>0\) for all \(i\). Then, for any event \(B\),

        \[
            \mathbb P(B)=\sum_{i=1}^n
            \mathbb P(B\mid A_i)\mathbb P(A_i)
        \]

        This result is called the *law of total probability*. It allows us to calculate the probability of an event by considering all the different situations in which that event can occur.

        In particular, if we divide the sample space into an event \(A\) and its complement \(A^c\), with \(0<\mathbb P(A)<1\), we obtain

        \[
            \mathbb P(B)
            =
            \mathbb P(B\mid A)\mathbb P(A)
            +
            \mathbb P(B\mid A^c)\mathbb P(A^c).
        \]

    ???- proof "Proof"

        Since the events \(A_1,\ldots,A_n\) cover the entire sample space, every outcome belonging to \(B\) must also belong to one of these events. Therefore, we can write

        \[
            B=\bigcup_{i=1}^n(B\cap A_i).
        \]

        Since the events \(A_i\) are mutually exclusive, the intersections \(B\cap A_i\) are also mutually exclusive. We can therefore add their probabilities:

        \[
            \mathbb P(B)
            =
            \sum_{i=1}^n\mathbb P(B\cap A_i).
        \]

        Recall from the definition of conditional probability that

        \[
            \mathbb P(B\mid A_i)
            =
            \frac{\mathbb P(B\cap A_i)}{\mathbb P(A_i)}.
        \]

        Rearranging gives

        \[
            \mathbb P(B\cap A_i)
            =
            \mathbb P(B\mid A_i)\mathbb P(A_i).
        \]

        Substituting this into our earlier expression, we obtain

        \[
            \mathbb P(B)
            =
            \sum_{i=1}^n
            \mathbb P(B\mid A_i)\mathbb P(A_i),
        \]

        which proves the proposition.

    ???- question "Identifying fields with high phosphorus runoff risk ☠️☠️"
        A municipality wants to identify agricultural fields with a high risk of phosphorus runoff so that it can target conservation measures effectively.

        Based on historical data, 20% of fields are considered high-risk. The municipality uses an environmental assessment that correctly identifies high-risk fields 80% of the time. However, the assessment also incorrectly identifies 10% of low-risk fields as high-risk.

        Suppose a field receives a positive assessment, indicating high runoff risk.

        1. What is the probability that the assessment gives a positive result for a randomly selected field?
        2. Use Bayes' theorem to calculate the probability that the field actually has high runoff risk, given the positive assessment.
        3. Compare your result with the original 20% probability. What does this tell us about the usefulness of the assessment for targeting conservation measures?

        *Hint:* Let \(H\) denote the event that a field has high runoff risk and \(+\) the event that the assessment is positive. You can calculate \(\mathbb P(+)\) using the law of total probability.

    ???- question "Extension: Using the assessment to make a decision ☠️☠️☠️"

        The municipality can install a phosphorus-removal filter at the field's drainage outlet to reduce phosphorus entering the nearby lake.

        Installing the filter costs €3,000. The municipality estimates that the filter would provide environmental benefits worth €9,000 if the field has high runoff risk. If the field has low runoff risk, the benefits are assumed to be negligible.

        The municipality wants to maximize its expected net benefit, defined as the environmental benefits minus the costs of the filter. If it does not install the filter, the net benefit is zero.

        The municipality can also conduct the environmental assessment before making its decision. The assessment costs €200.

        **Answer the following questions:**

        1. **Decision without the assessment:** Should the municipality install the filter based only on the original 20% probability of high runoff risk? Calculate the expected net benefit.
        2. **Decision after the assessment:** Suppose the municipality conducts the assessment. Should it install the filter if the result is positive? What if the result is negative? Use Bayes' theorem to update the probabilities and calculate the expected net benefits.
        3. **Value of information:** Calculate the expected net benefit of conducting the assessment and then making the optimal decision based on its result. Remember to account for the €200 assessment cost. Is conducting the assessment worthwhile? Then compute the threshold cost where it is beneficial to treat each field and separately fields that receive positive result from the test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected values and higher moments

    You should be familiar with the concepts of expected value and variance as well as computing these for certain distributions. For some distributions it is very easy to compute the expected values. For example, let \(B\) be a Bernoulli random with probability of success \(p\). Then the expected value is computed as

    \[
        \mathbb{E}[B]=0\cdot(1-p)+1\cdot p=p
    \]

    and the variance

    \[
        \begin{aligned}
            \text{Var}(B)&=\mathbb{E}[(B-\mathbb{E}[B])^2] \\
            &=(0-p)^2(1-p)+(1-p)^2p\\
            &=p^2(1-p)+1^2p-2p^2+p^3 \\
            &=p^2-p^3+p-2p^2+p^3 \\
            &=p-p^2 \\
            &=p(1-p).
        \end{aligned}
    \]

    For other this may be considerably harder. We are not going to need higher moments of the distributions, but also those can be used to characterize properties of the distributions.

    The expected value is defined as follows.

    !!! definition "Expected value"

        The *expected value* of a random variable \(X\), denoted by \(\mathbb E[X]\), describes the average value we would observe if we repeated the random experiment many times.

        For a discrete random variable \(X\) with probability mass function \(p_X(x)\), the expected value is defined as

        \[
            \mathbb E[X]=\sum_x x\,p_X(x).
        \]

        For a continuous random variable \(X\) with probability density function \(f_X(x)\), the expected value is defined as

        \[
            \mathbb E[X]=\int_{-\infty}^{\infty}x f_X(x)\,dx.
        \]

        We assume that the corresponding sum or integral is absolutely convergent, so that the expected value is finite.

    One particularly useful property of expected values is that they are *linear*. This allows us to calculate the expected value of a sum without first determining the probability distribution of the sum.

    !!! proposition "Linearity of expectation"

        Let \(X\) and \(Y\) be random variables with finite expected values, and let \(\alpha,\beta\in\mathbb R\). Then

        \[
            \mathbb E[\alpha X+\beta Y]
            =\alpha\mathbb E[X]+\beta\mathbb E[Y].
        \]

        Importantly, this result holds even if \(X\) and \(Y\) are not independent.


    ???- proof "Proof"

        For simplicity, consider a finite sample space \(\Omega\). By the definition of expected value, we can write

        \[
        \begin{aligned}
            \mathbb E[\alpha X+\beta Y]
            &=\sum_{\omega\in\Omega}
            (\alpha X(\omega)+\beta Y(\omega))\mathbb P(\{\omega\})\\
            &=\alpha\sum_{\omega\in\Omega}X(\omega)\mathbb P(\{\omega\})\\
            &\quad+\beta\sum_{\omega\in\Omega}Y(\omega)\mathbb P(\{\omega\})\\
            &=\alpha\mathbb E[X]+\beta\mathbb E[Y].
        \end{aligned}
        \]

        The second equality follows from the distributive property and the linearity of summation.

        Notice that we did not need to assume that \(X\) and \(Y\) are independent.

        The same result holds more generally for random variables with finite expected absolute values, using the linearity of integration in place of summation.

    This property is useful when the quantity we are interested in consists of several components. For example, if the total cost of maintaining an inventory consists of purchasing, holding, and shortage costs, we can calculate the expected total cost by adding the expected values of these individual costs.

    We have also already encountered *conditional expectations* in the newsvendor example. The following proposition formalizes how we can calculate an expected value by dividing the possible outcomes into different situations.

    !!! proposition "Law of total expectation"

        Let \(A_1,\ldots,A_n\) be mutually exclusive events that together cover the entire sample space, with \(\mathbb P(A_i)>0\) for every \(i\). For a random variable \(X\) with finite expected value,

        \[
            \mathbb E[X]
            =\sum_{i=1}^n
            \mathbb E[X\mid A_i]\mathbb P(A_i).
        \]

        Here, \(\mathbb E[X\mid A_i]\) is the expected value of \(X\) given that the event \(A_i\) has occurred.


    ???- proof "Proof"

        We will prove the result for a discrete random variable \(X\).

        Recall that the conditional probability of \(X=x\), given an event \(A_i\), is

        \[
            \mathbb P(X=x\mid A_i)
            =\frac{\mathbb P(\{X=x\}\cap A_i)}
            {\mathbb P(A_i)}.
        \]

        Using this, the conditional expected value can be expressed as

        \[
        \begin{aligned}
            \mathbb E[X\mid A_i]
            &=\sum_x x\,\mathbb P(X=x\mid A_i)\\
            &=\sum_x x\,
            \frac{\mathbb P(\{X=x\}\cap A_i)}
            {\mathbb P(A_i)}.
        \end{aligned}
        \]

        Multiplying both sides by \(\mathbb P(A_i)\), we obtain

        \[
            \mathbb E[X\mid A_i]\mathbb P(A_i)
            =\sum_x x\,\mathbb P(\{X=x\}\cap A_i).
        \]

        Now, summing over all the events \(A_1,\ldots,A_n\) gives

        \[
        \begin{aligned}
            &\sum_{i=1}^n
            \mathbb E[X\mid A_i]\mathbb P(A_i)\\
            &=\sum_{i=1}^n\sum_x
            x\,\mathbb P(\{X=x\}\cap A_i)\\
            &=\sum_x x\sum_{i=1}^n
            \mathbb P(\{X=x\}\cap A_i)\\
            &=\sum_x x\,\mathbb P(X=x)\\
            &=\mathbb E[X].
        \end{aligned}
        \]

        The second-to-last equality follows from the law of total probability, since the events \(A_1,\ldots,A_n\) form a partition of the sample space.

        Thus, we have shown that the expected value can be calculated as a weighted average of conditional expected values.

        The same result also holds for continuous random variables, with integrals replacing the sums.

    Notice the similarity to the law of total probability. We consider all possible situations, calculate the conditional expected value in each situation, and weight these values by the probabilities of the situations.


    Similarly variance and standard defiation are defined in the box below.

    !!! definition "Variance and standard deviation"

        The *variance* of a random variable \(X\) describes how much its values tend to differ from their expected value.

        Assuming that \(X\) has a finite second moment, the variance is defined as

        \[
            \operatorname{Var}(X)
            =\mathbb E\left[(X-\mathbb E[X])^2\right].
        \]

        In other words, variance is the expected squared distance between the random variable and its mean.

        For a discrete random variable, this becomes

        \[
            \operatorname{Var}(X)
            =\sum_x (x-\mathbb E[X])^2p_X(x),
        \]

        and for a continuous random variable,

        \[
            \operatorname{Var}(X)
            =\int_{-\infty}^{\infty}
            (x-\mathbb E[X])^2f_X(x)\,dx.
        \]

        Since variance measures squared deviations, its units are the square of the units of \(X\). To express variability in the original units, we use the *standard deviation*, defined as

        \[
            \sigma_X=\sqrt{\operatorname{Var}(X)}.
        \]


    Next we'll take a look at couple useful results that can simplify the computation of expected values and variances in many cases.


    !!! proposition "Alternative expression for variance"

        The variance of a random variable \(X\) with a finite second moment can also be expressed as

        \[
            \operatorname{Var}(X)
            =\mathbb E[X^2]-\mathbb E[X]^2.
        \]

        This expression is often useful for calculating variances.

    ???- proof "Proof"

        The result follows directly from the definition of variance and elementary algebra.

        Recall that \(\mathbb E[X]\) is a constant. Therefore, using the linearity of expectation, we obtain

        \[
        \begin{aligned}
            \operatorname{Var}(X)
            &=\mathbb E\left[(X-\mathbb E[X])^2\right]\\
            &=\mathbb E\left[X^2-2X\mathbb E[X]+\mathbb E[X]^2\right]\\
            &=\mathbb E[X^2]-2\mathbb E[X]\mathbb E[X]+\mathbb E[X]^2\\
            &=\mathbb E[X^2]-2\mathbb E[X]^2+\mathbb E[X]^2\\
            &=\mathbb E[X^2]-\mathbb E[X]^2.
        \end{aligned}
        \]

        Thus, the alternative expression is equivalent to the original definition of variance.


    !!! proposition "Expected value of a function of a random variable"

        Let \(X\) be a random variable and \(g\) a function such that \(g(X)\) is also a random variable with a finite expected absolute value.

        If \(X\) is a discrete random variable with probability mass function \(p_X(x)\), then

        \[
            \mathbb E[g(X)]=\sum_x g(x)p_X(x).
        \]

        Similarly, if \(X\) is a continuous random variable with probability density function \(f_X(x)\), then

        \[
            \mathbb E[g(X)]
            =\int_{-\infty}^{\infty}g(x)f_X(x)\,dx.
        \]

        In other words, we can calculate the expected value of \(g(X)\) directly using the probability distribution of \(X\), without first determining the distribution of \(g(X)\).

    ???- proof "Proof"

        **Discrete case.**

        Let \(Y=g(X)\). By the definition of expected value,

        \[
            \mathbb E[Y]=\sum_y y\,\mathbb P(Y=y).
        \]

        Since \(Y=g(X)\), the event \(Y=y\) occurs whenever \(X\) takes a value \(x\) for which \(g(x)=y\). Therefore,

        \[
            \mathbb P(Y=y)
            =\sum_{x:g(x)=y}p_X(x).
        \]

        Substituting this into the expected value gives

        \[
        \begin{aligned}
            \mathbb E[g(X)]
            &=\sum_y y\sum_{x:g(x)=y}p_X(x)\\
            &=\sum_x g(x)p_X(x).
        \end{aligned}
        \]

        Thus, the result holds for discrete random variables.

        We will not prove the result for continuous random variables, since a general proof requires additional mathematical tools from real analysis and measure theory. However, the corresponding expression also holds for continuous random variables under the assumptions of the proposition.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dynamical systems

    So far, we have mainly considered decisions where we choose some quantity and evaluate the resulting costs or benefits. However, many management problems involve systems that evolve over time. For example, the amount of inventory in a warehouse changes when new products arrive and existing products are sold. Similarly, the amount of water in a reservoir changes as water flows in and out.

    These are examples of *dynamical systems*.

    ### Discrete-time dynamical systems

    A discrete-time dynamical system describes how the state of a system changes between consecutive time periods.

    !!! definition "Discrete-time dynamical system"

        A discrete-time dynamical system can be expressed as

        \[
            x_{t+1}=f(x_t,u_t),
        \]

        where \(x_t\) is the *state* of the system at time \(t\), \(u_t\) represents an input or decision, and \(f\) describes how the system evolves.

    Consider a simple inventory system. Let \(I_t\) denote the inventory at the beginning of period \(t\), \(Q_t\) the quantity received during the period, and \(D_t\) the demand. Assuming that unmet demand is recorded as backorders, the inventory evolves according to

    \[
        I_{t+1}=I_t+Q_t-D_t.
    \]

    This equation simply states that the inventory in the next period equals the current inventory plus replenishments minus demand.

    For example, if we begin with 100 units, receive 30 new units, and face demand for 50 units, the inventory in the next period is

    \[
        I_{t+1}=100+30-50=80.
    \]

    We can repeatedly apply the same equation to determine how the system evolves over several periods.

    ### Continuous-time dynamical systems

    Some systems evolve continuously rather than in discrete periods. In these situations, the rate of change can be described using a differential equation.

    For example, consider a reservoir with water volume \(V(t)\), inflow rate \(r(t)\), and outflow rate \(d(t)\). Its volume changes according to

    \[
        \frac{dV(t)}{dt}=r(t)-d(t).
    \]

    This describes essentially the same balance relationship as our inventory equation, but in continuous time.

    We will not develop the general theory of dynamical systems or methods for solving differential equations. For our purposes, the important idea is that we can describe how a system changes over time using relatively simple mathematical equations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stochastic processes

    In the previous examples, we assumed that the quantities affecting the system were known. However, in many applications, quantities such as demand, customer arrivals, and rainfall are uncertain.

    For example, suppose that the demand \(D_t\) in our inventory model is now a random variable. The inventory equation becomes

    \[
        I_{t+1}=I_t+Q_t-D_t.
    \]

    Although the equation has not changed, we can no longer determine the future inventory level exactly before observing demand. Instead, the inventory level itself becomes a random quantity.

    This leads us to the concept of a *stochastic process*.

    !!! definition "Stochastic process"

        A stochastic process is a collection of random variables indexed by time.

        For a discrete-time process, we can write

        \[
            \{X_t\}_{t=0}^{\infty},
        \]

        Each random variable describes the state or outcome of the process at a particular time.

    We have already encountered one important example: the Poisson process, which describes random events occurring over time.

    In queueing models, we might use a Poisson process to describe customer arrivals and another probability distribution to describe service times. Together, these determine how the number of customers in the system evolves.

    ### Steady-state behaviour

    When studying dynamical systems, we are often interested in their long-run behaviour rather than their state at one particular moment.

    For deterministic systems, an *equilibrium* is a state that remains unchanged over time. For example, an equilibrium \(x^*\) of the system \(x_{t+1}=f(x_t)\) satisfies

    \[
        x^*=f(x^*).
    \]

    For stochastic systems, the situation is slightly different. Even if the state continues to change randomly, its probability distribution may eventually become independent of time. We call such a distribution a *steady-state* or *stationary distribution*.

    This is especially useful in queueing theory, where we often want to calculate long-run average waiting times and queue lengths.

    For example, in a simple queue with Poisson arrivals at rate \(\lambda\), exponentially distributed service times with rate \(\mu\), and one server (an M/M/1 queue), a stationary queue-length distribution exists when

    \[
        \lambda<\mu.
    \]

    In other words, the server must be able to process customers faster than they arrive on average.

    We will examine these models and their assumptions more carefully when we study queueing theory.
    """)
    return


if __name__ == "__main__":
    app.run()
