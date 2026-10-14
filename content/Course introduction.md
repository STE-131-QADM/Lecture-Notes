# Welcome to STE-131 Management Science – Quantitative Analysis for Decision Making!

## Course topics and materials

This website serves as lecture notes throughout the course and provides complementary material in addition to the course textbook, [*An Introduction to Management Science: Quantitative Approaches to Decision Making* by Anderson et al.](https://helka.helsinki.fi/permalink/358UOH_INST/1r0ctnh/alma9934145505206253)

As the title of the course suggests, this course is about analysing data and processes and using this analysis as a basis for making decisions. The course consists of three broad themes:

1. Linear and mixed-integer optimization
2. Queue and inventory models
3. Forecasting and embedding statistical inference into decision-making

The first part will take a significant amount of time. In the second part, we will use optimization techniques to derive key properties of queue and inventory models using tools from mathematical optimization. The last part will also build on the tools acquired earlier in the course. We will use statistical methods indirectly to estimate and interpret model parameters, but also directly by using forecasting results to parameterize optimization problems.

## Decision-making as a field of study

Decision-making can be studied from many different perspectives. In quantitative applications of decision-making we are required to somehow rank the different actions available in a given situation.

A function that assigns a value to a given action may have different names depending on the context, such as a *utility function* or an *objective function*. Its values can have a straightforward interpretation. For example, a profit function could measure the amount of money obtained from a particular combination of inputs in euros; an objective could measure the time required for a particular logistics network to operate in minutes; or it could measure the amount of land conserved in square metres.

However, the value need not always have a direct measurement scale. In some cases, we may only have ordinal information about the ranking of different solutions. This is the case, for example, in many economic applications of utility functions.

Regardless of the exact form and properties of this function, our goal is usually to find inputs that either maximize or minimize it, although there are also approaches to decision-making that are not formulated directly as optimization problems.[^1] This often leads to problems of the form

$$
    \max_{x\in\text{dom}f_0} f_0(x).
$$

Usually, the problem is also complicated by the fact that we are not completely free to choose $x$: some constraints apply. For example, companies need to take their budget constraints into account. The problem therefore typically takes the form

$$
    \begin{align*}
        \max_{x\in\text{dom}f_0}&\ f_0(x) \\
        \text{subject to}& \\
        & f(x)\leq b.
    \end{align*}
$$

Here, the functions $f$ and $h$ encode problem-specific constraints. This formulation also allows us to analyse the problem mathematically when we make additional assumptions about the functional forms and regularity properties of $f_0$, $f$, and $h$.

Optimization is therefore of central importance when analysing the actions available in a decision-making process. It also introduces many interesting questions that can be asked after an optimal action has been identified. For example, we are often interested in how much the result would change if there were a small change in a constraint or in some other specification of the problem. Such questions can be analysed rigorously within the mathematical optimization framework. The same general formulation can be applied in countless different situations.

For example, in inventory management, we are often interested in questions such as how much inventory to hold or how much to order at a given time. Optimal quantities can be derived using methods from mathematical optimization.

In queueing theory, we can derive many quantities of interest that describe the behaviour of a queueing process. While having this information is useful in itself, it is even more useful if we can use it to make informed decisions about how resources should be allocated. This can be achieved, for example, by minimizing waiting time subject to resource constraints.

The tools introduced in data science courses are also often used together with optimization methods. For example, a company may be interested in forecasting demand for some future period. However, having a forecast alone does not yet determine what action should be taken. There must also be a way to react to that information. Determining the best response may, for example, require solving an optimization problem that maximizes expected profit.

By the end of this course, you should feel comfortable applying this type of analysis to a range of applied problems, particularly in the field of natural resource management. This includes developing a good understanding of how to formulate such problems mathematically, how to solve them using appropriate programming tools, and how to analyse them theoretically in order to determine whether the chosen methods are suitable for the problem at hand.

While these tools are powerful, the person constructing the model ultimately bears responsibility for the assumptions used to represent the phenomenon being studied. It is therefore essential not only to understand how the methods work, but also to recognise situations in which their underlying assumptions are not appropriate.

[^1]: One such approach is the [Free Energy Principle](https://web.archive.org/web/20260610001030/https://www.noumenal.ai/post/wtf-is-the-fep-a-short-explainer-on-the-free-energy-principle).