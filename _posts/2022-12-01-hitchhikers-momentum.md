---
layout: distill
title: A Hitchhiker's Guide to Momentum
description: Polyak momentum is one of the most iconic methods in optimization. Despite it's simplicity, it features rich dynamics that depend both on the step-size and momentum parameter. In this blog post we identify the different regions of the parameter space and discuss their convergence properties using the theory of Chebyshev polynomials.
date: 2022-12-01
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2022-12-01-hitchhikers-momentum.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Gradient Descent with Momentum
  - name: How fast is Momentum?
  - name: The Robust Region
  - name: The Lazy Region
  - name: Knife's Edge
  - name: Putting it All Together

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  /* see http://drz.ac/2013/01/17/latex-theorem-like-environments-for-the-web/ and http://felix11h.github.io/blog/mathjax-theorems */
  .theorem {
    display: block;
    margin: 12px 0;
    font-style: italic;
  }
  .theorem:before {
    content: "Theorem.";
    font-weight: bold;
    font-style: normal;
  }
  .theorem[text]:before {
    content: "Theorem (" attr(text) ") ";
  }

  .corollary {
    display: block;
    margin: 12px 0;
    font-style: italic;
  }
  .corollary:before {
    content: "Corollary.";
    font-weight: bold;
    font-style: normal;
  }
  .corollary[text]:before {
  content: "Corollary (" attr(text) ") ";
  }

  .lemma {
      display: block;
      margin: 12px 0;
      font-style: italic;
  }
  .lemma:before {
      content: "Lemma.";
      font-weight: bold;
      font-style: normal;
  }
  .lemma[text]:before {
    content: "Lemma (" attr(text) ") ";
  }

  .definition {
    display: block;
    margin: 12px 0;
    font-style: italic;
  }
  .definition:before {
    content: "Definition.";
    font-weight: bold;
    font-style: normal;
  }
  .definition[text]:before {
    content: "Definition (" attr(text) ") ";
  }

  .remark {
    display: block;
    margin: 12px 0;
    font-style: italic;
  }
  .remark:before {
    content: "Remark.";
    font-weight: bold;
    font-style: normal;
  }
  .remark[text]:before {
    content: "Remark (" attr(text) ") ";
  }

  .lemma[text]:before {
    content: "Lemma (" attr(text) ") ";
  }

  .proof {
      display: block;
      font-style: normal;
      margin: 0;
  }
  .proof:before {
      content: "Proof.";
      font-style: italic;
  }
  .proof:after {
      content: "\25FC";
      float:right;
      font-size: 1.8rem;
  }

  .wrap-collapsible {
    margin-bottom: 1.2rem 0;
  }

  input[type='checkbox'] {
    display: none;
  }

  .lbl-toggle {
    text-align: center;
    padding: 0.6rem;
    cursor: pointer;
    border-radius: 7px;
    transition: all 0.25s ease-out;
  }

  .lbl-toggle::before {
    content: ' ';
    display: inline-block;
    border-top: 5px solid transparent;
    border-bottom: 5px solid transparent;
    border-left: 5px solid currentColor;
    vertical-align: middle;
    margin-right: .7rem;
    transform: translateY(-2px);
    transition: transform .2s ease-out;
  }

  .toggle:checked + .lbl-toggle::before {
    transform: rotate(90deg) translateX(-3px);
  }

  .collapsible-content {
    max-height: 0px;
    overflow: hidden;
    transition: max-height .25s ease-in-out;
  }

  .toggle:checked + .lbl-toggle + .collapsible-content {
    max-height: none;
    overflow: visible;
  }

  .toggle:checked + .lbl-toggle {
    border-bottom-right-radius: 0;
    border-bottom-left-radius: 0;
  }

  .collapsible-content .content-inner {
    /* background: rgba(250, 224, 66, .2); */
    /* border-bottom: 1px solid rgba(250, 224, 66, .45); */
    border-bottom-left-radius: 7px;
    border-bottom-right-radius: 7px;
    padding: .5rem 1rem;
  }

  .center {
      display: block;
      margin-left: auto;
      margin-right: auto;
  }

  .framed {
    border: 1px var(--global-text-color) dashed !important;
    padding: 20px;
  }
  
  d-article {
    overflow-x: visible;
  }

  .underline {
    text-decoration: underline;
  }
---
  
<!-- some latex shortcuts -->
<div style="display: none">
$$
\def\argmin{\mathop{\mathrm{arg\,min}}}
\def\xx{\pmb{x}}
\def\HH{\pmb{H}}
\def\bb{\pmb{b}}
\def\EE{ \mathbb{E} }
\def\RR{ \mathbb{R} }
\def\lmax{L}
\def\lmin{\mu}
\def\defas{\stackrel{\text{def}}{=}}
\definecolor{colormomentum}{RGB}{27, 158, 119}
\definecolor{colorstepsize}{RGB}{217, 95, 2}
\def\mom{ {\color{colormomentum}{m}} }
\def\step{ {\color{colorstepsize}h} }
$$
</div>


{% include figure.html path="assets/img/2022-12-01-hitchhikers-momentum/rate_convergence_momentum.png" class="img-fluid" %}


## Gradient Descent with Momentum


Gradient descent with momentum,<d-cite key="polyak1964some"></d-cite> also known as heavy ball or momentum for short, is an optimization method
designed to solve unconstrained minimization problems of the form
\begin{equation}
\argmin_{\xx \in \RR^d} \, f(\xx)\,,
\end{equation}
where the objective function $$f$$ is differentiable and we have access to its gradient $$\nabla f$$. In this method
the update is a sum of two terms. The first term is the
difference between the current and the previous iterate $$(\xx_{t} - \xx_{t-1})$$, also known as _momentum term_. The second term is the gradient $$\nabla f(\xx_t)$$ of the objective function.

<p class="framed">
  <b class="underline">Gradient Descent with Momentum</b><br>
  <b>Input</b>: starting guess \(\xx_0\), step-size \(\step > 0\) and momentum
    parameter \(\mom \in (0, 1)\).<br>
  \(\xx_1 = \xx_0 - \dfrac{\step}{\mom+1} \nabla f(\xx_0)\) <br>
  <b>For</b> \(t=1, 2, \ldots\) compute
  \begin{equation}\label{eq:momentum_update}
  \xx_{t+1} = \xx_t + \mom(\xx_{t} - \xx_{t-1}) - \step\nabla
  f(\xx_t)
  \end{equation}
</p>


Despite its simplicity, gradient descent with momentum exhibits unexpectedly rich dynamics that we'll explore on this post. 


The origins of momentum can be traced back to Frankel's method in the 1950s for solving linear system of equations.<d-cite key="frankel1950convergence"></d-cite> It was later generalized by Boris Polayk to non-quadratic objectives<d-cite key="polyak1964some"></d-cite>. In recent years there has been a resurgence in interest in this venerable method, as a stochastic variant of this method, where the gradient is replaced by a stochastic estimate, is one of the most popular methods for deep learning. This has led in recent years to a flurry fo research --and improved understanding -- of this stochastic variant. Although this blog posts limits itself with the deterministic variant, the interested reader is encouraged to explore following references. A good starting point is the paper by <a href="https://arxiv.org/abs/1712.07628">Sutskever et al.</a>,<d-cite key="sutskever2013importance"></d-cite> which was among the firsts to highlight the importance of momentum for deep learning optimization. More recent progress include an analysis of the last iteration of the method by <a href="https://arxiv.org/abs/2104.09864">Tao et al.</a><d-cite key="tao2021the"></d-cite> and a paper by <a href="https://arxiv.org/abs/2106.07587">Liu et al.</a><d-cite key="Liu2020Accelerating"></d-cite> that develops accelerated variants for over-parameterized models.


Coming back to the subject of our post, the (non-stochastic) gradient descendent with momentum method, a paper that also explores the dynamics of momentum is Gabriel Goh's <a href="https://distill.pub/2017/momentum/">Why Momentum Really Works</a>.<d-cite key="goh2017momentum"></d-cite> There are subtle but important differences between both analysis. The landscape described in the section <a href="https://distill.pub/2017/momentum/#momentum2D">"The Dynamics of Momentum"</a> describe the improvement along the direction _of a single eigenvector_. This partial view produces some misleading conclusions. For example, along the direction of a single eigenvector, the largest improvement is achieved with zero momentum and a step-size of 1 over the associated eigenvalue. This conclusion however doesn't hold in higher dimensions, where as we will see, the momentum term that yields the fastest convergence is non-zero.


## How fast is Momentum?

Momentum is _fast_. So fast that it's often the default choice of machine learning practitioners. But can we quantify this more precisely?

Throughout the post we'll assume that our objective function $$f$$ is a quadratic objective of the form
\begin{equation}\label{eq:opt}
f(\xx) \defas \frac{1}{2}(\xx - \xx^\star) \HH (\xx - \xx^\star)~,
\end{equation}
where $$\HH$$ is a symmetric positive definite matrix and $$\xx^\star$$ is the minimizer of the objective. We'll assume that the eigenvalues of $$\HH$$ are in the interval $$[\mu, L]$$.


The measure we'll use to quantify the speed of convergence is the rate of convergence. This is the worst-case relative improvement in the iterate suboptimality at iteration $$t$$, defined as
\begin{equation}\label{eq:convergence_rate}
r_t \defas \sup_{\xx_0, \text{eigs}(\HH) \in [\mu, L]} \frac{\\|\xx_{t} - \xx^\star\\|}{\\|\xx_{0} - \xx^\star\\|}\,.
\end{equation}
This is a worst-case measure because of all problem instances, we take worst possible initialization $$\xx_0$$ and matrix $$\HH$$ with eigenvalues in the interval $$[\mu, L]$$.


This is a measure of how much progress is made (in the worst-case) at iteration $$t$$. The smaller the value of $$r_t$$, the faster the algorithm converges. Since all algorithms that we consider converge exponentially fast, for large enough $$t$$ the error is of the order of $$\mathcal{O}{(\text{constant}^t)}$$. Hence the most informative quantity is the value of $$\text{constant}$$ in this expression. We'll call this quantity the <i>asymptotic rate of convergence</i>, and denote it:
\begin{equation}
r_{\infty} \defas \limsup_{t \to \infty} \sqrt[t]{r_t}\,.
\end{equation}
This is the quantity we'll be discussing throughout the post and what we'll use to compare the speed of momentum for different values of its hyperparameters.



### A connection between optimization methods and polynomials

To compute easily the asymptotic rate of convergence for all admissible values of step-size and momentum, we'll use a connection between optimization of quadratic functions and the theory of orthogonal polynomials. This theory was extensively used in the early days of numerical analysis <d-cite key="Rutishauser1959"></d-cite> and provides an elegant and simple way to compute asymptotic rates (and non-asymptotic ones, althought not the topic of this blog post) from known results in the theory of orthogonal polynomials. We favor this technique for its simplicity and elegance, although ones ones that also be used with identical results. Other techniques include the linear operator technique used by Polyak,<d-cite key="polyak1964some"></d-cite> the estimate sequences technique pioneered by Nesterov<d-cite key="nesterov1983method"></d-cite> or the use of Lyapunov functions.<d-cite key="JMLR:v22:20-195">



The main result that will allow us to make the link between optimization and orthogonal polynomials is the following result. It's origins seem unclear, although a proof can be found in the 1959 monograph of Rutishauser.<d-cite key="Rutishauser1959"></d-cite> 


<p class="lemma">
Consider the following polynomial \(P_t\) of degree \(t\), defined recursively as:
\begin{equation}
\begin{split}
&amp;P_{t+1}(\lambda) = (1 + \mom - \step \lambda ) P_{t}(\lambda) -
\mom P_{t-1}(\lambda)\\
&amp;P_1(\lambda) = 1 - \frac{\step}{1 + \mom} \lambda\,, ~ P_0(\lambda) = 1\,,~ 
\end{split}\label{eq:def_residual_polynomial2}
\end{equation}
Then we can write the suboptimality at iteration \(t\) as
\begin{equation}
\xx_t - \xx^\star = P_t(\HH) \left( \xx_0 - \xx^\star \right) \,,
\end{equation}
where \(P_t(\HH)\) is the matrix obtained from evaluating the (originally rel-valued) polynomial \(P_t\) at the matrix \(\HH\).
</p>


This last identity will allow us to easily compute convergence rates. In particular, plugging it into the definition of the convergence rate \eqref{eq:convergence_rate} we get that the rate is determined by the absolute value of the residual polynomial over the $$[\mu, L]$$ interval:
\begin{align}
r_t &amp;=  \sup_{\xx_0, \text{eigs}(\HH) \in [\mu, L]} \frac{\\|P_t(\HH) \left( \xx_0 - \xx^\star \right)\\|}{\\|\xx_{0} - \xx^\star\\|} \\\ 
&amp; = \sup_{\text{eigs}(\HH) \in [\mu, L]} \\|P_t(\HH)\\| \\\ 
&amp; = \sup_{\lambda \in [\mu, L]} \lvert P_t(\lambda) \rvert\,.
\end{align}
We've now reduced the problem of computing the convergence rate to the problem of computing the absolute value of a polynomial over a given interval. This is a problem that has been extensively studied in the theory of orthogonal polynomials. In particular, we'll use known bounds on Chebyshev polynomials of the first and second kind, as the residual polynomial of momentum can be written as a convex combination of these two polynomials. This fact is proven in the next result, which is a generalization of equation (II.29) in (Rutishauser 1959).<d-cite key="Rutishauser1959"></d-cite>



<p class="lemma">
The residual polynomial of momentum can be written in terms of Chebyshev polynomials of the first and second kind as
\begin{align}
P_t(\lambda) = \mom^{t/2} \left( {\small\frac{2\mom}{1+\mom}}\, T_t(\sigma(\lambda)) + {\small\frac{1 - \mom}{1 + \mom}}\,U_t(\sigma(\lambda))\right)\,.
\end{align}
where \(\sigma(\lambda) = {\small\dfrac{1}{2\sqrt{\mom}}}(1 + \mom - \step\,\lambda)\,\) is a linear function that we'll refer to as the <span class="underline">link function</span> and  \(T_t\) and \(U_t\) are the Chebyshev polynomials of the first and second kind respectively.
</p>

<div class="wrap-collapsible-XXX"> <input id="collapsible3" class="toggle" type="checkbox"> <label for="collapsible3" class="lbl-toggle" tabindex="0"><b>Show proof</b></label><div class="collapsible-content"><div class="content-inner"><div class="proof" id="proof-variance">
<p>
  Let's denote by \(\widetilde{P}_t\) the right hand side of the above equation, that is,
  \begin{equation}
  \widetilde{P}_{t}(\lambda) \defas \mom^{t/2} \left( {\small\frac{2
  \mom}{1 + \mom}}\,
  T_t(\sigma(\lambda))
  + {\small\frac{1 - \mom}{1 + \mom}}\,
  U_t(\sigma(\lambda))\right)\,.
  \end{equation}
  Our goal is to show that \(P_t = \widetilde{P}_t\) for all \(t\).
</p>
<p>
  For \(t=1\), \(T_1(\lambda) = \lambda\) and \(U_1(\lambda) = 2\lambda\), so we have
  \begin{align}
  \widetilde{P}_1(\lambda) &amp;= \sqrt{\mom} \left(\tfrac{2
  \mom}{1 + \mom} \sigma(\lambda) + \tfrac{1 - \mom}{1 + \mom} 2
  \sigma(\lambda)\right)\\
  &amp;= \frac{2 \sqrt{\mom}}{1 + \mom} \sigma(\lambda) = 1 - \frac{\step}{1 + \mom} \lambda\,,
  \end{align}
  which corresponds to the definition of \(P_1\) in \eqref{eq:def_residual_polynomial2}.
</p>
<p>
  Assume it's true for any iteration up to \(t\), we will show it's true for \(t+1\). Using the three-term recurrence of Chebyshev polynomials we have
  \begin{align}
  &amp;\widetilde{P}_{t+1}(\lambda) = \mom^{(t+1)/2} \left( {\small\frac{2 \mom}{1 + \mom}}\,
  T_{t+1}(\sigma(\lambda))
  + {\small\frac{1 - \mom}{1 + \mom}}\, U_{t+1}(\sigma(\lambda))\right) \\
  &amp;= \mom^{(t+1)/2} \Big( {\small\frac{2
  \mom}{1 + \mom}}\,
  (2 \sigma(\lambda) T_{t}(\sigma(\lambda)) - T_{t-1}(\sigma(\lambda))) \nonumber\\
  &amp;\qquad\qquad
  + {\small\frac{1 - \mom}{1 + \mom}}\, (2 \sigma(\lambda)
  U_{t}(\sigma(\lambda)) - U_{t-1}(\sigma(\lambda)))\Big)\\
  &amp;= 2 \sigma(\lambda) \sqrt{\mom} P_t(\lambda) - \mom P_{t-1}(\lambda)\\
  &amp;= (1 + \mom - \step \lambda) P_t(\lambda) -
  \mom P_{t-1}(\lambda)
  \end{align}
  where the third identity follows from grouping polynomials of same degree and the
  induction hypothesis. The last expression is the recursive definition of \(P_{t+1}\) in
  \eqref{eq:def_residual_polynomial2}, which proves the desired \(\widetilde{P}_{t+1} =
  {P}_{t+1}\).
</p>


</div></div></div></div>



### Tools of the trade: the two faces of Chebyshev polynomials


A key feature that we'll use extensively about Chebyshev polynomials is that they behave very differently inside and outside the interval $$[-1, 1]$$.  Inside this interval (shaded blue region) the magnitude of these polynomials stays close to zero, while outside it explodes:


{% include figure.html path="assets/img/2022-12-01-hitchhikers-momentum/two_phases_chebyshev.gif" class="img-fluid" %}

Let's make this observation more precise.

**Inside** the $$[-1, 1]$$ interval, Chebyshev polynomials admit the [trigonometric definitions](https://en.wikipedia.org/wiki/Chebyshev_polynomials#Trigonometric_definition) $$T_t(\cos(\theta)) = \cos(t \theta)$$ and $$U_{t}(\cos(\theta)) = \sin((t+1)\theta) / \sin(\theta)$$ and so they have an oscillatory behavior with values bounded in absolute value by 1 and $$t+1$$ respectively.


**Outside** of this interval the Chebyshev polynomials of the first kind admit the <a href="https://en.wikipedia.org/wiki/Chebyshev_polynomials#Explicit_expressions">explicit form</a> for $$|\xi| \ge 1$$:
\begin{align}
T_t(\xi) &amp;= \dfrac{1}{2} \Big(\xi-\sqrt{\xi^2-1} \Big)^t + \dfrac{1}{2} \Big(\xi+\sqrt{\xi^2-1} \Big)^t \\\\ 
U_t(\xi) &amp;= \frac{(\xi + \sqrt{\xi^2 - 1})^{t+1} - (\xi - \sqrt{\xi^2 - 1})^{t+1}}{2 \sqrt{\xi^2 - 1}}\,.
\end{align}
We're interested in convergence rates, so we'll look into $$t$$-th root asymptotics of the quantities.<d-footnote>With little extra effort, it would be possible to derive non-asymptotic convergence rates, although I won't pursue this analysis here.</d-footnote> Luckily, these asymptotics are the same for both polynomials<d-footnote>Although we won't use it here, this \(t\)-th root asymptotic holds for (almost) all orthogonal polynomials, not just Chebyshev polynomials. See for instance reference below</d-footnote> <d-cite key="stahl1990nth"></d-cite> and taking limits we have that
\begin{equation}
\lim_{t \to \infty} \sqrt[t]{|T_t(\xi)|} = \lim_{t \to \infty} \sqrt[t]{|U_t(\xi)|} = |\xi| + \sqrt{\xi^2 - 1}\,.
\end{equation}



## The Robust Region

Let's start first by considering the case in which the image of $$\sigma$$ is in the $$[-1, 1]$$ interval. 
This is the most favorable case. In this case, the Chebyshev polynomials are bounded in absolute value by 1 and $$t+1$$ respectively.
Since the Chebsyshev polynomials are evaluated at $$\sigma(\cdot)$$, this implies that $$\lvert \sigma(\lambda)\rvert \leq 1$$. We'll call the set of step-size and momentum parameters for which the previous inequality is verified the _robust region_.

Let's visualize this region in a map. Since $$\sigma$$ is a linear function, its extremal values are reached at the edges:
\begin{equation}
  \max_{\lambda \in [\lmin, \lmax]} |\sigma(\lambda)| = \max\{|\sigma(\lmin)|, |\sigma(\lmax)|\}\,.
\end{equation}
Using $$\lmin \leq \lmax$$ and that $$\sigma(\lambda)$$ is decreasing in $$\lambda$$, we can simplify the condition $$\lvert \sigma(\lambda)\rvert \leq 1$$ to $$\sigma(\lmin) \leq 1$$ and $$\sigma(L) \geq -1$$, which in terms of the step-size and momentum correspond to:
\begin{equation}\label{eq:robust_region}
\frac{(1 - \sqrt{\mom})^2}{\lmin} \leq \step \leq \frac{(1 + \sqrt{\mom})^2}{L} \,.
\end{equation}
These two conditions provide the upper and lower bound of the robust region.


{% include figure.html path="assets/img/2022-12-01-hitchhikers-momentum/sketch_robust_region.png" class="img-fluid" %}


### Asymptotic rate

Let $$\sigma(\lambda) = \cos(\theta)$$ for some $$\theta$$, which is always possible since $$\sigma(\lambda) \in [-1, 1]$$. In this regime, Chebyshev polynomials verify the identities $$T_t(\cos(\theta)) = \cos(t \theta)$$ and $$U_t(\cos(\theta)) = \sin((t+1)\theta)/\sin(\theta)$$ , which replacing in the definition of the residual polynomial gives
\begin{equation}
P_t(\sigma^{-1}(\cos(\theta))) = \mom^{t/2} \left[ {\small\frac{2\mom}{1+\mom}}\, \cos(t\theta) + {\small\frac{1 - \mom}{1 + \mom}}\,\frac{\sin((t+1)\theta)}{\sin(\theta)}\right]\,.
\end{equation}

Since the expression inside the square brackets is bounded in absolute value by $$t+2$$, taking $$t$$-th root and then limits we have $$\limsup_{t \to \infty} \sqrt[t]{\lvert P_t(\sigma^{-1}(\cos(\theta)))\rvert} = \sqrt{\mom}$$ for <i>any</i> $$\theta$$. This gives our first asymptotic rate:


<p class="framed" style="text-align: center">
  The asymptotic rate in the robust region is \(r_{\infty} = \sqrt{\mom}\).
</p>

This is nothing short of magical. It would seem natural &ndash;and this will be the case in other regions&ndash; that the speed of convergence should depend on both the step-size and the momentum parameter. Yet, this result implies that it's not the case in the robust region. In this region, the convergence <i>only</i> depends on the momentum parameter $\mom$. Amazing.<d-footnote>This insensitivity to step-size has been leveraged by Zhang et al. 2018 to develop a momentum tuner </d-footnote> <d-cite key="zhang2017yellowfin"></d-cite>

This also illustrates why we call this the <i>robust</i> region. In its interior, perturbing the step-size in a way that we stay within the region has no effect on the convergence rate. The next figure displays the asymptotic rate (darker is faster) in the robust region. 


{% include figure.html path="assets/img/2022-12-01-hitchhikers-momentum/rate_robust_region.png" class="img-fluid" %}


## The Lazy Region

Let's consider now what happens outside of the robust region. In this case, the convergence will depend on the largest of $$\{\lvert\sigma(\lmin)\rvert, \lvert\sigma(L)\rvert\}$$. We'll consider first the case in which the maximum is $$\lvert\sigma(\lmin)\rvert$$ and leave the other one for next section. 

This region is determined by the inequalities $$\lvert\sigma(\lmin)\rvert > 1$$ and $$\lvert\sigma(\lmin)\rvert \geq \lvert\sigma(L)\rvert$$.
Using the definition of $$\sigma$$ and solving for $$\step$$ gives the equivalent conditions
\begin{equation}
\step \leq \frac{2(1 + \mom)}{L + \lmin} \quad \text{ and }\quad \step \leq \frac{(1 - \sqrt{\mom})^2}{\lmin}\,.
\end{equation}
Note the second inequality is the same one as for the robust region \eqref{eq:robust_region} but with the inequality sign reversed, and so the region will be on the oposite side of that curve. We'll call this the <i>lazy region</i>, as in increasing the momentum will take us out of it and into the robust region.


{% include figure.html path="assets/img/2022-12-01-hitchhikers-momentum/sketch_lazy_region.png" class="img-fluid" %}



### Asymptotic rate

As we saw earlier, outside of the $$[-1, 1]$$ interval both Chebyshev have simple $$t$$-th root asymptotics.
Using this and that both kinds of Chebyshev polynomials agree in sign outside of the $$[-1, 1]$$ interval we can compute the asymptotic rate as
\begin{align}
\lim_{t \to \infty} \sqrt[t]{r_t} &amp;= \sqrt{\mom} \lim_{t \to \infty} \sqrt[t]{ {\small\frac{2\mom}{\mom+1}}\, T_t(\sigma(\lmin)) + {\small\frac{1 - \mom}{1 + \mom}}\,U_t(\sigma(\lmin))} \\\\ 
&amp;= \sqrt{\mom}\left(|\sigma(\lmin)| + \sqrt{\sigma(\lmin)^2 - 1} \right) \\\\ 
\end{align}
This gives the asymptotic rate for this region


<p class="framed" style="text-align: center">
  In the lazy region the asymptotic rate is \(r_{\infty} = \sqrt{\mom}\left(|\sigma(\lmin)| + \sqrt{\sigma(\lmin)^2 - 1} \right)\). 
</p>

Unlike in the robust region, this rate depends on both the step-size and the momentum parameter, which enters in the rate through the link function $$\sigma$$. This can be observed in the color plot of the asymptotic rate 



{% include figure.html path="assets/img/2022-12-01-hitchhikers-momentum/rate_lazy_region.png" class="img-fluid" %}


## Knife's Edge


The robust and lazy region occupy most (but not all!) of the region for which momentum converges. There's a small region that sits between the lazy and robust regions and the region where momentum diverges. We call this region the <i>Knife's edge</i>

For parameters not in the robust or lazy region, we have that $$|\sigma(L)| > 1$$ and $$|\sigma(L)| > |\sigma(\lmin)|$$. Using the asymptotics of Chebyshev polynomials as we did in the previous section, we have that the asymptotic rate is $$\sqrt{\mom}\left(|\sigma(L)| + \sqrt{\sigma(L)^2 - 1} \right)$$. The method will only converge when this asymptotic rate is below 1. Enforcing this results in $$\step \lt 2 (1 + \mom) / L$$. Combining this condition with the one of not being in the robust or lazy region gives the characterization:
\begin{equation}
\step \lt \frac{2 (1 + \mom)}{L}  \quad \text{ and } \quad \step \geq \max\Big\\{\tfrac{2(1 + \mom)}{L + \lmin}, \tfrac{(1 + \sqrt{\mom})^2}{L}\Big\\}\,.
\end{equation}


{% include figure.html path="assets/img/2022-12-01-hitchhikers-momentum/sketch_knife_edge.png" class="img-fluid" %}


### Asymptotic rate

The asymptotic rate can be computed using the same technique as in the lazy region. The resulting rate is the same as in that region but with $$\sigma(L)$$ replacing $$\sigma(\lmin)$$:


<p class="framed"  style="text-align: center">
  In the Knife's edge region the asymptotic rate is \(\sqrt{\mom}\left(|\sigma(L)| + \sqrt{\sigma(L)^2 - 1} \right)\).
</p>

Pictorially, this corresponds to 

{% include figure.html path="assets/img/2022-12-01-hitchhikers-momentum/rate_knife_edge.png" class="img-fluid" %}


## Putting it All Together

This is the end of our journey. We've visited all the regions on which momentum converges.<d-footnote>There's a small convergent region with <i>negative</i> momentum parameter that we haven't visited. Although not typically used for minimization, negative momentum has found applications in smooth games <a href="https://arxiv.org/abs/1807.04740">(Gidel et al., 2020)</a>.</d-footnote> The only thing left to do is to combine all the asymptotic rates we've gathered along the way.


<p class="theorem"> The asymptotic rate \(\limsup_{t \to \infty} \sqrt[t]{r_t}\) of momentum is
\begin{alignat}{2}
  &amp;\sqrt{\mom} &amp;&amp;\text{ if }\step \in \big[\frac{(1 - \sqrt{\mom})^2}{\lmin}, \frac{(1+\sqrt{\mom})^2}{L}\big]\\
&amp;\sqrt{\mom}(|\sigma(\lmin)| + \sqrt{\sigma(\lmin)^2 - 1})  &amp;&amp;\text{ if } \step \in \big[0, \min\{\tfrac{2(1 + \mom)}{L + \lmin}, \tfrac{(1 - \sqrt{\mom})^2}{\lmin}\}\big]\\
&amp;\sqrt{\mom}(|\sigma(L)| + \sqrt{\sigma(L)^2 - 1})&amp;&amp;\text{ if } \step \in \big[\max\big\{\tfrac{2(1 + \mom)}{L + \lmin}, \tfrac{(1 + \sqrt{\mom})^2}{L}\big\},  \tfrac{2 (1 + \mom) }{L} \big)\\
&amp;\geq 1 \text{ (divergence)} &amp;&amp; \text{ otherwise.}
\end{alignat}
</p>

Plotting the asymptotic rates for all regions we can see that Polyak momentum (the method with momentum $\mom = \left(\frac{\sqrt{L} - \sqrt{\lmin}}{\sqrt{L} + \sqrt{\lmin}}\right)^2$ and step-size $\step = \left(\frac{2}{\sqrt{L} + \sqrt{\lmin}}\right)^2$ which is asymptotically optimal among the momentum methods with constant coefficients) is at the intersection of the three regions.



{% include figure.html path="assets/img/2022-12-01-hitchhikers-momentum/rate_convergence_momentum.png" class="img-fluid" %}



## Reproducibility

All plots in this post were generated using the following Jupyer notebook: [[HTML]]({{'assets/html/2022-12-01-hitchhikers-momentum/hitchhikers-momentum.html' | relative_url}}) [[IPYNB]]({{'assets/html/2022-12-01-hitchhikers-momentum/hitchhikers-momentum.ipynb' | relative_url}})