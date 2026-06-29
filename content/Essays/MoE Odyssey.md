---
created: May 17, 2026
title: "[Translation] MoE Odyssey (MoE环游记)"
permalink: essays/MoE
type: technical
aliases:
  - moe
imageRaw: ../images/odyssey.png
unlisted: "true"
imageStyle: circle
---

**Author:** Jianlin Su (苏剑林) | [kexue.fm](https://kexue.fm)

*This is a compilation and AI-generated translation of Jianlin Su's wonderful series of blog posts "MoE环游记".*


## [Part 1: Starting from Geometric Interpretation](https://kexue.fm/archives/10699)

By Jianlin Su | 2025-02-08

A couple of years ago, in a stroke of inspiration, I started a "Transformer Upgrade Path" series, where I shared various improvements to mainstream Transformer architectures along with my personal reflections, which received recognition from some readers. Starting with this article, we follow the same style to introduce another mainstream architecture: MoE (Mixture of Experts).

The popularity of MoE goes without saying. The recently viral [DeepSeek-V3](https://papers.cool/arxiv/2412.19437) uses an MoE architecture, GPT-4 is rumored to also be MoE, and many recently released domestic models have adopted MoE as well. However, although research on MoE has a long history, its practical applications remained lukewarm for a long time. It was roughly from early last year's [*Mixtral of Experts*](https://papers.cool/arxiv/2401.04088) that MoE began to attract widespread attention. Its notable advantage is a large parameter count while keeping both training and inference costs significantly lower.

At the same time, MoE also has some challenges, such as training instability, load imbalance, and suboptimal performance — these were the main reasons it did not gain popularity in earlier years. However, with the increased attention over the past two years, these problems have been largely resolved, and we will discuss them one by one in subsequent articles.

### Problem Definition

First, I should point out that I will introduce MoE using my own line of reasoning. Where necessary, I will attach the corresponding references, but I will not systematically trace the origins of the MoE architecture. I hope readers will bear with me.

We know that Transformer models consist of Attention layers and MLP layers, and MoE replaces the MLP layers. MLP layers come in two varieties: FFN (FeedForward Network) and GLU (Gated Linear Unit). The mainstream choice is GLU, but for simplicity we will use FFN as our example:

$$\boldsymbol{y} = f(\boldsymbol{x}\boldsymbol{W}^{(A)})\boldsymbol{W}^{(B)}$$

where $\boldsymbol{x}\in\mathbb{R}^{d}$ is the input vector (a row vector), $\boldsymbol{W}^{(A)}\in\mathbb{R}^{d\times D},\boldsymbol{W}^{(B)}\in\mathbb{R}^{D\times d}$ are two parameter matrices, and $f$ is an element-wise activation function. Let $n$ be an integer that divides $D$. Then the above can be equivalently written using block matrices as:

$$\boldsymbol{y} = f\big(\boldsymbol{x}\begin{bmatrix}\boldsymbol{W}^{(A)}_1 & \boldsymbol{W}^{(A)}_2 & \cdots & \boldsymbol{W}^{(A)}_n\end{bmatrix}\big)\begin{bmatrix}\boldsymbol{W}^{(B)}_1 \\ \boldsymbol{W}^{(B)}_2 \\ \vdots \\ \boldsymbol{W}^{(B)}_n\end{bmatrix} = \sum_{i=1}^n \underbrace{f(\boldsymbol{x}\boldsymbol{W}^{(A)}_i)\boldsymbol{W}^{(B)}_i}_{\boldsymbol{v}_i}$$

where $\boldsymbol{W}^{(A)}_i = \boldsymbol{W}^{(A)}_{[:,(i-1)c:ic]}, \boldsymbol{W}^{(B)}_i = \boldsymbol{W}^{(B)}_{[(i-1)c:ic,:]}, c= D/n$, with slicing following Python conventions. From this we can see that FFN can be equivalently expressed as the sum of $n$ vectors $\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_n$, where each vector represents the output of a small model $f(\boldsymbol{x}\boldsymbol{W}^{(A)}_i)\boldsymbol{W}^{(B)}_i$. Each small model has the same computational cost, and these small models are the "Experts" in MoE.

The question MoE poses is:

> Can we select just $k$ vectors whose sum approximates the sum of all $n$ vectors? This would reduce computation to $k/n$ of the original.

### Sorting by Norm

This problem was actually explored in [*The Road to Low-Rank Approximation (Part 3): CR*](https://kexue.fm/archives/10427). Written as a mathematical formula:

$$\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_n\in\{0,1\}}\left\Vert\sum_{i=1}^n \lambda_i \boldsymbol{v}_i - \sum_{i=1}^n\boldsymbol{v}_i\right\Vert^2\quad\text{s.t.}\quad \sum_{i=1}^n \lambda_i = k$$

Let $\gamma_i = 1 - \lambda_i$, then it can be rewritten as:

$$\mathop{\text{argmin}}_{\gamma_1,\gamma_2,\cdots,\gamma_n\in\{0,1\}}\left\Vert\sum_{i=1}^n \gamma_i \boldsymbol{v}_i\right\Vert^2\quad\text{s.t.}\quad \sum_{i=1}^n \gamma_i = n - k$$

Finding the exact solution to this problem is quite difficult, but there is a simple approximate solution: when the $\boldsymbol{v}_i$ are pairwise orthogonal, we have:

$$\left\Vert\sum_{i=1}^n \gamma_i \boldsymbol{v}_i\right\Vert^2 = \sum_{i=1}^n \gamma_i^2 \Vert\boldsymbol{v}_i\Vert^2 = \sum_{i=1}^n \gamma_i \Vert\boldsymbol{v}_i\Vert^2$$

The optimal solution to the above is obviously to set $\gamma_i = 1$ for the $n-k$ vectors with the smallest norms $\Vert\boldsymbol{v}_i\Vert$, which is equivalent to selecting the $k$ vectors with the largest norms to approximate the sum of all $n$ vectors. When the $\boldsymbol{v}_i$ do not satisfy the pairwise orthogonality condition, we still use this as an approximate solution. Its geometric interpretation is also intuitive: vectors with larger norms are less likely to be cancelled out during summation, and thus play a more prominent role.

Furthermore, in [*The Road to Low-Rank Approximation (Part 3): CR*](https://kexue.fm/archives/10427) we also discussed a probabilistic sampling-based approximation process. Under the assumption of minimum variance, the optimal sampling probabilities are also proportional to the norms. So overall, sorting by vector norm is a simple yet effective strategy.

### The Emergence of MoE

Now we have a strategy — "select the $k$ vectors with the largest norms" — but upon closer reflection we realize it is not practical: to select the $k$ vectors with the largest norms, we must compute all the norms, which means computing all of the $\boldsymbol{v}_i$ first. But our original goal was to reduce the computation of $\boldsymbol{v}_i$!

To resolve this contradiction, we need to redesign each Expert model so that its norm can be computed at low cost. What does this mean? First, we normalize $\boldsymbol{v}_i$ to get $\boldsymbol{e}_i = \boldsymbol{v}_i/\Vert\boldsymbol{v}_i\Vert$, so that every $\boldsymbol{e}_i$ has the same norm. Then we define:

$$\underbrace{[\rho_1,\rho_2,\cdots,\rho_n]}_{\boldsymbol{\rho}} = h(\boldsymbol{x}\boldsymbol{W}^{(R)})\quad\in\mathbb{R}_{\geq 0}^n$$

where $\boldsymbol{W}^{(R)}\in\mathbb{R}^{d\times n}$ is a parameter matrix, and $h(\cdot)$ is an activation function mapping $\mathbb{R}\to\mathbb{R}_{\geq 0}$. In plain terms, this is just a linear transformation from $d$ dimensions to $n$ dimensions followed by an activation function, so it has relatively low computational cost. This component is called the "Router" in MoE.

What is the role of $\boldsymbol{\rho}$? It predicts the norm of each Expert! In other words, we use $\rho_i$ as the norm of the $i$-th Expert, and $\rho_i \boldsymbol{e}_i$ is the complete Expert. It is decomposed into two parts: the low-cost norm $\rho_i$ and the high-cost direction $\boldsymbol{e}_i$. To save computation, we first compute $\boldsymbol{\rho}$, select the top $k$, and only then compute the corresponding $\boldsymbol{e}_i$, finally multiplying by $\rho_i$ and summing:

$$\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i$$

This is the fundamental formula of the MoE model. Since only the Top-$k$ portion is retained in the computation, it is essentially a Sparse model. The original FFN, or the model when $k=n$, is typically called the corresponding Dense model.

### Summary of the Approach

Whether or not readers are familiar with MoE, the above process may seem somewhat unfamiliar, because this is my own independently conceived route to understanding MoE. However, because its geometric meaning is more explicit, it should fundamentally be easier to understand.

Let us reorganize the entire line of reasoning:

> 1. A standard Dense FFN model can be equivalently rewritten as the sum of $n$ Expert vectors $\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_n$.
>
> 2. To save computation, we attempt to select $k$ vectors whose sum approximates the original sum of $n$ vectors.
>
> 3. After formulating this as a mathematical optimization problem, we find that the selection rule is to pick the $k$ vectors with the largest norms.
>
> 4. Directly computing the norms of all $n$ Experts and then selecting $k$ does not actually save computation, so we need to redesign the Experts.
>
> 5. We normalize $\boldsymbol{v}_i$ to get $\boldsymbol{e}_i$, then use a separate small model (the Router) to predict the norm $\rho_i$. The final Expert is $\rho_i \boldsymbol{e}_i$.
>
> 6. At this point, we can first compute all $\rho_i$, select the top $k$, and only then compute $\boldsymbol{e}_i$, achieving the goal of saving computation.

### Why Go Through All This

Some readers might wonder: why go through this seemingly complex process? Isn't the standard MoE already easy to understand? The general MoE form is:

$$\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{v}_i$$

That is, there is no normalization of $\boldsymbol{v}_i$ before summation, and $\rho_i$ does not carry the meaning of a norm — it is purely a scoring model (the Router) used to rank Experts. But why does multiplying $\rho_i$ onto the Expert enable the Router to learn to correctly rank Experts? I have found that only [*Sparse Backpropagation for MoE Training*](https://papers.cool/arxiv/2310.00811) provides an explanation for this, but it is still somewhat lacking in intuition.

Under the geometric perspective of this article, however, many questions become immediately clear. After reparameterizing the Expert as $\rho_i \boldsymbol{e}_i$, the Dense model corresponds to summing all $\rho_i \boldsymbol{e}_i$, while MoE corresponds to summing after selecting the Top-$k$ by $\rho_i$. This is a theoretically guaranteed approximation of the Dense model. We do not consider how the Router should choose Experts; we simply try to approximate the Dense model as closely as possible at each step. This can be said to be the optimal choice for **both** having large parameters **and** having low computation.

Now that the geometric meaning of $\rho_i$ is a norm rather than a probability, the activation function $h(\cdot)$ has no normalization requirement. Besides Softmax, functions like Sigmoid and ReLU can also be considered, as can the smooth Top-$k$ approximation introduced in [*The Sequel to Softmax: Finding a Smooth Approximation to Top-K*](https://kexue.fm/archives/10373). Using a non-normalized activation function for the Router helps avoid destructive competition between Experts when $k > 1$, and can sometimes achieve better results.

One final note: we previously defined $\boldsymbol{e}_i = \boldsymbol{v}_i/ \Vert\boldsymbol{v}_i\Vert$ with the purpose of making all $\boldsymbol{e}_i$ have equal norms. In practice, L2 normalization is not strictly required — other equivalent operations can be used, such as RMS Norm with the gamma parameter fixed at 1, which better fits our output conventions.

### Article Summary

This article derives and understands MoE starting from the best approximation of a Dense model, arriving at a specific form of MoE that includes one additional Normalize step compared to existing MoE, but makes the geometric interpretation of MoE much more apparent. Of course, regardless of whether normalization is used or not, the MoE journey has only just begun — more challenges lie ahead.

---

## [Part 2: Not Scarcity but Inequality](https://kexue.fm/archives/10735)

By Jianlin Su | 2025-02-21

In the previous article, "MoE Odyssey: Part 1, Starting from Geometric Intuition," we introduced a geometric interpretation of MoE, aiming to derive and understand MoE by starting from the best approximation of a Dense model. At the end of that article, we also mentioned that writing down the MoE computation formula is merely the beginning — training a practically effective MoE model requires filling in many details, such as the Load Balance problem discussed in this article.

Load balance — "not scarcity but inequality"[^Confucius] — put plainly, means making every Expert work, and making them all do as close to the same amount of work as possible, to avoid some Experts wasting compute. Load balance is both a requirement for fully utilizing training compute and a requirement for maximizing the potential of MoE's large parameter count.

### Requirements Analysis

We know that the basic form of MoE is

$$\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i$$

For traditional MoE, $\boldsymbol{\rho}$ is a probability distribution (Router), $\boldsymbol{e}_i=\boldsymbol{v}_i$, and $\boldsymbol{v}_i$ is the output of a small FFN (Expert); for the geometric MoE we derived in the previous article, $\boldsymbol{\rho}$ has no normalization requirement — it predicts the magnitude of the Expert — while $\boldsymbol{e}_i=\boldsymbol{v}_i/\Vert\boldsymbol{v}_i\Vert$ predicts the direction of the Expert.

Regardless of which format of MoE is used, the actual performance is roughly the same — it is merely a difference in perspective. However, note that although the MoE formula gives the impression of "for each Token encountered, find the corresponding Expert to compute," during actual training the process is reversed: compute is first allocated to each Expert, and then Tokens are routed to their assigned Expert for parallel computation. This is precisely why the scoring function $\boldsymbol{\rho}$ is called a Router.

Given this, if Expert assignment is imbalanced, the following situation may arise: some Experts (Dead Experts) remain almost permanently idle, wasting compute; other Experts have too many Tokens to process and simply cannot keep up, resorting to Token Drop (i.e., abandoning processing of some Tokens). Theoretically, the emergence of Dead Experts means the MoE has not achieved its expected parameter count — you spend the memory budget of a large-parameter model but end up training only the effective capacity of a small-parameter model.

Therefore, from both the training and performance perspectives, we want to ensure load balance among Experts.

### Auxiliary Loss

The standard approach to promoting load balance is to add a related loss function, commonly referred to as "Aux Loss (Auxiliary Loss)." The Aux Loss in mainstream use today can be traced back to the 2020 paper *"GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding."*

Before introducing Aux Loss, we need to introduce some new concepts. First, we have already mentioned that for a general MoE, $\boldsymbol{\rho}$ is not necessarily a probability distribution. We denote the normalized $\boldsymbol{\rho}$ as $\boldsymbol{p}=[p_1,p_2,\cdots,p_n]$, and its Top-$k$ version as $\boldsymbol{f}=[f_1,f_2,\cdots,f_n]$, where

$$p_i = \frac{\rho_i}{\sum_{i=1}^n \rho_i},\qquad f_i = \left\{\begin{aligned}1/k, \quad i\in \mathop{\text{argtop}}\nolimits_k \boldsymbol{\rho} \\ 0, \quad i\not\in \mathop{\text{argtop}}\nolimits_k \boldsymbol{\rho}\end{aligned}\right.$$

Next, we define $\boldsymbol{P}=\mathbb{E}[\boldsymbol{p}],\boldsymbol{F}=\mathbb{E}[\boldsymbol{f}]$, where $\mathbb{E}$ denotes averaging over all Tokens across all samples. It is easy to see that $\boldsymbol{F}$ is the current load distribution of the Experts, while $\boldsymbol{P}$ serves as a smooth approximation of $\boldsymbol{F}$.

With these notations, we can write the Aux Loss as:

$$\mathcal{L}_{\text{aux}} = \boldsymbol{F}\cdot \boldsymbol{P} = \sum_{i=1}^n F_i P_i \tag{1}$$

Most literature defines Aux Loss with an additional factor of $n$, i.e., their Aux Loss equals $n \mathcal{L}_{\text{aux}}$ here. Additionally, some large-scale MoE models may compute Aux Loss per device to achieve intra-device balance and reduce inter-device communication — these are case-by-case adaptations. However, more recent experiments have shown that forcing local balance may very likely hurt the model's final performance.

### Straight-Through Estimation

I wonder if anyone has noticed a curious phenomenon: whether in the original source, subsequent literature, or tutorial articles — in all the materials the author has read — Aux Loss is cited without proof, as if everyone agrees that the above Aux Loss promoting balance is a self-evident fact. But is it really so obvious?

The author certainly did not find it obvious, so below I present a derivation approach for Equation $(1)$, from which we can also customize other forms of Aux Loss. First, define the uniform distribution $\boldsymbol{Q}=(1/n,1/n,\cdots,1/n)$. We just said that $\boldsymbol{F}$ is the current load distribution, so load balance is equivalent to $\boldsymbol{F}=\boldsymbol{Q}$. Then the following is a fairly intuitive Aux Loss:

$$\mathcal{L}_{\text{aux}} = \frac{1}{2}\Vert\boldsymbol{F} - \boldsymbol{Q}\Vert^2 = \frac{1}{2}\sum_{i=1}^n (F_i - 1/n)^2 \tag{2}$$

The problem is that $\boldsymbol{F}$ is produced by $\mathop{\text{argtop}}_k$, which means the above is not a directly usable differentiable objective. How do we solve this problem? The answer is the STE (Straight-Through Estimator) trick — designing separate functions for forward and backward passes. Specifically, $\boldsymbol{F}$ is non-differentiable, but $\boldsymbol{P}$ as its smooth approximation is differentiable, so we simply replace $\boldsymbol{F}$ with $\boldsymbol{P}$ during backpropagation:

$$\mathcal{L}_{\text{aux}} = \frac{1}{2}\Vert \boldsymbol{P} + \text{sg}[\boldsymbol{F}-\boldsymbol{P}] - \boldsymbol{Q}\Vert^2 = \frac{1}{2}\sum_{i=1}^n (P_i + \text{sg}[F_i - P_i] - 1/n)^2 \tag{3}$$

Here $\text{sg}[]$ is the stop-gradient operator, which preserves the forward output unchanged but forces the gradient to zero. After this modification, $\mathcal{L}_{\text{aux}}$ becomes a practically viable Aux Loss. Let us compute its gradient:

$$\begin{aligned}
\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{aux}} =&\, \frac{1}{2}\nabla_{\boldsymbol{\theta}}\sum_{i=1}^n (P_i + \text{sg}[F_i - P_i] - 1/n)^2 \\
=&\, \sum_{i=1}^n (P_i + \text{sg}[F_i - P_i] - 1/n) \nabla_{\boldsymbol{\theta}}(P_i + \text{sg}[F_i - P_i] - 1/n)\\
=&\, \sum_{i=1}^n (F_i - 1/n) \nabla_{\boldsymbol{\theta}}P_i = \nabla_{\boldsymbol{\theta}}\sum_{i=1}^n (F_i - 1/n) P_i\\
=&\, \nabla_{\boldsymbol{\theta}}\left(\sum_{i=1}^n F_i P_i\right)
\end{aligned}$$

Here $\boldsymbol{\theta}$ denotes the model parameters. The final result shows that the gradient of Equation $(3)$ equals the gradient of Equation $(1)$, which means using Equation $(1)$ as the Aux Loss is gradient-equivalent to Equation $(3)$. This is how Equation $(1)$ came to be used as the Aux Loss.

However, Equation $(1)$ only has meaning as an equivalent gradient — it does not have meaning as a loss, and is not a true loss function. For example, when $\boldsymbol{F} = \boldsymbol{P}$, we can compute that Equation $(1)$ equals $1/n$, but in fact we can construct an $\boldsymbol{F}$ not equal to $\boldsymbol{P}$ that makes it less than $1/n$. So Equation $(1)$ is not like a normal loss where smaller is always better, and its minimum is not achieved when $\boldsymbol{F} = \boldsymbol{P}$.

### General Form

The above derivation actually provides a general approach for constructing Aux Losses: **first construct a loss based on $\boldsymbol{F}$ that meets the requirements, then in implementation replace $\boldsymbol{F}$ with $\boldsymbol{P} + \text{sg}[\boldsymbol{F}-\boldsymbol{P}]$.**

For example, we know that maximum entropy can also push a distribution toward uniformity, so we can also use the negative entropy to construct an Aux Loss:

$$\mathcal{L}_{\text{aux}} = \sum_{i=1}^n (P_i + \text{sg}[F_i - P_i])\log(P_i + \text{sg}[F_i - P_i])$$

The above can be used directly in code implementation. Of course, if we pursue simplification, we can similarly compute the gradient, which yields:

$$\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{aux}} = \nabla_{\boldsymbol{\theta}}\sum_{i=1}^n(P_i + \text{sg}[F_i - P_i]) \log(P_i + \text{sg}[F_i - P_i]) = \nabla_{\boldsymbol{\theta}}\sum_{i=1}^n P_i \log F_i$$

In both gradient simplification steps, we used the following identity:

$$\sum_{i=1}^n \nabla_{\boldsymbol{\theta}}P_i = \nabla_{\boldsymbol{\theta}}\sum_{i=1}^n P_i = \nabla_{\boldsymbol{\theta}}1 = \boldsymbol{0}$$

This relies on the fact that $\boldsymbol{P}$ is a probability distribution and that the target distribution $\boldsymbol{Q}$ is uniform. If we do not pursue a simplified equivalent result but instead directly use the $\boldsymbol{F}\to \boldsymbol{P} + \text{sg}[\boldsymbol{F}-\boldsymbol{P}]$ form of the Aux Loss, then we are not constrained by these two conditions.

For example, regarding $\boldsymbol{P}$ as a smooth approximation of $\boldsymbol{F}$, we only used the property that "when $P_i$ is large, $F_i$ is typically also large," so using the non-normalized $\mathbb{E}[\boldsymbol{\rho}]$ as $\boldsymbol{P}$ usually works fine as well. This point may be critical in certain special scenarios (e.g., when $\boldsymbol{\rho}$ can be both positive and negative), since normalization into a probability distribution is then impossible. Similarly, the objective $\Vert\boldsymbol{F} - \boldsymbol{Q}\Vert^2$ can obviously push $\boldsymbol{F}$ toward any target distribution $\boldsymbol{Q}$ we desire, not necessarily uniform.

### Summary

This article introduced the load balance problem in MoE and presented a general approach for constructing Aux Losses. Besides Aux Loss, there are other methods for promoting load balance, which we will discuss next time.

[^Confucius]: *Translator's Note:* This quote and the title of this part allude to a classical Chinese saying from the Analects of Confucius: "不患寡而患不均" — "worry not about scarcity, but about inequality."

---

## [Part 3: A Different Approach to Assignment](https://kexue.fm/archives/10757)

By Jianlin Su | 2025-03-05

In this article, we continue exploring the load balancing problem in MoE. In the previous article, "MoE Odyssey: Part 2 - Not Scarcity but Inequality," we mainly discussed the approach of promoting load balance through Aux Loss. While Aux Loss is simple and intuitive, it has an obvious drawback — the weight is hard to tune — too low and it fails to promote balance, too high and it easily harms LM Loss. The community has therefore been continuously searching for alternatives.

What we will share in this article is the approach called "Loss-Free," proposed by DeepSeek in [*Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*](https://papers.cool/arxiv/2408.15664). Compared to DeepSeek's many dazzling open-source works, this paper may not seem particularly eye-catching, but in my view, its potential academic influence may far exceed their other work, because the proposed method is not only simple and effective, but also extremely general — a true classic.

### General Idea

When facing load imbalance, Aux Loss addresses it by using an additional loss to guide the Router toward balanced scores. Loss-Free instead takes a different approach to assignment: rather than changing the Router's existing scores, it changes the assignment mechanism $\mathop{\text{argtop}}_k \boldsymbol{\rho}$ itself.

There have been prior efforts in this direction. For example, in 2021 Facebook proposed [BASE Layer](https://papers.cool/arxiv/2103.16716), which treats expert assignment as a [linear assignment problem](https://en.wikipedia.org/wiki/Assignment_problem) — finding an assignment that maximizes the Router's total score subject to load balance constraints. This can be solved using the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) and similar methods. However, this approach requires knowing the scores of all tokens, so for autoregressive LLMs it is only applicable during training; inference still has to use $\mathop{\text{argtop}}_k \boldsymbol{\rho}$, creating an inconsistency between training and inference. Furthermore, due to current algorithmic limitations, it only applies to the $k=1$ case.

In contrast, Loss-Free's approach is very simple and effective. It observes a key fact: we can always introduce a bias term $\boldsymbol{b}$ such that the assignment from $\mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}$ is balanced. So it modifies the MoE formulation to:

$$\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i\qquad\to\qquad \boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}} \rho_i \boldsymbol{e}_i$$

Here $\boldsymbol{b}$ is an input-independent vector determined during training. Once training is complete it remains fixed, so it can also be used during inference — in other words, training and inference have a consistent form. Note that the coefficient multiplying $\boldsymbol{e}_i$ is still $\rho_i$ rather than $\rho_i + b_i$, meaning $\boldsymbol{b}$ only participates in the assignment process and not in the MoE forward computation. Therefore we have no special requirements on the sign of $\boldsymbol{b}$ or $\boldsymbol{\rho} + \boldsymbol{b}$.

### Hand-Crafted Gradients

How do we train $\boldsymbol{b}$? We know that the optimization direction for $\boldsymbol{b}$ should naturally promote load balance. Following the notation from the previous article, we first define $\boldsymbol{f}=[f_1,f_2,\cdots,f_n]$:

$$f_i = \left\{\begin{aligned}1/k, \quad i\in \mathop{\text{argtop}}\nolimits_k \boldsymbol{\rho}+\boldsymbol{b} \\
0, \quad i\not\in \mathop{\text{argtop}}\nolimits_k \boldsymbol{\rho}+\boldsymbol{b}\end{aligned}\right.$$

and $\boldsymbol{F}=\mathbb{E}[\boldsymbol{f}]$, where $\boldsymbol{F}$ is naturally the current load distribution of experts under the bias $\boldsymbol{b}$. Next we define the uniform distribution as $\boldsymbol{Q}=(1/n,1/n,\cdots,1/n)$, so that load balance amounts to minimizing:

$$\mathcal{L}_{\text{aux}} = \frac{1}{2}\Vert\boldsymbol{F} - \boldsymbol{Q}\Vert^2 = \frac{1}{2}\sum_{i=1}^n (F_i - 1/n)^2$$

This objective is non-differentiable, but with the experience from the previous article, we know that STE (Straight-Through Estimator) can solve this problem. The key to STE is finding a differentiable quantity that has the same monotonic trend as $\boldsymbol{F}$ to serve as a smooth approximation. Here our optimization parameter is only $\boldsymbol{b}$, and it happens to have exactly the property we want (increasing $b_i$ makes $i$ more likely to be selected, thus making $F_i$ larger). So the answer is readily apparent:

$$\mathcal{L}_{\text{aux}} = \frac{1}{2}\Vert\boldsymbol{b} + \text{sg}[\boldsymbol{F}-\boldsymbol{b}] - \boldsymbol{Q}\Vert^2 = \frac{1}{2}\sum_{i=1}^n (b_i + \text{sg}[F_i - b_i] - 1/n)^2$$

Its gradient is:

$$\nabla_{\boldsymbol{b}}\mathcal{L}_{\text{aux}} = \frac{1}{2}\nabla_{\boldsymbol{b}}\Vert\boldsymbol{b} + \text{sg}[\boldsymbol{F}-\boldsymbol{b}] - \boldsymbol{Q}\Vert^2 = \boldsymbol{F} - \boldsymbol{Q}$$

So updating $\boldsymbol{b}$ with gradient descent (SGD) gives:

$$\boldsymbol{b}\leftarrow \boldsymbol{b} - \gamma (\boldsymbol{F} - \boldsymbol{Q})$$

where $\gamma$ is the learning rate for $\boldsymbol{b}$. However, Loss-Free ultimately chooses a slightly different update rule — Sign Gradient Descent (SignSGD):

$$\boldsymbol{b}\leftarrow \boldsymbol{b} - \gamma \mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})\tag{1}$$

This result is actually quite intuitive: if $F_i$ is larger than $1/n$, then decrease $b_i$ a little; otherwise increase $b_i$ a little.

### An Improved Version

Beyond the sign gradient descent with $\mathop{\text{sign}}$, I have found that directly applying RMS Norm to $\boldsymbol{F} - \boldsymbol{Q}$ (i.e., Normalized SGD) often achieves better balance for the same $\gamma$:

$$\boldsymbol{b}\leftarrow \boldsymbol{b} - \gamma\frac{\boldsymbol{F} - \boldsymbol{Q}}{\text{RMS}(\boldsymbol{F} - \boldsymbol{Q})}$$

where $\text{RMS}$ is the "Root Mean Square," defined as:

$$\text{RMS}(\boldsymbol{F} - \boldsymbol{Q}) = \sqrt{\frac{1}{n}\sum_{i=1}^n (F_i - Q_i)^2}$$

It is easy to see that both $\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})$ and $\frac{\boldsymbol{F} - \boldsymbol{Q}}{\text{RMS}(\boldsymbol{F} - \boldsymbol{Q})}$ have an RMS of 1, so they are roughly the same scale, which means we can use the same $\gamma$.

Put simply, the problem with $\mathop{\text{sign}}$ is that it uses the same update magnitude regardless of how far $F_i$ is from the target $Q_i$. This causes $F_i$ values that are already close to $Q_i$ to easily deviate from the balance they had already achieved, producing oscillation. RMS Norm, on the other hand, preserves the relative magnitudes of $F_i - Q_i$, making the update magnitudes more adaptive. Theoretically this is more conducive to promoting balance, and in practice it usually performs better.

### A Unified Lineage

When the original paper introduces Loss-Free, it does not include the Aux Loss derivation above but directly gives the update rule $(1)$, giving the impression that it "hand-crafted" the gradient $\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})$ for $\boldsymbol{b}$. This is also the origin of the name "Loss-Free."

However, as the derivation in this article shows, update rule $(1)$ can be fully derived from the Aux Loss perspective — the two are of a unified lineage. While Loss-Free's most direct benefit appears to be eliminating the need to tune the Aux Loss weight, it actually still has a learning rate parameter $\gamma$ to tune. Although the original paper has already searched and found the default value $\gamma=0.001$, this hyperparameter undeniably exists.

In my view, Loss-Free's essential innovation is not the absence of Aux Loss, but rather the isolation of optimization parameters between Aux Loss and LM Loss, thereby achieving the dual goals of load balance and model capability. The most critical step is recognizing the fact that "a single bias term suffices for load balance," then letting Aux Loss optimize only the newly introduced bias $\boldsymbol{b}$ while LM Loss optimizes the remaining parameters, minimizing the negative impact of Aux Loss on LM Loss.

In contrast, the conventional Aux Loss approach requires all parameters to promote load balance, and LM Loss also optimizes all parameters. The two optimization directions may not be fully compatible, making it relatively more difficult to find an optimal balance point. Therefore, Loss-Free's approach of isolating the optimization parameters of the two losses based on "a single bias term suffices for load balance" is an elegant solution to the load balancing problem.

### Practical Details

Although Loss-Free is already simple and clear enough, there are some details to pay attention to in practice.

First, for each batch of data, we should first update the model parameters according to LM Loss, and then update $\boldsymbol{b}$ according to rule $(1)$. This is because the update of $\boldsymbol{b}$ depends on the statistical information $\boldsymbol{F}$ over all tokens. If we update $\boldsymbol{b}$ before updating the rest of the model parameters, there is in principle a risk of leaking future information. Although intuitively a single vector $\boldsymbol{b}$ cannot leak much information, this risk still exists and should be avoided.

Second, we mentioned that the original paper has tuned $\gamma=0.001$, but this result may be tied to the paper's choice of Sigmoid as the Router activation function for $\boldsymbol{\rho}$. The reason is not hard to see: after Sigmoid, each $\rho_i$ is relatively independent and lies within $(0,1)$, and $\gamma=0.001$ means the update magnitude per step is about one thousandth. If we switch to Softmax, ReLU, or other activation functions, we may need to re-tune $\gamma$.

To address this issue, my recommendation is to decouple the activation functions used for the Gate and the Bias, namely:

$$\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}} \rho_i \boldsymbol{e}_i\qquad\to\qquad \boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}^{(\sigma)} + \boldsymbol{b}} \rho_i^{(h)} \boldsymbol{e}_i$$

where $\boldsymbol{\rho}^{(\sigma)} = \sigma(\boldsymbol{x}\boldsymbol{W}^{(R)}), \boldsymbol{\rho}^{(h)} = h(\boldsymbol{x}\boldsymbol{W}^{(R)})$, $\sigma(\cdot)$ is the Sigmoid function, and $h(\cdot)$ is any monotonic function with non-negative range. In plain terms, the scores that $\boldsymbol{b}$ is added to use Sigmoid activation, allowing us to reuse $\gamma=0.001$. As for the Gate that multiplies the Expert output, we can use other activation functions, as long as their monotonicity is consistent with Sigmoid.

Additionally, because update rule $(1)$ uses the $\text{sign}$ function, it is possible to train $b_i$ values with absolute values greater than 1, and the overall absolute values may keep growing. This is all normal and has no impact on model performance. In fact, $\boldsymbol{b}$ has a redundant degree of freedom, because adding the same constant to all $b_i$ leaves $\mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}$ unchanged. This extra degree of freedom can be used for other interesting purposes (to be continued in the next installment).

### Further Reflections

Beyond MoE load balancing, the Loss-Free philosophy can be applied to many similar problems. For example, codebook collapse in VQ-VAE can be solved using the same approach, and compared to previously introduced "rotation tricks" and "linear transformation tricks," it appears more natural and general. In fact, the opening assessment that "Loss-Free's potential academic influence may far exceed other work" is precisely based on considerations of Loss-Free's generality.

Setting aside specific application contexts, from a mathematical perspective, Loss-Free's contribution can be understood as providing a method for solving assignment problems via gradient descent. A classic linear assignment problem can be expressed as:

$$\min_f \sum_{i=1}^n c_{i, f(i)}$$

where $c_{i,j}$ is a given cost function and $f$ is a bijection from $\{1,2,\cdots,n\}$ to itself. In the context of this article, $c_{i,j}$ corresponds to the scores of $n$ tokens and $n$ experts, and the desired $f$ is simply a load-balanced assignment. The general approach to solving such problems is to search for an optimal solution within the space satisfying the constraints. Loss-Free takes the opposite approach, first constructing an optimal solution that may not satisfy the constraints:

$$f(i) = \mathop{\text{argmin}}_j c_{i,j}$$

This solution is certainly optimal in terms of scores but may not satisfy the bijection condition — not satisfying bijection is equivalent to load imbalance. So we introduce a bias:

$$f(i) = \mathop{\text{argmin}}_j c_{i,j} + b_j$$

$b_j$ is initialized to zero and then updated according to rule $(1)$. The update rule simply says: whichever $j$ appears too many times, decrease the corresponding $b_j$; otherwise increase it — until a bijection is achieved.

### Summary

This article introduced the Loss-Free method for MoE load balancing, proposed by DeepSeek. Its core idea is achieving load balance through introducing a simple bias term. We further explored its connection to Aux Loss, as well as its potential for application to similar mathematical problems.

---

## [Part 4: Invest More Where It's Difficult](https://kexue.fm/archives/10815)

By Jianlin Su | 2025-03-28

In the previous two articles we discussed load balancing. When introducing the Loss-Free approach in "MoE Odyssey: Part 3, A Different Approach to Allocation," I left a cliffhanger: the Bias term it introduces has a redundant degree of freedom, and this degree of freedom can be used for something else that is interesting. In this article, we discuss exactly that.

We know that MoE selects only the top-$k$ most-matching Experts for each token to perform computation, thereby increasing parameter count while saving computation. However, upon careful reflection, this strategy clearly has room for improvement: intuitively, not all tokens are equally difficult, so a more reasonable approach would be to allocate more computational resources to difficult tokens and fewer resources to easy tokens. This might maximize performance under the same limited resource budget.

The extra degree of freedom in the Bias term mentioned above happens to provide a simple way to achieve this goal.

### Design Philosophy

First, let us review the basic form of MoE:

$$\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i$$

Load imbalance is a common problem in MoE training. Researchers have proposed Aux Loss to address this, which we covered in "MoE Odyssey: Part 2, Not Scarcity but Inequality." Additionally, in "MoE Odyssey: Part 3, A Different Approach to Allocation," we introduced the Loss-Free approach proposed by DeepSeek, which modifies MoE to:

$$\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}} \rho_i \boldsymbol{e}_i$$

It then achieves load balancing by adjusting the newly introduced Bias term $\boldsymbol{b}$. To allow each token to select a dynamic number of Experts, my proposed approach slightly modifies the Loss-Free formulation:

$$\boldsymbol{y} = \sum_{i\in \mathop{\text{argwhere}}\ \boldsymbol{\rho} + \boldsymbol{b} > 0} \rho_i \boldsymbol{e}_i$$

That is, any Expert satisfying $\rho_i + b_i > 0$ is selected. This way, the number of Experts chosen per token is naturally dynamic, and it eliminates the need for sorting — in some sense, it even becomes simpler.

### Optimization Objectives

The optimization of $\boldsymbol{b}$ has two objectives: first, like Loss-Free, to achieve **load balance**; second, to control the **average** number of Experts selected per token to be $k$, which we can call **budget control** — otherwise one could simply set $b_i = \infty$ to select all Experts, but that is not what we want.

Load balancing still follows the Loss-Free training procedure. Define the notation $\boldsymbol{f} = [f_1, f_2, \cdots, f_n]$:

$$f_i = \begin{cases}1, & \rho_i + b_i > 0 \\ 0, & \rho_i + b_i \leq 0\end{cases}$$

Then let $\tilde{\boldsymbol{F}}=\mathbb{E}[\boldsymbol{f}]$, so $\boldsymbol{F} = \tilde{\boldsymbol{F}}/|\tilde{\boldsymbol{F}}|$ is the current Expert distribution, where $|\tilde{\boldsymbol{F}}|$ is the sum of all components of $\tilde{\boldsymbol{F}}$. The update formula proposed by Loss-Free is:

$$\boldsymbol{b}\leftarrow \boldsymbol{b} - \gamma \mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})\tag{1}$$

where $\boldsymbol{Q}=(1/n, 1/n, \cdots, 1/n)$ is the target uniform distribution. As we have mentioned multiple times, $\boldsymbol{b}$ has a redundant degree of freedom: adding the same constant to all components of $\boldsymbol{b}$ does not change the sorting result. Given this, we can modify the update rule $(1)$ to:

$$\boldsymbol{b}\leftarrow \boldsymbol{b} - \gamma \left[\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q}) - \overline{\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})}\right]\tag{2}$$

Here, the overline on a vector denotes the mean of all its components (a scalar), and subtracting a scalar from a vector means subtracting it from each component. This ensures that $\boldsymbol{b}$ always satisfies $\overline{\boldsymbol{b}}=0$, without changing the load-balancing effect. We can then reserve the $\overline{\boldsymbol{b}}$ degree of freedom for budget control.

How should we understand this? Clearly, if we add the same positive number to all $b_i$, the probability of satisfying $\rho_i + b_i > 0$ increases, thereby increasing the total budget. So the approach is simple: first compute the current average budget, which is easily seen to be $|\tilde{\boldsymbol{F}}|$. If it is greater than $k$, decrease $\boldsymbol{b}$ slightly; otherwise increase it. Integrating this into Equation $(2)$:

$$\boldsymbol{b}\leftarrow \boldsymbol{b} - \gamma \left[\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q}) - \overline{\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})} + \mathop{\text{sign}}(|\tilde{\boldsymbol{F}}|- k)\right]\tag{3}$$

If we only want to ensure the budget does not exceed $k$, rather than requiring it to equal $k$ exactly, we can modify it so that no change is made when $|\tilde{\boldsymbol{F}}| < k$:

$$\boldsymbol{b}\leftarrow \boldsymbol{b} - \gamma \left[\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q}) - \overline{\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})} + \mathop{\text{sign}}(\max(|\tilde{\boldsymbol{F}}|- k,0))\right]\tag{4}$$

### Attempting Simplification

Examining Equation $(3)$ carefully, we see that it does two things: first, it pushes $\boldsymbol{F}=\tilde{\boldsymbol{F}}/|\tilde{\boldsymbol{F}}|$ toward $\boldsymbol{Q}$; second, it pushes $|\tilde{\boldsymbol{F}}|$ toward $k$. This appears to be combinable into a single objective: pushing $\tilde{\boldsymbol{F}}$ toward $\tilde{\boldsymbol{Q}}=k\boldsymbol{Q}=(k/n,k/n,\cdots,k/n)$. Thus Equation $(3)$ can be simplified to:

$$\boldsymbol{b}\leftarrow \boldsymbol{b} - \gamma \mathop{\text{sign}}(\tilde{\boldsymbol{F}} - \tilde{\boldsymbol{Q}})\tag{5}$$

I experimented with both Equation $(3)$ and Equation $(5)$, and found that their final performance is largely similar. However, Equation $(5)$ exhibits much greater fluctuation in both the load-balance and budget-control metrics during early training. Therefore, readers who prioritize stability may prefer Equation $(3)$ or $(4)$, while those who prefer simplicity can consider Equation $(5)$.

Since $\mathop{\text{sign}}$ only preserves the sign of $\tilde{F}_i - \tilde{Q}_i$ while discarding the magnitude, I also tried replacing $\mathop{\text{sign}}$ with RMS Norm:

$$\boldsymbol{b}\leftarrow \boldsymbol{b} - \gamma (\tilde{\boldsymbol{F}} - \tilde{\boldsymbol{Q}})/\Vert\tilde{\boldsymbol{F}} - \tilde{\boldsymbol{Q}}\Vert_{RMS}$$

Here $\Vert\cdot\Vert_{RMS}$ denotes the square root of the sum of squares of the components. Clearly the RMS of $\mathop{\text{sign}}$ is 1, and after RMS Norm the RMS is also 1, so both updates have the same magnitude and can use the same $\gamma$. Since RMS Norm preserves the relative magnitudes of $\tilde{F}_i - \tilde{Q}_i$, smaller errors lead to smaller updates, resulting in slightly less fluctuation than $\mathop{\text{sign}}$ — though the improvement is marginal.

Of course, replacing $\mathop{\text{sign}}$ with RMS Norm for increased stability is a general technique. Equations $(1)$, $(2)$, $(3)$, and $(4)$ can all undergo this replacement — it comes down to personal taste, as the stability gain is slight.

### Initialization

Having resolved the update rule for $\boldsymbol{b}$, let us consider the initialization of $\boldsymbol{b}$. This is an interesting but not critically important issue.

With the conventional approach of zero-initializing $\boldsymbol{b}$ and using Sigmoid activation for $\boldsymbol{\rho}$, the initial stage will select all $n$ Experts, clearly exceeding the $\leq k$ budget. This will cause a great deal of token dropping. However, if we are not overly particular about this, it is not a serious problem: the model's other parameters typically use warmup while $\boldsymbol{b}$ typically does not, so within the first few warmup steps the model will automatically resolve this issue.

If we do care about this, we can control the initial budget by adjusting the initialization of $\boldsymbol{b}$. Suppose the router's input is a $d$-dimensional vector with zero mean and unit variance (approximately true given RMSNorm), and the router's weight initialization variance is $\sigma^2$. Then the router's logits are approximately zero-mean with variance $\sigma^2 d$. With this information, we can use a normal approximation simulation plus binary search to estimate an initial $\boldsymbol{b}$:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def b_init(n, k, d, sigma, eps=0.1):
    b1, b2 = -1, 0
    std = sigma * d**0.5
    logits = np.random.randn(10000, n) * std
    scores = sigmoid(logits)
    while True:
        b = (b1 + b2) * 0.5
        c = ((scores + b) > 0).sum(1).mean()
        if -eps < c - k < eps:
            return b
        elif c > k:
            b2 = b
        else:
            b1 = b

b_init(32, 4, 1024, 6e-3)
```

The code considers Sigmoid activation, so the search interval is $[-1, 0]$. For other activation functions, please adjust accordingly. The recommendation here is the same as in "MoE Odyssey: Part 3, A Different Approach to Allocation": the $\boldsymbol{\rho}$ to which $\boldsymbol{b}$ is added can uniformly use Sigmoid activation, while the $\boldsymbol{\rho}$ that multiplies the Expert outputs can use a different activation function.

### Related Work

Before this article, several works have attempted MoE designs with dynamic Expert selection. Below I briefly list some works I found and offer some brief commentary from a personal aesthetic perspective.

The more straightforward approaches are [AdaMoE](https://papers.cool/arxiv/2406.13233) and [MoE++](https://papers.cool/arxiv/2410.07348). They mix in some low-computation-cost Experts among the standard ones, such as blank Experts, copy Experts, and constant Experts, while also encouraging load balance. When a token selects one of these simple Experts, it is equivalent to selecting fewer standard Experts, thus indirectly achieving a dynamic count. The advantage is that this can reuse the existing Top-$k$ MoE infrastructure, but it also lacks some flexibility.

Another straightforward idea is to replace Top-$k$ selection with Top-$p$, from ["Harder Tasks Need More Experts: Dynamic Routing in MoE Models"](https://papers.cool/arxiv/2403.07652). This conversion looks natural, but in practice it has quite a few problems. For instance, it cannot precisely control the average budget, because when $\boldsymbol{\rho}$ approaches a uniform distribution, the Top-$p$ proportion becomes very large. The original paper therefore adds an entropy loss to push $\boldsymbol{\rho}$ away from the uniform distribution. Overall, my personal feeling is that the problems it introduces are more apparent than the benefits.

A rather unique approach is [Ada-K Routing](https://papers.cool/arxiv/2410.10456), which adds a new module to predict the number of Experts to activate and then trains it with reinforcement learning. This is fine in principle, but introducing reinforcement learning undoubtedly increases training complexity. [DA-MoE](https://papers.cool/arxiv/2409.06669) uses attention scores to identify important tokens and allocates more Experts to them, but this feels insufficiently fundamental — since MoE is not in principle limited to FFN layers, once it is applied to the attention layer, there would be no attention scores available.

The work most similar in form to this article is probably [ReMoE](https://papers.cool/arxiv/2412.14711). It also uses a zero threshold to select Experts, but it relies on Aux Loss to achieve load balance and budget control, while also mixing in hand-crafted gradient ideas to control the Aux Loss weight. Overall it feels somewhat more patchwork. This article instead continues the Loss-Free philosophy, using the extra degree of freedom of $\boldsymbol{b}$ to regulate the threshold, thereby achieving dynamic Expert counts with minimal modifications.

### Summary

This article proposes an MoE design with dynamic Expert selection. The main idea is to slightly modify the Loss-Free MoE formulation and then adjust the Bias term's update rule, leveraging its extra degree of freedom to simultaneously achieve load balancing and budget control.

---

## [Part 5: Rethinking Uniform Distribution](https://kexue.fm/archives/10945)

By Jianlin Su | 2025-05-16

If Meta's LLaMA series established the standard architecture for dense models, then DeepSeek is perhaps the one who laid the foundation for the standard MoE architecture. Of course, this does not mean DeepSeek invented MoE, nor that its MoE cannot be surpassed, but rather that some of the improvements DeepSeek proposed for MoE likely represent directions with significant performance gains, gradually becoming standard practice for MoE. These include the Loss-Free load balancing scheme introduced in "MoE Odyssey: Part 3, A Different Approach to Allocation," as well as the Shared Expert and Fine-Grained Expert strategies that this article will introduce.

Speaking of load balancing, it is undoubtedly an extremely important objective for MoE. Parts 2 through 4 of this series can be said to revolve entirely around it. However, some readers have gradually come to realize that there is an essential question here that remains unanswered: **Setting aside efficiency requirements, is uniform distribution necessarily the direction that yields the best performance?** This article carries that question forward to understand Shared Expert and Fine-Grained Expert.

### Shared Expert

Let us once again review the basic form of MoE:

$$\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i$$

In addition, the Loss-Free method from "MoE Odyssey: Part 3" replaces $\mathop{\text{argtop}}_k \boldsymbol{\rho}$ with $\mathop{\text{argtop}}_k \boldsymbol{\rho}+\boldsymbol{b}$, and in "MoE Odyssey: Part 4, Invest More Where It's Difficult" we generalized it to $\mathop{\text{argwhere}}\ \boldsymbol{\rho}+\boldsymbol{b} > 0$. However, these variants are all orthogonal to the Shared Expert technique, so we will only use the most basic form as our example going forward.

Shared Expert modifies the above equation to:

$$\boldsymbol{y} = \sum_{i=1}^s \boldsymbol{e}_i + \sum_{i\in \mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s:]}} \rho_{i+s} \boldsymbol{e}_{i+s}$$

That is, the original top-$k$ out of $n$ selection is changed to top-$(k-s)$ out of $(n-s)$, while the other $s$ experts are always selected — these are called "Shared Experts." When this concept first appeared, we jokingly called them "permanent members of the Security Council." The remaining $n-s$ experts are called "Routed Experts." The number of Shared Experts $s$ is not large, typically 1 or 2; too many would cause the model to "neglect" the remaining Routed Experts.

It should be noted that with or without Shared Experts, the total number of experts remains $n$ and the number of activated experts remains $k$, so Shared Expert in principle does not increase model parameter count or inference cost. Nevertheless, [DeepSeekMoE](https://papers.cool/arxiv/2401.06066) and our own experiments show that Shared Expert can still improve model performance to a certain extent.

### Multiple Interpretations

We can understand Shared Expert from multiple perspectives. For example, from the residual perspective, the Shared Expert technique effectively changes the learning target from each individual expert to the residual between each expert and the Shared Expert. This reduces learning difficulty and provides better gradients. In DeepSeek's words: by compressing common knowledge into these Shared Experts, redundancy among Routed Experts is reduced, parameter efficiency is improved, and each Routed Expert is ensured to focus on unique aspects.

If we compare Routed Experts to subject teachers in a middle school, then the Shared Expert is like the "homeroom teacher." If a class only has subject teachers, each one will inevitably share some administrative work. Setting up the homeroom teacher role concentrates these common administrative tasks in one person, allowing subject teachers to focus on teaching their subjects and improving teaching efficiency.

We can also understand this from a geometric perspective. The unavoidable commonality among experts means geometrically that their vectors have angles less than 90 degrees, which contradicts the "pairwise orthogonal" assumption for expert vectors that we used when proposing the geometric interpretation of MoE in "MoE Odyssey: Part 1, Starting from Geometric Meaning." Although this assumption can still be understood as an approximate solution when it does not hold, it is naturally better when it does hold. We can interpret the Shared Expert as the mean of the Routed Experts; by learning the residuals after subtracting the mean, the orthogonality assumption becomes easier to satisfy.

### Scaling Factor

We write the Shared Expert equation more generally as:

$$\boldsymbol{y} = \sum_{i=1}^s \boldsymbol{e}_i + \lambda\sum_{i\in \mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s:]}} \rho_{i+s} \boldsymbol{e}_{i+s}$$

Because Routed Experts carry weights $\rho_{i+s}$ while Shared Experts do not, and because the number of Routed Experts is typically much larger than the number of Shared Experts (i.e., $n - s \gg s$), the scale between the two may become imbalanced. Therefore, to prevent one from overwhelming the other, setting an appropriate $\lambda$ is particularly important. In this regard, we proposed in ["Muon is Scalable for LLM Training"](https://papers.cool/arxiv/2502.16982) that a proper $\lambda$ should make the norms of the two components approximately equal during the initialization phase.

Specifically, we assume that each expert has the same norm during initialization (without loss of generality, we can set it to 1) and satisfies pairwise orthogonality. We then assume the router's logits follow a standard normal distribution (i.e., zero mean, unit variance; of course, one could consider other variances if deemed necessary). Under these assumptions, the total norm of the $s$ Shared Experts is $\sqrt{s}$, while the total norm of the Routed Experts is:

$$\lambda\sqrt{\sum_{i\in \mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s:]}} \rho_{i+s}^2}$$

By setting this equal to $\sqrt{s}$, we can estimate $\lambda$. Since different MoE routers can vary considerably due to choices of activation function, whether to renormalize, etc., we do not attempt to find an analytical solution but instead use numerical simulation directly:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return (p := np.exp(x)) / p.sum()

def scaling_factor(n, k, s, act='softmax', renorm=False):
    factors = []
    for _ in range(10000):
        logits = np.random.randn(n - s)
        p = np.sort(eval(act)(logits))[::-1][:k - s]
        if renorm:
            p /= p.sum()
        factors.append(s**0.5 / (p**2).sum()**0.5)
    return np.mean(factors)

scaling_factor(162, 8, 2, 'softmax', False)
scaling_factor(257, 9, 1, 'sigmoid', True)
```

Quite coincidentally, the simulation results from this script align very well with the settings of DeepSeek-V2 and DeepSeek-V3. For DeepSeek-V2 with $n=162, k=8, s=2$, using softmax activation without renormalization, the above script's simulation result is approximately 16, and DeepSeek-V2's $\lambda$ is exactly 16 ([source](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json#L48)). For DeepSeek-V3 with $n=257, k=9, s=1$, using sigmoid activation with renormalization, the script's result is approximately 2.83, while DeepSeek-V3's $\lambda$ is 2.5 ([source](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json#L57)).

### Non-Uniformity

Returning to the question posed at the beginning: is balance necessarily the direction that yields the best performance? It appears that Shared Expert provides a reference answer: not necessarily. Because Shared Expert can also be interpreted as certain experts always being activated, the overall result is a non-uniform expert distribution:

$$\boldsymbol{F} = \frac{1}{s+1}\bigg[\underbrace{1,\cdots,1}_{s},\underbrace{\frac{1}{n-s},\cdots,\frac{1}{n-s}}_{n-s}\bigg]$$

In fact, non-uniform distributions are ubiquitous in the real world, so it should be easy to accept that uniform distribution is not the optimal direction. Using the middle school teacher analogy again, the number of teachers for each subject within a school is actually non-uniform — there are typically the most for Chinese, math, and English, followed by physics, chemistry, and biology, with even fewer for physical education and art (and they are frequently "sick"). For more examples of non-uniform distributions, one can look up Zipf's law.

In summary, the non-uniformity of the real world inevitably leads to non-uniformity in natural language, which in turn leads to the sub-optimality of uniform distribution. Of course, from the perspective of training models, uniform distribution is still easier to parallelize and scale. Therefore, separating out some Shared Experts while still hoping the remaining Routed Experts are uniform is a compromise that is friendly to both sides for achieving non-uniformity, rather than directly forcing Routed Experts to align with a non-uniform distribution.

The above discussion concerns training — what about inference? During the inference phase, one can estimate the actual distribution of Routed Experts in advance, and there is no need to consider backpropagation. So with careful optimization, it is theoretically possible to achieve no loss in efficiency. However, since current MoE inference infrastructure is all designed for uniform distribution, and there are practical constraints such as limited single-GPU memory, we still prefer Routed Experts to be uniform for better inference efficiency.

### Fine-Grained Expert

Besides Shared Expert, another improvement proposed by [DeepSeekMoE](https://papers.cool/arxiv/2401.06066) is Fine-Grained Expert. It points out that with both total parameters and activated parameters held constant, finer expert granularity generally leads to better performance.

For example, suppose we originally have top-$k$ out of $n$ Routed Experts. Now we shrink each expert to half its size and switch to top-$2k$ out of $2n$. The total parameters and activated parameters remain the same, but the latter typically performs better. The original paper's explanation is that this enriches the diversity of expert combinations:

$$\binom{n}{k} \ll \binom{2n}{2k} \ll \binom{4n}{4k} \ll \cdots$$

Of course, we can have other interpretations — for instance, by splitting experts into smaller units, each expert can focus on a narrower domain of knowledge, achieving more refined knowledge decomposition, and so on. However, note that Fine-Grained Expert is not without cost: as $n$ grows larger, load imbalance among experts tends to worsen, and communication and coordination costs between experts also increase. Thus $n$ cannot grow without bound; there is a sweet spot where both performance and efficiency are favorable.

Regarding the effectiveness of Fine-Grained Expert, the author here proposes another less obvious explanation that relates to the theme of this article: **A larger number of finer-grained experts can better approximate the non-uniformity of the real world.** Consider the illustration below: suppose knowledge can be divided into one large category and one small category, and each expert is a circle. If we use 2 large circles to cover them, there will be some omission and waste. But if we instead use 8 small circles with the same total area, the coverage can be much more precise, and thus performance is better.

*[Figure: Fine-grained coverage is more precise]*

### Article Summary

This article introduced the Shared Expert and Fine-Grained Expert strategies for MoE, and pointed out that both, to some extent, reflect the sub-optimality of load balancing toward uniform distribution.

---

## [Part 6: Optimal Assignment for Load Balancing](https://kexue.fm/archives/11619)

By Jianlin Su | 2026-02-22

We know that Load Balance is a fundamental and critical component of the MoE architecture, directly affecting model efficiency and performance. This series has already introduced two mainstream approaches to achieving load balancing in two previous articles: the classic Aux Loss approach presented in "MoE Odyssey: 2, Inequality Rather Than Scarcity Is the Problem," and the Loss-Free approach proposed by DeepSeek in "MoE Odyssey: 3, A Different Approach to Assignment." Each has its strengths and limitations.

This article will explore a third approach: **optimal assignment**, which treats load balancing as a linear programming problem under equality constraints. In its final form, it still belongs to the Loss-Free category, but it is based on fundamentally different principles and provides a more accurate, hyperparameter-free update method.

### Method Review

Among the two existing methods, the Aux Loss approach is relatively straightforward — its core idea is "penalize wherever imbalance occurs," imposing penalties on load imbalance through regularization terms. However, Aux Loss has two problems: first, the penalty coefficient is difficult to tune — too large and it interferes with the main loss optimization, too small and the balancing effect is poor; second, Aux Loss relies on STE (Straight-Through Estimator), which means its gradients are suboptimal and may introduce unknown effects beyond load balancing.

To address this, DeepSeek proposed the second approach, Loss-Free, which introduces an additional bias term to assist with ranking, as shown below:

$$\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i\qquad\to\qquad \boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}} \rho_i \boldsymbol{e}_i$$

Note that $\boldsymbol{b}$ is only used to adjust the ranking of Experts — what gets multiplied onto the Expert output is still $\rho_i$ — so it does not directly participate in the model's computation and will not produce interfering gradients. However, the fact that $\boldsymbol{b}$ has no gradient means we need to design a custom update rule for it. The intuition is straightforward: the larger $b_i$ is, the higher the probability that the $i$-th Expert gets selected. So it first computes the current load distribution $\boldsymbol{F}$; if $F_i$ exceeds the expected value $1/n$, then $b_i$ is decreased, otherwise $b_i$ is increased:

$$\boldsymbol{b} \leftarrow \boldsymbol{b} - \gamma\mathop{\text{sign}}(\boldsymbol{F} - 1/n)$$

Overall, Loss-Free is less "invasive" to the model and can indeed be called more concise and elegant, but it is not without flaws. Although it does not have the penalty coefficient of Aux Loss, it still has a $\gamma$ parameter to tune — this is essentially the learning rate for $\boldsymbol{b}$. The paper recommends $\gamma=10^{-3}$, which is actually tightly coupled with using Sigmoid activation for $\boldsymbol{\rho}$; once you switch to a different activation function, $\gamma$ needs to be re-tuned.

Furthermore, even with Sigmoid activation, some layers have rather "pathological" distributions of $\boldsymbol{\rho}$, making the model sensitive to $\gamma$, and a constant $\gamma$ struggles to achieve load balance. This situation is not uncommon — for instance, if the first few layers of a model use MoE, they are often difficult to balance, which is why the "first_k_dense" operation exists; similarly, when the model is large or the total number of experts $n$ is large, individual layers may also be difficult to balance.

### Linear Programming

Let us describe the problem more precisely: suppose there are $m$ tokens, and the $i$-th token's router scores for $n$ experts are $\boldsymbol{s}_i = (s_{i,1},s_{i,2},\cdots,s_{i,n})$, giving a total of $mn$ scores. These scores may be positive or negative and are not necessarily within a predefined range. We wish to devise an assignment scheme based on these scores, determining which experts each token should activate.

We stipulate that each token selects $k$ experts. A basic approach is to select the $k$ experts with the highest scores, but this may lead to load imbalance — certain experts may be activated significantly more or fewer times than others. So we further stipulate that each expert is activated exactly $mk/n$ times. Under these two constraints, we seek the assignment scheme that maximizes the total score, formalized as:

$$\max_{x_{i,j}\in\{0,1\}} \sum_{i,j} x_{i,j}s_{i,j} \qquad\text{s.t.}\qquad \sum_j x_{i,j} = k,\quad \sum_i x_{i,j} = \frac{mk}{n}\tag{1}$$

Here $x_{i,j}=1$ indicates that the $i$-th token selects the $j$-th expert, and $x_{i,j}=0$ means it does not. Note that $mk/n$ must be an integer for both equality constraints to hold strictly — we assume this condition is satisfied. Having each expert activated exactly $mk/n$ times represents the ideal perfectly uniform state; in practice one can certainly relax this, but for theoretical derivation we adopt the strictest constraint.

Since $x_{i,j}$ can only take values 0 or 1 in the above, this is an integer programming problem. Discrete optimization is generally difficult, so we consider its relaxed version:

$$\max_{x_{i,j}\in[0,1]} \sum_{i,j} x_{i,j}s_{i,j} \qquad\text{s.t.}\qquad \sum_j x_{i,j} = k,\quad \sum_i x_{i,j} = \frac{mk}{n}\tag{2}$$

Now $x_{i,j}$ can take any real value in $[0,1]$, with all other conditions unchanged. Since $x_{i,j}$ is linear in both the objective function and the constraint equations, this is a linear programming problem over a bounded region.

### Minimax

Constrained optimization is generally not easy either, so we consider the unconstrained $\max$-$\min$ form (i.e., the "Lagrange multiplier method"):

$$\max_{x_{i,j}\in[0,1]}\min_{\alpha_i,\beta_j} \sum_{i,j} x_{i,j}s_{i,j} - \sum_i \alpha_i\left(\sum_j x_{i,j} - k\right) - \sum_j \beta_j\left(\sum_i x_{i,j} - \frac{mk}{n}\right)\tag{3}$$

If $\sum_j x_{i,j} = k$ and $\sum_i x_{i,j} = mk/n$, then the above is equivalent to equation $(2)$, and the maximum is a finite value; but if either constraint is violated, the $\min$ step can drive the value to negative infinity, making the maximum negative infinity. Since negative infinity is not as large as a finite value, only the first case can occur — meaning this is equivalent to equation $(2)$.

The objective $(3)$ is linear in $x_{i,j}$, $\alpha_i$, and $\beta_j$. A linear function is both convex and concave, and $[0,1]$ is a convex set, so it satisfies the conditions of the Minimax theorem. We can therefore swap the order of $\max$ and $\min$, obtaining:

$$\min_{\alpha_i,\beta_j} \max_{x_{i,j}\in[0,1]} \sum_{i,j} x_{i,j}(s_{i,j} - \alpha_i - \beta_j) +  k\sum_i \alpha_i + \frac{mk}{n}\sum_j \beta_j\tag{4}$$

In the above, we have separated the terms involving $x_{i,j}$, $\alpha_i$, and $\beta_j$. Upon careful inspection, the $\max$ step can actually be solved directly: when $s_{i,j} - \alpha_i - \beta_j > 0$, we set $x_{i,j}=1$ to maximize the objective; when it is less than 0, we set $x_{i,j}=0$; when it equals 0, $x_{i,j}$ can take any value with the same result. That is:

$$\left\{\begin{aligned}&\,x_{i,j}^* = 1, &\, s_{i,j} - \alpha_i - \beta_j > 0 \\
&\,x_{i,j}^* = 0, &\, s_{i,j} - \alpha_i - \beta_j < 0 \\
&\,x_{i,j}^* \in  [0,1], &\, s_{i,j} - \alpha_i - \beta_j = 0
\end{aligned}\right.$$

The case $s_{i,j} - \alpha_i - \beta_j = 0$ is a special situation. Assuming its probability of occurrence is negligible, $x_{i,j}^*$ is either 0 or 1. Of course, even when $s_{i,j} - \alpha_i - \beta_j = 0$, due to the arbitrariness of $x_{i,j}^*$, we can set it to 0 or 1 according to the constraints. This shows that although form $(2)$ is a relaxation of the original problem $(1)$, its optimal solution is also the optimal solution of the original problem — the two are completely equivalent.

### Divide and Conquer

Substituting the above $x_{i,j}^*$ into equation $(4)$, we get $x_{i,j}^*(s_{i,j} - \alpha_i - \beta_j) = \max(0, s_{i,j} - \alpha_i - \beta_j)$, so the optimization objective $(4)$ simplifies to:

$$\min_{\alpha_i,\beta_j} \sum_{i,j} \max(0, s_{i,j} - \alpha_i - \beta_j) +  k\sum_i \alpha_i + \frac{mk}{n}\sum_j \beta_j$$

We will use alternating minimization to solve this: first fix $\beta_j$ and solve for $\alpha_i$, then fix $\alpha_i$ and solve for $\beta_j$, alternating between the two. Since $\alpha_i$ and $\beta_j$ have obvious symmetry, these two steps are essentially the same problem. Let us first look at fixing $\beta_j$ and solving for $\alpha_i$; the problem is equivalent to:

$$\min_{\alpha_i} \sum_{i,j} \max(0, s_{i,j} - \alpha_i - \beta_j) +  k\sum_i \alpha_i$$

We can also observe that each $\alpha_i$ is accumulated independently, so we can decompose this into $m$ independent sub-optimization problems. This means we can temporarily drop the subscript $i$ and further simplify the problem to:

$$\min_{\alpha}  k\alpha + \sum_j \max(0, s_j - \beta_j - \alpha)$$

Sort all $s_j - \beta_j$ in descending order as $s_{\sigma_1} - \beta_{\sigma_1} \geq s_{\sigma_2} - \beta_{\sigma_2} \geq \cdots \geq s_{\sigma_n} - \beta_{\sigma_n}$, where the $j$-th largest element is $s_{\sigma_j} - \beta_{\sigma_j}$. Now suppose we know that $s_{\sigma_l} - \beta_{\sigma_l} \geq \alpha \geq s_{\sigma_{l+1}} - \beta_{\sigma_{l+1}}$; then the objective function equals:

$$k\alpha + \sum_{j=1}^l (s_{\sigma_j} - \beta_{\sigma_j} - \alpha) = \left\{\begin{aligned}
&\,\sum_{j=1}^k (s_{\sigma_j} - \beta_{\sigma_j}) + \sum_{j=k+1}^l \underbrace{(s_{\sigma_j} - \beta_{\sigma_j} - \alpha)}_{\geq 0},&\, l \geq k \\
&\,\sum_{j=1}^k (s_{\sigma_j} - \beta_{\sigma_j}) - \sum_{j=l+1}^k \underbrace{(s_{\sigma_j} - \beta_{\sigma_j} - \alpha)}_{\leq 0},&\, l \leq k \\
\end{aligned}\right.$$

This shows that whether $l > k$ or $l < k$, the objective function is amplified. Therefore the minimum can only be achieved at $l=k$, at which point the result does not depend on the specific value of $\alpha$ — meaning $\alpha^*$ can be any number between the $k$-th largest and $(k+1)$-th largest elements of $s_j - \beta_j$. By convention, we take $\alpha^*$ to be the $(k+1)$-th largest element.

### Alternating Iteration

Restoring the subscript $i$, we obtain: for any given $i$, $\alpha_i^*$ is the $(k+1)$-th element when all $s_{i,j} - \beta_j$ are sorted in descending order. Similarly, fixing $\alpha_i$ and solving for $\beta_j$ yields: for any given $j$, $\beta_j^*$ is the $(mk/n+1)$-th element when all $s_{i,j} - \alpha_i$ are sorted in descending order. We alternate between these two steps as the final solution algorithm.

Suppose we have obtained sufficiently accurate $\boldsymbol{\alpha}^*$ and $\boldsymbol{\beta}^*$. Then from the preceding analysis, $x_{i,j}^*$ automatically satisfies the 0-or-1 property and the constraints. Since $x_{i,j}^*=1$ corresponds to $s_{i,j} - \alpha_i^* - \beta_j^* > 0$, combined with the constraint $\sum_j x_{i,j}^* = k$, we can conclude that for each token $i$, the selected experts are necessarily the Top-$k$ of $\boldsymbol{s}_i - \boldsymbol{\beta}^*$.

This tells us that only $\boldsymbol{\beta}^*$ is needed during inference — $\boldsymbol{\alpha}^*$ is merely an intermediate variable in the solution process that can be discarded after training. This point is crucial because the size of $\boldsymbol{\beta}$ is a fixed $n$, while the size of $\boldsymbol{\alpha}$ is $m$ (the global batch size), which changes dynamically and is therefore not a suitable inference format. The solution procedure leveraging this property is shown below, where $\mathop{\text{des\_sort}}$ denotes descending sort.

$$\begin{array}{|l|}
\hline
\text{Quantile Balancing (QB): Alternating solution algorithm for problem }(1) \\[4pt]
\hline
\text{Input: Score matrix }\boldsymbol{s}\in\mathbb{R}^{m\times n} \\
\text{Output: Assignment scheme }\boldsymbol{x}\in\{0,1\}^{m\times n} \\[4pt]
\hline
\begin{array}{ll}
1: & \text{Initialize }\boldsymbol{\beta} = \boldsymbol{0}_{1\times n} \\
2: & \textbf{For }t=1,2,\cdots,T\textbf{ do } \\
3: & \qquad \boldsymbol{\alpha} \leftarrow \mathop{\text{des\_sort}}(\boldsymbol{s} - \boldsymbol{\beta}, \text{axis=1})_{[:, k:k+1]} \\
4: & \qquad \boldsymbol{\beta} \leftarrow \mathop{\text{des\_sort}}(\boldsymbol{s} - \boldsymbol{\alpha}, \text{axis=0})_{[mk/n:mk/n+1]} \\
5: & \text{Output } x_{i,j}=1 \text{ if } j\in\mathop{\text{argtop}}_k \boldsymbol{s}_i - \boldsymbol{\beta} \text{ else } 0
\end{array} \\
\hline
\end{array}$$

An additional improvement is that we can use the concept of "quantile" to unify "the $(k+1)$-th largest element among $n$ numbers" and "the $(mk/n+1)$-th largest element among $m$ numbers" — they are both the "$1-k/n$ quantile" along their respective dimensions. Numerical frameworks such as NumPy, JAX, and PyTorch all have a "quantile" function implementation; using it avoids performing a full sort of the data, saving some computational complexity.

It is precisely for this reason that we call this algorithm **Quantile Balancing (QB)**.

### Watch Out for the Trap

But it is not time to celebrate yet — there is an inconspicuous trap here that, if fallen into, may yield fundamentally incorrect results and requires special caution.

As mentioned earlier, QB inference only uses $\boldsymbol{\beta}$, and storing only $\boldsymbol{\beta}$ during training suffices as well. So from the Loss-Free perspective, QB provides a new bias update method — calling it "Quantile Bias" would not be inappropriate. Whether it is the original SignSGD or the Quantile method in this article, they both depend on the scores of all tokens, so the order must not be confused: **one must use the old $\boldsymbol{\beta}$ to select experts for the current batch of data, and only then update the $\boldsymbol{\beta}$ value — this ensures there is no information leakage problem.**

Some readers might wonder: how can a bias vector that does not directly participate in forward computation leak anything? Indeed, intuitively the leaked information is very little, but the risk nonetheless exists. Perhaps for training small models we can try the leaking version, but for training large models one should not take this risk — because large models are powerful, powerful enough to amplify any subtle bug. Therefore, maintaining training-inference consistency and eliminating any information leakage risk is a fundamental principle when training large models.

Combining with actual training scenarios, QB can be further adjusted: instead of zero-initializing and iterating $T$ times, we can start from the previous step's bias and iterate only once per step. This avoids overfitting to the current batch while reducing computation. Selecting Top-$k$ based on $\boldsymbol{s} - \boldsymbol{\beta}$ is something the original MoE already does; now it merely changes to selecting Top-$(k+1)$, adding almost no cost. So the additional step is using $\boldsymbol{s} - \boldsymbol{\alpha}$ to find the "$1-k/n$ quantile" along $\text{axis=0}$.

$$\begin{array}{|l|}
\hline
\text{Quantile Balancing (QB): Practical form} \\[4pt]
\hline
\text{Input: Score matrix }\boldsymbol{s}\in\mathbb{R}^{m\times n}\text{, previous }\boldsymbol{\beta}\in\mathbb{R}^n \\
\text{Output: Assignment scheme }\boldsymbol{x}\in\{0,1\}^{m\times n}\text{, new }\boldsymbol{\beta}\in\mathbb{R}^n \\[4pt]
\hline
\begin{array}{ll}
1: & x_{i,j}=1 \text{ if } j\in\mathop{\text{argtop}}_k \boldsymbol{s}_i - \boldsymbol{\beta} \text{ else } 0 \\
2: & \boldsymbol{\alpha} \leftarrow \mathop{\text{des\_sort}}(\boldsymbol{s} - \boldsymbol{\beta}, \text{axis=1})_{[:, k:k+1]} \\
3: & \boldsymbol{\beta} \leftarrow \mathop{\text{des\_sort}}(\boldsymbol{s} - \boldsymbol{\alpha}, \text{axis=0})_{[mk/n:mk/n+1]} \\
4: & \text{Output } \boldsymbol{x},\boldsymbol{\beta}
\end{array} \\
\hline
\end{array}$$

However, even with just one iteration, this additional step is still relatively expensive — it needs to find the $(mk/n+1)$-th largest element among $m$ elements, where $m$ equals "global number of samples times sequence length," typically starting at millions. Under various parallelism strategies and gradient accumulation, exact implementation is usually unacceptable. A compromise is to divide the samples into the largest mini-batches we can accept, compute a separate $\boldsymbol{\beta}$ for each mini-batch according to the formula, and then average them as the final result.

After resolving these issues, QB is almost entirely advantageous: first, it has no hyperparameters like learning rates to tune; second, it balances extremely fast and is especially good at handling extreme cases — for example, using it to train a full-MoE model, even the first MoE layer becomes very well balanced. However, for layers where the original SignSGD rule already achieves balance, QB typically offers no additional advantage.

### Demo Code

Here is a demo code snippet for interested readers to experiment with:

```python
import numpy as np

def quantile_bias(s, k, T=5):
    """Alternating quantile to find optimal bias
    Principle: https://kexue.fm/archives/11619
    """
    m, n = s.shape
    beta = np.zeros((1, n))
    for _ in range(T):
        alpha = np.quantile(s - beta, 1 - k / n, axis=1, keepdims=True)
        # alpha = alpha.clip(0, np.inf)  # BIP would add this step
        beta = np.quantile(s - alpha, 1 - k / n, axis=0, keepdims=True)
        # beta = beta.clip(0, np.inf)  # BIP would add this step
    return beta

def max_min_avg_vio(s, k):
    """Compute max_vio, min_vio, and avg_vio
    Where max_vio >= 0, avg_vio >= 0, -1 <= min_vio <= 0; all three closer to 0 means better balance
    """
    m, n = s.shape
    topk = np.argsort(-s, axis=1)[:, :k]
    f = np.bincount(topk.reshape(-1), minlength=n)
    f = f / f.sum() * n - 1
    return f.max(), f.min(), np.abs(f).mean()

m, n, k = 100000, 256, 8
s = np.random.rand(m, n) + np.random.rand(n)  # Simulate non-uniform scores

b = quantile_bias(s, k, 5)
max_min_avg_vio(s, k)  # max_vio, min_vio, avg_vio with direct top-k
max_min_avg_vio(s - b, k)  # max_vio, min_vio, avg_vio with top-k after subtracting bias
```

### Related Work

The earliest work viewing MoE load balancing from the optimal assignment perspective is the paper "BASE Layers: Simplifying Training of Large, Sparse Models," but the complete general solution was proposed in the paper "Binary-Integer-Programming Based Algorithm for Expert Load Balancing in Mixture-of-Experts Models" (abbreviated as BIP). QB is an improvement upon BIP.

Compared to QB's objective $(1)$, BIP changes the equality constraints to inequality constraints:

$$\max_{x_{i,j}\in\{0,1\}} \sum_{i,j} x_{i,j}s_{i,j} \qquad\text{s.t.}\qquad \sum_j x_{i,j} \leq k,\quad \sum_i x_{i,j} \leq \frac{mk}{n}\tag{5}$$

Then repeating the preceding series of derivations, at the $\max$-$\min$ form step, an additional constraint $\alpha_i\geq 0, \beta_j\geq 0$ appears — that is, we must consider:

$$\max_{x_{i,j}\in[0,1]}\min_{\alpha_i\geq 0,\beta_j\geq 0} \sum_{i,j} x_{i,j}s_{i,j} - \sum_i \alpha_i\left(\sum_j x_{i,j} - k\right) - \sum_j \beta_j\left(\sum_i x_{i,j} - \frac{mk}{n}\right)$$

for it to be equivalent to the relaxed version of problem $(5)$. Next, $\max$ and $\min$ can still be swapped, and repeating a similar derivation process, the iteration rules for $\boldsymbol{\alpha}$ and $\boldsymbol{\beta}$ gain an additional $\max(0,\cdot)$ truncation operation to ensure non-negativity:

$$\begin{aligned}
\boldsymbol{\alpha} \leftarrow &\, \max(0, \mathop{\text{des\_sort}}(\boldsymbol{s} - \boldsymbol{\beta}, \text{axis=1})_{[:, k:k+1]}) \\
\boldsymbol{\beta} \leftarrow &\, \max(0, \mathop{\text{des\_sort}}(\boldsymbol{s} - \boldsymbol{\alpha}, \text{axis=0})_{[mk/n:mk/n+1]})
\end{aligned}$$

However, the author's testing found that this truncation operation significantly slows down load balancing, and frequently produces cases where it can only "rob the rich" but cannot "help the poor" (experts with extremely high frequency get suppressed, but experts with extremely low frequency cannot be rescued). This indicates that the truncation operation is entirely counterproductive. Therefore, QB directly changes the original problem to equality constraints, which removes the non-negativity constraints on $\boldsymbol{\alpha}$ and $\boldsymbol{\beta}$, making the solution maximally simple.

Additionally, BIP appears to commit the mistake mentioned in the previous section — its approach is to first update $\boldsymbol{\beta}$ and then select Top-$k$, which violates the causal principle (leaking future information) and the training-inference consistency principle (across samples). Nevertheless, BIP's complete analysis of load balancing from the optimal assignment perspective remains highly instructive.

After searching, the author found that after BIP, several other works have explored this direction. They overlap somewhat with this article but are not entirely identical; we will not discuss each one in detail here:

> "Maximum Score Routing For Mixture-of-Experts"[^MaxScore]
>
> "Selective Sinkhorn Routing for Improved Sparse Mixture of Experts"[^Sinkhorn]
>
> "MicroMoE: Fine-Grained Load Balancing for Mixture-of-Experts with Token Scheduling"[^MicroMoE]
>
> "A Theoretical Framework for Auxiliary-Loss-Free Load Balancing of Sparse Mixture-of-Experts in Large-Scale AI Models"[^TheoreticalFramework]

### Gradient Descent

Finally, we introduce a solution approach that lies between Loss-Free and QB. We know that in QB's iteration format, computing $\boldsymbol{\alpha}$ is relatively cheap — what is truly expensive is computing $\boldsymbol{\beta}$, which requires performing a certain kind of sorting across all tokens. Suppose $\boldsymbol{\alpha}$ is given; then the optimization objective for $\boldsymbol{\beta}$ is:

$$\min_{\beta_j} \underbrace{\sum_{i,j} \max(0, s_{i,j} - \alpha_i - \beta_j) +  \frac{mk}{n}\sum_j \beta_j}_{\triangleq\,\ell}$$

Besides using Quantile to find its optimal solution, is there a cheaper approach to find an approximate solution? Indeed there is! The objective function $\ell$ is clearly differentiable, so we can consider gradient descent. Its gradient is:

$$\frac{\partial\ell}{\partial\beta_j} = \frac{mk}{n} - \sum_{i=1}^m \chi(s_{i,j} - \alpha_i - \beta_j > 0)$$

where $\chi$ is the indicator function with $\chi(\text{True})=1, \chi(\text{False})=0$. Clearly this gradient is also relatively cheap to compute. With the gradient in hand, we can perform gradient descent. To align with Loss-Free, we consider SignSGD:

$$\beta_j \leftarrow \beta_j - \gamma\mathop{\text{sign}}\left(\frac{\partial\ell}{\partial\beta_j}\right)$$

Using this to replace the Quantile update of $\boldsymbol{\beta}$ in QB achieves cost comparable to Loss-Free, with empirical performance between the two (under the same Sigmoid activation and the same $\gamma$ as Loss-Free). In fact, we can further prove that if the $k$-th largest and $(k+1)$-th largest elements in each row of the score matrix $\boldsymbol{s} - \boldsymbol{\beta}$ are never equal, then this approach is strictly equivalent to Loss-Free.

### Conclusion

In this article, we explored the load balancing problem of MoE from the optimal assignment perspective and derived a new Aux-Loss-free load balancing algorithm called Quantile Balancing. It is more stable and accurate than existing Loss-Free approaches, is applicable to router scores of arbitrary range, and has no extra hyperparameters to tune.

[^MaxScore]: [Maximum Score Routing For Mixture-of-Experts](https://arxiv.org/abs/2508.12801)
[^Sinkhorn]: [Selective Sinkhorn Routing for Improved Sparse Mixture of Experts](https://arxiv.org/abs/2511.08972)
[^MicroMoE]: [MicroMoE: Fine-Grained Load Balancing for Mixture-of-Experts with Token Scheduling](https://arxiv.org/html/2511.16947v1)
[^TheoreticalFramework]: [A Theoretical Framework for Auxiliary-Loss-Free Load Balancing of Sparse Mixture-of-Experts in Large-Scale AI Models](https://arxiv.org/abs/2512.03915)

---

## [Part 7: A Minimalist Solution for Dynamic Activation](https://kexue.fm/archives/11626)

By Jianlin Su | 2026-02-23

In the previous article, "MoE Odyssey: Part 6, Optimal Assignment for Load Balancing," we achieved load balancing by solving the following optimal assignment problem:

$$\max_{x_{i,j}\in\{0,1\}} \sum_{i,j} x_{i,j}s_{i,j} \qquad\text{s.t.}\qquad \sum_j x_{i,j} = k,\quad \sum_i x_{i,j} = \frac{mk}{n}\tag{1}$$

where $\sum_j x_{i,j} = k$ means each token activates exactly $k$ experts, and $\sum_i x_{i,j} = mk/n$ means each expert is activated exactly $mk/n$ times. However, upon careful reflection, the former constraint is actually unnecessary for both training and inference. What we truly need is the latter, which implies that "*on average* each token activates $k$ experts" along with load balancing across experts. This is sufficient to achieve the goals of MoE, so in this article we consider the simplified problem:

$$\max_{x_{i,j}\in\{0,1\}} \sum_{i,j} x_{i,j}s_{i,j} \qquad\text{s.t.}\qquad \sum_i x_{i,j} = \frac{mk}{n}$$

### Dynamic Activation

Assuming the reader is already familiar with the previous article, the derivation here will be relatively brief. If any steps are unclear, please refer back to the previous installment for review. As before, we first consider the relaxed version of objective $(1)$:

$$\max_{x_{i,j}\in[0,1]} \sum_{i,j} x_{i,j}s_{i,j} \qquad\text{s.t.}\qquad \sum_i x_{i,j} = \frac{mk}{n}$$

Then we consider its equivalent $\max$-$\min$ form:

$$\max_{x_{i,j}\in[0,1]}\min_{\beta_j} \sum_{i,j} x_{i,j}s_{i,j}  - \sum_j \beta_j\left(\sum_i x_{i,j} - \frac{mk}{n}\right)$$

Similarly, the order of $\max$ and $\min$ is interchangeable here. Swapping them and rearranging slightly gives:

$$\min_{\beta_j}\max_{x_{i,j}\in[0,1]} \sum_{i,j} x_{i,j}(s_{i,j} - \beta_j) + \frac{mk}{n} \sum_j \beta_j\tag{2}$$

The $\max$ step can be completed first, yielding:

$$\left\{\begin{aligned}&\,x_{i,j}^* = 1, &\, s_{i,j} - \beta_j > 0 \\
&\,x_{i,j}^* = 0, &\, s_{i,j} - \beta_j < 0 \\
&\,x_{i,j}^* \in  [0,1], &\, s_{i,j} - \beta_j = 0
\end{aligned}\right.$$

This tells us that any expert with $s_{i,j} - \beta_j > 0$ is activated, and the number of activations is not fixed. This is formally consistent with "MoE Odyssey: Part 4, Invest More Where It's Hard," except that previously we designed it by intuition, whereas here it is derived from the more fundamental objective $(1)$.

### One-Step Solution

Substituting $x_{i,j}^*$ back into $(2)$ gives $x_{i,j}^*(s_{i,j} - \beta_j) = \max(0, s_{i,j} - \beta_j)$, so the optimization objective for $\beta_j$ simplifies to:

$$\min_{\beta_j} \sum_{i,j} \max(0, s_{i,j} - \beta_j) + \frac{mk}{n} \sum_j \beta_j\tag{3}$$

Each $\beta_j$ term is independent, so this decomposes into $m$ independent sub-optimization problems. This means we can temporarily drop the subscript $j$:

$$\min_{\beta} \frac{mk}{n}\beta + \sum_i \max(0, s_i - \beta)$$

Sort all $s_i$ in descending order as $s_{\sigma_1}\geq s_{\sigma_2} \geq \cdots \geq s_{\sigma_m}$, then assume we already know $s_{\sigma_l}\geq\beta\geq s_{\sigma_{l+1}}$. We can then remove the $\max$ to get:

$$\frac{mk}{n}\beta + \sum_{i=1}^l (s_{\sigma_i} - \beta) = \left\{\begin{aligned}
&\,\sum_{i=1}^{mk/n} s_{\sigma_i} + \sum_{i=mk/n+1}^l \underbrace{(s_{\sigma_i} - \beta)}_{\geq 0},&\, l \geq mk/n \\
&\,\sum_{i=1}^{mk/n} s_{\sigma_i} - \sum_{\hphantom{ab}i=l+1\hphantom{ab}}^{mk/n} \underbrace{(s_{\sigma_i} - \beta)}_{\leq 0},&\, l \leq mk/n \\
\end{aligned}\right.$$

This shows that both $l > mk/n$ and $l < mk/n$ increase the objective, so the minimum is attained at $l=mk/n$. That is, $\beta^*$ lies between the $mk/n$-th and $(mk/n+1)$-th largest elements of $s_i$. By convention, we take the $(mk/n+1)$-th largest element. Restoring the subscript $j$: for a given $j$, $\beta_j^*$ is the $(mk/n+1)$-th largest element when $s_{i,j}$ is sorted in descending order, or equivalently, the "$1-k/n$ quantile."

Note that the only unknown variable is now $\boldsymbol{\beta}$, so there is no alternating iteration step — a single Quantile computation directly yields the perfectly balanced optimal solution! We continue to call this **Quantile Balancing (QB)**.

### Practical Considerations

If the reader is already familiar with "MoE Odyssey: Part 6, Optimal Assignment for Load Balancing," they will notice that this article is essentially a streamlined version of it. However, simplicity does not mean triviality — it is actually more elegant and efficient than Top-$k$ style MoE: first, it activates experts by checking whether $\boldsymbol{s} - \boldsymbol{\beta}$ is greater than 0, which is more efficient than selecting Top-$k$; second, it obtains the load-balanced optimal solution in a single Quantile step, which is also more efficient and elegant.

Of course, a one-step optimal solution is very likely to overfit the current training batch. To avoid this problem, the improved approach considered here is to apply EMA (Exponential Moving Average) between the current batch's optimal solution and the historical solution (the Top-$k$ version from the previous article can also use EMA). Note that we must still be careful about the information leakage trap: we must first use the old $\boldsymbol{\beta}$ to activate experts, and only then update $\boldsymbol{\beta}$.

The algorithm flow is as follows (in the author's experiments, $\lambda=0.9$):

$$\begin{array}{|l|}
\hline
\text{Quantile Balancing (QB): Dynamic Activation Version} \\[4pt]
\hline
\text{Input: Score matrix }\boldsymbol{s}\in\mathbb{R}^{m\times n}\text{, previous }\boldsymbol{\beta}\in\mathbb{R}^n\text{, decay rate }\lambda \\
\text{Output: Assignment }\boldsymbol{x}\in\{0,1\}^{m\times n}\text{, updated }\boldsymbol{\beta}\in\mathbb{R}^n \\[4pt]
\hline
\begin{array}{ll}
1: & x_{i,j}=1 \text{ if } s_{i,j} - \beta_j > 0 \text{ else } 0 \\
2: & \boldsymbol{\beta} \leftarrow \lambda\boldsymbol{\beta} + (1-\lambda)\mathop{\text{des\_sort}}(\boldsymbol{s}, \text{axis=0})_{[mk/n:mk/n+1]} \\
3: & \text{Output } \boldsymbol{x},\boldsymbol{\beta}
\end{array} \\
\hline
\end{array}$$

In practice, we similarly face the issue that finding the $1-k/n$ quantile of $\boldsymbol{s}$ is expensive — it requires finding the $(mk/n+1)$-th largest element among $m$ elements, where $m$ equals "global batch size * sequence length." Due to various parallelism strategies and gradient accumulation constraints, an exact implementation is typically unacceptable. So we again consider a compromise: split the samples into the largest mini-batches we can afford, compute a separate $\boldsymbol{\beta}$ for each mini-batch, and average them as the final result.

### Expert Choice

In fact, this dynamic version of QB has a very simple interpretation, namely "Expert Choice":

> You want each expert to be activated exactly $mk/n$ times? Then just let the experts choose the tokens! That is, each expert selects its Top-$mk/n$ tokens — which is simply finding the largest $mk/n$ elements along $\text{axis=0}$.

However, naive Expert Choice has a fatal flaw: it requires comparing all tokens globally, leading to cross-sample, cross-sequence selection. This violates causality (seeing future information during training) and breaks train-inference consistency (inference results depend on batch size). This is precisely why this method has been difficult to deploy in practice.

The elegance of this article lies in converting it into a bias form: using each expert's $(mk/n+1)$-th largest element (equivalently, the $1-k/n$ quantile) as the threshold $\boldsymbol{\beta}$, we reformulate Expert Choice as Token Choice — from "expert selects Top-$mk/n$" to "activate whenever $s_{i,j} - \beta_j > 0$." More critically, we delay the update of $\boldsymbol{\beta}$ until after the activation decision, perfectly avoiding the information leakage problem.

Remarkably, this bias form that fixes the defects of Expert Choice emerges naturally from the dual problem of the original linear program — one cannot help but appreciate its aesthetic beauty.

### Initialization Strategy

To prevent information leakage, during training we need to first use the old $\boldsymbol{\beta}$ to determine which experts each token activates, and only then update $\boldsymbol{\beta}$. This makes the first few training steps sensitive to the initialization of $\boldsymbol{\beta}$. For example, if $\boldsymbol{s}$ is entirely positive but $\boldsymbol{\beta}$ is zero-initialized, then on the first training step every token would activate all experts, causing computation to explode.

In "MoE Odyssey: Part 4, Invest More Where It's Hard" we already provided a simulation script to estimate a suitable initialization, but here we already know that the optimal $\boldsymbol{\beta}$ is the "$1-k/n$ quantile." We can therefore make some reasonable assumptions about the initial scores and derive an analytical initialization formula. Specifically, we assume the router's initial logits follow $\mathcal{N}(0,\sigma^2)$, so its $1-k/n$ quantile is:

$$\sigma \cdot \Phi^{-1}\left(1 - \frac{k}{n}\right)$$

This is precisely the optimal $\boldsymbol{\beta}$ for logits from this distribution, and we use it as the initialization for $\boldsymbol{\beta}$, where $\Phi^{-1}$ is the quantile function (inverse CDF) of the standard normal distribution.

Additionally, the router may include an activation function. If the activation function is element-wise and monotonically increasing, such as Sigmoid, it does not change the ordering, so we simply apply the corresponding activation function to the above expression. For Softmax it is slightly more complex: it first exponentiates then normalizes. Assuming the normalization denominator is approximately constant, it becomes similar to the previous case — we just need to exponentiate the above expression and divide by a denominator estimated via quantile-based simulated sampling:

$$\frac{\exp\left[\sigma \cdot \Phi^{-1}\left(1 - \frac{k}{n}\right)\right]}{\sum\limits_{i=1}^n \exp\left[\sigma \cdot \Phi^{-1}\left(1 - \frac{i}{n+1}\right)\right]}$$

As for estimating $\sigma$, it is quite straightforward. Assume the router's input dimension is $d$ with RMS Norm applied, and that the router's weight matrix is initialized from $\mathcal{N}(0,\tilde{\sigma}^2)$. Then the router logits approximately follow a normal distribution with mean approximately 0 and variance approximately $\tilde{\sigma}^2 d$, i.e., $\sigma\approx \tilde{\sigma} \sqrt{d}$. These are classical results from initialization theory.

### Demo Code

This section again provides demonstration code for readers to experiment with:

```python
import numpy as np

def quantile_bias(s, k):
    """Dynamic QB: one-step quantile yields the optimal bias
    Reference: https://kexue.fm/archives/11626
    """
    m, n = s.shape
    beta = np.quantile(s, 1 - k / n, axis=0)
    return beta

def max_min_avg_vio(s):
    """Compute max_vio, min_vio, and avg_vio
    where max_vio >= 0, avg_vio >= 0, -1 <= min_vio <= 0;
    all three closer to 0 means better balance
    """
    m, n = s.shape
    f = (s > 0).mean(0)
    f = f / f.sum() * n - 1
    return f.max(), f.min(), np.abs(f).mean()

def avg_std_active(s):
    """Compute the mean and std of activation counts per expert
    avg closer to k is better, std closer to 0 is better
    """
    m, n = s.shape
    f = (s > 0).mean(0) * n
    return f.mean(), f.std()

m, n, k = 100000, 256, 8
s = np.random.rand(m, n) + np.random.rand(n)  # Simulate non-uniform scores

b = quantile_bias(s, k)
max_min_avg_vio(s - b)  # Should be all zeros
avg_std_active(s - b)  # Should be (k, 0)

from scipy.stats import norm

sigma = 1
s = np.random.randn(m, n) * sigma  # Simulate initialization
avg_std_active(s)  # Approximately (128, 0.4), far exceeding budget k
b0 = np.zeros(n) + norm.ppf(1 - k / n) * sigma  # norm.ppf is the standard normal quantile function
avg_std_active(s - b0)  # Approximately (8, 0.1), close to budget k

s2 = 1 / (1 + np.exp(-s))  # Sigmoid activation
b0_2 = 1 / (1 + np.exp(-b0))  # Sigmoid activation
avg_std_active(s2 - b0_2)  # Still approximately (8, 0.1), close to budget k

s3 = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)  # Softmax activation
q = (np.arange(n) + 1) / (n + 1)  # Uniformly spaced points (quantiles)
b0_3 = np.exp(b0) / np.exp(sigma * norm.ppf(q)).sum()  # Exp activation divided by simulated denominator
avg_std_active(s3 - b0_3)  # Approximately (7.5, 0.1), close to budget k
```

### Gradient Descent

Finally, if one wants to avoid Quantile computation entirely, we can still consider gradient descent. First, denote the loss function from objective $(1)$ as $\ell$:

$$\min_{\beta_j} \underbrace{\sum_{i,j} \max(0, s_{i,j} - \beta_j) + \frac{mk}{n} \sum_j \beta_j}_{\triangleq\,\ell}$$

It is differentiable, with gradient:

$$\frac{\partial\ell}{\partial\beta_j} = \frac{mk}{n} - \sum_{i=1}^m \chi(s_{i,j} - \beta_j > 0)$$

where $\chi$ is the indicator function: $\chi(\text{True})=1,\chi(\text{False})=0$. Computing this gradient is cheaper than a global Quantile, making it suitable for readers who are more cost-sensitive. With the gradient in hand, we can perform gradient descent. We again consider SignSGD, aligning with Loss-Free:

$$\beta_j \leftarrow \beta_j - \gamma\mathop{\text{sign}}\left(\frac{\partial\ell}{\partial\beta_j}\right)$$

This can replace the Quantile update step for $\boldsymbol{\beta}$, and since this is inherently a long-term iterative algorithm, the EMA can also be removed. In fact, this is precisely Equation (9) from "MoE Odyssey: Part 4, Invest More Where It's Hard" — we have re-derived it from the perspective of "optimal assignment + dual objective + gradient descent."

### Summary

This article continues the exploration of Quantile Balancing (QB) from the previous installment. By removing the constraint "each token can only activate $k$ experts" from the optimal assignment problem, the solution form is significantly simplified — only a single Quantile step is needed to achieve load balancing. Each token only needs to check the sign of the biased score to decide which experts to activate, eliminating the overhead of Top-$k$ sorting.

---

## [Part 8: Where Does DeepSeek V4's tid2eid Come From?](https://kexue.fm/archives/11681)

By Jianlin Su | 2026-05-15

Anyone who has trained MoE models knows that if you replace the entire MLP portion of the model with MoE, the first few MoE layers closest to the embedding are often very difficult to load-balance. To address this, [DeepSeek V3](https://papers.cool/arxiv/2412.19437), including our [Kimi K2](https://papers.cool/arxiv/2507.20534), adopted the strategy of "first_k_dense" — as the name suggests, the first $k$ layers use conventional dense GLU instead of MoE. By [DeepSeek V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf), this strategy was changed to "first_k_hash".

The "hash" here refers to "Hash Routing", proposed in [Hash Layers For Large Sparse Models](https://papers.cool/arxiv/2106.04426), which assigns an Expert to each Token via a pre-determined mapping table called tid2eid. This article explores how to generate this tid2eid.

### Background

First, it should be pointed out that for the first few MoE layers, if one uses the Quantile Balancing method proposed by the author in "MoE Odyssey: 6. Optimal Assignment for Load Balance" and "MoE Odyssey: 7. A Minimalist Solution for Dynamic Activation", the imbalance problem can already be largely mitigated.

As for V4's adoption of Hash Routing, it can be seen as an attempt in a different direction for the same problem. The idea is that for the first few MoE layers, there is not much contextual information, so selecting Experts based on input vectors may not differ much from selecting directly based on Token Id, which carries no contextual information at all. Since this is the case, we might as well hard-code which Experts each Token should activate in the first few layers based on Token Id — this is the origin of "tid2eid (Token Id to Expert Id)".

So how do we generate this tid2eid table? Can we just do it completely randomly? That doesn't seem to work, because each Token's occurrence frequency differs — purely random assignment would actually lead to imbalance. DeepSeek has not published the details of how this table is generated, so we can only speculate based on our own reasoning.

### Mathematical Formulation

Suppose there are $m$ distinct Tokens, and the $i$-th Token has occurrence frequency $p_i$. Without loss of generality, assume they are sorted in descending order, i.e., $p_i \geq p_{i+1}$. We use $x_{i,j} \in \{0, 1\}$ to denote whether Token $i$ should activate Expert $j$ ($0$ means not activated, $1$ means activated). There are $n$ Experts in total, and each Token selects $k$ Experts. Then we actually need to solve the system of equations:

$$
x_{i,j}\in\{0, 1\},\qquad\sum_{j=1}^n x_{i,j} = k,\qquad \sum_{i=1}^m p_i x_{i,j}\approx \frac{k}{n}
$$

Note that for the last equation we used the approximate sign $\approx$. This is because if we change it to $=$, the system of equations may not strictly have a solution. So we keep the $\approx$ notation, meaning the two sides should be as close as possible. We can also write it in optimization form:

$$
\min_{x_{i,j}\in\{0, 1\}} \sum_{j=1}^n \left(\sum_{i=1}^m p_i x_{i,j} - \frac{k}{n}\right)^2 \qquad\text{s.t.}\qquad\sum_{j=1}^n x_{i,j} = k
$$

A fairly straightforward solution to this problem is a greedy algorithm:

> Process Tokens one by one from highest to lowest frequency. First compute each Expert's currently assigned load, then select the $k$ least-loaded Experts as the activated Experts for the current Token, and finally update the Expert load distribution.

### Reference Implementation

Although the greedy algorithm is greedy in principle, in most cases it yields a near-optimal solution. A reference implementation is as follows:

```python
import numpy as np

# Simulated distribution
m, n, k = 80000, 128, 4
p = 1 / (10 + np.arange(m))
p /= p.sum()

# Greedy processing
x, f = np.zeros((m, k), dtype='int32'), np.zeros(n)
for i in range(m):
    j = f.argsort()[:k]  # Select the k least-loaded
    f[j] += p[i] / k  # Update load distribution
    x[i] = j  # Record in tid2eid

# Evaluate balance
max_vio, min_vio = f.max() * n - 1, f.min() * n - 1
```

Note that this code contains no randomness whatsoever — it is in principle a deterministic algorithm. What if we want different tid2eid tables for different early layers? We can consider adding some randomness, for example replacing `range(m)` with `np.random.permutation(m)`, i.e., not processing in descending frequency order. This still yields a usable solution, just with slightly worse balance. We can run it multiple times and pick the solutions with the lowest max_vio.

### Extreme Cases

Now let us consider an extreme case: $p_1 \gg k / n$, meaning some Token's frequency far exceeds the balanced level. In this case, no matter how tid2eid is arranged, load balance is impossible to achieve. In other words, making decisions based solely on a single Token Id can never achieve balance. How should we handle this situation?

The answer is simple: switch to a Hash method that depends on more input information. Previously we made decisions based only on the current Token Id, but in reality each Token exists within a sequence and has context. If the previous approach was "1-gram to expert", we can now consider combining it with preceding Tokens to form "2-gram to expert", "3-gram to expert", and so on.

Taking the 2-gram $(a, b)$ as an example, a simple Hash method is: choose a prime $q$ greater than $m$, compute $(aq + b) \mod n$ as the activated Expert, and repeat $k$ times with different primes to get $k$ Experts. Because increasing the gram count greatly increases the number of input combinations, and these map to a finite set of $n$ values, each value is very likely to be "filled up". So as long as the Hash function is not particularly bad, the result is almost perfectly balanced.

Therefore, the advantage is that there is no need to consider frequencies, nor to store a tid2eid table in advance. As for the disadvantage, the Hash function computation becomes slightly more complex, and the same Token might be assigned to duplicate Experts — but neither of these is a particularly serious problem.

### Summary

This article briefly reviews the basic idea of Hash Routing in DeepSeek V4, with a focus on the construction principle of its tid2eid mapping table.

---

## [Part 9: The Gate Normalization Debate](https://kexue.fm/archives/11782)

By Jianlin Su | 2026-06-28

Looking back at MoE history, we find that in the early years the Router almost universally used Softmax activation when serving as a Gate multiplied onto the Expert output, and it remains one of MoE's standard forms to this day. However, to accommodate the [Loss-Free](https://kexue.fm/archives/10757) load balancing scheme, DeepSeek switched the activation function to Sigmoid and demonstrated that this is also a highly competitive approach, sparking deeper exploration and experimentation with the Router's form.

Even within Softmax there are two subtly different practices: apply Softmax first and then select Top-$k$, or select Top-$k$ first and then apply Softmax? The latter can also be understood as re-normalizing after selecting the Top-$k$, i.e., Re-Norm. So: should the Gate's activation function normalize? And if so, should we normalize before Top-$k$ selection or Re-Norm after? This is the topic of the present article.

### Problem Statement

We know that the general form of MoE is

$$\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i \tag{MoE-1}$$

Here $\boldsymbol{\rho}$ actually plays two roles: when used to select the Top-$k$ Experts, its role is the Router; when multiplied onto the Expert output, its role is the Gate. From the perspective of MoE design, $\boldsymbol{\rho}$'s core role is clearly the Router, while the Gate's purpose is to provide gradients to the Router during training.

The question we discuss can also be understood as: how to more scientifically construct $\boldsymbol{\rho}=(\rho_1, \rho_2, \cdots, \rho_n)$ so that the Router receives better gradients. For a long time, the standard answer has been Softmax:

$$\rho_i = \frac{e^{s_i}}{\sum_{j=1}^n e^{s_j}}$$

where $\boldsymbol{s}=(s_1, s_2, \cdots, s_n)$ are the logits projected directly by a linear layer. However, while this answer is "standard," the author has not found any real explanation for it — everyone seems to simply accept and perpetuate it, which left the author long puzzled about the training mechanism of MoE.

### Other Choices

As mentioned in the introduction, DeepSeek tried Sigmoid activation for Loss-Free load balancing, later using it in [DeepSeek-V3](https://papers.cool/arxiv/2412.19437). Its success demonstrates that non-Softmax activation can also work well, inspiring people to try more general approaches. For example, [ReMoE](https://papers.cool/arxiv/2412.14711) uses ReLU activation, and the geometric perspective from ["MoE Odyssey: Part 1, Starting from Geometric Interpretation"](https://kexue.fm/archives/10699) allows any non-negative activation function.

Additionally, regarding MoE's form, there is the Re-Norm option, which modifies (MoE-1) to:

$$\boldsymbol{y} = \frac{\sum\limits_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i}{\sum\limits_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i} \tag{MoE-2}$$

That is, re-normalizing the selected Top-$k$ $\rho_i$ values. For Softmax, this is equivalent to selecting Top-$k$ from $\boldsymbol{s}$, setting the unselected entries to $-\infty$, and then applying Softmax. Re-Norm's advantage is making the forward computation numerically more stable, but note that with Re-Norm, $k$ must be greater than 1 — otherwise $\boldsymbol{\rho}$ will have no gradient at all and cannot be trained.

Taking stock of current practice across various groups, these MoE variants all perform roughly similarly, with none clearly dominant. Since practice cannot distinguish a winner, let us explore theoretically which form is more scientifically justified.

### Design Principles

Our goal is to find a first principle that is closer to the essence, and use it to derive the gating mechanism of current MoE.

So the first question is naturally: what is this "principle"? For simplicity, first consider $k=1$. We know that MoE's most important feature is sparsity — first determine which Expert to activate via a Router, then compute only that Expert, thereby increasing parameter count while controlling computation. If this were the sole consideration, the naive model would be:

$$\boldsymbol{f}\left(\boldsymbol{e}_{\mathop{\text{argmax}}\boldsymbol{\rho}}\right)$$

That is, pick the highest-scoring Expert from the Router $\boldsymbol{\rho}$ and activate it. This form works perfectly for inference, but during training the Router receives no gradient and cannot be updated, so we must find a way to design gradients for the Router. How do we design gradients for the Router? To answer this, we must first think clearly about: what kind of Router do we want?

Since we can only activate 1 Expert, we naturally want it to be the best-performing one. If $\ell$ denotes the loss function, our desire can be written as:

$$\mathop{\text{argmax}} \boldsymbol{\rho} = \mathop{\text{argmin}}\, [\ell(\boldsymbol{e}_1),\ell(\boldsymbol{e}_2),\cdots,\ell(\boldsymbol{e}_n)] \tag{Target}$$

This is the design principle we are looking for.

### Objective Transformation

However, the target (Target) is not yet a loss function we can directly use for training — it requires further transformation. To this end, we construct two distributions. The first is a target distribution $\boldsymbol{q}=(q_1,q_2,\cdots,q_n)$ built from the loss function:

$$q_i = \frac{e^{-\ell(\boldsymbol{e}_i)/\tau}}{\sum_{j=1}^n e^{-\ell(\boldsymbol{e}_j)/\tau}}$$

This distribution is independent of the Router; for the Router's learning it serves as the "target distribution." The second distribution is a predicted distribution $\boldsymbol{p}$ built from $\boldsymbol{\rho}$. There are many possibilities here: $\boldsymbol{\rho}$ itself might already be a distribution $\boldsymbol{p}$ (if $\boldsymbol{\rho}$ is already normalized), or $\boldsymbol{p}$ could be the Softmax of $\boldsymbol{\rho}$ (treating $\boldsymbol{\rho}$ as logits), or some other normalization scheme. In short, $\boldsymbol{p}$ is some probabilistic representation of the Router; let its parameters be $\boldsymbol{\theta}$.

We transform the target (Target) into bringing $\boldsymbol{p}$ closer to $\boldsymbol{q}$, thereby providing gradients for $\boldsymbol{\theta}$. For this, we consider minimizing the KL divergence:

$$KL(\boldsymbol{p}\Vert \boldsymbol{q}) = \sum_{i=1}^n p_i \log \frac{p_i}{q_i}$$

With slight rearrangement:

$$KL(\boldsymbol{p}\Vert \boldsymbol{q}) = - \mathcal{H}(\boldsymbol{p}) + \frac{1}{\tau}\sum_{i=1}^n p_i \ell(\boldsymbol{e}_i) - \log \sum_{i=1}^n e^{-\ell(\boldsymbol{e}_i)/\tau}$$

This objective has three terms. The first is negative entropy $-\mathcal{H}(\boldsymbol{p})$; minimizing it means maximizing entropy, which encourages the model to explore sufficiently — we can consider that load balancing already plays a similar role, so we set it aside. The third term is independent of $\boldsymbol{p}$ (and thus of $\boldsymbol{\theta}$). So the effective loss function is:

$$\mathcal{L} = \sum_{i=1}^n p_i \ell(\boldsymbol{e}_i)$$

### Straight-Through Estimation

Taking the gradient of this effective loss:

$$\nabla_{\boldsymbol{\theta}}\mathcal{L} = \sum_{i=1}^n \nabla_{\boldsymbol{\theta}} p_i \cdot \ell(\boldsymbol{e}_i) = \sum_{i=1}^n p_i \nabla_{\boldsymbol{\theta}} \log p_i \cdot \ell(\boldsymbol{e}_i) = \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \ell(\boldsymbol{e}_i)]$$

The key step here is using $\nabla_{\boldsymbol{\theta}} p_i = p_i \nabla_{\boldsymbol{\theta}} \log p_i$ to isolate a factor of $p_i$, enabling the sum to be converted into an expectation, which can then be estimated via sampling to achieve MoE's sparse computation goal. Some readers may have already recognized this — it is precisely REINFORCE from policy gradients! (See [*"Optimization Through Sampling: A Unified View of Differentiable and Non-Differentiable Optimization"*](https://kexue.fm/archives/7521), [*"Policy Gradients and Zeroth-Order Optimization: Converging Paths"*](https://kexue.fm/archives/7737))

The problem with REINFORCE is high noise. Intuitively, this is because it places $p_i$ outside the loss function $\ell$; if possible, we would prefer a "reparameterization" form with $p_i$ inside $\ell$. To derive such a form, we use REINFORCE's invariance to baseline subtraction:

$$\begin{aligned}
\mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \ell(\boldsymbol{e}_i)] =&\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot (\ell(\boldsymbol{e}_i) - \ell(\boldsymbol{0}))] \\[4pt]
\approx&\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \langle\nabla_{\boldsymbol{e}_i} \ell(\boldsymbol{e}_i), \boldsymbol{e}_i - \boldsymbol{0}\rangle] \\[4pt]
= &\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}} \langle\nabla_{\boldsymbol{e}_i} \ell(\boldsymbol{e}_i), \log p_i \cdot \boldsymbol{e}_i\rangle] \\[4pt]
= &\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}} \ell((\log p_i + [1 - \log p_i]_{\text{sg}}) \cdot\boldsymbol{e}_i)] \\[4pt]
= &\, \nabla_{\boldsymbol{\theta}} \mathbb{E}_{i\sim \boldsymbol{p}} [\ell((\log p_i + [1 - \log p_i]_{\text{sg}}) \cdot\boldsymbol{e}_i)]
\end{aligned}$$

Here the $\approx$ applies a first-order Taylor approximation at $\boldsymbol{e}_i$, and $[\cdot]_{\text{sg}}$ denotes stop-gradient. We arrive at a Straight-Through Estimator (STE) that "uses 1 in the forward pass and $\log p_i$ in the backward pass" to provide gradients for the Router.

### The Final Form

While STE provides a workable training scheme, the inconsistency between forward and backward passes typically yields only suboptimal results. Here a remarkably elegant improvement is to change each Expert from $\boldsymbol{e}_i$ to $p_i\boldsymbol{e}_i$! Repeating the above derivation:

$$\begin{aligned}
\mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \ell(p_i\boldsymbol{e}_i)] =&\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot (\ell(p_i\boldsymbol{e}_i) - \ell(\boldsymbol{0}))] \\[4pt]
\approx&\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \langle\nabla_{p_i\boldsymbol{e}_i} \ell(p_i\boldsymbol{e}_i), p_i\boldsymbol{e}_i - \boldsymbol{0}\rangle] \\[4pt]
= &\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}} \langle\nabla_{p_i \boldsymbol{e}_i} \ell(p_i \boldsymbol{e}_i),  p_i \boldsymbol{e}_i\rangle] \\[4pt]
= &\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}} \ell(p_i \boldsymbol{e}_i)] \\[4pt]
= &\, \nabla_{\boldsymbol{\theta}} \mathbb{E}_{i\sim \boldsymbol{p}} [\ell(p_i \boldsymbol{e}_i)]
\end{aligned}$$

This transformation is exquisite and worth savoring carefully. By changing the Expert from $\boldsymbol{e}_i$ to $p_i\boldsymbol{e}_i$, we eliminate the stop-gradient, achieving consistency between forward and backward passes, and theoretically raising the ceiling on model performance.

Now we can answer the question posed at the beginning:

> If we desire a top-down probabilistic derivation, then the Router when serving as Gate should be normalized, but should **not** use Re-Norm.

### To Sample or Not to Sample

A detail worth noting: $\mathbb{E}_{i\sim \boldsymbol{p}}$ implies we should sample from $\boldsymbol{p}$, but in practice we usually just select Top-$k$ directly. How should we understand this discrepancy?

This is fundamentally a tradeoff between diversity and stability. Random sampling encourages the model to explore more fully, but introduces additional gradient variance and instability. Directly selecting Top-$k$ is more stable, but risks the model falling into suboptimal solutions or even collapsing. Fortunately, modern load balancing strategies are mature enough to encourage comprehensive exploration to some degree, so Top-$k$ selection remains mainstream.

If one wants sampling while maintaining stability, one can extend slightly beyond Top-$k$ rather than sampling completely freely. For example: first select Top-$(k+c)$, then randomly pick $k$ from those $k+c$ Experts. Or add mild noise to the logits of $\boldsymbol{p}$ before selecting Top-$k$. This increases randomness without straying too far from the original Top-$k$, balancing both concerns.

### Related Work

To be clear, the derivation in this article is not entirely new — it was distilled and modified by the author from Liu Liyuan's paper [*Sparse Backpropagation for MoE Training*](https://papers.cool/arxiv/2310.00811). That paper also has a prequel [*Bridging Discrete and Backpropagation: Straight-Through and Beyond*](https://papers.cool/arxiv/2304.08612) and a sequel [*GRIN: GRadient-INformed MoE*](https://papers.cool/arxiv/2409.12136).

Although these are papers from 2023–2024, for anyone wanting to deepen their understanding of the MoE Router, this trilogy is highly recommended reading. They provide a unified probabilistic framework for designing gradients for various discretization operations. Of course, the probabilistic framework also has its limitations — it is rather formalistic and can feel somewhat "constraining" in practice.

For example, when $k = 2$, if we straightforwardly extend the previous results, we should have:

$$\mathbb{E}_{i,j\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i p_j \cdot \ell(p_i p_j (\boldsymbol{e}_i + \boldsymbol{e}_j))] \approx \nabla_{\boldsymbol{\theta}} \mathbb{E}_{i,j\sim \boldsymbol{p}} [\ell(p_i p_j (\boldsymbol{e}_i + \boldsymbol{e}_j))]$$

That is, treating the joint distribution $p_i p_j$ and expert-pair sum $\boldsymbol{e}_i + \boldsymbol{e}_j$ as the basic unit, reducing Top-2 to Top-1. However, the MoE form we actually use is $\ell(p_i \boldsymbol{e}_i + p_j\boldsymbol{e}_j)$, which does not lend itself to a clean probabilistic derivation.

In this case, a more "relaxed" interpretation might be to treat it as an analogue of MaxPooling without insisting on a probabilistic explanation — or alternatively, to understand it through the geometric interpretation from ["MoE Odyssey: Part 1, Starting from Geometric Interpretation"](https://kexue.fm/archives/10699). In summary, the probabilistic framework proves the viability of a certain approach, but does not in principle rule out the viability of other approaches.

### Summary

This article attempts to explore the design of MoE's Router and Gate from first principles, providing a probabilistic justification for gate normalization.

---

