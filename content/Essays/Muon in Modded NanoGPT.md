---
draft: "false"
created: November 24, 2025
title: Muon in Modded NanoGPT
aliases:
  - AdaptiveMuon
  - muon
  - Adaptive Muon
  - optimal learning rates
image: ../images/Muon/descent.png
imageAlt: Descent of Everest
permalink: essays/muon
modified: November 30, 2025
---

The Muon optimizer was developed for Modded NanoGPT speedrun, which has the expressed goal of training a GPT-style model as fast as possible [^jordan-muon] [^moddednanogpt]. Since last summer, through the contributions of various researchers and engineers, the train time has decreased from 45 minutes to nearly 2 minutes. In this blog post, I'll walk through some Muon-related innovations from recent records and review the surrounding literature behind these techniques.

[^jordan-muon]: [Keller Jordan et al 2024b](https://kellerjordan.github.io/posts/muon/) *Muon: An optimizer for hidden layers in neural networks*

[^moddednanogpt]: [Keller Jordan et al 2024a](https://github.com/KellerJordan/modded-nanogpt) *Modded NanoGPT*. The competition is for "the fastest algorithm to use 8 NVIDIA H100 GPUs to train a language model that attains 3.28 cross-entropy loss on the FineWeb validation set."

[^larry2025a]: [Larry Dial 2025](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months) *How the NanoGPT Speedrun WR dropped by 20% in 3 months*

---
%%
want to review some key Muon modifications that have help drive down the Modded NanoGPT speedrun to nearly 2 minutes. added to Muon in the In this blog post, I want to examine

The Muon optimizer was invented by Keller Jordan, ... , & friends for the Modded NanoGPT speedrun competition. .. lots of papers.. but Keller Jordan says he "only trusts speedruns". With this in mind, here some changes . Survey on related literature

Aggregate related literature

- the problem is i dont want to talk about everything lol

There have been several recent papers highlighting the benefits of an adaptive estimation term inside of Muon. Given the recent adoption of one of these methods (Normuon) in the Modded NanoGPT speedrun, I want to examine the optimal learning rate and batch size for adaptive muon.
 %%
%% In this blog I examine how optimal learning rate changes as batch size, and so is quickly asymptotic as opposed to Adam's square-root law $\eta \propto \sqrt{B}$. I also survey how this relates to Critical Batch Size and learning rate schedules for Muon.  %%



## (1) Adaptive Muon
The Modded NanoGPT speedrun uses an adaptive variant of Muon called NorMuon [^normuon]. Several recent papers have explored similar adaptive extensions [^adamuon] [^adago] [^frans]. In this section, I want to explain what "adaptive Muon" means and why it helps.

%% I want to provide an overview of the literature on adaptive Muon variants, why it is effective, and why you might consider adopting it for yourself. %%

[^frans]: [Frans et al 2025](https://arxiv.org/abs/2510.25000) *What Really Matters in Matrix-Whitening Optimizers?*
[^adago]: [Zhang et al 2025](https://arxiv.org/abs/2509.02981) *AdaGrad Meets Muon*
[^normuon]: [Li et al 2025](https://arxiv.org/abs/2510.05491) *NorMuon*
[^adamuon]:[Si et al 2025](https://arxiv.org/abs/2507.11005) *AdaMuon*

Start with the simplest case: Stochastic Gradient Descent with Momentum (SGDM) updates parameters using an exponential moving average (EMA) of gradients. Adam (ADaptive Moment Estimation) builds on this by also tracking the EMA of squared gradients per parameter. We divide by this second EMA to produce a variance-corrected update.

%% performs steps by computing the exponential moving average (EMA) of gradients. Adam (ADaptive Moment Estimation) additionally tracks the EMA of the norm of each parameter. Dividing by this second EMA gives us a variance-corrected update.  %%

Adaptive variance techniques estimate two distinct properties about the gradient:
1) Stochastic noise: the variance within a single batch, estimated *per-parameter*. At a fixed batch size, certain features may be very noisy, while other features may have high signal. Adaptivity allows noisier gradient estimates to get dampened, effectively giving each parameter its own adaptive learning rate.
2) Curvature: how the gradient changes as we move through parameter space. Various papers examine how second order statistics approximate the hessian (the second moment) of the loss landscape—that is, the per-parameter curvature [^cohen] [^kustner].  Normalizing by this estimate makes the optimizer take smaller steps into high-curvature regions, preventing overshooting and descent into "sharp" minima.

[^kustner]: [Kustner et al 2024](https://arxiv.org/abs/2402.19449) *Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models*
[^cohen]: [Cohen et al 2024](https://arxiv.org/abs/2410.24206) *Understanding Optimization in Deep Learning with Central Flows*, with a shorter accompanying blogpost [here](https://centralflows.github.io/part3/).


%% as well. This "variance EMA" is an estimate of the noise we expect from sampling the gradient of each parameter. At each step, divide the update by variance so that high-variance terms update slowly and low-variance terms update quickly.

Muon tracks the EMA gradient of full parameters and normalizes the EMA through a method known as *orthogonalization* which you can variously read about elsewhere [^jordan24-muon] [^mahdid] [^bernstein]. Notably, this normalization is with respect to the matrix itself, not with respect to the estimated variance of the distribution. That is, Muon's normalization method does not increase our certainty about the gradient. This provides motivation for adding an adaptive estimator on top of Muon: %%



%% Muon's orthogonalization corrects for the former, in that it's optimization step normalizes all spectral directions. Therefore, it will nat %%
Despite not being a variance-adaptive method, Muon is "curvature-aware" [^kovalev][^anonymous26][^su25], which addresses (2). Intuitively, this comes from orthogonalizing the update: after orthogonalization, Muon's update is "well-rounded" in parameter space, avoiding directions of steep change [^spectral]. Formally, each update is perfectly-conditioned (all spectral values are $1$), which keeps the weights themselves well-conditioned (the spectral values are low and near each other)[^Boreiko]. Muon is effectively performing gradient descent down a restricted submanifold of the full parameter space. Relatedly, Jeremy Bernstein has proposed *Manifold Muon*, which tweaks Muon such that the weights remain perfectly-conditioned  [^bernstein25].


[^spectral]: *Spectral directions* are the left and right singular vectors in the SVD decomposition of a matrix. They correspond to the directions the matrix stretches the input/output space. The amount each direction is stretched corresponds to *singular values* of the matrix. Orthogonalization finds a matrix with identical spectral directions but with all the singular value equal to $1$. The resulting transformation is *isometric* between the input and output spaces: distances, lengths, and angles are preserved.


[^anonymous26]: [Anonymous ICLR Conference Submission 2025](https://openreview.net/forum?id=go388T3QjQ) *Long-tailed Learning with Muon Optimizer*
[^su25]: [Su 2025](https://arxiv.org/abs/2511.00674) *Isotropic Curvature Model for Understanding Deep Learning Optimization*
[^kovalev]: [Kovalev 2025](https://arxiv.org/abs/2503.12645) *Understanding Gradient Orthogonalization for Deep Learning via Non-Euclidean Trust-Region Optimization*

[^bernstein25]: [Jeremy Bernstein 2025](https://thinkingmachines.ai/blog/modular-manifolds/) *Modular Manifolds*

To the best of my knowledge, there is no argument that Muon solves (1)—e.g. that orthogonalization corrects for noise in the gradient estimation. To account for this, we want to estimate the noise along each spectral direction and correct for it. After orthogonalization, the columns of the update matrix correspond to the spectral directions, so we can estimate spectral variance by calculating column-wise RMS. For this reason, variance adaptation methods for Muon only require variance estimates along a single dimension[^frans].

```python {5-6}
def adaptive_muon_update(grad, momentum1, momentum2, beta1, beta2):
	momentum1.lerp_(grad, 1 - beta1)
	update = grad.lerp_(momentum1, beta1)
	update = orthogonalize(update)
	momentum2.lerp_((update ** 2).mean(dim=-1, keepdim=True), 1 - beta2)
	update /= momentum2.sqrt()
	update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
	return update
```

**Algorithm 1: Typical-ish Adaptive Muon update (differences from standard Muon highlighted).**

Note that the above technique estimates variance for the $i$th strongest spectral component, whose direction will fluctuate over subsequent steps. Future work might account for this fluctuation, or find alternative ways to estimate spectral noise.

The particular variant of Adaptive Muon that is adopted in Modded NanoGPT, *NorNuon*, maintains the Frobenius norm of the update after dividing by variance [^normuon]. Whether we track variance columnwise or rowwise is determined by which of the parameter's dimensions is larger.


%% 1D (or even scalar) variance factor  [^normuon] [^adamuon] [^adago] [^frans] . There are quite a few reasons for this, including that
* noise estimates are correlated across columns/rows (which roughly correspond to spectral directions in the orthogonal basis)
* per-parameter updates will disrupt the spectral conditioning of orthogonalization.
* Lower rank estimates reduce memory/FLOPs (useful in Modded NanoGPT!)
. For this reason, we want to estimate the variance after we have orthogonalized the matrix. From here, we compute rowwise/columnwise norms
Instead, 1D methods maintain variance estimates for each spectral direction by rank
that is, a

In my opinion, variance estimator for Muon would be in the spectral
The spectral normalization step in Muon has been argued to solve both of these, but it is demonstrated

If we intuitively understand Muon as taking equal steps in all spectral directions, it seems less likely we'll end up in sharp crevasses.

We can sketch an argument that Muon's spectral normalization is a form of curvature-aware normalization. For
This argument is formalized in a few places, though I have not reviewed how careful these proofs are %%


%% Therefore, the primary benefit of an adaptive term on Muon may be (1) %%


%% Several recent works have pointed out that orthogonalization approximations Gauss-Newton/Fisher preconditioning [^frans] []
Muon's update is constrained such that all spectral directions have the same magnitude. This means that, while descending the loss landscape, we are going through an extremely  %%

%% Franz Cesista notes that Muon is interpretable as a simpler algorithm (msign)  [^leloy]

At the time, it was unclear why adding an adaptive method to Muon would benefit from a lower learning rate.

In fact, we may expect the opposite. why? adaptive methods lower variance. low variance -> higher step size  %%

%% ## (2) Optimal Learning Rates

In minibatch training, larger batches average over more samples, which reduces the expected variance in our gradient sample. A good step size will adhere to the curvature of the loss landscape: we want to get close to the minima without overshooting. Increasing the learning rate will amplify noise in our gradient estimate, but decreasing the learning rate will mean it will take many steps to converge.

One understanding is that there is an optimal error budget per step, and we should choose a learning rate $\eta$ that is inverse to the expected variance of the gradient.
%%

## (2) Batch size scheduling
Choosing a batch size involves a tradeoff:
1) Smaller batch sizes are more token-efficient. For a fixed token budget, above a certain point, increasing the batch size will worsen final model performance.
2) Training at higher batch sizes is faster for various reasons: GPUs parallelize over the batch dimension, DDP is easy, every step incurs overhead (optimizers, comms, etc.)

A *critical batch size* $B_{\text{critical}}$  balances both of these considerations: low enough to be token-efficient, and high enough to be speed-efficient.

<img src="../images/Muon/cbs-revisited.png"  class="plot" style="width: 95%; height: auto; flex-shrink: 0;">

**Figure 1: Critical batch sizes for Adam at various token budgets [^ai2]. $B_{\text{critical}}$, chosen as the greatest batch size exceeding below 1% of the lowest loss, is marked in red. Batch size here is in units of 4096 Tokens.**

%% For small token budgets, $B_{\text{critical}}$ will be low, so it is possible the power-law scaling $\eta \propto B^p$ is accurate. At a larger scale, however, $\eta_{\text{saturated}}$ decreases (Fig1) and $B_{\text{critical}}$ increases (Fig3), so you should expect that the optimal learning rate for $B_{\text{critical}}$ is approximately $\eta_{\text{saturated}}$.  %%

Since the critical batch size increases as the total token budget increases, it is safe to increase the batch size throughout the course of training.  Batch size literature that has focused on Muon has discovered it's critical batch size is significantly higher than Adam's, which may allow for higher batch sizes [^essentialai].  Both Kimi K2 and GLM 4.5's recipes for training with Muon involve some level of batch size increase through training[^kimi]  [^glm].

To accurately determine the critical batch size, we need to determine what the optimal learning rates are at each batch size. Larger batches average over more samples, which reduces gradient variance. For SGD, we can directly model the relationship between the optimal learning rate $\eta$ and batch size $B$ [^Mccandlish]:

$$
\eta \propto 1 - \frac{1}{1 + B \cdot \text{SNR}(G)}
$$

where $\text{SNR}(G)$ is the signal-to-noise ratio of our gradient estimate[^snr]. When $B \cdot SNR(G) << 1$, this is approximately $B \cdot SNR(G)$, which gives us our "linear scaling law".

Adam dampens update variance, so its optimal scaling is closer to $\eta \sim \sqrt{B}$[^mccandlish] [^granziol] before also becoming asymptotic[^li]:

$$
\eta \propto \frac{\sqrt{B \cdot \text{SNR}(G)}}{1+ {B \cdot \text{SNR}(G)}} \qquad \text{for} \; B \cdot SNR(G) << 1
$$


A power-law relationship ($\eta \propto B^p$) is often assumed to hold for Muon as well[^sato][^ryu], though in practice it will asymptote quickly:

[^sato]: [Sato et al 2025](https://arxiv.org/abs/2507.01598) *Convergence Bound and Critical Batch Size of Muon Optimizer*.
[^ryu]: [Simo Ryu 2025](https://x.com/cloneofsimo/status/1907731069878825400) *Adam vs Shampoo vs Muon on MNIST. all follow the lr ~ sqrt(BS) law.*

%%
(see [[#(4) Related work]]). Muon orthogonalizes (i.e. spectrally normalizes) the update before stepping. Some literature proposes that orthogonalization can amplify noise [^adago], especially when the underlying gradient is ill-conditioned.

which provides some additional motivation for adding variance adaptation.




Due to this, I propose that as batch size increase, the optimal learning rate will converge to $\eta_{\text{saturated}}$. We can test this hypothesis by sweeping large batch sizes:

Adaptive Muon rapidly converges to the saturated learning rate.
(Faster than non-adaptive muon? would need to test)
The rate of saturation is controlled by the noise in the update. Implying that the adaptive term

In order to apply batch size increases inside of Modded NanoGPT, I needed to study how the critical batch size increases throughout the (very short) duration of training. In order to do this accurately, I needed to batch each batch size according to its optimal batch size.  %%

<img src="../images/Muon/validation_loss_heatmap_medium_light.png" class="theme-image-light plot" style="width: 100%; height: auto; flex-shrink: 0;">
<img src="../images/Muon/validation_loss_heatmap_medium_dark.png" class="theme-image-dark plot" style="width: 100%; height: auto; flex-shrink: 0;">


**Figure 2: Sweeping Adaptive Muon at a learning rate x log(batch size) grid at three different token budgets. The lowest loss per batch size is selected in red.**

At token budgets of $\approx 134M$, $536M$, $1073M$ tokens, the optimal learning rate seems to converge around $\eta \approx 0.035$, $0.025$, and $0.015$, respectively. Note that $B = 4 \times 2^{14}$ does not appear to converge at the highest token budget for any of the swept learning rates.

Below the asymptote/saturated batch size, we can approximate the value $p$ in $\eta \propto B^p$ via the log-slope of the red boxes. At this granularity, $p \approx 0.6-1.0$ seems approximately correct. Convergence to the optimal learning rate appears to be very fast for this variant of Muon.


  [^glm]: [Zeng et al](https://arxiv.org/abs/2508.06471) *GLM-4.5*
  [^kimi]: [Moonshot AI, Liu et al 2025](https://arxiv.org/abs/2502.16982) *Muon is Scalable for LLM Training*


%% Boreiko et al (2025) performs a hyperparameter sweep in which Muon's optimal learning rate is stable at various batch sizes [^Boreiko].

For low batch sizes, Muon's optimal learning rate may be well approximated via a power-law $\eta \propto B^p$ [^muonp]. Since orthogonalization keeps update variance constant regardless of batch size, the optimal learning rate must depend on other factors. I hypothesize that since small batches are noisy estimates of the underlying distribution, they will produce gradients with highly distinct directions. Even though each update matrix (the orthogonalized gradient) has the same expected variance, the variance *between* successive updates is high. A lower learning rate can act as an implicit smoothing, similar to Muon's momentum coefficients $\beta$. Sufficiently large batch sizes will produce gradients with spectral directions that are aligned with the actual data ($B \to \infty$). A complete picture of learning rate scaling should consider inter- and intra-step variances.  %%




<img src="../images/Muon/loss_vs_batch_size_light.png" class="theme-image-light plot" style="width: 100%; height: auto; flex-shrink: 0;">
<img src="../images/Muon/loss_vs_batch_size_dark.png" class="theme-image-dark plot" style="width: 100%; height: auto; flex-shrink: 0;">

**Figure 3: Results from the previous sweep (Fig1) while choosing the optimal learning rate for each batch size. For a fixed token budget, increasing batch size will tend to decrease final validation loss. $B_{\text{critical}}$ is the highest acceptable batch size given some tolerance for loss.**

The above graphs demonstrate an incredible stability in the critical batch size for our Adaptive Muon optimizer, especially at higher token budget. Consider the flatness in the curve for the highest token budget ($\approx 1T$ parameters): $1024$ steps at a A batch size of $2^{20}$ tokens converges to nearly the same loss at $1024$ steps as a batch size of $2^{18}$ tokens in $4096$ steps!


%%
Practically, I do not recommend doing this when training with Muon. As stated earlier, Kimi K2's training recipe did not increase learning rate when applying a batch size increase [^kimi].

Additionally, I recommend that your Muon hyperparameter sweeps shouldn't assume the square-root law holds. For large batch sizes, the optimal batch size may not require serious tuning.

TODO: Insert Modded NanoGPT PR here  %%


## (3) Faster orthogonalization: Polar Express and beyond
The Newton-Schulz iterative algorithm approximates orthogonalization via a quintic polynomial iteration. This iterative approach is agnostic to the conditioning of the underlying matrix. However, if we notice that the conditioning of the matrix improves in each iteration, we can find an optimal polynomial for that iteration. Ansel et al provide optimal coefficients at each iteration step via their algorithm Polar Express [^ansel].

Last month, a paper from Shulgin et al demonstrated that a more precise orthogonalization improves Muon convergence, especially when accompanied with appropriate learning rate tuning:

<img src="../images/Muon/shulgin-heatmap.png"  class="plot" style="width: 100%; height: auto; flex-shrink: 0;">

**Figure 4: Validation Loss at two levels of convergence for orthogonalization. Caption from Shulgin: "the optimal learning rate couples with approximation quality \[...\] higher precision → higher optimal LR + wider stability,"**

It follows that using Polar Express over Newton Schulz represents an improvement in convergence, and using it led to a new record in Modded NanoGPT.

On ongoing effort in the Modded NanoGPT speedrun is being made for even faster orthogonalization via *Almost-Orthogonal Layers*, though this has not been incorporated yet [^Boission].

[^Boission]: [Thibaut Boissin 2025](https://github.com/KellerJordan/modded-nanogpt/pull/155) *Modded NanoGPT PR#155*
[^ansel]: [Ansel et al 2025](https://arxiv.org/abs/2505.16932) *The Polar Express*

## (4) Cautious weight decay
Decoupled weight decay has been noted as a crucial technique for training with Muon. The Kimi team writes: "While vanilla Muon initially converges faster, we observed that some model weights grew too large over time, potentially limiting the model’s long-term performances. Adding weight decay addressed this issue - the results demonstrate that Muon with weight decay outperforms both vanilla Muon and AdamW" [^kimi].

Inside Modded NanoGPT, weight decay on Muon was doing more harm than good. In October, a variant of decoupled weight decay known as *Cautious Weight Decay* was discovered [^chen25], which only decays parameters that increase in magnitude in the update step:

```python {2}
def apply_update(param, update, learning_rate, weight_decay):
	param *= (update * param) >= 0
	update += weight_decay * param
	return param - learning_rate * update
```
**Algorithm 2: Cautious weight decay (difference from decoupled weight decay highlighted).**

This technique proved highly effective in Modded NanoGPT when paired with a schedule that decreases weight decay to $0$ by the end of training.

[^chen25]: [Chen et al 2025](https://arxiv.org/abs/2510.12402) *Cautious Weight Decay*
## (5) Distributed and batched computation
The implementation of Muon has been optimized in order to distribute the implementation over 8 devices. While some of these engineering tricks are interesting (e.g. batched orthogonalization), they do not represent algorithmic differences from the original form of Muon. For additional information I direct you to Larry Dial's blog post [^larry2025a] and the Modded NanoGPT repo [^moddednanogpt].


## (6) Implementation Notes

> "I only trust speedruns." —Keller Jordan [^jordan24a]

[^jordan24a]: [Keller Jordan 2025](https://x.com/kellerjordan0/status/1890178773586489716) *The reason I didn't write a proper arxiv paper for Muon is because I simply don't think there's any relationship between the ability to publish a paper with lots of good-looking results about a new optimizer, and whether that optimizer actually works. I only trust speedruns.*


I hope this post is broadly useful for pretraining with Muon. I want to highlight some important considerations for the Modded NanoGPT recipe:
- The model is very small (<124M active params) and has a unique architecture.
- Adam is used instead of Muon on a few parameters: the linear output head, the embedding dimension, and a few scalar constants, though this is "standard" for Muon.
- Adam is stepped with twice the number of gradient accumulation steps as Muon. Effectively, it has a 2x batch size than Muon. In the experiment in [[#(2) Batch size scheduling|Section 2]], I've scaled the Adam learning rate according to the square-root law.
%%

In the hyperparameter sweep, batch sizes $> 64 \times 16384$ tokens seemed too large for my token budget. In particular, the number of total steps was low, which increased variability. The Modded NanoGPT speedrun uses a batch size of $16 \times 16384$ for a token budget of approx $570M$ tokens.

On September 30th, the Modded NanoGPT speedrun was optimized by decreasing Muon's batch size by 33%, shaving $\approx 1.7$ seconds off the record time. On October 24th, *Normuon*, a variant of Adaptive Muon, was added to Modded NanoGPT, which sped up convergence by approximately $\approx 0.8$ seconds. A few days later, I cut the learning rate in half and saved $\approx 1.9$ more seconds. Reviewing the relationship between learning rate and batch size can help us understand why this worked, and how we can squeeze out additional benefits.  %%
%%
## (4) Related work
The following chart has been tweeted in evidence that $\eta \propto \sqrt{B}$ holds for Muon:

On this log(batch size) x log(learning rate) chart, the interpretation is that a power law relationship $\eta \propto B^p$ will have a linear "loss valley" with slope $p$. I've cropped and annotated the chart below to demonstrate what I mean:

Since the above experiment was performed on MNIST at a low $(<10^4$ samples) batch size, it makes sense that a power law is approximately correct. In my view, however, this chart seems to hint at eventual convergence:

Sato et al (2025)[^sato] performs a hyperparameter sweep that assumes the square-root scaling law will hold for Muon. Since the maximum batch size tried in this paper was only $8192$ tokens, it is likely that the square-root law was approximately correct.

Shah et al (2025) [^essentialai] seemingly uses the same learning rate for Muon across a sweep of batch sizes (their section 2.3), though it is unclear if this was the result of tuning.

Shat et al and Sato et al take an opposite view on whether the Critical Batch Size is higher for Muon than for Adam. The former claims that Muon has a higher Critical Batch Size than Adam (and is therefore more token-efficient). I believe they are likely to be correct, since their experiments were performed on larger models ($500M$ parameters) and larger batch sizes (up to $16M$).

Boreiko et al (2025)[^Boreiko] performs a hyperparameter sweep on variations of Muon in of NanoGPT in which they find that "optimal learning rate is the same for the smallest and the largest models". Unfortunately, they do not seem to have any explicit results about the relationship of learning rate and batch size.  %%
%%
```
Jeremy Bernstein proves that Muon's optimal learning rate is stable across various model sizes [^bernstein]:
<img src="../images/Muon/bernstein.png" class="plot" style="width: 95%; height: auto; flex-shrink: 0;">

**Figure 7: From Bernstein's *Deriving Muon*. This experiment was conducted on various sizes of MLPs being trained on CIFAR. "Dualization" here refers to orthogonalizing the gradient.**
```


Li et al 2024 [^Li] has a great overview on learning rate scaling for adaptive optimizers. They propose that $\eta$  is asymptotic for Adam as well. They attribute this convergence due to the second moment's effect on the gradient, which normalizes the update in such a way that the variance of the update will also saturate at sufficiently high batch size. Practically, I believe Muon's learning rate saturates much earlier than Adam's, though this argument requires future work.
 %%

This post summarizes the work of many people on the Modded NanoGPT speedrun. Section 1 primarily corresponds to the work of an author of *NorMuon*, Zichong Li. Sections 2, 3, and 4 correspond to records added by myself. Section 5 is the result of many people over many iterations, though in the last few months Larry Dial especially.

---

<br>
Thank you to Prime Intellect, who sponsors my research. If this blog post was useful for you, you can cite:

```
@misc{
	srivastava2025,
	author = {Varun Srivastava},
	title = {Optimal Learning Rates for Muon},
	year = {2025},
	url = {https://varunneal.github.io/essays/muon}
}
```

[^McCandlish]: [McCandlish et al 2018](https://arxiv.org/pdf/1812.06162) *An Empirical Model of Large-Batch Training*
[^granziol]: [Granziol et al 2020](https://arxiv.org/pdf/2006.09092) *Learning Rates as a Function of Batch Size* contains a proof.
[^Li]: [Li et al 2024](https://arxiv.org/abs/2405.14578) *Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling*

[^Boreiko]: [Boreiko et al 2025](https://openreview.net/forum?id=ppmyFtr9EW)  *Towards Understanding Orthogonalization in Muon*
[^ai2]: [Ai2, Merrill et al 2025](https://arxiv.org/abs/2505.23971) *Critical Batch Size Revisited*
[^Bernstein]: [Bernstein 2025](https://jeremybernste.in/writing/deriving-muon) *Deriving Muon*.
[^essentialAI]: [Essential AI, Shah 2025](https://arxiv.org/abs/2505.02222) *Practical Efficiency of Muon for Pretraining*.

[^muonp]: I did not conduct my experiments in high enough resolution to establish what $p$ should be, though I suspect $p > 0.5$, as per Fig5.


[^su]: [Su 2025](https://kexue.fm/archives/11416) *Muon Optimizer Guide* has a comprehensive overview on different per-shape learning rate strategies.
[^larry]: [Larry 2025](https://github.com/KellerJordan/modded-nanogpt/pull/136#issuecomment-3536835289)
[^wen]: [Wen et al 2025](https://arxiv.org/pdf/2509.02046) *Fantastic Optimizers and Where to Find Them* performs a hyperparameter sweep on various optimizers including Muon. The optimal configs from their sweep can be found [here](https://github.com/WhenWen/marin/tree/kaiyue/optimizers/experiments/optimizer_sweep/Analysis/Results/).
[^jordan-warmup]: [Jordan 2024](https://x.com/kellerjordan0/status/1845867151946899852) "Removed learning rate warmup, since the optimizer (Muon) doesn't need it"

[^mahdid]: [Mahdid 2025](https://www.yacinemahdid.com/p/muon-optimizer-explained-to-a-toddler) *muon optimizer explained to a toddler*
[^snr]: I'm being somewhat imprecise with how we define the signal to noise ratio. McCandlish et al [^mccandlish] defines a critical batch size $B_{\text{noise}} =  \text{Cov}(G) / \mathbb{E}(G)^2$ for their analysis of SGD. I'm using its inverse as the SNR.
