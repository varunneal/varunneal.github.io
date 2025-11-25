---
draft: "false"
created: November 24, 2025
hidden: "true"
---
(Draft)

In this blog I propose that Muon's optimal learning rate saturates quickly as you scale batch size, and so is quickly asymptotic as opposed to Adam's square-root law $\eta \propto \sqrt{B}$. I also survey how this relates to Critical Batch Size and learning rate schedules for Muon. 

## (1) Optimal Learning Rate
In minibatch training, larger batches average over more samples, which reduces gradient variance. If we want each step to have a fixed variance budget, we should choose a learning rate $\eta$ that is inverse to the expected variance of the gradient. For SGD, the optimal learning rate increases linearly with batch size (up to a point). Adam dampens update variance, so its optimal scaling is closer to $\eta \approx \sqrt{B}$[^sqrtadam].


This relationship is often assumed to hold for Muon as well (see [[#(4) Related work]]), but we should expect a difference. When we scale Adam's step size by $\sqrt{B}$, it's because we expect the update's variance to be proportional to $B^{-0.5}$. Muon, however, orthogonalizes (i.e. spectrally normalizes) the update before stepping. As Jeremy Bernstein points out, after orthogonalization the expected variance is constant with respect to input shape—the batch size has no impact on update variance.[^Bernstein] 

Due to this, I propose that as batch size increase, the optimal learning rate will converge to $\eta_{\text{saturated}}$. We can test this hypothesis by sweeping large batch sizes:
<img src="../images/Muon/validation_loss_heatmap_medium_light.png" class="theme-image-light plot" style="width: 100%; height: auto; flex-shrink: 0;">
<img src="../images/Muon/validation_loss_heatmap_medium_dark.png" class="theme-image-dark plot" style="width: 100%; height: auto; flex-shrink: 0;">


**Figure 1: Sweeping Modded NanoGPT on a learning rate x log(batch size) grid at three different token budgets. The lowest loss per batch size is selected in red. See [[#(3) Implementation Notes]] for more details.** 

At token budgets of $\approx 134M$, $536M$, $1073M$ tokens, the optimal learning rate seems to converge around $\eta \approx 0.035$, $0.025$, and $0.015$, respectively. Note that $B = 4 \times 2^{14}$ does not appear to converge at the highest token budget for any of the swept learning rates. 

There appears to be additional empirical validation for this asymptotic scaling law. Notably, Kimi K2's Moonlight technical report implies they do not increase Muon's learning rate when they increase the batch size[^kimi] . Boreiko et al (2025) performs a hyperparameter sweep in which Muon's optimal learning rate is stable at various batch sizes [^Boreiko].  

For low batch sizes, Muon's optimal learning rate may be well approximated via a power-law $\eta \propto B^p$ [^muonp]. Since orthogonalization keeps update variance constant regardless of batch size, the optimal learning rate must depend on other factors. I hypothesize that since small batches are noisy estimates of the underlying distribution, they will produce gradients with highly distinct directions. Even though each update matrix (the orthogonalized gradient) has the same expected variance, the variance *between* successive updates is high. A lower learning rate can act as an implicit smoothing, similar to Muon's momentum coefficients $\beta$. Sufficiently large batch sizes will produce gradients with spectral directions that are aligned with the actual data ($B \to \infty$). A complete picture of learning rate scaling should consider inter- and intra-step variances. 

## (2) Relationship to Critical Batch Size
Choosing a batch size involves a tradeoff:
1) Smaller batch sizes are more token-efficient. For a fixed token budget, above a certain point, increasing the batch size will worsen final model performance.
2) Training at higher batch sizes is faster for various reasons: GPUs parallelize over the batch dimension, DDP is easy, every step incurs overhead (optimizers, comms, etc.)

A *critical batch size* $B_{\text{critical}}$  balances both of these considerations: low enough to be token-efficient, and high enough to be speed-efficient.

<img src="../images/Muon/cbs-revisited.png"  class="plot" style="width: 100%; height: auto; flex-shrink: 0;">

**Figure 2: Loss curves for Adam from *Critical Batch Size Revisited* [^ai2]. $B_{\text{critical}}$, chosen as the greatest batch size exceeding below 1% of the lowest loss, is marked in red. Batch size here is in units of 4096 Tokens.**


<img src="../images/Muon/loss_vs_batch_size_light.png" class="theme-image-light plot" style="width: 100%; height: auto; flex-shrink: 0;">
<img src="../images/Muon/loss_vs_batch_size_dark.png" class="theme-image-dark plot" style="width: 100%; height: auto; flex-shrink: 0;">

**Figure 3: Results from the previous sweep (Fig1) while choosing the optimal learning rate for each batch size. For a fixed token budget, increasing batch size will tend to decrease final validation loss. $B_{\text{critical}}$ is the highest acceptable batch size given some tolerance for loss. Similar curves are given in Shah et al [^essentialAI] and in Sato et al[^sato].** 

For small token budgets, $B_{\text{critical}}$ will be low, so it is possible the power-law scaling $\eta \propto B^p$ is accurate. At a larger scale, however, $\eta_{\text{saturated}}$ decreases (Fig1) and $B_{\text{critical}}$ increases (Fig3), so you should expect that the optimal learning rate for $B_{\text{critical}}$ is approximately $\eta_{\text{saturated}}$. 

Since the critical batch size increases as the total token budget increases, it is safe to increase the batch size throughout the course of training. Previous critical batch size literature has focused on Adam, and recommends scaling the learning rate alongside batch size [^ai2]. Practically, I do not recommend doing this when training with Muon. As stated earlier, Kimi K2's training recipe did not increase learning rate when applying a batch size increase [^kimi]. 

Additionally, I recommend that your Muon hyperparameter sweeps shouldn't assume the square-root law holds. For large batch sizes, the optimal batch size may not require serious tuning.

TODO: Insert Modded NanoGPT PR here 


## (3) Implementation Notes 

I hope this post is broadly applicable to pretraining with Muon, but my experimentation was conducted on the Modded NanoGPT architcture, which uses a optimized variant of Muon. I want to highlight some important considerations:
- The model is very small (<124M active params) and has a unique architecture. 
- Adam is used instead of Muon on a few parameters: the linear output head, the embedding dimension, and a few scalar constants, though this is "standard" for Muon. 
- Adam is stepped over twice the number of gradient accumulation steps than Muon. Effectively, it has a 2x batch size than Muon. In the above experiments, I've scaled the Adam learning rate according to the square-root law. 
- Modded NanoGPT uses a specialized version of Muon:
	- *Normuon*: A 1d momentum buffer tracks variance along one dimension of each parameter. This adds an adafactor like adaptivity to stepsizes.
	- *Cautious weight decay*: A masked version of decoupled weight decay
	- *Polar express*: A faster-converging sign-iteration function than Newton-Schultz
	- *Shape transformations*: The orthogonalization step is dependent on the shape of the input parameters. Each attention weight Q, K, and V are treated separately, however, they are not split by head. 
	- Also related to shape, learning rate is scaled by $\max(1, \sqrt{d_{out} / d_{in}})$ per parameter. This is motivated by keeping the the variance of the orthogonalized update independent of parameter shape [^bernstein][^su]. 

In the hyperparameter sweep, batch sizes $> 64 \times 16384$ tokens seemed too large for my token budget. In particular, the number of total steps was low, which increased variability. The Modded NanoGPT speedrun uses a batch size of $16 \times 16384$ for a token budget of approx $570M$ tokens. 

## (4) Related work
The following chart has been tweeted in evidence that $\eta \propto \sqrt{B}$ holds for Muon:

<img src="../images/Muon/simo-0.png" class="plot" style="width: 95%; height: auto; flex-shrink: 0;">

**Figure 4: Original caption: *Adam vs Shampoo vs Muon on MNIST. all follow the lr ~ sqrt(BS) law*.[^ryu]**

On this log(batch size) x log(learning rate) chart, the interpretation is that a power law relationship $\eta \propto B^p$ will have a linear "loss valley" with slope $p$. I've cropped and annotated the chart below to demonstrate what I mean: 

<img src="../images/Muon/simo-2.jpg" class="plot" style="width: 95%; height: auto; flex-shrink: 0;">

**Figure 5: Fig4 cropped to just include Adam and Muon. I've drawn a red line indicating the valley of minima losses as we increase batch size.** 

Since the above experiment was performed on MNIST at a low $(<10^4$ samples) batch size, it makes sense that a power law is approximately correct. In my view, however, this chart seems to hint at eventual convergence:

<img src="../images/Muon/simo-1.jpg" class="plot" style="width: 95%; height: auto; flex-shrink: 0;">

**Figure 6: I've redrawn the Muon red line to be concave, which I believe has a slightly better fit.**

Sato et al (2025)[^sato] performs a hyperparameter sweep that assumes the square-root scaling law will hold for Muon. Since the maximum batch size tried in this paper was only $8192$ tokens, it is likely that the square-root law was approximately correct. 

Shah et al (2025) [^essentialai] seemingly uses the same learning rate for Muon across a sweep of batch sizes (their section 2.3), though it is unclear if this was the result of tuning. 

Shat et al and Sato et al take an opposite view on whether the Critical Batch Size is higher for Muon than for Adam. The former claims that Muon has a higher Critical Batch Size than Adam (and is therefore more token-efficient). I believe they are likely to be correct, since their experiments were performed on larger models ($500M$ parameters) and larger batch sizes (up to $16M$).

Boreiko et al (2025)[^Boreiko] performs a hyperparameter sweep on variations of Muon in of NanoGPT in which they find that "optimal learning rate is the same for the smallest and the largest models". Unfortunately, they do not seem to have any explicit results about the relationship of learning rate and batch size. 

Jeremy Bernstein proves that Muon's optimal learning rate is stable across various model sizes [^bernstein]:
<img src="../images/Muon/bernstein.png" class="plot" style="width: 95%; height: auto; flex-shrink: 0;">

**Figure 7: From Bernstein's *Deriving Muon*. This experiment was conducted on various sizes of MLPs being trained on CIFAR. "Dualization" here refers to orthogonalizing the gradient.**


Li et al 2024 [^Li] has a great overview on learning rate scaling for adaptive optimizers. They propose that $\eta$  is asymptotic for Adam as well. They attribute this convergence due to the second moment's effect on the gradient, which normalizes the update in such a way that the variance of the update will also saturate at sufficiently high batch size. Practically, I believe Muon's learning rate saturates much earlier than Adam's, though this argument requires future work.

---
Thank you to Prime Intellect, who sponsors my research. If this blog post was useful for you, you can cite:

```
@misc{
	srivastava2025, 
	author = {Varun Srivastava}, 
	title = {Optimal Learning Rates for Muon}, 
	year = {2025}, 
	url = {https://varunneal.github.io/Essays/Optimal-Learning-Rates-for-Muon} 
}
```

[^sqrtadam]: [Granziol et al 2020](https://arxiv.org/pdf/2006.09092) *Learning Rates as a Function of Batch Size* is probably the earliest proof of this law, though [McCandlish et al 2018](https://arxiv.org/pdf/1812.06162) demonstrate it empirically. 
[^Li]: [Li et al 2024](https://arxiv.org/abs/2405.14578) *Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling* has a great overview on this topic. Additionally, they propose that batch size is also asymptotic in Adam for sufficiently high batch size. They attribute this convergence due to the second moment of Adam, which normalizes the update in such a way that the variance of the update will also saturate at sufficiently high batch size.
[^ryu]: [Ryu 2025](https://x.com/cloneofsimo/status/1907731069878825400) I do not mean to pick on Simo Ryu, whose chart I found quite useful. 
[^Boreiko]: [Boreiko et al 2025](https://openreview.net/forum?id=ppmyFtr9EW)  *Towards Understanding Orthogonalization in Muon*
[^ai2]: [Ai2, Merrill et al 2025](https://arxiv.org/abs/2505.23971) *Critical Batch Size Revisited* 
[^Bernstein]: [Bernstein 2025](https://jeremybernste.in/writing/deriving-muon) *Deriving Muon*. 
[^essentialAI]: [Essential AI, Shah 2025](https://arxiv.org/abs/2505.02222) *Practical Efficiency of Muon for Pretraining*. 
[^sato]: [Sato et al 2025](https://arxiv.org/abs/2507.01598) *Convergence Bound and Critical Batch Size of Muon Optimizer*. 
[^muonp]: I did not conduct my experiments in high enough resolution to establish what $p$ should be, though I suspect $p > 0.5$, as per Fig5. 
[^kimi]: [Moonshot AI, Liu et al 2025](https://arxiv.org/abs/2502.16982) *Muon is Scalable for LLM Training*. Excerpt: "`33B to 5.2T tokens:` In this stage, the learning rate decays from 4.2e-4 to 4.2e-5 in a cosine style. We keep the batch size at 2048 until 200B tokens, and then doubled to 4096 for the remaining". This paper also proposes a power law between computational budget and optimal learning rate and batch size for Muon (their Table 9). 
[^jordan]: [Jordan 2025](https://x.com/kellerjordan0/status/1845867151946899852) "Removed learning rate warmup, since the optimizer (Muon) doesn't need it"
[^su]: [Su 2025](https://kexue.fm/archives/11416) *Muon Optimizer Guide* has a comprehensive overview on different per-shape learning rate strategies. 
[^larry]: [Larry 2025](https://github.com/KellerJordan/modded-nanogpt/pull/136#issuecomment-3536835289) 
[^wen]: [Wen et al 2025](https://arxiv.org/pdf/2509.02046) *Fantastic Optimizers and Where to Find Them* performs a hyperparameter sweep on various optimizers including Muon. The optimal configs from their sweep can be found [here](https://github.com/WhenWen/marin/tree/kaiyue/optimizers/experiments/optimizer_sweep/Analysis/Results/). 