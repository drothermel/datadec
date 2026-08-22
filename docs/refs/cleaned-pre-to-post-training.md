---
title: Cleaned Pre-to-Post Training
---

<callout>
	## Related Work of Previous Projects/Questions
	### Topic 1: Continual Learning (Plasticity + Multi-Power Laws) {toggle="true"}
		**My description of the initial question**: “Can features of a loss curve be used to estimate or predict certain properties relevant to ‘success’ however that is defined?”
		Started out with the work by Sutton’s student and Clare Lyle about plasticity (considering experiments on the scale of CIFAR-10).
		<callout color="gray_bg">
			- [Shibhansh Dohare](https://app.notion.com/p/3c2de135cd1f8034bc8ed1e80221f6b8)  et al., [Loss of plasticity in deep continual learning](https://app.notion.com/p/a3e975b6036d4d729bed0d594de740e0) (Nature 2024; earlier arXiv version “Maintaining Plasticity…”, 2306.13812) — Sutton’s student is [Shibhansh Dohare](https://app.notion.com/p/3c2de135cd1f8034bc8ed1e80221f6b8) . They show that standard deep-learning methods gradually lose plasticity in continual-learning settings until they learn no better than a shallow network, demonstrated on ImageNet (repurposed as task sequences) and RL problems, and propose continual backpropagation (selectively reinitializing dormant/unuseful units during training). The incremental-CIFAR experiments you remember are in their codebase.”
			- [Clare Lyle](https://app.notion.com/p/3c2de135cd1f80f9a315ff7bbdf9894e)  et al., [Understanding Plasticity in Neural Networks](https://app.notion.com/p/3c2de135cd1f809ebd9fe2463f494efd)  (ICML 2023, arXiv 2303.01486) — a systematic empirical analysis finding plasticity loss is deeply connected to changes in loss-landscape curvature, often occurring without saturated units; and the follow-up [Disentangling the Causes of Plasticity Loss in Neural Networks](https://app.notion.com/p/6bde1d83f6de4c56ba273e25fb0aaf2b)  (arXiv 2402.18762).
		</callout>
		Ended up at llm pretraining and papers like the Multi-Power law paper that predicts test loss from train loss and downstream accuracy from test loss.
		<callout color="gray_bg">
			- [Kairong Luo](https://app.notion.com/p/f2491f5a6e1a412eb4bced7331c50c4f)  et al., [A Multi-Power Law for Loss Curve Prediction Across Learning Rate Schedules](https://app.notion.com/p/29ba77d7fdf949e1a1f9bdb4df8c35a4)  (arXiv 2503.12811, ICLR 2025) — predicts the full pretraining loss curve at every intermediate step across LR schedules, using a power law on the sum of learning rates plus extra power-law terms for the decay-induced loss drop; fitted on a few runs, it extrapolates to unseen schedules and even discovers a schedule beating cosine (resembling WSD).
			- For loss → downstream accuracy: [Yangyi Chen](https://app.notion.com/p/756ac5715d434d17a62fae97caf17189) et al., [Scaling Laws for Predicting Downstream Performance in LLMs](https://app.notion.com/p/59133e03cc774153b83012321d143718)  (arXiv 2410.08527) — the two-stage “FLP” pipeline: FLOPs → pretraining loss → downstream performance. Related: [Samir Yitzhak Gadre](https://app.notion.com/p/9ef7ea1367cf4c2b9c916bfe173265c6)  et al. 2024 ([Language models scale reliably with over-training and on downstream tasks](https://app.notion.com/p/0be09c486c0d4cb4a37823b841b48c63) ), where downstream accuracy is predicted as an exponential function of training loss, and [Akshita Bhagia](https://app.notion.com/p/64073767dcbb48d48b51bb5ce1ec6ed1)  et al.’s model-ladders paper ([Establishing Task Scaling Laws via Compute-Efficient Model Ladders](https://app.notion.com/p/a46c70a5768e4964b656aeb958b2ed8a) ) which maps compute → task NLL → accuracy.
		</callout>
		<callout color="gray_bg">
			#### Similarities
			> **What low-dimensional summary of training dynamics is sufficient to forecast a capability?**
			- The plasticity answer so far is “no single statistic — curvature comes closest” (Lyle)
			- The pretraining answer is “a surprisingly simple functional of the LR schedule” (multi-power law) plus a sigmoid/exponential link to accuracy — with the caveat that hard accuracy metrics can look emergent, showing no progress above chance until the loss crosses a threshold, which is where the loss-to-accuracy mapping gets fragile.
			Both sub-communities treat the loss curve (or statistics derivable during training) as a *measurable signal that predicts a latent capability you actually care about*.
			- “In the plasticity literature, the latent quantity is future trainability — can the network still reduce loss on the *next* task?”
				- Lyle’s work explicitly hunts for cheap-to-compute training statistics (curvature, feature rank, dead units, weight norm) that correlate with or cause that ability. 
			- In the scaling-law literature, the latent quantity is final loss or benchmark accuracy, predicted from early/partial loss curves or small-model runs. 
			Both are fundamentally about *optimization dynamics under non-stationarity.*
			- LR decay in pretraining is itself a controlled non-stationarity.
			- The multi-power law’s decay term is essentially modeling how the optimizer’s response to the schedule shapes the curve — a dynamics question the plasticity people would recognize.
			There’s a growing literal bridge: does plasticity loss appear in LLM-scale training.
			- [Can Scale Save Us From Plasticity Loss in Large Language Models?](https://app.notion.com/p/6b9dcac7ded64529b9b29fd319075b9d) by [J. Fernando Hernandez-Garcia](https://app.notion.com/p/41d682e64893426a922671aa41c11ec7) et. al.
			- Matters for continual pretraining and mid-training regimes — exactly where the two threads collide.
		</callout>
		<callout color="gray_bg">
			#### Differences
			- *Regime:* plasticity work assumes an explicitly non-stationary data stream (task sequences, RL bootstrapping); scaling-law work assumes a single stationary distribution where the only “non-stationarity” is the LR schedule.
			- *Target of prediction:* plasticity work predicts a property of the *learner* (future adaptability); scaling work predicts a property of the *outcome* (final loss/accuracy). Success in one is “can it keep learning,” in the other “how good will it be when we stop.”
			- *Methodological flavor:* plasticity is mechanistic/causal — intervening on curvature, resets, normalization to see what restores learning. Scaling laws are phenomenological — fit a parametric form, extrapolate, and mostly stay agnostic about mechanism.
			- *Scale and stakes:* CIFAR/ImageNet/Atari with many cheap seeds vs. Llama-scale runs where the whole point is that you can’t afford to run the experiment, so prediction substitutes for experimentation.
		</callout>
</callout>
<empty-block/>