# Empirical neural tangent kernel — reference topic

**Kind:** reference (standing accumulator). Entries are dated and quoted close to verbatim;
related-work claims are unverified unless a citation with an identifier is given.

Why it matters here: the empirical NTK (eNTK) — its Gram matrix, spectrum, and how fast it
moves during training — is a candidate *landscape / dynamics readout* for the projects that
track optimization state alongside accuracy: `../../potential-projs/landscape-geometry.md`,
`../../potential-projs/trajectory-statistics.md`, and the CNN ablation ladder in
`../../past-projects/cnn-deconstruction-ladder.md` ("track key metrics about the optimization
landscape as I go"). It was also Danielle's first "learn with agents" sprint (2025
seven-track record in `project-approach-principles.md`).

Interactive version of the source: <https://www.perplexity.ai/search/15452705-722c-49aa-8fe3-ad348c1b781a>
(Perplexity page with the generated interactive web app; undated, ~spring 2025; intake
2026-08-22). Danielle's prompt was not pasted; the response describes itself as "an
interactive educational overview of the empirical NTK" with a seven-section app.

---

## Undated (~spring 2025) — Educational overview (condensed)

**Definition.** For $f(x;\theta)$, $\Theta(x,x';\theta) = \nabla_\theta f(x;\theta)\cdot
\nabla_\theta f(x';\theta)$ — the inner product of parameter-gradients at two inputs; "how
similarly the network's gradients respond to these inputs," i.e. how strongly learning at
one point couples to the other. For a two-layer ReLU net $f = m^{-1/2}\sum_j a_j\,
\sigma(w_j^\top x)$: $\partial f/\partial a_j = m^{-1/2}\sigma(w_j^\top x)$,
$\partial f/\partial w_j = m^{-1/2} a_j\,\sigma'(w_j^\top x)\,x$.

**Infinite width.** The kernel converges (law of large numbers over the init) to a
deterministic, *time-invariant* limit computable in closed form from architecture,
activation, and init; training becomes kernel regression with that kernel (Jacot et al.
2018, arXiv 1806.07572 — the one identifier-bearing citation in the response).

**Empirical NTK.** The finite-width kernel at the current parameters. Unlike the limit it
*evolves* during training; that evolution "captures the feature learning capability that
distinguishes finite networks from their infinite-width approximations" and is "why finite
networks often outperform their infinite-width counterparts."

**Training dynamics.** Under gradient flow on squared loss,
$\dot f(x) = -\Theta(x,X)\,(f(X)-y)$: the NTK is the propagator from training residuals to
any query point. With eigenpairs $(\lambda_i, v_i)$ of $\Theta(X,X)$ the error component
along $v_i$ decays as $e^{-\lambda_i t}$ — large-eigenvalue modes (smooth, global
patterns) are learned first; small-eigenvalue modes (complex, localized) last. The
condition number $\lambda_{\max}/\lambda_{\min}$ is offered as a predictor of training
difficulty; "well-conditioned … with sufficiently large minimum eigenvalues" as a design
heuristic.

**Computation.** $N\times N$ Gram matrix from per-example Jacobians ($N$ points, $P$
params); avoid materializing the Jacobian via JVP/VJP contractions. Tooling named:
`neural_tangents.empirical_ntk_fn` (JAX), the PyTorch `torch.func` NTK tutorial. Cost
"quadratic in data points and linear in parameters."

**Limitations named.** The infinite-width limit misses hierarchical feature learning;
extensions (Tensor Programs II for arbitrary architectures; "beyond NTK" work) address
finite-width feature learning.

**The app.** Seven sections: intuitive intro; definition walk-through; visualization of
loss, function shape, and spectrum on real training data; an optimization-trajectory
explorer with eigenvalue sliders; intuition walkthroughs; practical applications; future
directions. Live at the Perplexity link above.

## Intake notes

- The text is a correct textbook summary at the level of the standard tutorials it cites
  (Lilian Weng 2022, inference.vc, RBC Borealis, the PyTorch tutorial). Citations are a mix
  of blogs, slides, and a few papers; apart from Jacot et al. only arXiv 2104.03093,
  2305.14585, 2406.18800, 2502.02870 carry identifiers and none were checked. Nothing
  here should be cited without going to the primary source.
- What it omits that matters for using the eNTK *as a measurement*: (i) the eNTK's
  early-training motion is large and then slows — Fort et al. 2020, "Deep learning versus
  kernel learning" (arXiv 2010.15110; unverified) report that the kernel changes most in the
  first few epochs and that the network becomes close to linear-in-parameters afterwards,
  which is exactly the kind of "early dynamics" signal EDP and the critical-period line
  (`critical-periods.md`) care about; (ii) for multi-output nets the NTK is
  $NC\times NC$ (or trace-reduced), and the "$N\times N$" description is the
  scalar-output case; (iii) sampling — the full Gram matrix on tens of thousands of
  examples is not needed; a few hundred sampled points give the spectrum shape and kernel
  alignment; (iv) the NTK–target alignment (e.g. centered kernel alignment of $\Theta$ with
  $yy^\top$) is the quantity most often used as a trainability / generalization readout,
  and it is absent from the overview.
- Candidate readouts for the ladder / GEO, if the eNTK is adopted: top-$k$ spectrum and
  effective rank at checkpoints; kernel velocity $\|\Theta_t-\Theta_{t-1}\|_F/\|\Theta_t\|_F$;
  kernel–target alignment; all on a fixed probe set of a few hundred examples. These are
  cheap relative to the runs the ladder already plans.
