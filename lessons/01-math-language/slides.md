# Lesson 1 — Math language for attention and diffusion

## Slide 1 — The two derivation targets

- Transformer: learned similarity weights mix value vectors.
- DDPM: repeated Gaussian perturbations collapse to one Gaussian marginal.
- Skill for both: track shapes, means, covariances, and assumptions.

## Slide 2 — Shapes before symbols

- (X:n\times d\), (W_Q:d\times d_k\), (Q:n\times d_k\)
- (QK^T:n\times n\): one score per query–key pair
- Row-softmax (A:n\times n\); (AV:n\times d_v\)

## Slide 3 — Softmax

\[p_i=e^{z_i}/\sum_j e^{z_j}\]

- (p_i>0,\sum_i p_i=1)
- (\mathrm{softmax}(z+c\mathbf1)=\mathrm{softmax}(z))
- Jacobian: (\operatorname{diag}(p)-pp^T)

## Slide 4 — Attention

\[\mathrm{Attention}(Q,K,V)=\mathrm{softmax}_{row}(QK^T/\sqrt{d_k})V\]

- Similarity is query–key dot product.
- Normalized weights form a convex mixture of values.
- Scale keeps initial logit variance order one under an i.i.d. model.

## Slide 5 — One diffusion step

\[x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t,\quad\epsilon_t\sim N(0,I)\]

- Conditional mean contracts toward zero.
- Independent Gaussian perturbation increases uncertainty.
- (\bar\alpha_t=\prod_{s\le t}\alpha_s\)

## Slide 6 — Closed-form marginal

\[q(x_t|x_0)=N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)\]
\[x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon\]

- Mean coefficients multiply.
- Independent noise covariances add.
- Result follows by induction; it is exact.

## Slide 7 — Derive it, then code it

- Hand derive the (t-1\to t) mean and covariance.
- Numerically compare recursive and direct samples.
- Inspect attention rows: sums, entropy, and output convex hull.

## Slide 8 — Paper anchors

- Vaswani et al. (2017): §3.2.1, Eq. 1
- Ho et al. (2020): §2.1, Eqs. 1 and 4
- Next: derive scaled attention and masking, not just quote the equation.
