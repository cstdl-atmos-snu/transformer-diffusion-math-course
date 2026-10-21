# Lesson 1 — Mathematical language for attention and diffusion

## Learning goals

By the end, you should be able to (i) track dimensions in a batched attention calculation, (ii) explain why softmax is invariant to a common logit shift, (iii) derive the exact marginal of a linear Gaussian diffusion chain, and (iv) distinguish an exact identity from a modeling approximation.

## 1. Notation and shape discipline

Vectors are columns unless explicitly written as row batches. A sequence batch is usually stored as a matrix (X\in\mathbb R^{n\times d}), with tokens in rows. A linear projection is therefore (Q=XW_Q), where (W_Q\in\mathbb R^{d\times d_k}). If (Q,K\in\mathbb R^{n\times d_k}), then (QK^T\in\mathbb R^{n\times n}): each entry is one query–key compatibility score. Multiplication by (V\in\mathbb R^{n\times d_v}) after row-wise normalization returns (n\times d_v).

The notation (\mathcal N(\mu,\Sigma)) denotes a Gaussian with mean vector (\mu) and covariance matrix (\Sigma). In the scalar isotropic case, (\mathcal N(0,I)) has independent unit-variance components. (I) is always the identity matrix of the matching dimension.

## 2. Softmax as a normalized exponential map

For (z\in\mathbb R^m),
\[
\mathrm{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]
The entries are positive and sum to one. For any scalar (c),
\[
\mathrm{softmax}(z+c\mathbf 1)_i
=\frac{e^{z_i+c}}{\sum_j e^{z_j+c}}
=\frac{e^c e^{z_i}}{e^c\sum_j e^{z_j}}
=\mathrm{softmax}(z)_i.
\]
Thus only relative logits matter. Numerically, subtracting \(\max_i z_i\) before exponentiation changes no probabilities and avoids overflow.

The Jacobian is
\[
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j),\qquad p=\mathrm{softmax}(z).
\]
In matrix form, (J=\operatorname{diag}(p)-pp^T). It is positive semidefinite and has null vector \(\mathbf 1\), reflecting shift invariance.

## 3. Scaled dot-product attention as weighted averaging

Given one query (q\in\mathbb R^{d_k}), keys (k_j\in\mathbb R^{d_k}), and values (v_j\in\mathbb R^{d_v}), define
\[
\ell_j=\frac{q^T k_j}{\sqrt{d_k}},\qquad
a_j=\frac{e^{\ell_j}}{\sum_r e^{\ell_r}},\qquad
o=\sum_j a_jv_j.
\]
The output is a convex combination of value vectors. For all queries at once, (Q,K\in\mathbb R^{n\times d_k}), (V\in\mathbb R^{n\times d_v}):
\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}_{\mathrm{row}}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
\tag{1.1}
\]
The (1/\sqrt{d_k}) scale prevents dot-product variance from growing with dimension under a simple independent, zero-mean, unit-variance model. If (q_r,k_r\) are independent with unit variance, then (q^Tk=\sum_{r=1}^{d_k}q_rk_r) has variance (d_k). Dividing by \(\sqrt{d_k}\) gives order-one variance. This is a motivation under assumptions, not a universal theorem about trained activations.

### A useful interpretation

Row-wise softmax turns each query into weights over all keys. Attention is therefore data-dependent averaging: the values are mixed according to similarities computed from learned query/key projections. It resembles kernel smoothing, but the similarity function and feature maps are learned, and ordinary softmax dot-product attention is not generally a symmetric positive-definite kernel.

## 4. Linear Gaussian transitions and a closed-form marginal

Consider a discrete noising chain
\[
q(x_t\mid x_{t-1})=\mathcal N\!\left(\sqrt{\alpha_t}x_{t-1},\,(1-\alpha_t)I\right),\quad 0<\alpha_t<1,
\tag{1.2}
\]
or equivalently sample it as
\[
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\,\epsilon_t,
\quad \epsilon_t\sim\mathcal N(0,I).
\tag{1.3}
\]
Let \(\bar\alpha_t=\prod_{s=1}^t\alpha_s\). We claim
\[
q(x_t\mid x_0)=\mathcal N\!\left(\sqrt{\bar\alpha_t}x_0,\,(1-\bar\alpha_t)I\right),
\tag{1.4}
\]
so one can sample any noise level directly:
\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\quad\epsilon\sim\mathcal N(0,I).
\tag{1.5}
\]

### Derivation by induction

For (t=1), \,\(\bar\alpha_1=\alpha_1\), and (1.4) is exactly (1.2) conditioned on (x_0). Assume at (t-1),
\[
x_{t-1}\mid x_0\sim\mathcal N\!\left(\sqrt{\bar\alpha_{t-1}}x_0,(1-\bar\alpha_{t-1})I\right).
\]
Write (x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0+\sqrt{1-\bar\alpha_{t-1}}\,\tilde\epsilon), where \(\tilde\epsilon\sim\mathcal N(0,I)\) is independent of \(\epsilon_t\). Substitution into (1.3) gives
\[
x_t=\sqrt{\alpha_t\bar\alpha_{t-1}}x_0
+\sqrt{\alpha_t(1-\bar\alpha_{t-1})}\,\tilde\epsilon
+\sqrt{1-\alpha_t}\,\epsilon_t.
\]
The mean coefficient is \(\sqrt{\bar\alpha_t}\). The two independent Gaussian noise terms have total covariance
\[
\left[\alpha_t(1-\bar\alpha_{t-1})+(1-\alpha_t)\right]I
=(1-\alpha_t\bar\alpha_{t-1})I
=(1-\bar\alpha_t)I.
\]
This proves (1.4). The independence assumption matters: without it, covariances would include cross terms.

## 5. Reparameterization and gradients

The direct sample (1.5) expresses a random variable as a deterministic differentiable function of (x_0,\bar\alpha_t) and exogenous noise \(\epsilon\). For a differentiable loss \(L(x_t)\),
\[
\nabla_{x_0}L(x_t)=\sqrt{\bar\alpha_t}\,\nabla_{x_t}L(x_t),
\]
holding sampled \(\epsilon\) fixed. This is the pathwise/reparameterization view of gradients. It allows stochastic sampling in the forward pass while propagating gradients through the deterministic transformation. In DDPM training, the forward process is fixed and gradients are taken through the neural predictor, not through a learned forward schedule in the original formulation.

## 6. What to carry forward

- Shape-check every product before interpreting it.
- Softmax probabilities depend on logit differences, not the common offset.
- Attention returns normalized, query-dependent mixtures of values.
- Independent Gaussian noise terms add their covariances; scalar variances add only in the isotropic case.
- The diffusion marginal (1.4) is exact given the linear Gaussian transition (1.2), not an approximation.

## Paper mapping

See [sources.md](sources.md) for exact paper locations. Equation (1.1) is Vaswani et al.'s Eq. 1; the diffusion transition and closed-form marginal correspond to Ho et al.'s Eqs. 1 and 4 (with \(\alpha_t=1-\beta_t\)). The induction above expands the paper's statement that the marginal has a closed form.
