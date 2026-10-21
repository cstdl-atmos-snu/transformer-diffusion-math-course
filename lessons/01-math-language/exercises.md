# Lesson 1 — Exercises and solutions

Try each problem before reading its solution.

## Problems

### 1. Shape audit

Let (X\in\mathbb R^{B\times n\times d}\), (W_Q,W_K\in\mathbb R^{d\times d_k}\), (W_V\in\mathbb R^{d\times d_v}\). Give the shapes of (Q,K,V,QK^T,A,AV) when attention is applied independently to each batch member. State the axis of softmax.

### 2. Softmax properties

Prove shift invariance. Derive \(\partial p_i/\partial z_j\). Show why the Jacobian annihilates \(\mathbf1\).

### 3. Why scale logits?

Assume independent coordinates with \(E[q_r]=E[k_r]=0\), \(E[q_r^2]=E[k_r^2]=1\), and independent (q_r,k_r). Compute \(E[q^Tk]\) and \(\mathrm{Var}(q^Tk)\). Explain the effect of dividing by \(\sqrt{d_k}\), and name one limitation of this argument.

### 4. Two-step diffusion marginal

For (x_1=\sqrt{\alpha_1}x_0+\sqrt{1-\alpha_1}\epsilon_1) and (x_2=\sqrt{\alpha_2}x_1+\sqrt{1-\alpha_2}\epsilon_2), derive (q(x_2|x_0)).

### 5. Reverse conditional versus marginal

Explain why the result (q(x_t|x_0)) does not by itself give (q(x_{t-1}|x_t)). What additional random variable or distribution is needed to compute that reverse conditional?

### 6. Numerical check

Design a test that compares samples from the recursive chain and the direct marginal. Which statistics should agree, and why should individual sample vectors not match?

## Solutions

### 1. Shape audit

For each batch member: (Q,K:B\times n\times d_k), (V:B\times n\times d_v), (QK^T:B\times n\times n), (A:B\times n\times n), and (AV:B\times n\times d_v). Softmax is over the final key-position axis, so each query row sums to one.

### 2. Softmax properties

\(p_i(z+c\mathbf1)=e^{z_i+c}/\sum_j e^{z_j+c}=p_i(z)\). Differentiating the quotient gives (\partial p_i/\partial z_j=p_i\delta_{ij}-p_ip_j=p_i(\delta_{ij}-p_j)\). Thus (J=\mathrm{diag}(p)-pp^T\), and (J\mathbf1=p-p(p^T\mathbf1)=p-p=0\), since probabilities sum to one.

### 3. Why scale logits?

By independence and zero means, (E[q_rk_r]=0), so (E[q^Tk]=0). The products are independent across coordinates and (E[(q_rk_r)^2]=E[q_r^2]E[k_r^2]=1), hence variance sums to (d_k). Scaling by (1/\sqrt{d_k}) gives variance one. The argument assumes independent, unit-variance coordinates; learned coordinates may be correlated or have other variances, and it describes initialization behavior rather than guaranteeing a trained distribution.

### 4. Two-step diffusion marginal

Substitute the first equation into the second:
\[x_2=\sqrt{\alpha_1\alpha_2}x_0+\sqrt{\alpha_2(1-\alpha_1)}\epsilon_1+\sqrt{1-\alpha_2}\epsilon_2.\]
Independent standard Gaussians combine into a Gaussian with variance (\alpha_2(1-\alpha_1)+(1-\alpha_2)=1-\alpha_1\alpha_2\). Therefore (x_2|x_0\sim N(\sqrt{\alpha_1\alpha_2}x_0,(1-\alpha_1\alpha_2)I)\).

### 5. Reverse conditional versus marginal

The forward marginal only links (x_t) to (x_0), having integrated out intermediate variables. The reverse conditional needs the joint distribution of adjacent states, which depends on the data prior (q(x_0)) after marginalizing over (x_0). In contrast, (q(x_{t-1}|x_t,x_0)) is tractable because the forward chain and (x_0) are both specified; this posterior is used in DDPM's variational derivation.

### 6. Numerical check

Fix (x_0), schedule, and (t). Draw many independent chains recursively and many independent \(\epsilon\sim N(0,I)\) direct samples. Compare empirical means to \(\sqrt{\bar\alpha_t}x_0\), empirical covariance to \((1-\bar\alpha_t)I\), and optionally a one-dimensional histogram/QQ plot. The two procedures use different noise draws, so samplewise equality is neither expected nor required; their conditional distributions should agree.
