# Course syllabus

**Level:** graduate self-study; assumes multivariable calculus, linear algebra, probability, and basic Python. A learner who needs a refresher should treat Lesson 1 as a diagnostic and fill gaps before continuing.

**Outcome:** independently derive the scaled dot-product attention equation, the DDPM forward marginal and variational objective, the noise-prediction loss, and the reverse-time score SDE / probability-flow ODE, while understanding assumptions, dimensions, and connections.

| # | Lesson | Derivation target | Primary-paper anchor |
|---|---|---|---|
| 1 | Mathematical language: tensors, softmax, Gaussians, and gradients | Shapes and invariances of attention; closed-form marginal of a linear Gaussian diffusion chain | Vaswani §3.2.1 Eq. 1; Ho §2 Eqs. 1–7 |
| 2 | Attention as content-based kernel smoothing | Derive (\mathrm{softmax}(QK^T/\sqrt{d_k})V\); explain scale and masking | Vaswani §3.2.1 Eqs. 1–3 |
| 3 | Multi-head attention, positional information, and the Transformer block | From head projections to concatenation; sinusoidal position encoding and residual block | Vaswani §§3.2–3.5, Eqs. 1–7 |
| 4 | Sequence learning objective and optimization | Teacher forcing, token cross-entropy, label smoothing, and the learning-rate schedule | Vaswani §§5.3–5.4, Eqs. 8–9 |
| 5 | Latent-variable models and variational inference | ELBO from Jensen's inequality; KL decomposition for a latent Markov chain | Ho §2, Eq. 3 and Appendix A |
| 6 | Gaussian identities for diffusion | Product of Gaussian transitions, posterior conditioning, and reparameterization | Ho §2, Eqs. 4–7; Appendix A |
| 7 | DDPM forward process and reverse parameterization | Derive (q(x_t\mid x_0)), (q(x_{t-1}\mid x_t,x_0)), and learned reverse chain | Ho §§2.1–2.2, Eqs. 4–11 |
| 8 | Variational bound to the noise-prediction loss | Decompose the ELBO; substitute the noise parameterization; identify weighting choices | Ho §3, Eqs. 12–16 |
| 9 | Denoising score matching and Langevin dynamics | Gaussian denoising target equals a scaled score; connect score estimation to sampling | Ho §3.2; Song & Ermon (2019) §3 |
| 10 | Continuous-time diffusion: SDE, reverse SDE, and probability-flow ODE | Derive reverse-time drift and the marginally equivalent ODE | Song et al. (2021) §§3.1–3.3, Eqs. 6–14 |
| 11 | Discretizations, samplers, and prediction parameterizations | Relate epsilon, score, and clean-sample predictions; analyze discretization error | Ho §4; Song et al. §§4–5 |
| 12 | Conditional generation and guidance | Derive conditional score identities and classifier guidance; separate exact identities from approximations | Dhariwal & Nichol (2021) §3; Ho & Salimans (2022) §2 |
| 13 | Transformer–diffusion connections and research extensions | Compare attention as normalized kernel averaging with denoising as conditional estimation; identify limits of the analogy | Vaswani §3; Ho §§2–3; selected primary papers |
| 14 | Capstone: reconstruct the original derivations | Re-derive and explain selected equations from paper assumptions to implementation | Vaswani §§3, 5; Ho §§2–3; Song et al. §3 |

## Assessment checkpoints

1. **After Lesson 4:** derive attention from tensor projections and implement a masked causal head.
2. **After Lesson 8:** derive the DDPM simplified objective, naming every dropped or fixed weight.
3. **After Lesson 11:** show how a discrete Gaussian schedule approximates a continuous SDE and compare samplers.
4. **Capstone:** write a paper-style derivation with assumptions, dimensions, limiting cases, and a numerical check.

## Recommended pacing

One lesson per week gives time for hand derivations and code experiments; Lessons 6–10 may need more than one week. No lesson should be considered complete solely because the notebook runs: the main criterion is being able to reconstruct its derivations unaided.
