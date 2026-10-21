# Lesson 1 — Primary-paper map

| Course item | Paper location | Relationship |
|---|---|---|
| Scaled dot-product attention, Eq. (1.1) | Vaswani et al. (2017), §3.2.1, Eq. 1 | Same operation and scaling; this course adds a shape audit and softmax properties. |
| Variance motivation for (1/\sqrt{d_k}) | Vaswani et al. (2017), §3.2.1, paragraph following Eq. 1 | The course reproduces the independent unit-variance argument and states its assumptions. |
| Forward transition, Eq. (1.2) | Ho et al. (2020), §2.1, Eq. 1, with \(\alpha_t=1-\beta_t\) | Equivalent parameterization. |
| Closed-form marginal, Eqs. (1.4)–(1.5) | Ho et al. (2020), §2.1, Eq. 4 | Course notes show the induction and covariance addition explicitly. |

## Original papers

- Ashish Vaswani et al. (2017), “Attention Is All You Need,” arXiv:1706.03762. [Paper](https://arxiv.org/abs/1706.03762) · [PDF](https://arxiv.org/pdf/1706.03762)
- Jonathan Ho, Ajay Jain, Pieter Abbeel (2020), “Denoising Diffusion Probabilistic Models,” arXiv:2006.11239. [Paper](https://arxiv.org/abs/2006.11239) · [PDF](https://arxiv.org/pdf/2006.11239)

## Reading instruction

Read Vaswani §3.2.1 and Ho §2.1 immediately after the notes. Compare notation carefully: this course uses (\alpha_t=1-\beta_t), and (\bar\alpha_t=\prod_{s=1}^{t}\alpha_s\). The course equations are numbered locally and are not the papers' equation numbers.
