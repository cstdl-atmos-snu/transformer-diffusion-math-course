# Mathematical Foundations of Transformers and Diffusion Models

Graduate self-study course for deriving the central equations in the original Transformer and DDPM papers, then connecting discrete diffusion to score-based continuous-time models.

## How to study

Work through lessons in order. For each lesson: read the mathematical notes, annotate every derivation without looking, review the short slides, solve the exercises before opening solutions, then run and modify the notebook. Keep a derivation log with assumptions, dimensions, and places where an equality uses a probabilistic identity.

## Lesson package

Each lesson is intended to contain `notes.md`, `slides.md`, `exercises.md` (with solutions), and a runnable Jupyter notebook. `sources.md` maps claims to exact sections/equations in primary papers. When a later lesson reveals a clearer connection, update the earlier explanation and record the change in `CHANGELOG.md`.

## Curriculum

See [SYLLABUS.md](SYLLABUS.md). Lessons 1 and 2 are scaffolded here; Lesson 1 is complete. The remaining lessons are queued in order and will use the same package contract.

## Environment

Python 3.10+, NumPy, Matplotlib, and Jupyter. Install with `python -m pip install -r requirements.txt`, then run `jupyter lab` from this directory.

## Primary-paper links

- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762), original paper.
- Ho, Jain & Abbeel (2020), [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), original paper.
- Song et al. (2021), [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), continuous-time synthesis.

## Versioning convention

Use one commit per coherent lesson or correction. Keep equations in LaTeX, state tensor shapes next to equations, and pin derivation conventions (row/column vectors, variance schedule indexing) so lessons remain internally consistent.
