# Final Project Report

CS8803 Deep Reinforcement Learning final report (Spring 2026).

**Authors:** Bo Feng (bfeng66@gatech.edu), Frank Yang (frank.yang@gatech.edu)

**Compiled PDF:** [`CS8803_DRL_Final_Report.pdf`](CS8803_DRL_Final_Report.pdf)

## Contents

| File | Description |
|------|-------------|
| `CS8803_DRL_Final_Report.pdf` | Compiled report (5 pages: 4 content + 1 references) |
| `report.tex` | LaTeX source |
| `references.bib` | BibTeX bibliography (8 entries) |
| `corl_2026.sty`, `corlabbrvnat.bst` | CoRL 2026 template files |
| `figures/training_curves.pdf` | Figure 1: reward vs. environment steps |
| `figures/make_training_curves.py` | Script that generates Figure 1 from `ray_results/*/progress.csv` |
| `eval_vs_random.log` | Raw log of the 20-match vs. Random evaluation (20W 0L 0D = 100%) |

## Reproducing

### Compile the PDF

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

### Re-generate the training-curves figure

Requires the local `ray_results/PPO_SP/...` and `ray_results/PPO_reward_shaped/...`
trial directories from training (each contains a `progress.csv`):

```bash
python figures/make_training_curves.py
```

### Re-run the vs.\ Random evaluation

From the repo root:

```bash
conda run -n soccertwos14 python scripts/eval_vs_random.py 20
```

The vs.\ CEIA evaluation uses `scripts/eval_vs_ceia.py` and was performed on PACE
(see `notes/training-log.md` for the full per-run history).
