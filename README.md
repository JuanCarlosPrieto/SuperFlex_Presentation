# SuperFlex paper-presentation package

A focused 20-minute Geometric Deep Learning presentation about **SuperFlex: Deformable Superquadrics for Point Cloud Decomposition** (ECCV 2026).

## Compile

From this folder:

```sh
pdflatex -interaction=nonstopmode SuperFlex_talk.tex
pdflatex -interaction=nonstopmode SuperFlex_talk.tex
```

Requires a standard LaTeX installation with Beamer, TikZ and AMS math. Tested with `pdflatex` in the current container.

## Run geometry demonstrations

Install `numpy`, `scipy`, `matplotlib`, `nbformat`, `jupyter`/`ipykernel` if needed. Open `SuperFlex_geometry_notebook.ipynb` with Jupyter and run all cells from this folder. The notebook recreates `assets/demo_*.png`, which are embedded in the slides. It does not run the trained network.

## Academic credits

Paper: Gabriel Tavernini, Elisabetta Fedele, Tiago Novello, Leonidas Guibas, Marc Pollefeys and Francis Engelmann (ECCV 2026), <https://arxiv.org/abs/2607.01015>. Project: <https://superflex3d.github.io/>.
