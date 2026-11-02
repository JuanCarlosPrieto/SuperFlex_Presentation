# SuperFlex paper-presentation package

A focused 20-minute Geometric Deep Learning presentation about **SuperFlex: Deformable Superquadrics for Point Cloud Decomposition** (ECCV 2026), prepared from the user-supplied paper and photographed study notes.

## Deliverables

- `SuperFlex_talk.pdf`: compiled **30-slide** 16:9 presentation (21 main-talk slides; 9 backup/Q&A slides).
- `SuperFlex_talk.tex`: fully editable, self-contained **Beamer LaTeX source** (with relative links to `assets/`).
- `SuperFlex_geometry_notebook.ipynb`: executed CPU-only Python notebook with geometry illustrations, code, and embedded figures.
- `Speaker_notes_and_QA.md`: slide-by-slide ~20-minute pacing, detailed technical explanations and anticipated questions.
- `assets/`: original generated diagrams and figures cropped from the user-provided paper.

## Compile

From this folder:

```sh
pdflatex -interaction=nonstopmode SuperFlex_talk.tex
pdflatex -interaction=nonstopmode SuperFlex_talk.tex
```

Requires a standard LaTeX installation with Beamer, TikZ and AMS math. Tested with `pdflatex` in the current container.

## Run geometry demonstrations

Install `numpy`, `scipy`, `matplotlib`, `nbformat`, `jupyter`/`ipykernel` if needed. Open `SuperFlex_geometry_notebook.ipynb` with Jupyter and run all cells from **this folder**. The notebook recreates `assets/demo_*.png`, which are embedded in the slides. It does **not** run the trained network.

## The official neural-network implementation

The authors provide source code and pretrained checkpoints at <https://github.com/GabrielTavernini/superflex>. Their README describes the `demo_viser.py` example and checkpoint download script; training is far more resource intensive. The published scores in this presentation come solely from the paper.

## Academic credits

Paper: Gabriel Tavernini, Elisabetta Fedele, Tiago Novello, Leonidas Guibas, Marc Pollefeys and Francis Engelmann (ECCV 2026), <https://arxiv.org/abs/2607.01015>. Project: <https://superflex3d.github.io/>. Paper figures are reproduced for scholarly discussion and identified as such. Original visualizations are attributed to the companion notebook.
