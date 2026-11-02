# SuperFlex — 20-minute speaker guide and 10-minute Q&A

**Talk language:** English. **Paper:** Tavernini et al., *SuperFlex: Deformable Superquadrics for Point Cloud Decomposition*, ECCV 2026. **Format:** slides 1–21 are the prepared talk; slides 22–30 are backup material and should not be presented unless needed during questions. **Target talk time:** 19:15–19:45 including transitions (approximately 20 minutes). Practice once while advancing the PDF and aim to finish slide 15 around minute 13.

The slides are **not a reproduction of the entire article**. Their purpose is to explain the geometry, data flow, distinct objectives, and why each component is needed. The published tables are clearly attributed; visual mathematics is recreated by the companion CPU notebook. The official implementation is separate.

## Run of show (main slides)

| # | Topic | Target duration | Running target |
|---|---|---:|---:|
| 1 | Title and claim | 0:20 | 0:20 |
| 2 | Shape abstraction task | 0:45 | 1:05 |
| 3 | Prior methods and SuperDec inheritance | 0:55 | 2:00 |
| 4 | Canonical implicit superellipsoid | 1:10 | 3:10 |
| 5 | Failure of rigid shape family | 0:40 | 3:50 |
| 6 | Tapering and bending | 1:05 | 4:55 |
| 7 | 19-parameter inverse pipeline | 0:55 | 5:50 |
| 8 | Radial distance and caveat | 1:15 | 7:05 |
| 9 | Actual encoder/decoder architecture | 1:15 | 8:20 |
| 10 | Chamfer problem | 0:45 | 9:05 |
| 11 | Differentiable volumetric IoU | 1:20 | 10:25 |
| 12 | Surface loss and robust kernel | 1:10 | 11:35 |
| 13 | Regularization | 0:55 | 12:30 |
| 14 | Test-time optimization | 1:00 | 13:30 |
| 15 | Completion under partial observations | 1:10 | 14:40 |
| 16 | Protocol | 0:40 | 15:20 |
| 17 | Main ShapeNet results | 0:55 | 16:15 |
| 18 | Loss ablations | 1:15 | 17:30 |
| 19 | Deformation and optimization gains | 0:45 | 18:15 |
| 20 | Robust partial-cloud gains | 0:50 | 19:05 |
| 21 | Takeaways | 0:45 | 19:50 |

### Slide 1 — Thesis

"SuperFlex reconstructs 3D objects as small sets of deformable superquadrics. I am going to concentrate on *why the representation and learning objective work*, rather than just presenting a benchmark ranking." The three qualitative columns from the original teaser correspond to reconstruction fidelity, expressiveness, and robustness.

### Slide 2 — The decomposition task

The input is an unordered set of 3D coordinates. The output is not a mesh and not a dense signed-distance volume. It is a **set of solids**, each compactly parameterized. The decomposition should be faithful and interpretable, but it should also avoid unnecessary pieces. Geometric parts need not align perfectly with semantic labels: this is a geometric abstraction rather than semantic part parsing.

### Slide 3 — What came before

Optimization methods fit primitives individually or iteratively. EMS explicitly handles noise/outliers; Marching Primitives exploits distance fields but incurs heavy per-object cost and needs sufficiently good SDF information. Neural methods amortize fitting across objects and make inference much faster. SuperDec is the closest architectural ancestor: point encoder plus Transformer queries for part parameters. Stress that **the paper does not claim to invent Transformer-based set prediction**. The *changes* are the new composite geometry, occupancy/surface losses, inference-time refinement, and robustness training.

### Slide 4 — Where the shape parameters live

Point to the canonical equation; $q=1$ is the surface and (in the valid interior) $q<1$ is the solid. The absolute values are mathematically needed for fractional powers of negative coordinates. For $\epsilon_1=\epsilon_2=1$, the formula is an ellipsoid. Decreasing $\epsilon_2$ makes the horizontal section squarer; decreasing $\epsilon_1$ makes the vertical longitudinal profile squarer. The scales $s_x,s_y,s_z$ change size without requiring a mesh with more vertices. **Avoid claiming all superquadrics are represented here**: this is the paper's superellipsoid family, which is the relevant branch for the method.

### Slide 5 — Why deformations help

A chair or arched structure may be represented with many symmetric pieces, even if a human recognizes one smoothly varying geometric part. Rigid primitives cannot bend a straight centerline or taper progressively. This creates an *expressivity-versus-compactness* trade-off. The next slide changes the primitive family itself rather than simply increasing the number of queries.

### Slide 6 — Tapering versus bending

Tapering changes the cross-section at height $z$ using a **linear scale factor**. The two coefficients $\tau_x,\tau_y$ can widen in one horizontal direction while narrowing in the other. Bending takes a straight primary axis into a circular arc. The curvature $k$ fixes the radius $1/|k|$; $\alpha$ determines **which azimuthal plane the arc occupies**. Even for equal curvature, an arc may bend toward $x$, toward $y$, or a diagonal. The full paper learns three bending-axis pairs, not one. The notebook illustrates the $z$-axis case only; do not represent it as a full inference engine.

### Slide 7 — The composition and counting

Count the 19 degrees of freedom aloud: translation 3, rotation 3, scale 3, shape exponents 2, taper 2, bending 3×2 = 6. When evaluating the implicit field at a real-world point, **undo pose**, then undo bending in the prescribed reverse order, then undo tapering, and evaluate the canonical $q_{can}$. The order matters because these transformations generally do not commute. The rotation is represented mathematically by a 3×3 matrix, but has only three independent degrees of freedom.

### Slide 8 — Radial distance, properly interpreted

For an undeformed superellipsoid, homogeneity gives $q(\lambda x)=\lambda^{2/\epsilon_1}q(x)$. The ray scale satisfying $q(\lambda_*x)=1$ is $q(x)^{-\epsilon_1/2}$. Multiply displacement along the center ray by its norm; the sign convention is positive outside and negative inside. The shortest Euclidean distance normally follows a different line. The paper calls a training term "SDF" but Eq. (3) evaluates a radial proxy, which is less expensive. **Precise caveat:** deforming the shape breaks the homogeneity used in this derivation; the same closed-form expression on the deformed implicit field is therefore not mathematically guaranteed to equal the exact deformed ray-intersection distance. The notebook demonstrates a specific counterexample numerically. You can leave this subtle point for questions if time is tight.

### Slide 9 — Network data flow

The encoder is PVCNN, transforming $N\times3$ coordinates to $N\times D$ point features. Initialize $P$ superquadric tokens via sinusoidal positional encodings; unlike the $N$ points, $P$ is the **maximum primitive-query budget**. Each of the $L$ Transformer-decoder blocks refines primitive queries through self-attention among queries and cross-attention with point features. Heads output the 19 geometric parameters $\Theta_p$, an existence probability $e_p\in[0,1]$, and per-point soft assignments $W\in\mathbb R^{N\times P}$. The softmax runs across primitive candidates for each input point; each point's assignment weights sum to one. In the experiments $N=4096,P=16,D=128,L=3$. Do not confuse $P$ predictions with 16 guaranteed active primitives: many are suppressed.

### Slide 10 — What Chamfer cannot ensure

Chamfer compares samples on the *input surface* and *predicted primitive surfaces*. It can provide sensible initial agreement, but it does not directly measure whether the solid union is filled properly. The paper specifically argues that primarily Chamfer-trained models can generate rounded profiles and gaps between primitives. This is the motivation for **two complementary supervision signals**, not a claim that Chamfer is always unusable: the robust completion stage later uses Chamfer for *matched primitive shapes*.

### Slide 11 — Differentiable IoU, line by line

First, $d_p$ becomes an occupancy probability through a temperature-controlled sigmoid. The smaller $T_{IoU}$, the sharper the transition. Second, the probability that none of the existing primitives occupies a location is $\prod_p(1-e_p I_p)$; its complement is the probabilistic union. For two always-existing parts, expand the product to obtain $I_1+I_2-I_1I_2$. Third, evaluate a soft intersection numerator and union denominator against ground-truth occupancy over a uniformly sampled 3D set; the paper takes negative log soft IoU. **Do not equate occupancy samples with point-cloud samples**: the former are queries in 3D space, and require ground-truth occupancy (from complete object geometry) during training.

### Slide 12 — Surface geometry and robust penalty

The volumetric term alone may overfill or overlap volumes. The second term acts on observed surface points and weights each candidate by the model's soft assignment $w_{ip}$. The bounded $\tanh$ penalty limits the effect of large errors. It is asymmetric: with $\omega=0.1$, negative distances (points inside a predicted primitive) receive smaller penalties than positive distances (points outside). State that “SDF loss” is the paper's name, but the numerical distance input is radial rather than a true Euclidean signed distance. This is the key to reconciling the formal name and the derivation.

### Slide 13 — Compactness losses

The complete training objective is a weighted sum of volumetric loss, surface loss, and three regularizers. Overlap discourages two candidate primitives from explaining the same volume; sparsity discourages use of an excessive number of parts; existence aligns $e_p$ with how heavily the network assigns points to part $p$. The last two mechanisms come from SuperDec. If challenged: the published overlap loss uses $\sum_p I_p$ and is not the same as geometric hard-overlap volume.

### Slide 14 — Refinement and latency

At inference, first threshold existence at 0.5 and retain active candidates. Apply 1000 Adam steps optimizing their geometric parameters. The SDF weighting no longer uses the network's segmentation matrix because the set is treated as a single global shape. The hard minimum of distances is replaced by a differentiable log-sum-exp soft minimum. The same global field also drives volumetric occupancy during refinement, rather than the initial per-query probabilistic OR. Report the cost plainly: IoU 0.72 to 0.87, but 0.0082 s feed-forward to 5 s/object refined. These timings cannot be assigned to the notebook geometry demos.

### Slide 15 — Structural completion requires pseudo-labels

Real point clouds can be incomplete even when every observed coordinate is accurate. Training only against visible points gives no direct reason to reconstruct unseen chair legs or the far side of a sofa. SuperFlex therefore builds high-quality *complete* primitive sets from full objects and treats them as pseudo-ground truth. A second network/fine-tuning regime sees occluded inputs and learns to output the same set. The geometric loss first matches predicted and target parts by Hungarian assignment, then computes surface-sampled Chamfer distances for matched **existing** objects. This handles permutation ambiguity and multiple possible equivalent parameterizations. An auxiliary undeformed-geometry loss encourages coarse shape recovery before emphasizing bends and tapering.

### Slide 16 — What is compared

The object-level benchmark uses 13 ShapeNet classes with canonical alignment and 4096 FPS-sampled points per shape. For real-world-style occlusions, ASE permits quantitative comparison because complete geometry is available; ScanNet++ supports qualitative scene visualizations. The network was trained on four NVIDIA GH200 GPUs in the paper. Explain that this is **training hardware**, not the documented minimal inference configuration. The user's CPU notebook generates the mathematical figures, not network predictions.

### Slide 17 — Main result: separate improvements

Compare SuperDec 0.59 to SuperFlex rigid 0.70: this largely isolates the new training supervision within the architecture lineage. Then compare 0.70 to deformable 0.72: added expressive degrees of freedom improve the final metric, albeit less dramatically. The deformable model averages 5.64 active primitives, with a similar ~8 ms reported feed-forward runtime. Marching Primitives uses 24.10 on average and ~163 seconds on the reported setup; these numbers highlight the speed versus optimization trade-off. Avoid saying all runtime values were measured on identical consumer hardware.

### Slide 18 — The ablation is the conceptual climax

The IoU-only version increases volumetric overlap to 32%, even though its IoU metric rises to 0.74. Adding the radial surface term reduces L2 Chamfer error markedly, 0.124 to 0.042, and overlap to 14%. The explicit overlap penalty further reduces that overlap to 8% at almost unchanged L2 and IoU. **Do not claim every loss monotonically improves every metric**: IoU drops from 0.74 to 0.72 with added surface alignment because the optimization trades pure volume agreement for local geometry and parsimony. The ablation is evidence that the full objective is balancing partially conflicting goals.

### Slide 19 — Two axes of additional refinement

The deformation ablation in Table 5 starts from the same undeformed predictions and tests distinct allowed deformation operators *during refinement*. It is not numerically the same setup as the main feed-forward Table 1. Taper alone and bending alone both help; together, 0.85 in that ablation. Table 2 measures the main optimized model, rising from 0.72 to 0.87. This is not contradictory: the procedures and initializations differ. The main presentation need only note the increasing model expressivity and the latency cost.

### Slide 20 — Incomplete observations

On ASE partial-cloud inputs, full-cloud-trained SuperDec and base SuperFlex both score approximately 0.20 IoU. Training SuperFlex with synthetic occlusions raises IoU to 0.48, and adaptation on ASE partial clouds raises it to 0.54. This shows the gain comes largely from a new *training signal*, not merely a better complete-shape loss. The paired ScanNet++ examples are qualitative—do not assign them quantitative scores from the ASE table.

### Slide 21 — Close

"Three aspects work together: a compact deformable primitive family, a reconstruction objective combining solid occupancy and surface precision, and pseudo-ground-truth training for structural completion." The main limits are approximate distance geometry, costly optional per-instance refinement, reliance on learned shape priors for occluded regions, and the unavoidable expressivity ceiling of a small primitive family. Finish on the broader geometric deep learning question: why should the latent representation be semantically and geometrically interpretable?

---

## Ten-minute Q&A: likely questions and precise answers

**1. Why are $\alpha$ and $k$ both necessary?**  Curvature $k$ specifies how strongly the centerline bends (arc radius $1/|k|$). Direction $\alpha$ specifies *where* in the cross-sectional plane it bends. The parameter pair lets bending direction change without reoptimizing global object rotation. Paper Table 6 reports IoU 0.83 without $\alpha$ versus 0.86 with it on a subset. See backup C.

**2. Isn't the “SDF loss” incorrectly named?**  The authors name the term $\mathcal L_{SDF}$ but define $d$ using their radial formula in Eq. (3). For rigid superellipsoids, the formula is exact as a signed **radial** distance, not exact Euclidean SDF. Once non-linear deformations are inserted in $q$, the homogeneous ray argument generally no longer holds, so it is best described as a radial-distance **surrogate**. The notebook numerically verifies this distinction. See backup B.

**3. Why does a 19-parameter vector include a rotation matrix, which has nine entries?**  The rotation lies in $SO(3)$, which is a three-dimensional manifold: three independent degrees of freedom. Its chosen computational representation may use a larger set of constrained coordinates.

**4. What's the point of the segmentation head if IoU already evaluates the union?**  It maps point-level evidence to particular primitive queries. The surface term needs differentiable weights $w_{ip}$ to assign each observed point's fitting error among the $P$ candidates. The global occupancy term, in contrast, sees the union and is agnostic to which part explains a particular location.

**5. Why does pure IoU produce more overlap?**  Many decompositions have similar global occupancy; two primitives can cover the same volume without substantially changing their union. IoU constrains the final union but does not inherently penalize redundant internal overlap. Explicit regularization removes this degeneracy.

**6. Is the method self-supervised or supervised?**  The first complete-cloud fitting stage uses geometry-based losses without annotated part decompositions, so the paper calls it self-supervised. The robust partial-cloud stage uses **supervised pseudo-target primitive sets**, produced from full geometry and refinement. These are different training regimes.

**7. Why Hungarian matching?**  The predicted primitives form an unordered set. Different orderings of the same parts should not incur arbitrary losses. Also, parameter-equivalent symmetries can make raw parameter MSE ambiguous. One-to-one matching followed by a surface-based loss emphasizes *geometric* agreement.

**8. Why does the robust stage use Chamfer after criticizing Chamfer?**  Chamfer is not inherently invalid. The concern is optimizing the entire shape purely through sampled-surface Chamfer. The robust stage has high-quality **individual primitive** pseudo-targets and uses Chamfer *after matching* corresponding surfaces, alongside a global IoU and rigid-shape prior.

**9. Why is test-time refinement significantly slower?**  A feed-forward network predicts all primitives in one pass. Refinement runs an iterative optimizer (1000 Adam iterations in the paper) on each object. The accuracy gain is substantial but optional. For real-time applications you would typically choose direct inference.

**10. Does SuperFlex have exact world-model or semantic part recovery?**  No. It predicts geometric primitives; colour-coded parts do not guarantee semantic labels or physically perfect segmentation. Its representational compactness is helpful for downstream reasoning, but semantic validity or physically correct completion is not guaranteed by the losses.

**11. Why is the bending order important?**  Rotation, tapering, and different bending operators generally do not commute. Undoing a deformation in the wrong order evaluates the wrong canonical point. The paper specifies inverse bending in the order $B_z^{-1}$, then $B_x^{-1}$, then $B_y^{-1}$, followed by $T^{-1}$, after inverse rigid pose. See backup A.

**12. Can the model be tested on a low-power laptop?**  The official public repository provides pretrained checkpoints and a `demo_viser.py`. The project indicates that the custom sampler build is needed for **training only**. Nonetheless, actual CPU-only inference compatibility and memory/time depend on the installation and were not validated in this package. The notebook's illustrations *are* lightweight CPU-only. You do not need to download the 73.4 GB ShapeNet dataset merely to use the authors' supplied demonstration examples.

**13. Why isn't the best IoU ablation also the best overall model?**  The IoU-only variant gets 0.74 but has worse L2 and 32% overlap. The authors target a *multiobjective* balance: volume fidelity, local surface fidelity, and few primitives. The full-loss method gets 0.72 with much better local error and lower redundancy.

**14. How is robustness actually evaluated?**  ASE is quantitative because it supplies complete object geometry. The input is partial, and outputs are evaluated against complete shapes. ScanNet++ examples show qualitative generalization, but the paper's Table 3 numbers are not ScanNet++ scores.

**15. Does SuperFlex introduce bending and tapering as entirely new mathematical concepts?**  No. These are established global deformations from classical geometric modeling (e.g., Barr and Solina & Bajcsy). SuperFlex integrates them into a compact **learned multi-primitive decomposition** with new loss and refinement methods.

## Technical caveats worth knowing but not putting on the main slides

- The official paper's full geometric chain uses 3-axis bending; the notebook demonstrates the single-$z$ case explicitly. The notebook labels this limitation.
- The paper calls its quantity radial distance even when deforming the primitive; do not claim it solves the nearest-boundary problem analytically.
- The occupancy loss requires occupancy $O(x)$ of the target during complete-cloud training; it is not obtained purely from the set of coordinate triples without an occupancy-providing source.
- The *existence penalty* follows average usage, while the *overlap penalty* in Eq. (11) is written in terms of per-primitive indicators $I_p$, not existence-gated occupancy $e_p I_p$.
- The supplementary material discusses a separate unsupervised robustness approach that fits only visible fragments; the main method uses complete pseudo-targets to hallucinate missing geometry.
- Tables 1, 2, 4, and 5 have different experimental conditions. Do not combine numbers from different tables as if they were the same training run.

## Suggested live demonstration

For a 20-minute talk, **do not run Jupyter live** unless you have confirmed everything in advance. The Beamer file already embeds exported plots. If your instructor permits an extra 20–30 seconds, briefly show the bending image and ask: “Which parameter bends more, and which one changes its direction?” Reveal $k$ versus $\alpha$ using the prepared visual. Keep the actual notebook in reserve for questions.

## Files and provenance

The main PDF and Beamer `.tex` form the presentation. `SuperFlex_geometry_notebook.ipynb` is executable with outputs embedded. The paper figures are cropped directly from the **user-provided** `SuperFlex.pdf`; every borrowed image carries a slide source caption. Python images are originals generated analytically from formulas cited in the source paper. No model training or evaluation experiments were executed, and no published performance figures were represented as independently verified results.
