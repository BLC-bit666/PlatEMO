Please analyze this CBS-RegionGAN / WGAN-GP package.

Context:
- We want a conditional generator p(x | coarse objective region), not a deterministic y -> x inverse map.
- Condition c should identify coarse objective boundary/region; z should represent decision diversity inside a region.
- WGAN-GP is the current mainline because BCE CGAN was unstable on online small boundary datasets.
- We do not want posterior filtering, local repair, extra evaluations, or multi-generation accumulated BMem as the main answer.

Please focus on:
1. Whether the current training-set construction creates a learnable thin boundary distribution or still creates a sparse/multi-component/thick distribution.
2. Whether query_boundary_populated and wgan_iter100 address different failure modes, and why random_iter100 metrics can outperform query_boundary_iter100.
3. Whether zDim=6 and the current z sampling appear too expressive, underused, or merely secondary relative to condition/training-set problems.
4. Whether the condition c has enough control over objective-region identity without becoming a precise y -> x inverse mapping.
5. A prioritized experimental plan with at most three main branches, including expected visual and CSV indicators.

Use source, CSV metrics, and the selected images together. Please distinguish direct evidence from inference.
