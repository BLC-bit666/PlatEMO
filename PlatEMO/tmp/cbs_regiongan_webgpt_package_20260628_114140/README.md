# CBS-RegionGAN / WGAN-GP Web GPT Review Package

This package is for asking Web GPT to review the current CBS-RegionGAN mainline and its recent ablation branches.

## Core Current Idea

The current mainline is not a deterministic inverse map from objective y to decision x. It is a conditional generator for p(x | coarse objective region): condition c describes a coarse objective-space region / boundary region, and random z should carry decision-space diversity inside that region. WGAN-GP is used because BCE CGAN was unstable for small online boundary training sets.

Important current constraints:

- Do not use multi-generation accumulated BMem.
- Do not rely on posterior filtering, local repair, or extra real evaluations to make generated samples look good.
- Current mainline parameters: CBS_RegionWGAN_GP, queryMode=random_all_w or boundary_populated in ablations, prevBMemMode=prev1_fair_union, nGen=30, zDim=6, hidden=[32 32], gpLambda=10, nCritic=5.
- Latest prev1_fair_union semantics: previous BMem contributes only feasible anchors before pairing; old infeasible pairs/gap rows are not reused directly.

## What Is Included

- `source/current_CBS_CGAN`: all current CBS-CGAN / CBS-RegionGAN / WGAN-GP source and tests in the working tree.
- `source/platemo_context`: minimal PlatEMO base classes, KNN helper, and LIR/DAS BC problem definitions.
- `Agent.md`: local current-state notes.
- `experiments/metrics`: compact CSV evidence and selected full event summaries for key branches.
- `experiments/selected_images`: 50 selected domain figures showing the current failure modes and branch differences.
- `metadata`: git status, current diffs, source file list, and image selection manifest.

## Key Experiment Branches In This Package

- mainline_prev_anchor_repair: current corrected prev1_fair_union domain figures from the early mainline inspection.
- query_boundary_iter75_fixed: query_boundary_iter75 rerendered with the correct domain-background plotting style.
- query_boundary_populated: branch that changes query condition selection toward populated boundary regions.
- wgan_iter100: branch that increases WGAN training iterations.
- random_iter100 and query_boundary_iter100: no-plot comparison branches included through metrics.
- z_sigma_025 and z_zero_sample: z-sampling ablations included through images and compact metrics.
- refcap5_min32_CGAN_vs_WGAN: early CGAN/WGAN comparison images.

## Main Question For Review

The orange training boundary points often sit on or near a feasible/infeasible boundary, but generated red points can still spread, drift, or leave the thin boundary cloud. Please diagnose whether the main blocker is dataset geometry, condition semantics, query strategy, z sampling, WGAN training volume, or model capacity, and propose the smallest next experiment/change that can distinguish these causes.
