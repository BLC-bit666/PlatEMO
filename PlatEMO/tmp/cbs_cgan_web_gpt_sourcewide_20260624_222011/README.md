# CBS-CGAN / BDG Web GPT Review Package

Created: 20260624_222011
Workspace: /Users/lanai/Code/Matlab/PlatEMO/PlatEMO
Current branch: endpoint-yb-norm

This package is intentionally source-heavy and image-selective.

## Contents

- source/current_worktree/PlatEMO
  - Broad current local source snapshot: Algorithms, Problems, Metrics, GUI, docs, manuals, fix notes, root MATLAB files.
  - Data/, tmp/, hidden tool state, and .DS_Store are excluded.
- source/focused
  - Quick access to current CBS-CGAN, current CCMO-GAN-BDG, and CPS-MOEA/KNN.m.
- source/committed_snapshots
  - Historical committed snapshots from HEAD, UC-GAN, UC-GAN-2, 4212a3d5, cf503f63, 7cbe0144 where relevant paths exist.
  - This is included so reviewers can inspect the committed mainline and old boundary archive pair + KNN implementation.
- experiments/all_non_png
  - CBS-CGAN non-PNG outputs: CSV/MAT/manifest/log/config-like files.
  - Selected old CCMO_GAN_BDG non-PNG CSV/log/text evidence only. Full Data/CCMO_GAN_BDG is 7.1G and is not included.
- experiments/selected_png
  - Representative PNGs only, approximately 70 images.
  - Includes A/B raw failure examples, baseline/mechanism/epoch/pair/endpoint/y_b_norm contact sheets, and single-stage overfit diagnostics.
- metadata
  - Git status, diffs, source file list, selected image list, and grep hits for key mechanism terms.

## Counts

- Source files: 2715
- Non-PNG experiment files: 2984
- Selected PNG files: 72

## Recommended starting points

- source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN
- source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CCMO-GAN-BDG
- source/committed_snapshots/UC-GAN/Algorithms/Multi-objective optimization/CCMO-GAN-BDG
- metadata/key_mechanism_grep.txt
- experiments/selected_png/Data/CBS_CGAN
- experiments/all_non_png/Data/CBS_CGAN


## Corrected snapshot extraction

Committed snapshots were extracted from git top-level using PlatEMO/ path prefixes after validation.
Updated counts after corrected snapshot extraction:

- Source files: 2890
- Non-PNG experiment files: 2984
- Selected PNG files: 72
