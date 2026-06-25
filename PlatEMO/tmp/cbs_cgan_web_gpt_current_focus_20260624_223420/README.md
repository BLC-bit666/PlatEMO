# CBS-CGAN Current-Focus Review Package

This is the small package requested for Web GPT. It is centered on the current branch, current source code, current failure mode, and recent git history. Historical source is included only as background/comparison.

## Package Focus

- Current branch source: `source/current_branch_current_worktree/Algorithms/Multi-objective optimization/CBS-CGAN`
- Current problem evidence: `experiments/representative_png/Data/CBS_CGAN`
- Key CSV evidence: `experiments/key_csv/Data/CBS_CGAN`
- Current uncommitted changes: `metadata/current_branch_uncommitted_diff.patch`
- Recent git context: `metadata/recent_algorithm_git_log.txt` and `metadata/recent_algorithm_git_log_with_stat.txt`

## Historical Background

Only a few valuable historical snapshots are included:

- `HEAD`: latest committed state behind current working tree
- `UC-GAN`: compact pair-cell / KNN boundary archive background
- `4212a3d5`: earlier CGAN stage where generation still did not stick to boundary
- `cf503f63`: earlier state described as boundary data training effective but GAN generation ineffective

These snapshots are under `source/historical_snapshots`. They are not the center of the package; they are included to compare boundary archive construction and earlier GAN/CGAN design.

## Minimal PlatEMO Context

Only 9 PlatEMO context files are included:

- `Algorithms/ALGORITHM.m`
- `Problems/PROBLEM.m`
- `Algorithms/Multi-objective optimization/CPS-MOEA/KNN.m`
- `Problems/Multi-objective optimization/LIR-CMOP_BC/LIRCMOP5_BC.m`
- `Problems/Multi-objective optimization/LIR-CMOP_BC/LIRCMOP6_BC.m`
- `Problems/Multi-objective optimization/LIR-CMOP_BC/LIRCMOP7_BC.m`
- `Problems/Multi-objective optimization/LIR-CMOP_BC/LIRCMOP8_BC.m`
- `Problems/Multi-objective optimization/LIR-CMOP_BC/LIRCMOP9_BC.m`
- `Problems/Multi-objective optimization/LIR-CMOP_BC/LIRCMOP10_BC.m`

## Experiment Evidence

Included images are representative, not exhaustive:

- ABCD baseline comparison contact sheets
- mechanism sweep contact sheets
- epoch sweep contact sheets
- pair=3 vs pair=6 contact sheets
- endpoint vs default contact sheets
- endpoint + y_b_norm vs default contact sheets
- A_ref_only_adv and B_ref_only_adv_huber raw training reconstruction examples
- single-stage overfit diagnostics

No `.mat` snapshots or full experiment trees are included.

## Counts At Creation

- Source files: 150
- Representative PNG files: 56
- Key CSV files: 46
- Minimal PlatEMO files: 9

