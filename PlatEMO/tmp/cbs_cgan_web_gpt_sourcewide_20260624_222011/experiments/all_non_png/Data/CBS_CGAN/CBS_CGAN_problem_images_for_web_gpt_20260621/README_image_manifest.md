# CBS-CGAN problem image package

Purpose: provide a small visual evidence set for Web GPT analysis of the current CBS-CGAN problem: boundary-solution training for CGAN/GAN and direct generation of complete decision vectors whose objective values form a narrow feasible/infeasible boundary, not a thick point cloud.

Main text package to upload with this zip:

- `Data/CBS_CGAN/CBS_CGAN_current_fixmd_source_results_web_gpt_20260621_7M.txt`

## Current CBS-CGAN branch

Source experiment:

- `Data/CBS_CGAN/boundary_quality_FS_qpc1_runs1_20260621_110631`
- Settings recorded in `run_summary.csv`: 10 problems, `runs=1`, `N=100`, `D=30`, `maxFE=100000`, five plotted stages per problem: `FE=10000,30000,50000,70000,100000`.
- Each sheet shows feasible/infeasible objective-space domains, training set, and CGAN-generated points.

Included sheets:

- `current_cbs_cgan_qpc1/DASCMOP1_BC_run1_sheet.png`
- `current_cbs_cgan_qpc1/DASCMOP2_BC_run1_sheet.png`
- `current_cbs_cgan_qpc1/DASCMOP4_BC_run1_sheet.png`
- `current_cbs_cgan_qpc1/DASCMOP5_BC_run1_sheet.png`
- `current_cbs_cgan_qpc1/LIRCMOP5_BC_run1_sheet.png`
- `current_cbs_cgan_qpc1/LIRCMOP6_BC_run1_sheet.png`
- `current_cbs_cgan_qpc1/LIRCMOP7_BC_run1_sheet.png`
- `current_cbs_cgan_qpc1/LIRCMOP8_BC_run1_sheet.png`
- `current_cbs_cgan_qpc1/LIRCMOP9_BC_run1_sheet.png`
- `current_cbs_cgan_qpc1/LIRCMOP10_BC_run1_sheet.png`

## Prior branch/reference visuals

These images are included only as visual references for earlier branches and comparisons. They are not the current main result.

Included images:

- `prior_branch_reference/ALL_PROBLEMS_FE_contact_GND_keep80_localMAD_weightedCond.png`
  - Source: `Data/CCMO_GAN_BDG/GND_keep80_localMAD_nGen20_weightedCond_runs3_n100_fe100000_8w_20260614_191241/image_review_contact_sheets/ALL_PROBLEMS_FE_contact.png`
- `prior_branch_reference/DASCMOP2_old_epoch50_vs_current.png`
  - Source: `Data/CCMO_GAN_BDG/visual_compare_epoch50_vs_current_20260618_194646/DASCMOP2_BC_old_epoch50_vs_current.png`
- `prior_branch_reference/LIRCMOP8_old_epoch50_vs_current.png`
  - Source: `Data/CCMO_GAN_BDG/visual_compare_epoch50_vs_current_20260618_194646/LIRCMOP8_BC_old_epoch50_vs_current.png`

## Visual evidence to inspect

Use these images to inspect only objective facts:

- Whether the training set forms one narrow boundary skeleton or mixes multiple boundary/frontier segments.
- Whether CGAN-generated points are close to the training boundary distribution.
- Whether CGAN-generated points form a narrow line or a thick/scattered point cloud.
- Whether generated points fill missing reference regions or mostly stay near already sampled regions.
- Whether red generated points cross between separated boundary segments.

