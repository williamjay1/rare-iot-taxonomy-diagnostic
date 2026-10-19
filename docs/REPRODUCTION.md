# Reproduction notes

This is an archival code and result release, not a one-command reproduction package.
The experiment scripts were executed in a frozen local workspace with a specific path
contract. The repository preserves that contract through placeholders and documents the
required layout.

## Suggested layout

```
workspace/
  harmonized/
  extensions/
  temporal/
  external_grouped/
  pseudo_null/
  standards/
  primary_audit/
legacy_project/
  scripts/
release/
  data/
  results/
  figures/
```

`<WORKSPACE>` in the archived records maps to `workspace/`; `<LEGACY_PROJECT>` maps to
`legacy_project/`; `<RELEASE>` maps to this repository root.

## Execution order

1. Obtain the raw datasets listed in the repository README and place them under the
   workspace paths expected by the scripts.
2. Run the primary CICIoT2023 scripts in `code/primary/`.
3. Run `code/harmonized/` for the 37 predictor refits and paired comparisons.
4. Run `code/extensions/` for matched endpoint, calibration, alert-budget, scarcity,
   family-exclusion, and capture-order analyses.
5. Run `code/temporal/` for NF-ToN-IoT-v3 grouped and chronological roles.
6. Run `code/external_grouped/` and `code/pseudo_null/` for the independent-source and
   random-reference controls.
7. Run `code/standards/` for the CAPEC-informed control.
8. Run `code/primary_audit/` for uncertainty recomputation.
9. Run `code/figures/build_figures.py` to regenerate figures from the result tables.

The result tables in `results/` are the archived outputs of these stages. Re-running the
pipeline requires the raw data and may require path or dependency adjustments.

## Environment

Python 3.12 with the versions in `requirements.txt`. The manuscript reports the full
workstation and library versions. GPU use is not required.
