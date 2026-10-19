# Reproducibility package: Diagnosing Attack Taxonomy Utility in Rare IoT Intrusion Detection

Status: archival snapshot accompanying a manuscript under review. Author identities are withheld during double-blind review.

This repository contains the analysis code, derived result tables, source and role manifests, and figure material for the study. It does **not** redistribute raw third-party flow data or fitted model checkpoints, because of size and source licences. Raw datasets must be obtained from their providers.

## What is included

- `code/`: archival Python and PowerShell scripts for the primary CICIoT2023 analysis, matched 37 predictor refits, extensions, temporal NF-ToN-IoT-v3 analysis, external grouped allocations, pseudo-taxonomy controls, CAPEC-informed controls, uncertainty recomputation, and figure generation.
- `results/`: machine-readable CSV and JSON records underlying the manuscript tables and figures.
- `data/`: source manifests, role definitions, checksum audits, CAPEC 3.9 definitions, and the dataset metadata used for NF-ToN-IoT-v3.
- `figures/`: the 11 TIFF figures and available vector PDF/SVG versions.
- `docs/`: experiment protocols, analysis contracts, porting notes, and Zenodo instructions.
- `checksums/SHA256SUMS.txt`: SHA256 for every file in this repository.

## What is not included

- Raw CICIoT2023, IoTID20, N-BaIoT, or NF-ToN-IoT-v3 flow records.
- Full `.npz` prediction arrays and `.joblib` model checkpoints.
- Any confidential review material, author identity, or local absolute path.

The manuscript's data availability statement gives the source locations. `data/README.md` repeats them with licence notes.

## Key results

On the CICIoT2023 public subsample, equal taxonomy blending improved Macro F1 by **0.0218** and rare recall by **0.0827** across six grouped holdouts. The manuscript reports conditional paired intervals and approximate overlap-adjusted intervals, together with wrong-mapping, feature-partition, language-prior, NF-ToN-IoT-v3 grouped and chronological, archive, and device-holdout checks.

The positive result is local. The repository preserves the negative and bounded results rather than presenting them as successful transfer.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The scripts are an archival snapshot, not a one-command pipeline. They retain a frozen local path contract from the experiment workspace. `docs/PORTING.md` lists the placeholders and the recommended directory layout. `docs/REPRODUCTION.md` gives the execution order.

## Data sources

| Source | Role | Access | Licence |
|---|---|---|---|
| CICIoT2023 public subsample | Discovery | Hugging Face `lacg030175/CIC-IoT-2023-neto-subsample`, config `random`; derived from `lacg030175/CIC-IoT-2023-neto-full` | CC BY 4.0 for the mirror; original CIC terms |
| CICIoT2023 official archive | Scale and integrity audit | Canadian Institute for Cybersecurity, 63 MERGED_CSV files | CIC dataset terms |
| IoTID20 | External conflict-rule test | Third-party preprocessed release `KathiS/IoTID20_Preprocessed_File`; original Ullah and Mahmoud (2020) | Original dataset terms |
| N-BaIoT | Device-held-out ceiling check | UCI Machine Learning Repository | UCI dataset terms |
| NF-ToN-IoT-v3 | Independent grouped and chronological source | DOI `10.48610/44d7c5e` | University of Queensland noncommercial terms |
| CAPEC 3.9 | Standard-informed control | `https://capec.mitre.org/data/xml/capec_v3.9.xml` | MITRE terms |

## Citation

The accompanying article is under review. A Zenodo DOI will be added after the archive is published. `CITATION.cff` contains the package metadata and a placeholder author entry for the double-blind period.

## Licence

Code is released under the MIT License. Derived tables, metadata, and figures are released under CC BY 4.0. Raw third-party datasets are not redistributed and remain under their original licences. See `LICENSE` and `DATA_LICENSE.md`.
