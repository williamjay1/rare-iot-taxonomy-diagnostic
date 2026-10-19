# Data manifests

This directory contains source and role metadata, not raw flow data.

- `source_manifest.json`: dataset identifiers, source chain, and retrieval notes.
- `all_data_roles.csv`: role definitions for the discovery and matched refits.
- `input_manifest.csv`: input prediction-ledger records used for uncertainty recomputation.
- `nf_ton_download_manifest.json`: NF-ToN-IoT-v3 download record.
- `nf_ton_datacite_metadata.json`: DataCite metadata for the NF-ToN-IoT-v3 DOI.
- `frozen_source_hash_audit.csv`: source hash audit for frozen records.
- `checkpoint_manifest.csv`: model checkpoint inventory; checkpoints are not included.
- `evidence_manifest.json`: manifest of the evidence snapshot.
- `legacy_dependency_manifest.json`: dependency closure for the legacy primary pipeline.
- `capec_definitions.csv`: CAPEC 3.9 definitions used for the standard-informed control.

Raw datasets are not redistributed. See the repository README for source links and
licences.
