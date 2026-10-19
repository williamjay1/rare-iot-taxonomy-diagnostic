# Derived result tables

These CSV and JSON files are the machine-readable records behind the manuscript tables
and figures. They contain aggregate metrics, split-level effects, role manifests, audit
records, and threshold/protocol summaries. They do not contain raw flow records, full
prediction arrays, or fitted model checkpoints.

Subdirectories follow the experiment structure:

- `harmonized/`: 37 shared predictor refits and paired 46 predictor comparisons.
- `extensions/`: matched endpoint controls, calibration, alert budget, scarcity, capture order, and family exclusion.
- `temporal/`: NF-ToN-IoT-v3 grouped and chronological allocations.
- `external_grouped/`: NF-ToN-IoT-v3 grouped allocation summaries.
- `pseudo_null/`: expanded random reference and similarity summaries.
- `standards/`: CAPEC-informed control and source audits.
- `primary_audit/`: uncertainty recomputation and McNemar summaries.

Some provenance files retain placeholders such as `<WORKSPACE>` or `<RELEASE>` after
path sanitization. The placeholders document the original local layout without exposing
absolute paths. The numerical columns are unchanged.
