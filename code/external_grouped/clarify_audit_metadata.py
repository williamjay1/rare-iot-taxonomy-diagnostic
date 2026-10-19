"""Remove inherited chronological fields from random-allocation metadata."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
source = json.loads((BASE.parent / "temporal" / "data_audit.json").read_text())
for seed in (42, 43, 44):
    directory = BASE / f"allocation{seed}"
    target = directory / "data_audit.json"
    audit = {
        "source_rows": source["source_rows"],
        "features": source["features"],
        "allocation": "Predictor-grouped random allocation, not chronological",
        "role_audit": "role_audit.json",
        "source_audit": "../../temporal/data_audit.json",
    }
    assert json.loads(target.read_text())["features"] == audit["features"]
    target.write_text(json.dumps(audit, indent=2), encoding="utf-8")
print("Three metadata files clarified; no predictors, roles or model settings changed.")
