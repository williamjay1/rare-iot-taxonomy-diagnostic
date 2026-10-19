"""Record physical paths and read-only reference source digests, without fitting."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import run_harmonized_refit as run


def main():
    physical_source = run.ROOT.resolve()
    references = {}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.is_relative_to(physical_source) and path.suffix == ".py":
                references[str(path)] = run.sha256(path)
    path = run.WORK / "runtime_provenance.json"
    if path.exists():
        prior = json.loads(path.read_text())
        assert prior["read_only_reference_sources"] == references, "Reference sources changed"
    value = {"python_path": sys.executable, "python_resolved": str(Path(sys.executable).resolve()),
             "python_version": sys.version, "package_versions": run.benefit.package_versions(),
             "old_project_logical": str(run.ROOT), "old_project_physical": str(physical_source),
             "working_directory_physical": str(run.WORK.resolve()),
             "training_and_outputs_physical_d": run.WORK.resolve().drive.upper() == "D:",
             "bytecode_writing_disabled": sys.dont_write_bytecode,
             "read_only_reference_sources": references,
             "metric_inference_seconds_note": "Total utility+selection+test prediction stage, not a test-only hardware benchmark",
             "permission_proof": {
                 "flat_comparator_reference": str(Path(run.comparator.__file__).resolve()),
                 "newton_stacker_reference": str(Path(run.newton.__file__).resolve()),
                 "flat_temperature_and_stacker_fit_role": "Second split_development return: calibration in comparator scripts; utility in PRIMARY",
                 "flat_selection_role": "Third split_development return, diagnostics only; never fitted",
                 "benefit_gate": "Second return utility for regressor, third return selection for grid",
                 "outer": "Evaluation only for every arm"}}
    run.json_write(path, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
