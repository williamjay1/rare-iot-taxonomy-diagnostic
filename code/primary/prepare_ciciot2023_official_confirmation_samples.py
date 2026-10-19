from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import run_ciciot_agreement_gated_trf_confirmation as public_design


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "ciciot2023_official_validated"
    / "MERGED_CSV"
)
DEFAULT_MANIFEST = (
    ROOT
    / "results"
    / "jit_ciciot_official_archive_audit"
    / "official_member_manifest.csv"
)
DEFAULT_DUPLICATE_AUDIT = (
    ROOT
    / "results"
    / "jit_ciciot_official_split_duplicate_audit"
)
FILE_PATTERN = re.compile(r"Merged(\d{2})\.csv$", re.IGNORECASE)
ROLE_SEQUENCES = {
    "base": range(1, 46),
    "gate_train": range(46, 51),
    "gate_select": range(51, 55),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare deterministic capped development samples for the "
            "frozen official CICIoT2023 confirmation."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--duplicate-audit-dir",
        type=Path,
        default=DEFAULT_DUPLICATE_AUDIT,
        help=(
            "Completed duplicate audit used to exclude predictor hashes "
            "that occur in an earlier development role."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--attack-cap", type=int, default=25_000)
    parser.add_argument("--benign-cap", type=int, default=80_000)
    parser.add_argument("--gate-cap", type=int, default=15_000)
    parser.add_argument(
        "--out-dir",
        default="jit_ciciot2023_official_confirmation_samples",
    )
    return parser.parse_args()


def sequence_from_member(member: str) -> int:
    match = FILE_PATTERN.search(member)
    if match is None:
        raise ValueError(f"Unexpected source member: {member}")
    return int(match.group(1))


def canonical_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


def canonical_label_map() -> dict[str, str]:
    labels = set(
        pd.read_parquet(
            public_design.TRAIN_PATH,
            columns=["Label"],
        )["Label"].astype(str)
    )
    labels.update(
        pd.read_parquet(
            public_design.TEST_PATH,
            columns=["Label"],
        )["Label"].astype(str)
    )
    mapping = {canonical_key(label): label for label in sorted(labels)}
    if len(mapping) != len(labels):
        raise ValueError("Canonical public label keys are not unique.")
    mapping["benign"] = "BenignTraffic"
    return mapping


def splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        output = values.astype(np.uint64) + np.uint64(
            0x9E3779B97F4A7C15
        )
        output = (output ^ (output >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        output = (output ^ (output >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        return output ^ (output >> np.uint64(31))


def row_priorities(
    sequence: int,
    row_start: int,
    rows: int,
    seed: int,
) -> np.ndarray:
    row_ids = np.arange(
        row_start,
        row_start + rows,
        dtype=np.uint64,
    )
    identity = (
        row_ids
        ^ (np.uint64(sequence) << np.uint64(48))
        ^ np.uint64(seed)
    )
    return splitmix64(identity)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def retain_smallest(
    current: pd.DataFrame | None,
    candidate: pd.DataFrame,
    cap: int,
) -> pd.DataFrame:
    if current is None:
        combined = candidate
    else:
        combined = pd.concat([current, candidate], ignore_index=True)
    combined = (
        combined.sort_values(
            ["_priority", "_source_sequence", "_source_row"]
        )
        .drop_duplicates("_feature_hash", keep="first")
    )
    if len(combined) > cap:
        combined = combined.nsmallest(
            cap,
            ["_priority", "_source_sequence", "_source_row"],
            keep="first",
        )
    return combined.reset_index(drop=True)


def membership_mask(
    values: np.ndarray,
    sorted_reference: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(sorted_reference, values)
    valid = positions < len(sorted_reference)
    matched = np.zeros(len(values), dtype=bool)
    matched[valid] = (
        sorted_reference[positions[valid]] == values[valid]
    )
    return matched


def prepare_role(
    role: str,
    sequences: range,
    source: Path,
    manifest: pd.DataFrame,
    canonical: dict[str, str],
    feature_columns: list[str],
    chunk_size: int,
    seed: int,
    attack_cap: int,
    benign_cap: int,
    gate_cap: int,
    conflicting_feature_hashes: np.ndarray,
    excluded_prior_role_hashes: list[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reservoirs: dict[str, pd.DataFrame] = {}
    source_counts: dict[str, int] = {}
    observed_counts: dict[str, int] = {}
    excluded_conflict_counts: dict[str, int] = {}
    excluded_prior_counts: dict[str, int] = {}
    for sequence in sequences:
        row = manifest.loc[manifest["sequence"] == sequence]
        if len(row) != 1:
            raise ValueError(f"Missing manifest row for sequence {sequence}.")
        member = str(row.iloc[0]["member"])
        path = source / Path(member).name
        print(f"Sampling {member} for {role}")
        row_start = 0
        for chunk in pd.read_csv(
            path,
            chunksize=chunk_size,
            low_memory=False,
        ):
            columns = [str(column) for column in chunk.columns]
            if columns != feature_columns + ["Label"]:
                raise ValueError(f"Unexpected schema in {member}.")
            canonical_labels = chunk["Label"].astype(str).map(
                lambda value: canonical.get(canonical_key(value))
            )
            if canonical_labels.isna().any():
                values = sorted(
                    chunk.loc[canonical_labels.isna(), "Label"]
                    .astype(str)
                    .unique()
                    .tolist()
                )
                raise ValueError(f"Unmapped labels in {member}: {values}")
            source_rows = np.arange(
                row_start,
                row_start + len(chunk),
                dtype=np.int64,
            )
            priorities = row_priorities(
                sequence,
                row_start,
                len(chunk),
                seed,
            )
            row_start += len(chunk)
            for label, count in canonical_labels.value_counts().items():
                source_counts[str(label)] = (
                    source_counts.get(str(label), 0) + int(count)
                )
            feature_hash = pd.util.hash_pandas_object(
                chunk[feature_columns],
                index=False,
                categorize=True,
            ).to_numpy(dtype=np.uint64)
            conflict = membership_mask(
                feature_hash,
                conflicting_feature_hashes,
            )
            prior = np.zeros(len(chunk), dtype=bool)
            for reference in excluded_prior_role_hashes:
                prior |= membership_mask(feature_hash, reference)
            prior_only = prior & ~conflict
            for label, count in canonical_labels[
                conflict
            ].value_counts().items():
                excluded_conflict_counts[str(label)] = (
                    excluded_conflict_counts.get(str(label), 0)
                    + int(count)
                )
            for label, count in canonical_labels[
                prior_only
            ].value_counts().items():
                excluded_prior_counts[str(label)] = (
                    excluded_prior_counts.get(str(label), 0)
                    + int(count)
                )
            keep = ~(conflict | prior)
            chunk = chunk.loc[keep].copy()
            canonical_labels = canonical_labels.loc[keep]
            source_rows = source_rows[keep]
            priorities = priorities[keep]
            feature_hash = feature_hash[keep]
            chunk["Label"] = canonical_labels.to_numpy()
            chunk["_source_sequence"] = np.int16(sequence)
            chunk["_source_row"] = source_rows
            chunk["_priority"] = priorities
            chunk["_feature_hash"] = feature_hash

            for label, group in chunk.groupby("Label", sort=False):
                label = str(label)
                observed_counts[label] = (
                    observed_counts.get(label, 0) + len(group)
                )
                if role == "base":
                    cap = (
                        benign_cap
                        if label == "BenignTraffic"
                        else attack_cap
                    )
                else:
                    cap = gate_cap
                candidate = (
                    group.sort_values(
                        ["_priority", "_source_sequence", "_source_row"]
                    )
                    .drop_duplicates("_feature_hash", keep="first")
                    .head(cap)
                )
                reservoirs[label] = retain_smallest(
                    reservoirs.get(label),
                    candidate,
                    cap,
                )
        if row_start != int(row.iloc[0]["rows"]):
            raise ValueError(
                f"Row count changed for {member}: "
                f"{row_start} != {int(row.iloc[0]['rows'])}"
            )

    expected_labels = sorted(set(canonical.values()))
    missing_labels = sorted(set(expected_labels).difference(reservoirs))
    if missing_labels and role == "base":
        raise ValueError(f"Role {role} is missing labels: {missing_labels}")
    selected = pd.concat(
        [reservoirs[label] for label in sorted(reservoirs)],
        ignore_index=True,
    ).sort_values(
        ["Label", "_priority", "_source_sequence", "_source_row"]
    )
    counts = pd.DataFrame(
        [
            {
                "role": role,
                "label": label,
                "source_rows": source_counts.get(label, 0),
                "excluded_conflicting_label_rows": (
                    excluded_conflict_counts.get(label, 0)
                ),
                "excluded_prior_role_rows": (
                    excluded_prior_counts.get(label, 0)
                ),
                "eligible_rows": observed_counts.get(label, 0),
                "available_rows": observed_counts.get(label, 0),
                "selected_rows": len(reservoirs.get(label, ())),
            }
            for label in expected_labels
        ]
    )
    return selected.reset_index(drop=True), counts


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    manifest_path = args.manifest.resolve()
    duplicate_audit_dir = args.duplicate_audit_dir.resolve()
    out_dir = ROOT / "results" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    duplicate_audit_path = (
        duplicate_audit_dir / "official_split_duplicate_audit.json"
    )
    if not duplicate_audit_path.is_file():
        raise FileNotFoundError(duplicate_audit_path)
    unique_feature_hashes = {
        role: np.load(
            duplicate_audit_dir / f"{role}_unique_feature_hashes.npy",
            mmap_mode="r",
        )
        for role in ("base", "gate_train")
    }
    conflicting_feature_hashes = np.load(
        duplicate_audit_dir / "global_conflicting_feature_hashes.npy",
        mmap_mode="r",
    )

    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 63 or set(manifest["status"]) != {"complete"}:
        raise ValueError("A complete 63 file archive audit is required.")
    manifest["sequence"] = manifest["member"].map(sequence_from_member)
    if sorted(manifest["sequence"].tolist()) != list(range(1, 64)):
        raise ValueError("The official archive file sequence is incomplete.")

    first_path = source / "Merged01.csv"
    columns = [str(column) for column in pd.read_csv(first_path, nrows=0)]
    if columns[-1] != "Label":
        raise ValueError("The official label column must be last.")
    feature_columns = columns[:-1]
    canonical = canonical_label_map()

    all_counts: list[pd.DataFrame] = []
    output_records: list[dict[str, object]] = []
    for role, sequences in ROLE_SEQUENCES.items():
        if role == "base":
            excluded_prior_hashes: list[np.ndarray] = []
        elif role == "gate_train":
            excluded_prior_hashes = [unique_feature_hashes["base"]]
        else:
            excluded_prior_hashes = [
                unique_feature_hashes["base"],
                unique_feature_hashes["gate_train"],
            ]
        selected, counts = prepare_role(
            role,
            sequences,
            source,
            manifest,
            canonical,
            feature_columns,
            args.chunk_size,
            args.seed,
            args.attack_cap,
            args.benign_cap,
            args.gate_cap,
            conflicting_feature_hashes,
            excluded_prior_hashes,
        )
        output_path = out_dir / f"{role}.parquet"
        selected.to_parquet(
            output_path,
            index=False,
            compression="zstd",
        )
        all_counts.append(counts)
        output_records.append(
            {
                "role": role,
                "path": output_path.name,
                "source_rows": int(counts["source_rows"].sum()),
                "excluded_conflicting_label_rows": int(
                    counts["excluded_conflicting_label_rows"].sum()
                ),
                "excluded_prior_role_rows": int(
                    counts["excluded_prior_role_rows"].sum()
                ),
                "eligible_rows": int(counts["eligible_rows"].sum()),
                "rows": int(len(selected)),
                "bytes": int(output_path.stat().st_size),
                "sha256": sha256_file(output_path),
                "source_sequences": (
                    f"{min(sequences):02d}-{max(sequences):02d}"
                ),
            }
        )
        print(
            f"Wrote {output_path.name}: {len(selected):,} rows"
        )

    pd.concat(all_counts, ignore_index=True).to_csv(
        out_dir / "selected_label_counts.csv",
        index=False,
    )
    pd.DataFrame(output_records).to_csv(
        out_dir / "sample_manifest.csv",
        index=False,
    )
    metadata = {
        "purpose": "official CICIoT2023 confirmation development samples",
        "design_lock": (
            "docs/jit_20260729_v1/"
            "OFFICIAL_CICIOT2023_CONFIRMATION_LOCK.md"
        ),
        "source": str(source),
        "archive_manifest": str(manifest_path),
        "duplicate_audit_dir": str(duplicate_audit_dir),
        "seed": args.seed,
        "priority_method": "SplitMix64 of source sequence, source row and seed",
        "feature_columns": feature_columns,
        "canonical_labels": sorted(set(canonical.values())),
        "role_sequences": {
            role: list(sequences)
            for role, sequences in ROLE_SEQUENCES.items()
        },
        "caps": {
            "base_attack": args.attack_cap,
            "base_benign": args.benign_cap,
            "gate_train_per_label": args.gate_cap,
            "gate_select_per_label": args.gate_cap,
        },
        "final_test_sequences": list(range(55, 64)),
        "final_test_sampled": False,
        "development_cross_role_feature_hashes_excluded": True,
        "global_conflicting_feature_hashes_excluded": True,
        "development_samples_unique_by_exact_predictor_hash": True,
        "development_exclusion_rule": {
            "base": "none",
            "gate_train": "exclude hashes observed in base",
            "gate_select": (
                "exclude hashes observed in base or gate_train"
            ),
        },
    }
    (out_dir / "sample_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(output_records).to_string(index=False))


if __name__ == "__main__":
    main()
