from __future__ import annotations

import csv
import hashlib
import json
import stat
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = Path(r"F:\AcademicData\iot_ids_sci\raw\standards_sources_20261018_v1")
SOURCES = {
    "capec_v3.9.xml": "https://capec.mitre.org/data/xml/capec_v3.9.xml",
    "ciciot2023_official.html": "https://www.unb.ca/cic/datasets/iotdataset-2023.html",
    "attack_T1498.html": "https://attack.mitre.org/techniques/T1498/",
    "attack_T1499.html": "https://attack.mitre.org/techniques/T1499/",
    "attack_T1557_002.html": "https://attack.mitre.org/techniques/T1557/002/",
    "attack_T1595_002.html": "https://attack.mitre.org/techniques/T1595/002/",
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    RAW.mkdir(parents=True, exist_ok=True)
    entries = []
    for filename, url in SOURCES.items():
        path = RAW / filename
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                body = response.read(10 * 1024 * 1024 + 1)
                if len(body) > 10 * 1024 * 1024:
                    raise RuntimeError("Source exceeds bounded snapshot size")
            with path.open("xb") as stream:
                stream.write(body)
        body = path.read_bytes()
        path.chmod(stat.S_IREAD)
        entry = {
            "filename": filename,
            "url": url,
            "path": str(path),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        entries.append(entry)
        print(json.dumps(entry), flush=True)
    (ROOT / "source_manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    ns = {"c": "http://capec.mitre.org/capec-3"}
    tree = ET.fromstring((RAW / "capec_v3.9.xml").read_bytes())
    rows = []
    for node in tree.findall(".//c:Attack_Pattern", ns):
        description = node.find("c:Description", ns)
        rows.append({
            "capec_id": node.attrib["ID"],
            "name": node.attrib["Name"],
            "abstraction": node.attrib.get("Abstraction", ""),
            "status": node.attrib.get("Status", ""),
            "description": " ".join(" ".join(description.itertext()).split()) if description is not None else "",
            "url": f"https://capec.mitre.org/data/definitions/{node.attrib['ID']}.html",
            "source_version": tree.attrib.get("Version", "3.9"),
        })
    with (ROOT / "capec_definitions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    terms = ("flood", "fragment", "fingerprint", "scan", "upload", "browser", "backdoor", "spoof", "dictionary", "command injection", "sql injection", "cross-site")
    for row in rows:
        if any(t in row["name"].lower() for t in terms):
            print(row["capec_id"], row["name"], row["description"][:700])


if __name__ == "__main__":
    main()
