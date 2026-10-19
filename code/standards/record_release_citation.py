from __future__ import annotations

import hashlib
import html.parser
import json
import stat
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = Path(r"F:\AcademicData\iot_ids_sci\raw\standards_sources_20261018_v1")
URL = "https://capec.mitre.org/news/index.html"
PATH = RAW / "capec_release_news_20261019_v1.html"


class PlainHTML(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def main() -> None:
    if not PATH.exists():
        with urllib.request.urlopen(URL, timeout=45) as response:
            body = response.read(1024 * 1024 + 1)
        assert len(body) <= 1024 * 1024
        with PATH.open("xb") as stream:
            stream.write(body)
    body = PATH.read_bytes()
    parser = PlainHTML()
    parser.feed(body.decode("utf-8"))
    text = " ".join(" ".join(parser.parts).split())
    title = "CAPEC List Version 3.9 Now Available"
    start = text.index(title)
    assert "January 24, 2023" in text[start : start + 200]
    xml_path = RAW / "capec_v3.9.xml"
    catalog = ET.parse(xml_path).getroot()
    assert catalog.attrib["Version"] == "3.9"
    assert catalog.attrib["Date"] == "2023-01-24"
    PATH.chmod(stat.S_IREAD)
    record = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "3.9",
        "catalog_date_and_official_release_date": "2023-01-24",
        "release_announcement_title": title,
        "release_url": URL,
        "release_snapshot": str(PATH),
        "release_snapshot_bytes": len(body),
        "release_snapshot_sha256": hashlib.sha256(body).hexdigest(),
        "catalog_url": "https://capec.mitre.org/data/xml/capec_v3.9.xml",
        "catalog_path": str(xml_path),
        "catalog_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        "apa_reference": "The MITRE Corporation. (2023, January 24). Common Attack Pattern Enumeration and Classification (CAPEC) (Version 3.9) [Attack-pattern catalog]. https://capec.mitre.org/data/xml/capec_v3.9.xml",
        "scope": "Citation evidence added after design lock. Does not amend frozen mapping, roles, configuration, source_manifest, or experiment results.",
    }
    destination = ROOT / "results" / "ijismd_v15_standards_20261018" / "CAPEC_RELEASE_CITATION.json"
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
