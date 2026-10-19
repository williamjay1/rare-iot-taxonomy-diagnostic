"""Verify every recovered cell and record a bounded, reproducible layout diagnosis."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zipfile import ZipFile

from lxml import etree
import pdfplumber

ROOT = Path(__file__).resolve().parent
MAIN = ROOT.parents[1] / 'manuscript'
SOURCE = ROOT / 'snapshot/IJISMD_v15_editorial_working_draft.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{'+NS['w']+'}'
SOFFICE = Path(r'C:\Program Files\LibreOffice\program\soffice.com')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body(path):
    with ZipFile(path) as z:
        return etree.fromstring(z.read('word/document.xml')).find('w:body', NS)


def compact(value):
    return ''.join((value or '').split())


def rows(table):
    return [[''.join(cell.xpath('.//w:t/text()', namespaces=NS)) for cell in row.findall('w:tc', NS)]
            for row in table.findall('w:tr', NS)]


def main():
    original = body(SOURCE)
    expected = rows(original.findall('w:tbl', NS)[5])
    records = []
    for name in ('baseline', 'caption_pagebreak', 'no_repeat_header', 'no_cant_split',
                 'explicit_table_width', 'minimal_table'):
        directory = ROOT / 'probes' / name
        record = json.loads((directory / 'result.json').read_text())
        found = []
        with pdfplumber.open(directory / f'{name}.pdf') as pdf:
            for page_index, page in enumerate(pdf.pages):
                for table in page.extract_tables():
                    if table and len(table[0]) == 4:
                        data = table[1:] if [compact(c) for c in table[0]] == [compact(c) for c in expected[0]] else table
                        found.extend((page_index+1, row) for row in data)
        record['cell_rows'] = [{'row': i, 'expected_cells': row,
                                'pages': [page for page, actual in found
                                          if [compact(c) for c in actual] == [compact(c) for c in row]]}
                               for i, row in enumerate(expected[1:], 1)]
        record['complete_exact_cell_rows'] = all(row['pages'] for row in record['cell_rows'])
        record['matched_body_cells'] = 4*sum(bool(row['pages']) for row in record['cell_rows'])
        mutated = body(directory / f'{name}.docx')
        record['table6_source_cells_identical'] = rows(mutated.findall('w:tbl', NS)[0 if name == 'minimal_table' else 5]) == expected
        if name == 'no_repeat_header':
            record['warning'] = 'All body cells recovered, but header remains alone on page 12 and all data rows move to page 13. Not the preferred fix.'
        if name == 'caption_pagebreak':
            caption = mutated.findall('w:tbl', NS)[5].getprevious()
            br = caption.find('w:pPr/w:pageBreakBefore', NS)
            assert br is not None
            br.set(W+'val', '0')
            record['only_effective_body_change_is_caption_pagebreak'] = etree.tostring(mutated, method='c14n') == etree.tostring(original, method='c14n')
            assert record['only_effective_body_change_is_caption_pagebreak']
        records.append(record)
    assert next(r for r in records if r['probe'] == 'baseline')['matched_body_cells'] == 4
    assert next(r for r in records if r['probe'] == 'caption_pagebreak')['matched_body_cells'] == 36
    odt_dir = ROOT / 'odt_import_probe'
    odt_dir.mkdir(exist_ok=True)
    temp = odt_dir / 'temp'
    temp.mkdir(exist_ok=True)
    command = [str(SOFFICE), '-env:UserInstallation='+(odt_dir / 'profile').as_uri(), '--headless',
               '--convert-to', 'odt', '--outdir', str(odt_dir), str(ROOT / 'probes/baseline/baseline.docx')]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240,
                            env=dict(os.environ, TEMP=str(temp), TMP=str(temp)), creationflags=subprocess.CREATE_NO_WINDOW)
    (odt_dir / 'conversion.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
    odt = odt_dir / 'baseline.odt'
    assert odt.exists()
    ons = {'table':'urn:oasis:names:tc:opendocument:xmlns:table:1.0',
           'text':'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
    with ZipFile(odt) as archive:
        content = etree.fromstring(archive.read('content.xml'))
    mapping_tables = []
    for table in content.findall('.//table:table', ons):
        if 'Random pseudo maps' in ''.join(table.itertext()):
            mapping_tables.append(table)
    assert len(mapping_tables) == 1
    odt_rows = mapping_tables[0].findall('.//table:table-row', ons)
    assert len(odt_rows) == 10
    version = subprocess.run([str(SOFFICE), '--version'], capture_output=True, text=True,
                             creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
    report = {'status': 'REPRODUCED_WITH_MINIMAL_WORKAROUNDS',
              'scope': 'Only scratch artifacts under harmonized/audit; manuscript unchanged by this audit',
              'source_snapshot_sha256': sha(SOURCE),
              'main_docx_sha256_at_finish': sha(MAIN / 'staging_v15' / SOURCE.name),
              'main_pdf_sha256_at_finish': sha(MAIN / 'qa_projection' / SOURCE.with_suffix('.pdf').name),
              'libreoffice': version, 'executable': str(SOFFICE),
              'conversion_policy': 'Same existing manuscript renderer for exact defect reproduction; isolated D profiles, no installed or downloaded dependencies',
              'ooxml_import_preserves_all_10_rows_in_odt': len(odt_rows) == 10,
              'probes': records,
              'diagnosis': 'Reproducible page-continuation layout/export defect involving repeating header and non-splittable rows, not missing OOXML or failed data import. Exact LibreOffice internal defect not established.',
              'minimal_recommended_fix': 'Set page_break_before=True on Table 6 caption only. Keep header repetition and row non-splitting; verified all 36 body cells at page 13, total 34 pages.',
              'alternatives': ['Remove cantSplit on Table 6 rows to allow normal page continuation'],
              'nonpreferred_workaround': 'Removing tblHeader recovers all body cells, but leaves an orphan header on page 12. Prefer the caption break or removal of cantSplit.',
              'nonfix': 'Changing table preferred width from auto to explicit dxa 9360 alone still loses eight data rows'}
    (ROOT / 'TABLE_LAYOUT_DIAGNOSIS.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'probes'}, indent=2))
    print(json.dumps([{k:r[k] for k in ('probe', 'pages', 'matched_body_cells', 'complete_exact_cell_rows')}
                      for r in records], indent=2))


if __name__ == '__main__':
    main()
