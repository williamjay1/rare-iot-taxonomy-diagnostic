"""Single-variable layout probes against a frozen DOCX copy, never the main files."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree
import pdfplumber
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'snapshot/IJISMD_v15_editorial_working_draft.docx'
SOFFICE = Path(r'C:\Program Files\LibreOffice\program\soffice.com')
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + NS['w'] + '}'


def write_probe(name):
    with ZipFile(SOURCE) as archive:
        parts = {entry.filename: archive.read(entry.filename) for entry in archive.infolist()}
    document = etree.fromstring(parts['word/document.xml'])
    body = document.find('w:body', NS)
    table = body.findall('w:tbl', NS)[5]
    previous = table.getprevious()
    if name == 'minimal_table':
        keep = deepcopy(table)
        section = deepcopy(body.find('w:sectPr', NS))
        for child in list(body):
            body.remove(child)
        body.append(keep)
        body.append(section)
    elif name == 'no_cant_split':
        for element in table.findall('.//w:cantSplit', NS):
            element.getparent().remove(element)
    elif name == 'no_repeat_header':
        for element in table.findall('.//w:tblHeader', NS):
            element.getparent().remove(element)
    elif name == 'caption_pagebreak':
        properties = previous.find('w:pPr', NS)
        if properties is None:
            properties = etree.SubElement(previous, W+'pPr')
        pagebreak = properties.find('w:pageBreakBefore', NS)
        if pagebreak is None:
            pagebreak = etree.Element(W+'pageBreakBefore')
            properties.insert(0, pagebreak)
        pagebreak.set(W+'val', '1')
    elif name == 'explicit_table_width':
        width = table.find('w:tblPr/w:tblW', NS)
        width.set(W+'type', 'dxa')
        width.set(W+'w', '9360')
    elif name != 'baseline':
        raise ValueError(name)
    parts['word/document.xml'] = etree.tostring(document, xml_declaration=True, encoding='UTF-8', standalone=True)
    directory = ROOT / 'probes' / name
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (name+'.docx')
    with ZipFile(destination, 'w', compression=ZIP_DEFLATED) as archive:
        for part, data in parts.items():
            archive.writestr(part, data)
    return destination


def render(name):
    source = write_probe(name)
    output = source.parent
    temp = output / 'temp'
    temp.mkdir(exist_ok=True)
    profile = output / 'lo_profile'
    env = dict(os.environ, TEMP=str(temp), TMP=str(temp), PYTHONDONTWRITEBYTECODE='1')
    command = [str(SOFFICE), '-env:UserInstallation='+profile.as_uri(), '--headless',
               '--convert-to', 'pdf:writer_pdf_Export', '--outdir', str(output), str(source)]
    started = time.perf_counter()
    result = subprocess.run(command, env=env, capture_output=True, text=True,
                            timeout=240, creationflags=subprocess.CREATE_NO_WINDOW)
    (output / 'conversion.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
    pdf_path = source.with_suffix('.pdf')
    if not pdf_path.exists():
        raise RuntimeError((result.returncode, result.stdout, result.stderr))
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or '' for page in pdf.pages]
    text = '\n'.join(pages)
    (output / 'pdf_text.txt').write_text(text, encoding='utf-8')
    with ZipFile(SOURCE) as archive:
        document = etree.fromstring(archive.read('word/document.xml'))
    table = document.find('w:body', NS).findall('w:tbl', NS)[5]
    labels = [''.join(row.find('w:tc', NS).xpath('.//w:t/text()', namespaces=NS))
              for row in table.findall('w:tr', NS)[1:]]
    record = {'probe': name, 'seconds': time.perf_counter()-started,
              'returncode': result.returncode, 'pages': len(pages), 'source_copy': str(source),
              'pdf': str(pdf_path), 'labels': {label:[i+1 for i,p in enumerate(pages) if label in p]
                                              for label in labels},
              'numeric_needles': {needle:[i+1 for i,p in enumerate(pages) if needle in p]
                                  for needle in ('0.8958', '0.8814')}}
    record['all_mapping_labels_found'] = all(record['labels'].values())
    (output / 'result.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
    with pdfium.PdfDocument(pdf_path) as pdf:
        for index, page in enumerate(pages):
            if 'Mapping controls' in page or 'Random pseudo maps' in page or name == 'minimal_table':
                pdf[index].render(scale=1.25).to_pil().save(output / f'page_{index+1}.png')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('probes', nargs='*', default=['baseline', 'caption_pagebreak', 'no_repeat_header',
                                                   'no_cant_split', 'explicit_table_width', 'minimal_table'])
    args = parser.parse_args()
    for name in args.probes:
        render(name)
