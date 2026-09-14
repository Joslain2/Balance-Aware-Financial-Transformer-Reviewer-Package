#!/usr/bin/env python3
"""Verify included dataset files without displaying transaction-level records."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_csv(stream, expected: dict, scan: bool) -> dict:
    reader = csv.DictReader(stream)
    if reader.fieldnames != expected['columns']:
        raise ValueError('CSV columns differ from dataset manifest')
    result = {'columns_match': True}
    if scan:
        users = set()
        rows = 0
        for record in reader:
            if None in record or any(value is None for value in record.values()):
                raise ValueError('CSV row has an unexpected number of fields')
            rows += 1
            users.add(record['USER_ID'])
        if rows != expected['rows'] or len(users) != expected['users']:
            raise ValueError(f'Observed counts differ: {rows} rows, {len(users)} users')
        result.update(rows_verified=rows, users_verified=len(users))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-rows', action='store_true', help='Read all records to verify row and distinct-user counts')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/dataset_manifest.json').read_text())
    reports = []
    for entry in manifest['datasets']:
        path = (ROOT / entry['path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Dataset path must remain within the repository')
        if digest(path) != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {entry["path"]}')
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                members = [member for member in archive.infolist() if not member.is_dir()]
                if len(members) != 1 or members[0].filename != entry['csv_member']:
                    raise ValueError('Unexpected archive content')
                if archive.testzip() is not None:
                    raise ValueError('Archive integrity check failed')
                with archive.open(members[0]) as raw:
                    findings = validate_csv(io.TextIOWrapper(raw, encoding='utf-8-sig', newline=''), entry, args.scan_rows)
        else:
            with path.open(encoding='utf-8-sig', newline='') as stream:
                findings = validate_csv(stream, entry, args.scan_rows)
        reports.append({'dataset_kind': entry['dataset_kind'], 'path': entry['path'],
                        'sha256_matches': True, **findings})
    print(json.dumps({'status': 'PASS', 'transaction_values_displayed': False,
                      'counts_scanned': args.scan_rows, 'datasets': reports}, indent=2))


if __name__ == '__main__':
    main()
