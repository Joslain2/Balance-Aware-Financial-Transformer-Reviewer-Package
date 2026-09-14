#!/usr/bin/env python3
"""Generate an accounting-consistent synthetic input for software checks only."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import random
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--users', type=int, default=24)
    parser.add_argument('--events-per-user', type=int, default=180)
    parser.add_argument('--seed', type=int, default=314159)
    args = parser.parse_args()
    if args.users < 6 or args.events_per_user < 165:
        parser.error('Use at least 6 users and 165 daily events to populate all chronological splits')
    rng = random.Random(args.seed)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['USER_ID', 'COMPLETION_TIME', 'TAB', 'SERVICE', 'AMOUNT', 'MMT_FEE', 'BALANCE'])
    for user in range(args.users):
        balance = 50000
        for t in range(args.events_per_user):
            service = rng.choices(['CashIn', 'BuyBundles', 'CashOut'], weights=[3, 4, 3])[0]
            if balance < 7000:
                service = 'CashIn'
            if service == 'CashIn':
                amount, fee, direction = rng.randrange(5000, 12001), 0, 'credit'
                balance += amount
            else:
                amount = rng.randrange(100, 1001) if service == 'BuyBundles' else rng.randrange(2000, 5001)
                fee, direction = (10 if service == 'BuyBundles' else 100), 'debit'
                balance -= amount + fee
            when = datetime(2025, 9, 1, 12) + timedelta(days=t, minutes=user)
            writer.writerow([f'synthetic_user_{user:04d}', when.strftime('%d-%b-%y %I.%M.%S.%f %p'),
                             direction, service, f'{amount/100:.2f}', f'{fee/100:.2f}', f'{balance/100:.2f}'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo('synthetic_mobile_money.csv', date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, stream.getvalue().encode('utf-8'))
    print(json.dumps({'dataset_kind': 'synthetic_software_fixture', 'seed': args.seed,
                      'users': args.users, 'rows': args.users * args.events_per_user,
                      'output': str(args.output), 'scientific_evidence': False}, indent=2))


if __name__ == '__main__':
    main()
