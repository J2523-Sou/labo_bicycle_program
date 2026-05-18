from pathlib import Path
import csv

import fitparse

# FITファイルのパスをスクリプト基準で解決する
fit_file_path = Path(__file__).with_name('sample.fit')
csv_file_path = fit_file_path.with_suffix('.csv')

# ファイルの読み込み
fitfile = fitparse.FitFile(str(fit_file_path))

records = list(fitfile.get_messages('record'))

field_names = []
seen_fields = set()
for record in records:
    for data in record:
        if data.name not in seen_fields:
            seen_fields.add(data.name)
            field_names.append(data.name)

if 'timestamp' in seen_fields:
    field_names = ['timestamp'] + [name for name in field_names if name != 'timestamp']

with csv_file_path.open('w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=field_names)
    writer.writeheader()

    for record in records:
        row = {name: '' for name in field_names}
        for data in record:
            value = data.value
            if hasattr(value, 'isoformat'):
                value = value.isoformat(sep=' ')
            row[data.name] = value
        writer.writerow(row)

print(f'Saved CSV: {csv_file_path}')
print(f'Records: {len(records)}')