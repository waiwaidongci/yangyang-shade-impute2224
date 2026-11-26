"""Deterministic readings from simulated nursery benches and two shade meshes."""

from random import Random

MESHES = {'mesh30': 0, 'mesh60': 1, 'mesh80': 1.5}


def _readings(batch, partition, rows, mesh, rng):
    produced = []
    for j in range(rows):
        day_phase = (j // 2) / 5 + rng.uniform(-0.035, 0.035)
        shift = MESHES[mesh]
        produced.append({'reading_id': f'{batch}-{j + 1:02d}', 'batch_id': batch, 'partition': partition,
                         'shade_class': mesh,
                         'temp_c': round(16 + 12 * day_phase + 2 * shift + rng.gauss(0, 0.4), 2),
                         'rh_pct': round(81 - 30 * day_phase - 3 * shift + rng.gauss(0, 0.7), 2),
                         'par': round(120 + (455 + 340 * shift) * day_phase + rng.gauss(0, 9), 2)})
    return produced


def demo_rows():
    rng = Random(261129)
    rows = []
    # Training batches alternate meshes; 4 batches per mesh = 48 rows per mesh.
    for i in range(8):
        rows.extend(_readings(f'TR-{i + 1:02d}', 'train', 12, ('mesh30', 'mesh60')[i % 2], rng))
    # Calibration batches are one mesh each: mesh30 has 12, mesh60 only 8 (one shortfall to demo).
    for i in range(12):
        rows.extend(_readings(f'CA-{i + 1:02d}', 'calib', 12, 'mesh30', rng))
    for i in range(8):
        rows.extend(_readings(f'CA-{i + 13:02d}', 'calib', 12, 'mesh60', rng))
    # Holdout batches are independent from calibration and also one mesh each.
    for i in range(6):
        rows.extend(_readings(f'HO-{i + 1:02d}', 'holdout', 12, 'mesh30', rng))
    for i in range(3):
        rows.extend(_readings(f'HO-{i + 7:02d}', 'holdout', 12, 'mesh60', rng))
    # Pending batches carry the missing-field combinations handed over daily.
    for i in range(2):
        batch = f'PE-{i + 1:02d}'
        produced = _readings(batch, 'pending', 9, 'mesh30', rng)
        for j, row in enumerate(produced):
            row['shade_class'] = 'mesh80' if j == 7 else ('mesh30', 'mesh60')[j % 2]
            row[('temp_c', 'rh_pct', 'par')[j % 3]] = None
            if j == 5:
                row['temp_c'] = None
        rows.extend(produced)
    return rows
