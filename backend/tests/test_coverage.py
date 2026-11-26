"""Unit tests for joint-interval calibration and whole-batch holdout coverage."""

import json
import unittest
from statistics import pstdev

from src.domain.coverage import (MIN_CALIBRATION_BATCHES, STATUS_NO_REFERENCE, STATUS_READY,
                                 STATUS_SHORT, STATUS_UNSUGGESTED, assess, calibrated_radius)
from src.domain.fixtures import demo_rows
from src.domain.neighbors import fit_reference
from src.domain.project import analyze
from src.domain.records import normalize_rows
from src.domain.settings import parse_options


def _row(reading_id, batch, partition, shade, temp, rh, par):
    return {'reading_id': reading_id, 'batch_id': batch, 'partition': partition, 'shade_class': shade,
            'temp_c': temp, 'rh_pct': rh, 'par': par}


def _core(shade, calib_batches, holdout_batches):
    """Calibration and holdout batches that each belong to one shade class."""
    rows = []
    for i in range(calib_batches):
        batch = f'CA-{shade}-{i + 1:02d}'
        for j in range(6):
            rows.append(_row(f'{batch}-{j + 1:02d}', batch, 'calib', shade,
                             18 + 0.2 * j, 80 - 0.3 * j, 150 + 10 * j))
    for i in range(holdout_batches):
        batch = f'HO-{shade}-{i + 1:02d}'
        for j in range(6):
            rows.append(_row(f'{batch}-{j + 1:02d}', batch, 'holdout', shade,
                             18 + 0.2 * j, 80 - 0.3 * j, 150 + 10 * j))
    return rows


def _train_from(rows, shade):
    """A complete train partition with small, deterministic per-position jitter."""
    train = []
    for row in rows:
        if row['partition'] != 'calib' or row['shade_class'] != shade:
            continue
        j = int(row['reading_id'].rsplit('-', 1)[1]) - 1
        copy = {**row, 'reading_id': 'TR-' + row['reading_id'],
                'batch_id': 'TR-' + row['batch_id'], 'partition': 'train',
                'temp_c': row['temp_c'] + 0.003 * (j % 2),
                'rh_pct': row['rh_pct'] - 0.002 * (j % 3),
                'par': row['par'] + 0.5 * (j % 2)}
        train.append(copy)
    return train


class CalibrationRadiusTests(unittest.TestCase):
    def test_quantile_uses_ceil_rank_and_never_above_max(self):
        scores = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        # ceil(11 * 0.9) = 10 -> the 10th smallest, i.e. the largest.
        self.assertEqual(calibrated_radius(scores), 1.00)

    def test_rank_matches_hand_computation(self):
        scores = list(range(1, 21))  # ceil(21*0.9)=19 -> 19th value
        self.assertEqual(calibrated_radius(scores), 19)

    def test_short_calibration_returns_none(self):
        self.assertIsNone(calibrated_radius([0.1] * (MIN_CALIBRATION_BATCHES - 1)))


class AssessTests(unittest.TestCase):
    def test_ready_group_exposes_radius_and_half_width(self):
        core = _core('mesh30', 12, 3)
        rows = _train_from(core, 'mesh30') + core
        rows.append(_row('PE-01-01', 'PE-01', 'pending', 'mesh30', None, 80.0, 160.0))
        options = parse_options({})
        references = fit_reference([r for r in rows if r['partition'] == 'train'], options['neighbors'])
        group = assess(rows, references, options)[('mesh30', ('temp_c',))]
        self.assertEqual(group['status'], STATUS_READY)
        self.assertIsNotNone(group['standardized_radius'])
        scale = pstdev([r['temp_c'] for r in references['mesh30'].rows])
        self.assertAlmostEqual(group['half_widths']['temp_c'], group['standardized_radius'] * scale, places=6)
        self.assertEqual(group['calibration_batches'], 12)
        self.assertEqual(group['holdout_batches'], 3)
        self.assertEqual(group['holdout_covered_batches'], 3)

    def test_short_calibration_reports_shortfall(self):
        core = _core('mesh30', 8, 2)
        rows = _train_from(core, 'mesh30') + core
        rows.append(_row('PE-01-01', 'PE-01', 'pending', 'mesh30', None, 80.0, 160.0))
        options = parse_options({})
        references = fit_reference([r for r in rows if r['partition'] == 'train'], options['neighbors'])
        group = assess(rows, references, options)[('mesh30', ('temp_c',))]
        self.assertEqual(group['status'], STATUS_SHORT)
        self.assertIsNone(group['standardized_radius'])
        self.assertEqual(group['calibration_shortfall'], MIN_CALIBRATION_BATCHES - 8)
        self.assertIsNone(group['holdout_covered_batches'])

    def test_missing_holdout_suggestion_counts_as_uncovered(self):
        core = _core('mesh30', 12, 3)
        for row in core:
            if row['partition'] == 'holdout':
                row['rh_pct'] = None  # train only has temp+par, so the train copy must stay complete
        rows = _train_from(_core('mesh30', 12, 3), 'mesh30') + core
        rows.append(_row('PE-01-02', 'PE-01', 'pending', 'mesh30', None, None, 160.0))
        options = parse_options({})
        references = fit_reference([r for r in rows if r['partition'] == 'train'], options['neighbors'])
        group = assess(rows, references, options)[('mesh30', ('temp_c', 'rh_pct'))]
        self.assertEqual(group['status'], STATUS_UNSUGGESTED)
        self.assertGreater(group['holdout_missing_suggestion_batches'], 0)
        self.assertIsNone(group['holdout_covered_batches'])

    def test_ready_group_counts_a_batch_without_suggestion_as_uncovered(self):
        core = _core('mesh30', 12, 3)
        # Keep an interval (enough usable calibration batches) but make one holdout batch un-suggestable:
        # drop both rh_pct and par, leaving only temp_c as anchor so min_anchors=2 fails.
        for row in core:
            if row['partition'] == 'holdout' and row['batch_id'].endswith('-03'):
                row['rh_pct'] = None
                row['par'] = None
        rows = _train_from(_core('mesh30', 12, 3), 'mesh30') + core
        rows.append(_row('PE-01-01', 'PE-01', 'pending', 'mesh30', None, 80.0, 160.0))
        options = parse_options({})
        references = fit_reference([r for r in rows if r['partition'] == 'train'], options['neighbors'])
        group = assess(rows, references, options)[('mesh30', ('temp_c',))]
        self.assertEqual(group['status'], STATUS_READY)
        self.assertEqual(group['holdout_batches'], 3)
        self.assertEqual(group['holdout_missing_suggestion_batches'], 1)
        self.assertEqual(group['holdout_covered_batches'], 2)

    def test_unknown_shade_reports_no_reference(self):
        core = _core('mesh30', 12, 3)
        rows = _train_from(core, 'mesh30') + core
        rows.append(_row('PE-01-08', 'PE-01', 'pending', 'mesh80', 20.0, 70.0, None))
        options = parse_options({})
        references = fit_reference([r for r in rows if r['partition'] == 'train'], options['neighbors'])
        groups = assess(rows, references, options)
        self.assertEqual(groups[('mesh80', ('par',))]['status'], STATUS_NO_REFERENCE)
        self.assertIsNone(groups[('mesh80', ('par',))]['standardized_radius'])


class DemoPipelineTests(unittest.TestCase):
    def test_demo_result_is_json_safe_and_has_expected_coverage(self):
        result = analyze(demo_rows(), {})
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        tables = {table['key']: table for table in result['tables']}
        groups = tables['coverage_groups']['rows']
        ready = [g for g in groups if g['status'] == STATUS_READY]
        self.assertTrue(ready)
        self.assertTrue(all(g['holdout_batches'] > 0 for g in ready))
        self.assertTrue(all(g['holdout_covered_batches'] <= g['holdout_batches'] for g in ready))
        cells = tables['missing_cells']['rows']
        suggested = [c for c in cells if c['proposed_value'] is not None]
        self.assertTrue(suggested)
        # Every cell in a ready group gets a joint interval around its suggestion.
        ready_cells = [c for c in suggested if c['coverage_status'] == STATUS_READY]
        self.assertTrue(ready_cells)
        self.assertTrue(all(c['joint_low'] <= c['proposed_value'] <= c['joint_high'] for c in ready_cells))
        # Cells without a suggestion never carry an interval.
        self.assertTrue(all(c['joint_low'] is None and c['joint_high'] is None
                            for c in cells if c['proposed_value'] is None))

    def test_calib_rows_must_be_complete(self):
        rows = demo_rows()
        normalized = normalize_rows(rows)
        self.assertTrue(any(r['partition'] == 'calib' for r in normalized))
        broken = [dict(r) for r in rows]
        broken[0]['partition'] = 'calib'
        broken[0]['temp_c'] = None
        with self.assertRaises(ValueError):
            normalize_rows(broken)

    def test_calibration_and_holdout_are_separate_partitions(self):
        from src.domain.records import normalize_rows
        rows = [dict(r) for r in demo_rows()]
        # Move a single reading of one batch to another partition: the batch would straddle.
        rows[0]['partition'] = 'calib'
        with self.assertRaises(ValueError):
            normalize_rows(rows)


if __name__ == '__main__':
    unittest.main()
