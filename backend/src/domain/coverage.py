"""Calibrate joint intervals per shade class and missing-field combination.

Each calibration batch yields one score: the largest standardized error among
every masked cell of that batch.  The 90% joint radius is the conformal
quantile of those batch scores, so a whole holdout batch counts as covered only
when all of its cells stay inside the interval.
"""

from .neighbors import predict
from .settings import FIELDS

ALPHA = 0.1
TARGET_COVERAGE = 1 - ALPHA
# A 90% conformal quantile needs ceil((n + 1) * 0.9) <= n usable batches.
MIN_CALIBRATION_BATCHES = 9
ORDER = list(FIELDS)

STATUS_READY = '可定区间'
STATUS_SHORT = '校准批不足'
STATUS_UNSUGGESTED = '组合内无可用建议'
STATUS_NO_REFERENCE = '同档训练读数不足'
ACTION_BY_STATUS = {STATUS_READY: '按区间交接', STATUS_SHORT: '补校准批',
                    STATUS_UNSUGGESTED: '人工补测', STATUS_NO_REFERENCE: '补同档训练批'}


def label_of(combination):
    return '+'.join(combination)


def combination_of(row):
    return tuple(field for field in ORDER if row[field] is None)


def pending_groups(rows):
    """Distinct (shade class, missing-field combination) pairs among pending rows."""
    found = set()
    for row in rows:
        if row['partition'] != 'pending':
            continue
        combination = combination_of(row)
        if combination:
            found.add((row['shade_class'], combination))
    return sorted(found, key=lambda item: (item[0], len(item[1]),
                                            [ORDER.index(field) for field in item[1]]))


def batch_score(rows, combination, references, options):
    """Largest standardized error over one batch, or None when a cell has no suggestion."""
    worst = 0.0
    missing = 0
    for row in rows:
        masked = {**row, **{field: None for field in combination}}
        reference = references.get(row['shade_class'])
        for field in combination:
            guess = predict(masked, field, references, options['neighbors'], options['min_anchors'])
            scale = reference.scales[field] if reference is not None else 0.0
            if guess.value is None or scale <= 0:
                missing += 1
                continue
            worst = max(worst, abs(row[field] - guess.value) / scale)
    if missing:
        return None, missing
    return worst, 0


def calibrated_radius(scores):
    """Conformal 90% quantile of per-batch maxima; None below the minimum batch count."""
    ordered = sorted(scores)
    if len(ordered) < MIN_CALIBRATION_BATCHES:
        return None
    # ceil((n + 1) * 0.9) in exact integer arithmetic; index into the 1-based order statistic.
    rank = ((len(ordered) + 1) * 9 + 9) // 10
    return ordered[min(max(rank, 1), len(ordered)) - 1]


def _batches(rows, shade):
    batches = {}
    for row in rows:
        if row['shade_class'] == shade:
            batches.setdefault(row['batch_id'], []).append(row)
    return batches


def _batch_rows(partition, combination, batches, references, options, radius):
    rows = []
    for batch_id, members in sorted(batches.items()):
        score, missing = batch_score(members, combination, references, options)
        rows.append({'partition': partition, 'batch_id': batch_id, 'reading_rows': len(members),
                     'standardized_max_error': score, 'missing_suggestions': missing,
                     'within_interval': None if score is None or radius is None else score <= radius})
    return rows


def assess_group(shade, combination, calibration, holdout, references, options):
    reference = references.get(shade)
    calibration_batches = _batches(calibration, shade)
    calibration_rows = _batch_rows('calib', combination, calibration_batches, references, options, None)
    usable = [row['standardized_max_error'] for row in calibration_rows
              if row['standardized_max_error'] is not None]
    no_suggestions = bool(calibration_batches) and all(row['standardized_max_error'] is None
                                                       for row in calibration_rows)
    radius = calibrated_radius(usable) if reference is not None else None
    if reference is None:
        status = STATUS_NO_REFERENCE
    elif no_suggestions:
        status = STATUS_UNSUGGESTED
    elif radius is None:
        status = STATUS_SHORT
    else:
        status = STATUS_READY

    holdout_batches = _batches(holdout, shade)
    holdout_rows = _batch_rows('holdout', combination, holdout_batches, references, options, radius)
    for row in calibration_rows:
        if row['standardized_max_error'] is not None and radius is not None:
            row['within_interval'] = row['standardized_max_error'] <= radius
    holdout_covered = None
    if radius is not None:
        holdout_covered = sum(row['within_interval'] is True for row in holdout_rows)
    missing_batches = sum(row['missing_suggestions'] > 0 for row in holdout_rows)
    holdout_total = len(holdout_batches)
    scales = {field: reference.scales[field] for field in combination} if reference is not None else {}
    half_widths = {field: radius * scales[field] for field in combination} if radius is not None else {}
    return {
        'shade_class': shade,
        'combination': label_of(combination),
        'fields': list(combination),
        'status': status,
        'action': ACTION_BY_STATUS[status],
        'target_coverage': TARGET_COVERAGE,
        'standardized_radius': radius,
        'minimum_calibration_batches': MIN_CALIBRATION_BATCHES,
        'calibration_batches': len(calibration_batches),
        'usable_calibration_batches': len(usable),
        'calibration_shortfall': max(0, MIN_CALIBRATION_BATCHES - len(usable)) if status == STATUS_SHORT else 0,
        'calibration_rows': calibration_rows,
        'holdout_batches': holdout_total,
        'holdout_covered_batches': holdout_covered,
        'holdout_coverage': (holdout_covered / holdout_total) if holdout_covered is not None and holdout_total else None,
        'holdout_missing_suggestion_batches': missing_batches,
        'holdout_rows': holdout_rows,
        'scales': scales,
        'half_widths': half_widths,
    }


def assess(rows, references, options):
    """Evaluate every pending group against separate calibration and holdout batches."""
    calibration = [row for row in rows if row['partition'] == 'calib']
    holdout = [row for row in rows if row['partition'] == 'holdout']
    groups = {}
    for shade, combination in pending_groups(rows):
        groups[(shade, combination)] = assess_group(shade, combination, calibration, holdout,
                                                    references, options)
    return groups
