"""Generate per-cell handover items, preparation statistics, and tabular evidence."""

from .coverage import combination_of
from .neighbors import predict
from .settings import FIELDS

CELL_COLUMNS = ['reading_id', 'batch_id', 'field', 'shade_class', 'proposed_value', 'donors',
                'nearest_distance', 'reason', 'joint_low', 'joint_high', 'coverage_status',
                'holdout_coverage', 'action']
GROUP_COLUMNS = ['shade_class', 'combination', 'status', 'action', 'standardized_radius',
                 'minimum_calibration_batches', 'calibration_batches', 'usable_calibration_batches',
                 'calibration_shortfall', 'holdout_batches', 'holdout_covered_batches', 'holdout_coverage',
                 'holdout_missing_suggestion_batches']
SCORE_COLUMNS = ['shade_class', 'combination', 'batch_id', 'reading_rows',
                 'standardized_max_error', 'missing_suggestions', 'within_interval']


def make_table(key, title, columns, rows):
    data = [{name: round(value, 6) if isinstance(value, float) else value
             for name, value in row.items()} for row in rows]
    return {'key': key, 'title': title, 'columns': columns, 'rows': data}


def handover(rows, references, options, groups):
    cells = []
    for row in rows:
        for field in FIELDS:
            if row[field] is not None:
                continue
            guess = predict(row, field, references, options['neighbors'], options['min_anchors'])
            group = groups.get((row['shade_class'], combination_of(row)))
            low = high = None
            if guess.value is not None and group is not None and group['standardized_radius'] is not None:
                margin = group['half_widths'].get(field)
                if margin is not None:
                    low, high = guess.value - margin, guess.value + margin
            cells.append({'reading_id': row['reading_id'], 'batch_id': row['batch_id'], 'field': field,
                          'shade_class': row['shade_class'], 'proposed_value': guess.value,
                          'donors': guess.donors, 'nearest_distance': guess.nearest, 'reason': guess.reason,
                          'joint_low': low, 'joint_high': high,
                          'coverage_status': group['status'] if group is not None else '缺少缺测组合',
                          'holdout_coverage': group['holdout_coverage'] if group is not None else None,
                          'action': group['action'] if group is not None else '待定区间'})
    return cells


def _group_rows(groups):
    return [{key: group[key] for key in GROUP_COLUMNS} for group in groups.values()]


def _score_rows(groups, partition):
    rows = []
    for group in groups.values():
        for row in group['calibration_rows' if partition == 'calib' else 'holdout_rows']:
            rows.append({'shade_class': group['shade_class'], 'combination': group['combination'],
                         **{key: row[key] for key in SCORE_COLUMNS[2:]}})
    return rows


def assemble(cells, details, scores, references, groups):
    ready = sum(row['proposed_value'] is not None for row in cells)
    compared = sum(row['compared'] for row in scores)
    frozen = [{'shade_class': shade, 'field': field, 'training_rows': len(ref.rows),
               'training_scale': ref.scales[field], 'training_median': ref.medians[field]}
              for shade, ref in sorted(references.items()) for field in FIELDS]
    group_rows = _group_rows(groups)
    interval_groups = sum(group['standardized_radius'] is not None for group in groups.values())
    short_groups = sum(group['status'] == '校准批不足' for group in groups.values())
    unsuggested_groups = sum(group['status'] == '组合内无可用建议' for group in groups.values())
    holdout_total = sum(group['holdout_batches'] for group in groups.values()
                        if group['holdout_covered_batches'] is not None)
    holdout_covered = sum(group['holdout_covered_batches'] for group in groups.values()
                          if group['holdout_covered_batches'] is not None)
    batch_coverage = (holdout_covered / holdout_total) if holdout_total else None
    batch_note = f'整批留出覆盖 {holdout_covered}/{holdout_total}' if holdout_total else '暂无整批留出覆盖证据'
    short_note = f'{short_groups} 个组合校准批不足需补测' if short_groups else '校准批均可用'
    if unsuggested_groups:
        short_note += f'，{unsuggested_groups} 个组合在现有参数下无可用建议'
    return {
        'summary': f'{len(details)} 个留出读数单元完成遮挡，其中 {compared} 个可进行同口径对比。'
                   f'{len(cells)} 个待补单元中，{ready} 个有建议值，{len(cells)-ready} 个需人工补测。'
                   f'{len(groups)} 个缺测组合中 {interval_groups} 个已定九成联合区间，{batch_note}；'
                   f'{short_note}。',
        'metrics': {'留出遮挡单元': len(details), '有效配对覆盖率': round(compared / len(details), 6),
                    '可建议单元': ready, '待人工补测单元': len(cells) - ready,
                    '缺测组合数': len(groups), '可定联合区间组合数': interval_groups,
                    '校准不足组合数': short_groups, '无建议组合数': unsuggested_groups,
                    '整批留出批数': holdout_total,
                    '整批覆盖批数': holdout_covered,
                    '整批覆盖率': round(batch_coverage, 6) if batch_coverage is not None else None},
        'warnings': ['遮挡回测评估单字段随机缺测，设备断连造成的多列缺测可能没有足够近邻。',
                     '九成联合区间按遮阳网档位与缺测组合冻结于本次运行；晚到的校准数据只用于新一轮试算，'
                     '已签核交班表继续下载本次运行的原区间与依据。'],
        'tables': [
            make_table('missing_cells', '缺测建议交班', CELL_COLUMNS, cells),
            make_table('coverage_groups', '缺测组合联合区间', GROUP_COLUMNS, group_rows),
            make_table('calibration_scores', '校准批标准化误差', SCORE_COLUMNS,
                       _score_rows(groups, 'calib')),
            make_table('holdout_coverage', '独立留出整批覆盖', SCORE_COLUMNS,
                       _score_rows(groups, 'holdout')),
            make_table('holdout_mae', '分字段遮挡误差', ['field', 'attempted', 'compared', 'knn_mae', 'median_mae'], scores),
            make_table('masked_holdout', '留出遮挡明细', ['reading_id', 'batch_id', 'field', 'actual',
                       'predicted', 'baseline', 'reason'], details),
            make_table('frozen_reference', '训练尺度与基准', ['shade_class', 'field', 'training_rows',
                       'training_scale', 'training_median'], frozen),
        ],
    }
