#!/usr/bin/env python3
"""Analyze fixed-baseline frame exports; never select parameters with this script.

Reacquisition starts at the first delivered projected percept within 0.5 m of
current truth during a loss. This is supporting detector evidence, not proof of
physical visibility or detector identity. Unknown/absent truth ends an episode.
"""
import argparse
import json
import math
from pathlib import Path

RADIUS = 0.5


def analyze(rows):
    initial = established = 0.0
    acquired = lost = False
    supported_since = None
    losses = 0
    delays = []
    censored = []
    unsupported_recoveries = 0
    last_time = 0.0
    for row in rows:
        time = row['time'] * 1e-9
        last_time = time
        truth = row['truth']
        if truth is None:
            if supported_since is not None:
                censored.append(time - supported_since)
            acquired = lost = False
            supported_since = None
            continue
        estimate = row['estimate']
        correct = estimate is not None and math.dist(truth, estimate['position']) <= RADIUS
        supporting = any(math.dist(truth, p['percept_in_ground']['mean']) <= RADIUS
                         for p in row['percepts'])
        if acquired and not correct:
            if not lost:
                losses += 1
                lost = True
            established += row['seconds']
        elif not correct:
            initial += row['seconds']
        if lost and supporting and supported_since is None:
            supported_since = time
        if correct:
            if lost:
                if supported_since is None:
                    unsupported_recoveries += 1
                else:
                    delays.append(time - supported_since)
            acquired = True
            lost = False
            supported_since = None
    if supported_since is not None:
        censored.append(last_time - supported_since)
    return {
        'initial_correct_ball_unavailable_seconds': initial,
        'established_correct_ball_unavailable_seconds': established,
        'established_loss_runs': losses,
        'reacquisition_delays_seconds': delays,
        'censored_reacquisition_lower_bounds_seconds': censored,
        'recoveries_without_supporting_percept': unsupported_recoveries,
    }


def summarize(entries):
    delays = [d for e in entries for d in e['availability']['reacquisition_delays_seconds']]
    result = {
        k: sum(e['availability'][k] for e in entries)
        for k in ['initial_correct_ball_unavailable_seconds',
                  'established_correct_ball_unavailable_seconds',
                  'established_loss_runs', 'recoveries_without_supporting_percept']
    }
    result.update(reacquisitions=len(delays),
                  mean_reacquisition_seconds=sum(delays) / len(delays) if delays else None,
                  maximum_reacquisition_seconds=max(delays, default=None),
                  censored_reacquisitions=sum(len(e['availability']['censored_reacquisition_lower_bounds_seconds']) for e in entries))
    for metric, duration in [('close_range_position_rmse_metres', 'close'),
                             ('motion_lag_rms_seconds', 'motion')]:
        weighted = seconds = 0.0
        for entry in entries:
            score = entry['score']
            weight = (score['close_range_present_seconds'] - score['close_range_missing_seconds']
                      if duration == 'close' else score['moving_reference_seconds'])
            if score[metric] is not None:
                weighted += score[metric] ** 2 * weight
                seconds += weight
        result[metric] = math.sqrt(weighted / seconds) if seconds else None
    for metric in ['false_track_seconds', 'wrong_track_seconds', 'correct_track_missing_seconds',
                   'close_range_correct_track_missing_seconds']:
        result[metric] = sum(e['score'][metric] for e in entries)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    entries = []
    for index, (recording, comparison) in enumerate(zip(report['training_recordings'], report['training_per_recording'], strict=True)):
        path = args.report.parent / f'baseline-training-{index}.jsonl'
        with path.open() as frames:
            availability = analyze(json.loads(line) for line in frames)
        unavailable = availability['initial_correct_ball_unavailable_seconds'] + availability['established_correct_ball_unavailable_seconds']
        score = comparison['baseline']
        if not math.isclose(unavailable, score['correct_track_missing_seconds'], abs_tol=1e-6):
            raise ValueError(f'Frame/report availability mismatch in {recording}: {unavailable} vs {score["correct_track_missing_seconds"]}')
        entries.append({'recording': recording,
                        'family': Path(recording).parent.name.split('-retry')[0],
                        'score': score, 'availability': availability})
    result = {'definition': __doc__, 'report': str(args.report),
              'aggregate': summarize(entries),
              'families': {family: summarize([e for e in entries if e['family'] == family])
                           for family in sorted({e['family'] for e in entries})},
              'recordings': entries}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
