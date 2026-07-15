#!/usr/bin/env python3
"""
REMOVE: This utility only selected samples for the removed gate-trace JSON
feature. FedAvg and the learned-gate training path do not call it. Delete this
file after review.

List CVDN trace candidates by scan to help choose inst_idx values for SPM tracing.

Examples:
  python cvdn_src/list_trace_candidates.py --scan 5q7pvUzZiYa --splits train --max_rows 20
  python cvdn_src/list_trace_candidates.py --splits val_seen --contains kitchen --output_csv /tmp/cvdn_candidates.csv
"""

import argparse
import csv
import json
import os


def load_datasets(splits):
    data = []
    for split in splits:
        if split not in {'train', 'val_seen', 'val_unseen', 'test'}:
            raise ValueError(f'Unsupported split: {split}')
        fpath = os.path.join('datasets', 'CVDN', 'annotations', f'{split}.json')
        with open(fpath, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    return data


def format_instruction(item, history):
    dialog_history = item.get('dialog_history', [])
    target = item.get('target', '')

    if history == 'none':
        return ''
    if history == 'target' or len(dialog_history) == 0:
        return '<TAR> ' + target
    if history == 'oracle_ans':
        ora_a = dialog_history[-1]['message']
        return '<ORA> ' + ora_a + ' <TAR> ' + target
    if history == 'nav_q_oracle_ans':
        nav_q = dialog_history[-2]['message']
        ora_a = dialog_history[-1]['message']
        return '<NAV> ' + nav_q + ' <ORA> ' + ora_a + ' <TAR> ' + target
    if history == 'all':
        segments = []
        for turn in dialog_history:
            prefix = '<NAV>' if turn['role'] == 'navigator' else '<ORA>'
            segments.append(f"{prefix} {turn['message']}")
        segments.append(f'<TAR> {target}')
        return ' '.join(segments)
    raise ValueError(f'Unsupported history mode: {history}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='List CVDN trace candidates and their inst_idx values.'
    )
    parser.add_argument('--splits', type=str, default='train',
                        help='Comma-separated splits: train,val_seen,val_unseen,test')
    parser.add_argument('--scan', type=str, default='',
                        help='Optional scan filter')
    parser.add_argument('--inst_idx', type=int, default=-1,
                        help='Optional exact inst_idx filter')
    parser.add_argument('--history', type=str, default='all',
                        choices=['none', 'target', 'oracle_ans', 'nav_q_oracle_ans', 'all'],
                        help='How to reconstruct the instruction preview')
    parser.add_argument('--contains', type=str, default='',
                        help='Optional case-insensitive substring filter on the preview text')
    parser.add_argument('--max_rows', type=int, default=50,
                        help='Maximum number of rows to print')
    parser.add_argument('--preview_chars', type=int, default=120,
                        help='Maximum preview length for console output')
    parser.add_argument('--output_csv', type=str, default='',
                        help='Optional CSV path to save all matched rows')
    return parser.parse_args()


def main():
    args = parse_args()
    splits = [split.strip() for split in args.splits.split(',') if split.strip()]
    rows = []
    needle = args.contains.strip().lower()

    for item in load_datasets(splits):
        if args.scan and item.get('scan') != args.scan:
            continue
        if args.inst_idx >= 0 and int(item.get('inst_idx', -1)) != args.inst_idx:
            continue

        preview = format_instruction(item, args.history)
        if needle and needle not in preview.lower():
            continue

        rows.append({
            'split_hint': ','.join(splits),
            'scan': item.get('scan', ''),
            'inst_idx': int(item.get('inst_idx', -1)),
            'path_id': item.get('path_id', ''),
            'target': item.get('target', ''),
            'turns': len(item.get('dialog_history', [])),
            'preview': preview,
        })

    rows.sort(key=lambda row: (row['scan'], row['inst_idx']))

    if args.output_csv:
        with open(args.output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['split_hint', 'scan', 'inst_idx', 'path_id', 'target', 'turns', 'preview'],
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f'[INFO] matched rows: {len(rows)}')
    if not rows:
        return

    print('scan                     inst_idx   turns  target               preview')
    print('-' * 120)
    for row in rows[:args.max_rows]:
        preview = row['preview'].replace('\n', ' ').strip()
        if len(preview) > args.preview_chars:
            preview = preview[:args.preview_chars - 3] + '...'
        target = str(row['target'])
        if len(target) > 20:
            target = target[:17] + '...'
        print(f"{row['scan']:<24} {row['inst_idx']:>8}  {row['turns']:>5}  {target:<20} {preview}")

    first = rows[0]
    print('\n[INFO] example trace command:')
    print(
        f"TRACE_SCAN={first['scan']} TRACE_INST_IDX={first['inst_idx']} TRACE_PHASE=both ./run_SPM.bash"
    )
    if args.output_csv:
        print(f'[INFO] CSV saved to: {args.output_csv}')


if __name__ == '__main__':
    main()