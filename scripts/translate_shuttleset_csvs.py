#!/usr/bin/env python3
"""
Translate ShuttleSet CSV Shot Types from Chinese to English

This script translates the 'type' column in ShuttleSet CSV files from Chinese
to English using the mapping from ShuttleSet dataset analysis.

Usage:
    # Translate all CSVs in a directory
    python scripts/translate_shuttleset_csvs.py --input ShuttleSet/set --output ShuttleSet/set_english

    # Translate in-place (overwrites original files)
    python scripts/translate_shuttleset_csvs.py --input ShuttleSet/set --in-place

    # Dry run to preview translations
    python scripts/translate_shuttleset_csvs.py --input ShuttleSet/set --dry-run
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict

# Translation mapping from SHUTTLESET_DATASET_ANALYSIS.md
SHOT_TYPE_TRANSLATION: Dict[str, str] = {
    # Overhead Shots
    '殺球': 'Smash',
    '長球': 'Clear',
    '挑球': 'Lift',
    '過度切球': 'Overhead_Slice',

    # Drop Shots
    '放小球': 'Drop',
    '切球': 'Slice_Drop',
    '點扣': 'Steep_Smash',

    # Net Shots
    '擋小球': 'Block',
    '撲球': 'Net_Kill',
    '勾球': 'Cross_Net',

    # Drive/Push Shots
    '推球': 'Push',
    '平球': 'Drive',
    '後場抽平球': 'Rear_Drive',
    '小平球': 'Short_Drive',

    # Defensive Returns
    '防守回抽': 'Defensive_Drive',
    '防守回挑': 'Defensive_Lift',

    # Service
    '發短球': 'Short_Serve',
    '發長球': 'Long_Serve',

    # Unknown
    '未知球種': 'Unknown'
}


def translate_csv(input_path: Path, output_path: Path, dry_run: bool = False) -> Dict:
    """
    Translate shot types in a CSV file from Chinese to English

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        dry_run: If True, only count translations without writing

    Returns:
        Dictionary with translation statistics
    """
    stats = {
        'total_rows': 0,
        'translated': 0,
        'untranslated': 0,
        'chinese_types': {},
        'english_types': {}
    }

    # Read CSV
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    stats['total_rows'] = len(rows)

    # Translate 'type' column
    for row in rows:
        if 'type' in row:
            original_type = row['type']

            # Track original
            stats['chinese_types'][original_type] = stats['chinese_types'].get(original_type, 0) + 1

            # Translate
            if original_type in SHOT_TYPE_TRANSLATION:
                translated_type = SHOT_TYPE_TRANSLATION[original_type]
                row['type'] = translated_type
                stats['translated'] += 1
                stats['english_types'][translated_type] = stats['english_types'].get(translated_type, 0) + 1
            else:
                stats['untranslated'] += 1
                # Keep original if no translation found
                stats['english_types'][original_type] = stats['english_types'].get(original_type, 0) + 1

    # Write translated CSV (unless dry run)
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return stats


def translate_directory(input_dir: Path, output_dir: Path = None,
                        in_place: bool = False, dry_run: bool = False):
    """
    Translate all CSV files in a directory

    Args:
        input_dir: Input directory containing CSV files
        output_dir: Output directory (if not in_place)
        in_place: If True, overwrite original files
        dry_run: If True, only preview translations
    """
    if in_place:
        output_dir = input_dir
    elif output_dir is None and not dry_run:
        raise ValueError("Must specify output_dir or use --in-place")
    elif output_dir is None:
        # Dry run without output dir - use input dir as placeholder
        output_dir = input_dir

    # Find all CSV files recursively
    csv_files = list(input_dir.rglob('*.csv'))

    if len(csv_files) == 0:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")
    print(f"Mode: {'DRY-RUN' if dry_run else ('IN-PLACE' if in_place else 'COPY')}")
    print()

    # Aggregate statistics
    total_stats = {
        'files_processed': 0,
        'total_rows': 0,
        'translated': 0,
        'untranslated': 0,
        'chinese_types': {},
        'english_types': {}
    }

    for input_path in csv_files:
        # Determine output path
        rel_path = input_path.relative_to(input_dir)
        output_path = output_dir / rel_path

        # Translate
        try:
            stats = translate_csv(input_path, output_path, dry_run)

            # Update totals
            total_stats['files_processed'] += 1
            total_stats['total_rows'] += stats['total_rows']
            total_stats['translated'] += stats['translated']
            total_stats['untranslated'] += stats['untranslated']

            for chinese_type, count in stats['chinese_types'].items():
                total_stats['chinese_types'][chinese_type] = \
                    total_stats['chinese_types'].get(chinese_type, 0) + count

            for english_type, count in stats['english_types'].items():
                total_stats['english_types'][english_type] = \
                    total_stats['english_types'].get(english_type, 0) + count

            # Progress
            action = "Preview" if dry_run else ("Updated" if in_place else "Translated")
            print(f"✓ {action}: {rel_path} ({stats['total_rows']} rows, "
                  f"{stats['translated']} translated)")

        except Exception as e:
            print(f"✗ Error processing {rel_path}: {e}")

    # Summary
    print()
    print("="*60)
    print("TRANSLATION SUMMARY")
    print("="*60)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Total rows: {total_stats['total_rows']}")
    print(f"Translated: {total_stats['translated']}")
    print(f"Untranslated: {total_stats['untranslated']}")

    print()
    print("Chinese shot types found:")
    for chinese_type, count in sorted(total_stats['chinese_types'].items(),
                                     key=lambda x: x[1], reverse=True):
        translation = SHOT_TYPE_TRANSLATION.get(chinese_type, "NO TRANSLATION")
        print(f"  {chinese_type:12s} -> {translation:20s} ({count:5d} occurrences)")

    print()
    print("English shot types after translation:")
    for english_type, count in sorted(total_stats['english_types'].items(),
                                     key=lambda x: x[1], reverse=True):
        print(f"  {english_type:20s} ({count:5d} occurrences)")

    if dry_run:
        print()
        print("⚠️  DRY-RUN mode - no files were modified")
        print("   Remove --dry-run to execute translations")
    elif in_place:
        print()
        print("✓ Translation complete (files updated in-place)")
    else:
        print()
        print(f"✓ Translation complete (files written to {output_dir})")


def main():
    parser = argparse.ArgumentParser(
        description='Translate ShuttleSet CSV shot types from Chinese to English',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing CSV files')
    parser.add_argument('--output', type=str,
                       help='Output directory for translated CSVs (required if not --in-place)')
    parser.add_argument('--in-place', action='store_true',
                       help='Translate in-place (overwrite original files)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview translations without modifying files')

    args = parser.parse_args()

    # Validate arguments
    if not args.in_place and not args.output and not args.dry_run:
        parser.error("Must specify --output or use --in-place (or --dry-run for preview)")

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else None

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Run translation
    translate_directory(input_dir, output_dir, args.in_place, args.dry_run)

    return 0


if __name__ == '__main__':
    exit(main())
