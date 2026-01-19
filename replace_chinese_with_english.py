#!/usr/bin/env python3
"""
Replace all Chinese stroke names with English in metadata CSV files

This script updates:
1. drop_smash_metadata.csv
2. lift_smash_metadata.csv
3. multiclass_4stroke_metadata.csv

Replaces Chinese stroke names in 'type' column with English equivalents.
"""

import pandas as pd
from pathlib import Path

# Chinese to English mapping
TRANSLATIONS = {
    '殺球': 'Smash',
    '長球': 'Clear',
    '挑球': 'Lift',
    '放小球': 'Drop',
    '點扣': 'Drop_Smash',
    '擋小球': 'Block',
    '推球': 'Push',
    '平球': 'Drive',
    '勾球': 'Cross_Net',
    '撲球': 'Rush',
    '發短球': 'Short_Serve',
    '防守回挑': 'Defensive_Lift',
    '防守回抽': 'Defensive_Drive',
    '過度切球': 'Cut',
    '後場抽平球': 'Rear_Drive',
    '切球': 'Slice',
    '未知球種': 'Unknown',
    '發長球': 'Long_Serve',
    '小平球': 'Small_Drive',
    # Common Chinese terms in other columns
    '對手落地致勝': 'Opponent_winning_landing',
    '落地致勝': 'Winning_landing',
    '對手出界': 'Opponent_out_of_bounds',
    '出界': 'Out_of_bounds',
    '掛網': 'Net_error',
    '對手掛網': 'Opponent_net_error',
    '未過網': 'Did_not_cross_net',
    '對手未過網': 'Opponent_did_not_cross_net',
}

def replace_chinese_in_file(filepath):
    """Replace all Chinese text in a CSV file with English"""
    print(f"Processing: {filepath}")

    # Read CSV
    df = pd.read_csv(filepath)

    # Count Chinese occurrences before replacement
    chinese_count = 0
    for col in df.columns:
        if df[col].dtype == 'object':  # String columns only
            for chinese, english in TRANSLATIONS.items():
                count = df[col].astype(str).str.contains(chinese, regex=False).sum()
                if count > 0:
                    chinese_count += count
                    print(f"  Column '{col}': '{chinese}' → '{english}' ({count} occurrences)")
                    # Replace
                    df[col] = df[col].astype(str).str.replace(chinese, english, regex=False)

    if chinese_count == 0:
        print("  ✓ No Chinese text found (already English)")
    else:
        # Save back to file
        df.to_csv(filepath, index=False)
        print(f"  ✓ Replaced {chinese_count} Chinese terms with English")

    print()
    return chinese_count

def main():
    print("=" * 80)
    print("REPLACE CHINESE TEXT WITH ENGLISH")
    print("=" * 80)
    print()

    # Files to process
    files = [
        Path("data/processed/clips/drop_smash_metadata.csv"),
        Path("data/processed/clips/lift_smash_metadata.csv"),
        Path("data/processed/clips/multiclass_4stroke_metadata.csv"),
    ]

    total_replaced = 0

    for filepath in files:
        if filepath.exists():
            count = replace_chinese_in_file(filepath)
            total_replaced += count
        else:
            print(f"⚠️  File not found: {filepath}")
            print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total Chinese terms replaced: {total_replaced}")

    if total_replaced > 0:
        print()
        print("✓ All metadata files now use English text only")
        print()
        print("Verify the changes:")
        print("  head -5 data/processed/clips/drop_smash_metadata.csv")
    else:
        print()
        print("✓ All files already use English text")

    print()

if __name__ == "__main__":
    main()
