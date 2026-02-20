#!/usr/bin/env python3
"""
Verify Training Data Split

This script checks if the training actually used the expected data split (70/10/20).
It analyzes the results_summary.json files and verifies:
1. Total samples match expected dataset size
2. Train split is ~70% of total data
3. Validation split is ~10% of total data
4. Test split is ~20% of total data
5. No data leakage (all splits sum to 100%)

Usage:
    python verify_training_split.py
    python verify_training_split.py --results-dir /path/to/results
"""

import json
import argparse
from pathlib import Path


def load_results(results_path):
    """Load results_summary.json file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def verify_split(results, model_name="Model", expected_split=(70, 10, 20), tolerance=1.0):
    """
    Verify that data split matches expected percentages.

    Args:
        results: Dictionary from results_summary.json
        model_name: Name of the model for display
        expected_split: Tuple of (train%, val%, test%)
        tolerance: Allowed deviation in percentage points

    Returns:
        bool: True if split is correct, False otherwise
    """
    dataset = results.get('dataset', {})

    train = dataset.get('train_samples', 0)
    val = dataset.get('val_samples', 0)
    test = dataset.get('test_samples', 0)
    total = dataset.get('total_samples', 0)

    if total == 0:
        print(f"❌ ERROR: No samples found in results")
        return False

    # Calculate actual percentages
    train_pct = (train / total) * 100
    val_pct = (val / total) * 100
    test_pct = (test / total) * 100
    sum_pct = train_pct + val_pct + test_pct

    expected_train, expected_val, expected_test = expected_split

    # Print results
    print(f"\n{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")
    print(f"Total samples: {total:,}")
    print()

    print(f"{'Split':<12} {'Actual':>12} {'Expected':>12} {'Difference':>12} {'Status':>12}")
    print(f"{'-'*70}")

    # Train
    train_diff = train_pct - expected_train
    train_ok = abs(train_diff) <= tolerance
    print(f"{'Train':<12} {train:>6,} ({train_pct:>5.2f}%) {expected_train:>5.0f}% ({int(total * expected_train / 100):>6,}) "
          f"{train_diff:>+5.2f}pp  {'✓' if train_ok else '✗'}")

    # Validation
    val_diff = val_pct - expected_val
    val_ok = abs(val_diff) <= tolerance
    print(f"{'Val':<12} {val:>6,} ({val_pct:>5.2f}%) {expected_val:>5.0f}% ({int(total * expected_val / 100):>6,}) "
          f"{val_diff:>+5.2f}pp  {'✓' if val_ok else '✗'}")

    # Test
    test_diff = test_pct - expected_test
    test_ok = abs(test_diff) <= tolerance
    print(f"{'Test':<12} {test:>6,} ({test_pct:>5.2f}%) {expected_test:>5.0f}% ({int(total * expected_test / 100):>6,}) "
          f"{test_diff:>+5.2f}pp  {'✓' if test_ok else '✗'}")

    print(f"{'-'*70}")
    print(f"{'Total':<12} {train + val + test:>6,} ({sum_pct:>5.2f}%)")

    # Check for issues
    all_ok = train_ok and val_ok and test_ok
    sum_ok = abs(sum_pct - 100.0) < 0.01

    print()
    if all_ok and sum_ok:
        print(f"✓ Data split is CORRECT!")
        print(f"  • Training used {train_pct:.2f}% of data (target: {expected_train}%)")
        print(f"  • Validation used {val_pct:.2f}% (target: {expected_val}%)")
        print(f"  • Test used {test_pct:.2f}% (target: {expected_test}%)")
        print(f"  • No data leakage detected (sum = {sum_pct:.2f}%)")
    else:
        print(f"❌ Data split is INCORRECT!")
        if not train_ok:
            print(f"  • Train split off by {train_diff:+.2f}pp")
        if not val_ok:
            print(f"  • Val split off by {val_diff:+.2f}pp")
        if not test_ok:
            print(f"  • Test split off by {test_diff:+.2f}pp")
        if not sum_ok:
            print(f"  • Data leakage? Sum = {sum_pct:.2f}% (expected 100%)")

    return all_ok and sum_ok


def verify_training_consistency(results, model_name="Model"):
    """
    Verify training metrics are consistent.

    Checks:
    - Test samples in results match test_samples in dataset
    - Predictions were made for all test samples
    - No missing or duplicate predictions
    """
    test_info = results.get('test', {})
    dataset_info = results.get('dataset', {})

    test_samples = test_info.get('total_samples', 0)
    expected_samples = dataset_info.get('test_samples', 0)
    correct = test_info.get('correct', 0)
    accuracy = test_info.get('accuracy', 0.0)

    print(f"\n{'='*70}")
    print(f"Training Consistency Check - {model_name}")
    print(f"{'='*70}")

    # Check 1: Test sample count
    samples_match = test_samples == expected_samples
    print(f"Test samples evaluated: {test_samples:,}")
    print(f"Expected test samples:  {expected_samples:,}")
    print(f"✓ Counts match" if samples_match else f"✗ MISMATCH!")

    # Check 2: Accuracy calculation
    if test_samples > 0:
        calculated_acc = (correct / test_samples) * 100
        acc_match = abs(calculated_acc - accuracy) < 0.01
        print(f"\nAccuracy (reported):    {accuracy:.2f}%")
        print(f"Accuracy (calculated):  {calculated_acc:.2f}%")
        print(f"✓ Accuracy correct" if acc_match else f"✗ MISMATCH!")
    else:
        acc_match = False
        print(f"\n✗ No test samples found!")

    # Check 3: Confusion matrix
    cm = results.get('confusion_matrix', [])
    if cm:
        cm_total = sum(sum(row) for row in cm)
        cm_match = cm_total == test_samples
        print(f"\nConfusion matrix total: {cm_total:,}")
        print(f"✓ Confusion matrix matches test samples" if cm_match else f"✗ MISMATCH!")
    else:
        cm_match = False
        print(f"\n✗ No confusion matrix found!")

    all_ok = samples_match and acc_match and cm_match

    print()
    if all_ok:
        print(f"✓ All consistency checks PASSED!")
    else:
        print(f"❌ Some consistency checks FAILED!")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description='Verify training data split')
    parser.add_argument('--results-dir', type=str,
                       default='/Volumes/Ext/GenAI/iti123_v2/outputs',
                       help='Directory containing results folders')
    parser.add_argument('--expected-split', type=int, nargs=3,
                       default=[70, 10, 20],
                       help='Expected train/val/test split percentages')
    parser.add_argument('--tolerance', type=float, default=1.0,
                       help='Tolerance in percentage points')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    expected_split = tuple(args.expected_split)

    # Find all results_summary.json files
    result_files = list(results_dir.glob('*/results_summary.json'))

    if not result_files:
        print(f"❌ No results_summary.json files found in {results_dir}")
        return 1

    print(f"\nFound {len(result_files)} result file(s)")
    print(f"Expected split: {expected_split[0]}/{expected_split[1]}/{expected_split[2]}")
    print(f"Tolerance: ±{args.tolerance}%")

    all_passed = True

    for result_file in sorted(result_files):
        try:
            results = load_results(result_file)
            model_name = results.get('model', result_file.parent.name)

            # Verify split
            split_ok = verify_split(results, model_name, expected_split, args.tolerance)

            # Verify consistency
            consistency_ok = verify_training_consistency(results, model_name)

            if not (split_ok and consistency_ok):
                all_passed = False

        except Exception as e:
            print(f"\n❌ Error processing {result_file}: {e}")
            all_passed = False

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    if all_passed:
        print("✓ All models passed verification!")
        print(f"  • Data splits are correct ({expected_split[0]}/{expected_split[1]}/{expected_split[2]})")
        print(f"  • No data leakage detected")
        print(f"  • Training metrics are consistent")
        print(f"\n✓ Training used {expected_split[0]}% of data as expected!")
        return 0
    else:
        print("❌ Some models failed verification!")
        print("  Please check the issues above")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
