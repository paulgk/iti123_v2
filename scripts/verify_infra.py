#!/usr/bin/env python3
"""
Infrastructure Verification Script
Validates all Phase 1 infrastructure components.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

def check(name: str, condition: bool, details: str = "") -> Tuple[bool, str]:
    """Format a check result."""
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {name}"
    if details:
        msg += f": {details}"
    return condition, msg

def verify_python() -> List[Tuple[bool, str]]:
    """Verify Python environment."""
    results = []

    # Check Python version
    version = sys.version_info
    is_310 = version.major == 3 and version.minor == 10
    results.append(check(
        "Python 3.10",
        is_310,
        f"Found {version.major}.{version.minor}.{version.micro}"
    ))

    # Check TensorFlow
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        is_tf215 = tf_version.startswith("2.15")
        results.append(check("TensorFlow 2.15.x", is_tf215, tf_version))
    except ImportError as e:
        results.append(check("TensorFlow", False, str(e)))

    # Check MediaPipe
    try:
        import mediapipe as mp
        mp_version = mp.__version__
        is_mp0109 = mp_version == "0.10.9"
        results.append(check("MediaPipe 0.10.9", is_mp0109, mp_version))
    except ImportError as e:
        results.append(check("MediaPipe", False, str(e)))

    # Check protobuf (critical for MediaPipe)
    try:
        import google.protobuf
        pb_version = google.protobuf.__version__
        is_pb320 = pb_version.startswith("3.20")
        results.append(check("Protobuf 3.20.x", is_pb320, pb_version))
    except ImportError as e:
        results.append(check("Protobuf", False, str(e)))

    # Check MLflow
    try:
        import mlflow
        results.append(check("MLflow", True, mlflow.__version__))
    except ImportError as e:
        results.append(check("MLflow", False, str(e)))

    return results

def verify_git_lfs() -> List[Tuple[bool, str]]:
    """Verify Git LFS configuration."""
    results = []

    # Check Git LFS installed
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True, text=True
        )
        lfs_installed = result.returncode == 0
        results.append(check(
            "Git LFS installed",
            lfs_installed,
            result.stdout.strip() if lfs_installed else result.stderr
        ))
    except FileNotFoundError:
        results.append(check("Git LFS installed", False, "git-lfs not found"))

    # Check .gitattributes exists
    gitattributes = REPO_ROOT / ".gitattributes"
    results.append(check(
        ".gitattributes exists",
        gitattributes.exists()
    ))

    # Check LFS tracking patterns
    if gitattributes.exists():
        content = gitattributes.read_text()
        has_h5 = "*.h5" in content and "filter=lfs" in content
        has_pkl = "models/" in content and "filter=lfs" in content
        results.append(check("LFS tracks .h5 files", has_h5))
        results.append(check("LFS tracks models/*.pkl", has_pkl))

    return results

def verify_gcs() -> List[Tuple[bool, str]]:
    """Verify GCS configuration."""
    results = []

    # Check credentials
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    has_creds = creds_path and Path(creds_path).exists()
    results.append(check(
        "GCS credentials set",
        has_creds,
        creds_path if has_creds else "GOOGLE_APPLICATION_CREDENTIALS not set"
    ))

    # Check config file
    paths_config = REPO_ROOT / "config" / "paths.yaml"
    results.append(check("config/paths.yaml exists", paths_config.exists()))

    # Check gsutil available
    try:
        result = subprocess.run(
            ["gsutil", "version"],
            capture_output=True, text=True
        )
        results.append(check(
            "gsutil installed",
            result.returncode == 0,
            result.stdout.split('\n')[0] if result.returncode == 0 else ""
        ))
    except FileNotFoundError:
        results.append(check("gsutil installed", False, "gsutil not found"))

    # Try to verify bucket access (only if credentials set)
    if has_creds:
        try:
            from scripts.gcs_setup import verify_gcs_access
            gcs_result = verify_gcs_access()
            results.append(check(
                "GCS bucket accessible",
                gcs_result.get('bucket_exists', False),
                gcs_result.get('bucket_name', '')
            ))
        except Exception as e:
            results.append(check("GCS bucket accessible", False, str(e)))

    return results

def verify_mlflow() -> List[Tuple[bool, str]]:
    """Verify MLflow configuration."""
    results = []

    # Check config file
    mlflow_config = REPO_ROOT / "config" / "mlflow.yaml"
    results.append(check("config/mlflow.yaml exists", mlflow_config.exists()))

    # Check MLflow can initialize
    try:
        import mlflow
        from scripts.mlflow_config import setup_mlflow
        # Don't actually create experiment, just check config loads
        results.append(check("MLflow config loadable", True))
    except Exception as e:
        results.append(check("MLflow config loadable", False, str(e)))

    return results

def verify_sync_scripts() -> List[Tuple[bool, str]]:
    """Verify sync scripts exist and are executable."""
    results = []

    scripts = [
        "scripts/pull_data.sh",
        "scripts/push_results.sh",
        "scripts/sync_utils.py",
        "scripts/colab_setup.sh",
        "scripts/run_in_colab.py"
    ]

    for script in scripts:
        script_path = REPO_ROOT / script
        exists = script_path.exists()
        executable = os.access(script_path, os.X_OK) if exists else False
        results.append(check(
            f"{script} exists",
            exists
        ))
        if script.endswith('.sh'):
            results.append(check(
                f"{script} executable",
                executable
            ))

    return results

def verify_all() -> Dict[str, List[Tuple[bool, str]]]:
    """Run all verification checks."""
    return {
        "Python Environment": verify_python(),
        "Git LFS": verify_git_lfs(),
        "Google Cloud Storage": verify_gcs(),
        "MLflow": verify_mlflow(),
        "Sync Scripts": verify_sync_scripts()
    }

def print_results(results: Dict[str, List[Tuple[bool, str]]]) -> int:
    """Print results and return exit code (0 if all pass)."""
    all_pass = True

    print("\n" + "=" * 60)
    print("INFRASTRUCTURE VERIFICATION REPORT")
    print("=" * 60)

    for section, checks in results.items():
        print(f"\n## {section}")
        print("-" * 40)
        for passed, msg in checks:
            print(f"  {msg}")
            if not passed:
                all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("STATUS: ALL CHECKS PASSED")
    else:
        print("STATUS: SOME CHECKS FAILED")
        print("\nNote: Some checks may fail in local environment.")
        print("Re-run in Colab after setup for full verification.")
    print("=" * 60 + "\n")

    return 0 if all_pass else 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify Phase 1 Infrastructure")
    parser.add_argument("--python", action="store_true", help="Check Python only")
    parser.add_argument("--lfs", action="store_true", help="Check Git LFS only")
    parser.add_argument("--gcs", action="store_true", help="Check GCS only")
    parser.add_argument("--mlflow", action="store_true", help="Check MLflow only")
    parser.add_argument("--scripts", action="store_true", help="Check scripts only")
    args = parser.parse_args()

    # Run specific or all checks
    if any([args.python, args.lfs, args.gcs, args.mlflow, args.scripts]):
        results = {}
        if args.python:
            results["Python Environment"] = verify_python()
        if args.lfs:
            results["Git LFS"] = verify_git_lfs()
        if args.gcs:
            results["Google Cloud Storage"] = verify_gcs()
        if args.mlflow:
            results["MLflow"] = verify_mlflow()
        if args.scripts:
            results["Sync Scripts"] = verify_sync_scripts()
    else:
        results = verify_all()

    sys.exit(print_results(results))
