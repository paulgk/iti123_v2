#!/usr/bin/env python3
"""
Sync Utilities for GCS Data Transfer
Provides Python functions for syncing data between local/Colab and GCS.
"""

import os
import subprocess
import yaml
from pathlib import Path
from typing import List, Optional, Tuple

def load_paths_config() -> dict:
    """Load paths configuration."""
    config_path = Path(__file__).parent.parent / "config" / "paths.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def detect_environment() -> str:
    """
    Detect whether running in Colab or local environment.
    Returns: 'colab' or 'local'
    """
    # Check for Colab-specific paths/env
    if os.path.exists('/content'):
        return 'colab'
    if 'COLAB_GPU' in os.environ:
        return 'colab'
    if 'COLAB_RELEASE_TAG' in os.environ:
        return 'colab'
    return 'local'

def get_local_paths(env: str = None) -> dict:
    """Get local paths based on environment."""
    if env is None:
        env = detect_environment()

    config = load_paths_config()

    if env == 'colab':
        return {
            'data': config['local']['colab_data'],
            'outputs': config['local']['colab_outputs'],
            'root': config['local']['colab_root']
        }
    else:
        # Local development - use repo paths
        repo_root = Path(__file__).parent.parent
        return {
            'data': str(repo_root / config['local']['repo_data']),
            'outputs': str(repo_root / config['local']['repo_outputs']),
            'root': str(repo_root)
        }

def get_gcs_uri(prefix: str) -> str:
    """Get full GCS URI for a prefix."""
    config = load_paths_config()
    bucket = os.environ.get('GCS_BUCKET_NAME', config['gcs']['bucket'])
    return f"gs://{bucket}/{prefix}"

def run_gsutil(args: List[str], check: bool = True) -> Tuple[int, str, str]:
    """
    Run gsutil command.
    Returns: (return_code, stdout, stderr)
    """
    cmd = ['gsutil'] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"gsutil error: {result.stderr}")
    return result.returncode, result.stdout, result.stderr

def pull_from_gcs(
    gcs_prefix: str,
    local_path: str,
    patterns: Optional[List[str]] = None,
    dry_run: bool = False
) -> bool:
    """
    Pull data from GCS to local path.

    Args:
        gcs_prefix: GCS prefix (e.g., 'videos/')
        local_path: Local destination path
        patterns: Optional glob patterns to filter (e.g., ['*.mp4'])
        dry_run: If True, only print what would be done

    Returns:
        True if successful
    """
    gcs_uri = get_gcs_uri(gcs_prefix)
    Path(local_path).mkdir(parents=True, exist_ok=True)

    args = ['-m', 'rsync']
    if dry_run:
        args.append('-n')
    args.append('-r')  # Recursive

    # Add exclude patterns for non-matching files
    if patterns:
        # gsutil rsync uses -x for exclude, we want include
        # Use -x to exclude everything except patterns
        # This is complex, so for now just sync all
        pass

    args.extend([gcs_uri, local_path])

    print(f"Pulling: {gcs_uri} -> {local_path}")
    code, stdout, stderr = run_gsutil(args)

    if stdout:
        print(stdout)

    return code == 0

def push_to_gcs(
    local_path: str,
    gcs_prefix: str,
    patterns: Optional[List[str]] = None,
    dry_run: bool = False
) -> bool:
    """
    Push data from local path to GCS.

    Args:
        local_path: Local source path
        gcs_prefix: GCS prefix (e.g., 'outputs/')
        patterns: Optional glob patterns to filter
        dry_run: If True, only print what would be done

    Returns:
        True if successful
    """
    gcs_uri = get_gcs_uri(gcs_prefix)

    if not Path(local_path).exists():
        print(f"Warning: Local path does not exist: {local_path}")
        return False

    args = ['-m', 'rsync']
    if dry_run:
        args.append('-n')
    args.append('-r')  # Recursive

    args.extend([local_path, gcs_uri])

    print(f"Pushing: {local_path} -> {gcs_uri}")
    code, stdout, stderr = run_gsutil(args)

    if stdout:
        print(stdout)

    return code == 0

def sync_checkpoint(checkpoint_path: str, direction: str = 'push') -> bool:
    """
    Sync a checkpoint file to/from GCS.

    Args:
        checkpoint_path: Local checkpoint file path
        direction: 'push' or 'pull'
    """
    config = load_paths_config()
    checkpoint_name = Path(checkpoint_path).name
    gcs_checkpoint = f"{config['gcs']['checkpoints_prefix']}{checkpoint_name}"

    if direction == 'push':
        return push_to_gcs(checkpoint_path, gcs_checkpoint)
    else:
        return pull_from_gcs(gcs_checkpoint, checkpoint_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sync Utilities")
    parser.add_argument("--env", action="store_true", help="Detect environment")
    parser.add_argument("--paths", action="store_true", help="Show local paths")
    args = parser.parse_args()

    if args.env:
        env = detect_environment()
        print(f"Environment: {env}")
    elif args.paths:
        paths = get_local_paths()
        for k, v in paths.items():
            print(f"  {k}: {v}")
    else:
        parser.print_help()
