#!/usr/bin/env python3
"""
Terminal Script Runner for Colab
Executes Python scripts in terminal mode (not Jupyter).
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """Ensure environment is properly configured."""
    # Add repo root to path
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Set environment variables for Colab
    if os.path.exists('/content'):
        os.environ.setdefault('COLAB_ENV', 'true')

        # Suppress TensorFlow warnings in terminal mode
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    return repo_root

def run_script(script_path: str, args: list = None, capture_output: bool = False):
    """
    Run a Python script in terminal mode.

    Args:
        script_path: Path to Python script
        args: Additional arguments to pass to script
        capture_output: If True, capture and return output

    Returns:
        (return_code, stdout, stderr) if capture_output else return_code
    """
    repo_root = setup_environment()

    # Resolve script path
    script = Path(script_path)
    if not script.is_absolute():
        # Try relative to repo root
        script = repo_root / script_path
        if not script.exists():
            # Try relative to scripts/
            script = repo_root / 'scripts' / script_path

    if not script.exists():
        print(f"Error: Script not found: {script_path}")
        return 1 if not capture_output else (1, "", "Script not found")

    # Build command
    cmd = [sys.executable, str(script)]
    if args:
        cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    else:
        result = subprocess.run(cmd)
        return result.returncode

def run_command(command: str):
    """Run a shell command."""
    print(f"Running: {command}")
    return subprocess.run(command, shell=True).returncode

def activate_venv():
    """Activate Python 3.10 venv if in Colab."""
    venv_path = '/content/venv/bin/activate'
    if os.path.exists(venv_path):
        # This only works in shell context, not in Python
        print(f"Note: Run 'source {venv_path}' to activate venv")
        return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Python scripts in terminal mode for Colab"
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="Script to run (relative to repo root or scripts/)"
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Arguments to pass to script"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Just setup environment, don't run anything"
    )
    parser.add_argument(
        "--shell",
        type=str,
        help="Run a shell command instead of Python script"
    )

    args = parser.parse_args()

    if args.setup:
        repo_root = setup_environment()
        print(f"Environment configured. Repo root: {repo_root}")
        activate_venv()
        sys.exit(0)

    if args.shell:
        sys.exit(run_command(args.shell))

    if not args.script:
        parser.print_help()
        sys.exit(1)

    sys.exit(run_script(args.script, args.args))
