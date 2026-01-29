#!/usr/bin/env python3
"""
GCS Setup and Verification Script
Sets up bucket structure for badminton ML project.
"""

import os
import yaml
from pathlib import Path
from google.cloud import storage


def load_config():
    """Load paths configuration."""
    config_path = Path(__file__).parent.parent / "config" / "paths.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_bucket(bucket_name: str = None) -> storage.Bucket:
    """Get GCS bucket, using env var or config."""
    client = storage.Client()
    if bucket_name is None:
        bucket_name = os.environ.get('GCS_BUCKET_NAME')
        if not bucket_name:
            config = load_config()
            bucket_name = config['gcs']['bucket']
    return client.bucket(bucket_name)


def setup_gcs_bucket(bucket_name: str = None) -> dict:
    """
    Create folder structure in GCS bucket.
    Returns dict with created prefixes.
    """
    bucket = get_bucket(bucket_name)
    config = load_config()

    prefixes = [
        config['gcs']['videos_prefix'],
        config['gcs']['features_prefix'],
        config['gcs']['models_prefix'],
        config['gcs']['checkpoints_prefix'],
        config['gcs']['mlflow_prefix'],
    ]

    created = []
    for prefix in prefixes:
        # Create empty blob to establish "folder"
        blob = bucket.blob(prefix + '.keep')
        if not blob.exists():
            blob.upload_from_string('')
            created.append(prefix)
            print(f"Created: gs://{bucket.name}/{prefix}")
        else:
            print(f"Exists: gs://{bucket.name}/{prefix}")

    return {"bucket": bucket.name, "prefixes": prefixes, "created": created}


def verify_gcs_access(bucket_name: str = None) -> dict:
    """
    Verify GCS bucket access and structure.
    Returns verification results.
    """
    results = {
        "authenticated": False,
        "bucket_exists": False,
        "prefixes_exist": [],
        "errors": []
    }

    try:
        bucket = get_bucket(bucket_name)
        results["authenticated"] = True

        # Check bucket exists
        if bucket.exists():
            results["bucket_exists"] = True
            results["bucket_name"] = bucket.name
        else:
            results["errors"].append(f"Bucket {bucket.name} does not exist")
            return results

        # Check prefixes
        config = load_config()
        for prefix in [config['gcs']['videos_prefix'],
                       config['gcs']['features_prefix'],
                       config['gcs']['models_prefix']]:
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
            if blobs or bucket.blob(prefix + '.keep').exists():
                results["prefixes_exist"].append(prefix)

    except Exception as e:
        results["errors"].append(str(e))

    return results


def list_bucket_contents(bucket_name: str = None, prefix: str = "") -> list:
    """List contents of bucket under prefix."""
    bucket = get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GCS Setup for Badminton ML")
    parser.add_argument("--setup", action="store_true", help="Create bucket structure")
    parser.add_argument("--verify", action="store_true", help="Verify bucket access")
    parser.add_argument("--list", type=str, help="List contents under prefix")
    parser.add_argument("--bucket", type=str, help="Override bucket name")
    args = parser.parse_args()

    if args.setup:
        result = setup_gcs_bucket(args.bucket)
        print(f"\nSetup complete: {result}")
    elif args.verify:
        result = verify_gcs_access(args.bucket)
        print(f"\nVerification: {result}")
    elif args.list is not None:
        contents = list_bucket_contents(args.bucket, args.list)
        for item in contents[:20]:  # Limit output
            print(item)
    else:
        parser.print_help()
