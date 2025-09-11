#!/usr/bin/env python3
"""
Debug script to compare WandB API tag data with our stored data.
"""

import wandb
from datadec.wandb_eval.wandb_store import WandBStore


def debug_tags():
    print("=== WandB Tag Debug Comparison ===\n")

    # Connect to WandB API directly
    api = wandb.Api()

    # Connect to our database
    store = WandBStore("postgresql+psycopg://localhost/wandb_test")
    stored_runs = store.get_runs(entity="ml-moe", project="ft-scaling")

    print(f"Stored runs in database: {len(stored_runs)}")

    # Get a few sample runs from WandB API
    print("\n=== DIRECT API COMPARISON ===")
    api_runs = list(api.runs("ml-moe/ft-scaling"))
    print(f"API runs available: {len(api_runs)}")

    # Compare tags for first 5 runs
    print("\n=== SAMPLE TAG COMPARISON ===")

    for i, api_run in enumerate(api_runs[:5]):
        print(f"\nRun {i + 1}: {api_run.id}")
        print(f"  API tags: {api_run.tags}")

        # Find corresponding stored run
        stored_run = stored_runs[stored_runs["run_id"] == api_run.id]
        if not stored_run.empty:
            stored_tags = stored_run["wandb_tags"].iloc[0]
            print(f"  Stored tags: {stored_tags}")

            # Check if they match
            if api_run.tags:
                api_tag_str = ",".join(sorted(api_run.tags))
                if stored_tags != api_tag_str:
                    print("  ❌ MISMATCH!")
                else:
                    print("  ✅ Match")
            else:
                if stored_tags:
                    print(f"  ❌ API has no tags but we have: {stored_tags}")
                else:
                    print("  ✅ Both have no tags")
        else:
            print("  ⚠️ Run not found in database")

    # Check all unique tags from API
    print("\n=== ALL UNIQUE TAGS FROM API ===")
    all_api_tags = set()
    tagged_runs_count = 0

    for run in api_runs:
        if run.tags:
            tagged_runs_count += 1
            all_api_tags.update(run.tags)

    print(f"API runs with tags: {tagged_runs_count}/{len(api_runs)}")
    print(f"Unique tags from API: {len(all_api_tags)}")
    print("API tags:")
    for tag in sorted(all_api_tags):
        # Count runs with this tag
        count = sum(1 for run in api_runs if run.tags and tag in run.tags)
        print(f"  {tag}: {count} runs")


if __name__ == "__main__":
    debug_tags()
