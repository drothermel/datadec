#!/usr/bin/env python3
"""
On-demand WandB tag synchronization script.

Updates tags for runs in our database by re-downloading from WandB API.
Supports different sync modes for efficiency.
"""

import argparse
from datetime import datetime, timedelta
from datadec.wandb_eval.wandb_store import WandBStore
from datadec.wandb_eval.wandb_downloader import WandBDownloader
from sqlalchemy import text


def update_tags_in_batches(store: WandBStore, runs_list: list, batch_size: int = 50):
    """Update tags for a list of runs in batches."""
    # Single connection for all updates
    with store.engine.connect() as conn:
        for i in range(0, len(runs_list), batch_size):
            batch = runs_list[i : i + batch_size]
            print(
                f"\n=== Processing batch {i // batch_size + 1}/{(len(runs_list) + batch_size - 1) // batch_size} ({len(batch)} runs) ==="
            )

            # Collect all updates for this batch
            updates = []
            for j, run in enumerate(batch):
                if (i + j + 1) % 50 == 0 or (i + j + 1) == len(runs_list):
                    percent = ((i + j + 1) / len(runs_list)) * 100
                    print(
                        f"  Progress: {i + j + 1}/{len(runs_list)} ({percent:.1f}%) - {run.name[:50]}..."
                    )

                if run.tags:
                    tags_str = ",".join(run.tags)
                    updates.append({"tags": f'"{tags_str}"', "run_id": run.id})

            # Execute all updates in this batch
            if updates:
                conn.execute(
                    text(
                        "UPDATE wandb_runs SET raw_data = jsonb_set(raw_data, '{wandb_tags}', :tags) WHERE run_id = :run_id"
                    ),
                    updates,
                )
                conn.commit()
                print(f"  Updated {len(updates)} runs with tags")
            else:
                print("  No tags found in this batch")

            print(f"Batch {i // batch_size + 1} completed. Rate limiting pause...")
            import time

            time.sleep(1)  # Rate limiting between batches


def sync_tags(
    entity: str, project: str, mode: str = "recent", days: int = 7, force: bool = False
):
    """
    Synchronize WandB tags for runs in the database.

    Args:
        entity: WandB entity name
        project: WandB project name
        mode: Sync mode - "recent", "all", or "finished"
        days: Number of recent days to sync (for "recent" mode)
        force: Force refresh all runs regardless of state
    """
    print("=== WandB Tag Synchronization ===")
    print(f"Entity: {entity}")
    print(f"Project: {project}")
    print(f"Mode: {mode}")
    if mode == "recent":
        print(f"Days: {days}")
    print(f"Force refresh: {force}")
    print()

    store = WandBStore("postgresql+psycopg://localhost/wandb_test")
    downloader = WandBDownloader(store)

    if mode == "all":
        print("Performing full refresh of all runs...")
        all_runs = list(downloader.api.runs(f"{entity}/{project}"))

        print(f"Found {len(all_runs)} total runs")
        stats = {
            "total_runs": len(all_runs),
            "new_runs": 0,
            "updated_runs": len(all_runs),
        }

        update_tags_in_batches(store, all_runs)

    elif mode == "recent":
        print(f"Refreshing runs from last {days} days...")
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        all_runs = list(downloader.api.runs(f"{entity}/{project}"))
        recent_runs = []

        for run in all_runs:
            if run.created_at:
                run_date = (
                    run.created_at.isoformat()
                    if hasattr(run.created_at, "isoformat")
                    else str(run.created_at)
                )
                if run_date > cutoff_date:
                    recent_runs.append(run)

        print(f"Found {len(recent_runs)} runs from last {days} days")
        stats = {
            "total_runs": len(recent_runs),
            "new_runs": 0,
            "updated_runs": len(recent_runs),
        }

        update_tags_in_batches(store, recent_runs)

    elif mode == "finished":
        print("Refreshing all finished runs...")
        all_runs = list(downloader.api.runs(f"{entity}/{project}"))
        finished_runs = [run for run in all_runs if run.state == "finished"]

        print(f"Found {len(finished_runs)} finished runs")
        stats = {
            "total_runs": len(finished_runs),
            "new_runs": 0,
            "updated_runs": len(finished_runs),
        }

        update_tags_in_batches(store, finished_runs)

    print("\nSync completed:")
    print(f"  Total runs processed: {stats['total_runs']}")
    print(f"  New runs: {stats['new_runs']}")
    print(f"  Updated runs: {stats['updated_runs']}")

    # Show tag statistics
    runs_df = store.get_runs(entity=entity, project=project)
    if "wandb_tags" in runs_df.columns:
        tagged_runs = runs_df["wandb_tags"].notna().sum()
        total_runs = len(runs_df)
        print(
            f"  Runs with tags: {tagged_runs}/{total_runs} ({tagged_runs / total_runs * 100:.1f}%)"
        )

        # Show unique tags
        all_tags = set()
        for tag_string in runs_df["wandb_tags"].dropna():
            if isinstance(tag_string, str):
                tags = [tag.strip() for tag in tag_string.split(",")]
                all_tags.update(tags)
        print(f"  Unique tags found: {len(all_tags)}")
        if len(all_tags) <= 20:
            print(f"  Tags: {sorted(all_tags)}")


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize WandB tags for runs in database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync tags for runs from last 3 days
  python scripts/sync_wandb_tags.py ml-moe ft-scaling --mode recent --days 3
  
  # Sync all finished runs
  python scripts/sync_wandb_tags.py ml-moe ft-scaling --mode finished
  
  # Force refresh all runs (full sync)
  python scripts/sync_wandb_tags.py ml-moe ft-scaling --mode all
        """,
    )

    parser.add_argument("entity", help="WandB entity name")
    parser.add_argument("project", help="WandB project name")
    parser.add_argument(
        "--mode",
        choices=["recent", "finished", "all"],
        default="recent",
        help="Sync mode: recent (last N days), finished (all finished runs), all (force refresh everything)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of recent days to sync (only for 'recent' mode)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh regardless of existing state",
    )

    args = parser.parse_args()

    try:
        sync_tags(
            entity=args.entity,
            project=args.project,
            mode=args.mode,
            days=args.days,
            force=args.force,
        )
    except Exception as e:
        print(f"Error during sync: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
