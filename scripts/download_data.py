"""Download and process DataDecide datasets."""

import argparse
from datadecide_loader.data import prep_base_df


def main():
    """Download and process DataDecide datasets."""
    parser = argparse.ArgumentParser(description="Download and process DataDecide datasets.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to store the downloaded and processed data.",
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="Force re-downloading and processing of the data.",
    )
    args = parser.parse_args()

    print(f"Downloading and processing DataDecide datasets to {args.data_dir}...")
    prep_base_df(data_dir=args.data_dir, force_reload=args.force_reload, verbose=True)
    print("Done.")


if __name__ == "__main__":
    main()
