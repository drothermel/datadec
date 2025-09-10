#!/usr/bin/env python3


from datadec import analysis_helpers


def main():
    print("=== RUN NAME PARSING SAMPLE ===\n")
    sample_runs = analysis_helpers.load_random_run_sample(20)
    print(f"Showing parsing results for {len(sample_runs)} random runs:\n")
    for _, row in sample_runs.iterrows():
        run_name = row["run_name"]
        parsed_params = analysis_helpers.extract_hyperparameters(run_name)
        analysis_helpers.print_run_name_parsed(run_name, parsed_params)


if __name__ == "__main__":
    main()
