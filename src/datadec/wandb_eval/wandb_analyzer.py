from typing import Dict, List, Optional, Tuple

import pandas as pd

from datadec.wandb_eval import wandb_constants as wconsts
from datadec.wandb_eval import wandb_transforms as transforms
from datadec.wandb_eval.wandb_loader import WandBDataLoader


class WandBDataAnalyzer:
    def __init__(self, loader: WandBDataLoader = None):
        self.loader = loader or WandBDataLoader()

    def analyze_column_categorization(self, runs_df: pd.DataFrame) -> Dict:
        filtered_df = transforms.filter_broken_initial_testing_runs(runs_df)
        filtered_df = transforms.drop_wandb_constant_ignored_cols(filtered_df)
        filtered_df = transforms.convert_objects_and_normalize_dtypes(filtered_df)

        oe_cols, rest_cols = self._split_oe_cols_vs_rest(filtered_df.columns.tolist())
        pretrain_cols, rest_cols = self._split_pretrain_eval_cols_vs_rest(rest_cols)
        categorized_cols, rest_cols = self._categorize_columns_by_key_sets(rest_cols)
        object_cols, nonobject_cols = self._split_obj_vs_nonobj_cols(
            filtered_df[rest_cols]
        )

        assert len(object_cols) == 0, (
            f"Found {len(object_cols)} unexpected object columns after conversion: {object_cols}"
        )

        if nonobject_cols:
            nonobj_df = filtered_df[nonobject_cols]
            col_categories = self._filter_constant_and_nanconstant_cols(nonobj_df)
            truly_uncategorized = col_categories["other"]
        else:
            col_categories = {
                "all_nan": [],
                "all_constant": [],
                "constant_or_nan": [],
                "other": [],
            }
            truly_uncategorized = []

        return {
            "filtered_df": filtered_df,
            "object_cols": object_cols,
            "all_nan_cols": col_categories["all_nan"],
            "all_constant_cols": col_categories["all_constant"],
            "constant_or_nan_cols": col_categories["constant_or_nan"],
            "pretrain_cols": pretrain_cols,
            "oe_cols": oe_cols,
            "truly_uncategorized": truly_uncategorized,
            **categorized_cols,
        }

    def generate_comprehensive_report(
        self, runs_df: Optional[pd.DataFrame] = None
    ) -> str:
        if runs_df is None:
            runs_df, _ = self.loader.load_runs_and_history()

        result = self.analyze_column_categorization(runs_df)
        filtered_df = result["filtered_df"]

        report_lines = [
            "=== SEQUENTIAL FILTERING AND CATEGORIZATION ===\n",
            f"1. Initial runs: {len(runs_df)}",
            f"   After filtering early test runs: {len(filtered_df)}",
            "",
            "2. Semantic categorization:",
            f"   OE eval columns: {len(result['oe_cols'])}",
            f"   Pretrain eval columns: {len(result['pretrain_cols'])}",
            "",
            "3. Key-based semantic groups:",
        ]

        semantic_total = 0
        for cat in wconsts.KEY_SETS.keys():
            if result.get(cat):
                semantic_total += len(result[cat])
                report_lines.append(f"   {cat}: {len(result[cat])}")

        report_lines.extend(
            [
                "",
                "4. After semantic categorization:",
                f"   Object columns remaining: {len(result['object_cols'])}",
                f"   Total semantically categorized: {semantic_total + len(result['oe_cols']) + len(result['pretrain_cols'])}",
                "",
                "5. Constant/NaN filtering on remaining columns:",
                f"   All NaN columns: {len(result['all_nan_cols'])}",
                f"   All constant columns: {len(result['all_constant_cols'])}",
                f"   Constant or NaN columns: {len(result['constant_or_nan_cols'])}",
                f"   Truly uncategorized: {len(result['truly_uncategorized'])}",
            ]
        )

        if result["truly_uncategorized"]:
            report_lines.extend(
                [
                    "",
                    f"6. Truly uncategorized columns analysis ({len(result['truly_uncategorized'])}):",
                ]
            )
            for i, col in enumerate(result["truly_uncategorized"], 1):
                if col in filtered_df.columns:
                    col_data = filtered_df[col]
                    n_unique = col_data.nunique(dropna=False)
                    pct_non_nan = (col_data.notna().sum() / len(col_data)) * 100
                    sample_values = col_data.dropna().unique()[:5].tolist()
                    if len(sample_values) == 0:
                        sample_values = ["All NaN"]
                    report_lines.append(
                        f"   {i:2d}. {col:<35} | unique: {n_unique:3d} | non-NaN: {pct_non_nan:5.1f}% | sample: {sample_values}"
                    )

        report_lines.extend(
            [
                "",
                "=== WANDB TAGS ===",
            ]
        )

        if "wandb_tags" in filtered_df.columns:
            tags_data = filtered_df["wandb_tags"].dropna()
            if len(tags_data) > 0:
                unique_tags = tags_data.unique()
                report_lines.append(f"Unique wandb_tags ({len(unique_tags)}):")
                for tag in sorted(unique_tags):
                    report_lines.append(f"   {tag}")
            else:
                report_lines.append("No wandb_tags data available")
        else:
            report_lines.append("wandb_tags column not found")

        report_lines.extend(
            [
                "",
                "=== ALL-NAN COLUMNS ===",
            ]
        )
        if result["all_nan_cols"]:
            report_lines.append(f"All-NaN columns ({len(result['all_nan_cols'])}):")
            for col in result["all_nan_cols"]:
                report_lines.append(f"   {col}")
        else:
            report_lines.append("No all-NaN columns")

        report_lines.extend(
            [
                "",
                "=== CONSTANT/NAN COLUMNS WITH VALUES ===",
            ]
        )
        if result["constant_or_nan_cols"]:
            report_lines.append(
                f"Constant or NaN columns ({len(result['constant_or_nan_cols'])}):"
            )
            for col in result["constant_or_nan_cols"]:
                if col in filtered_df.columns:
                    col_data = filtered_df[col]
                    unique_values = col_data.dropna().unique()
                    if len(unique_values) > 0:
                        constant_value = unique_values[0]
                        report_lines.append(f"   {col:<35} = {constant_value}")
                    else:
                        report_lines.append(f"   {col:<35} = [All NaN]")

        return "\n".join(report_lines)

    def analyze_nan_patterns(self, df: pd.DataFrame) -> Dict:
        nan_analysis = {}
        for col in df.columns:
            total_rows = len(df)
            nan_count = df[col].isna().sum()
            nan_pct = (nan_count / total_rows) * 100
            nan_analysis[col] = {
                "nan_count": nan_count,
                "total_rows": total_rows,
                "nan_percentage": nan_pct,
            }
        return nan_analysis

    def analyze_wandb_tags(self, df: pd.DataFrame) -> Dict:
        if "wandb_tags" not in df.columns:
            return {"error": "wandb_tags column not found"}

        tags_data = df["wandb_tags"].dropna()
        if len(tags_data) == 0:
            return {"unique_tags": [], "count": 0}

        unique_tags = tags_data.unique()
        return {
            "unique_tags": sorted(unique_tags),
            "count": len(unique_tags),
            "sample_rows": len(tags_data),
        }

    def _split_oe_cols_vs_rest(self, cols: List[str]) -> Tuple[List[str], List[str]]:
        oe_cols = [col for col in cols if col.startswith("oe_eval_metrics/")]
        rest_cols = [col for col in cols if col not in oe_cols]
        return oe_cols, rest_cols

    def _split_pretrain_eval_cols_vs_rest(
        self, cols: List[str]
    ) -> Tuple[List[str], List[str]]:
        pretrain_cols = [col for col in cols if col.startswith("pretrain_eval")]
        rest_cols = [col for col in cols if col not in pretrain_cols]
        return pretrain_cols, rest_cols

    def _categorize_columns_by_key_sets(
        self, all_cols: List[str]
    ) -> Tuple[Dict, List[str]]:
        categorized = {name: [] for name in wconsts.KEY_SETS.keys()}
        remaining_cols = list(all_cols)
        for category_name, key_list in wconsts.KEY_SETS.items():
            matched_cols = [col for col in remaining_cols if col in key_list]
            categorized[category_name] = matched_cols
            remaining_cols = [col for col in remaining_cols if col not in matched_cols]
        return categorized, remaining_cols

    def _split_obj_vs_nonobj_cols(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        object_columns = []
        nonobject_columns = []
        for col in df.columns:
            if df[col].dtype == "object":
                object_columns.append(col)
            else:
                nonobject_columns.append(col)
        return object_columns, nonobject_columns

    def _filter_constant_and_nanconstant_cols(self, df: pd.DataFrame) -> Dict:
        all_nan_columns = []
        all_constant_columns = []
        constant_or_nan_columns = []
        other_columns = []

        for col in df.columns:
            if df[col].dtype == "object":
                assert False, "Filter object columns before finding constants"

            nunique_with_nan = df[col].nunique(dropna=False)
            has_nan = df[col].isna().any()

            if nunique_with_nan == 0:
                all_nan_columns.append(col)
            elif nunique_with_nan == 1:
                if has_nan:
                    all_nan_columns.append(col)
                else:
                    all_constant_columns.append(col)
            elif nunique_with_nan == 2 and has_nan:
                constant_or_nan_columns.append(col)
            else:
                other_columns.append(col)

        return {
            "all_nan": all_nan_columns,
            "all_constant": all_constant_columns,
            "constant_or_nan": constant_or_nan_columns,
            "other": other_columns,
        }


def main():
    analyzer = WandBDataAnalyzer()
    report = analyzer.generate_comprehensive_report()
    print(report)


if __name__ == "__main__":
    main()
