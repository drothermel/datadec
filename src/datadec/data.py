import itertools
import json
import math
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from datadec import features as dd_lrs
from datadec import constants as consts
from datadec import parsing


class DataDecidePaths:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir) / "datadecide"
        self.dataset_dir = self.data_dir / "datasets"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.ds_details_path = self.data_dir / "dataset_features.csv"

    def parquet_path(self, name: str) -> Path:
        return self.data_dir / f"{name}.parquet"

    def dataset_path(self, max_params_str: str) -> Path:
        return self.dataset_dir / f"dataset_{max_params_str}.pkl"

    # ------------ DataDecide Raw Paths ------------
    @property
    def ppl_eval_raw_path(self) -> Path:
        return self.parquet_path("ppl_eval")

    @property
    def downstream_eval_raw_path(self) -> Path:
        return self.parquet_path("downstream_eval")

    # ------------ DataDecide Parsed Paths ------------
    @property
    def step_to_token_compute_path(self) -> Path:
        return self.parquet_path("step_to_token_compute")

    @property
    def ppl_eval_parsed_path(self) -> Path:
        return self.parquet_path("ppl_eval_parsed")

    @property
    def downstream_eval_parsed_path(self) -> Path:
        return self.parquet_path("downstream_eval_parsed")

    @property
    def full_eval_ds_path(self) -> Path:
        return self.parquet_path("full_eval")

    @property
    def mean_eval_ds_path(self) -> Path:
        return self.parquet_path("mean_eval")

    @property
    def std_eval_ds_path(self) -> Path:
        return self.parquet_path("std_eval")


class DataDecideDefaults:
    def __init__(self):
        self.model_configs = {}
        self.fill_all_model_configs()

    @property
    def model_details_df(self):
        return pd.DataFrame(self.model_configs).T.infer_objects()

    def make_model_config(self, model_size_str: str, **kwargs):
        mc = {**consts.MODEL_CONFIG_BASE, "params": model_size_str, "model_size_str": model_size_str, **kwargs}
        mc["model_size"] = int(self.parse_model_size_str(mc["model_size_str"]))
        mc["batch_size"] = int(self.calc_batch_size(mc["model_size_str"]))
        mc["lr_max"] = self.calc_lr_max(mc["model_size_str"])
        mc["lr_final"] = consts.LR_FINAL_RATIO * mc["lr_max"]
        mc["warmup_tokens"] = int(self.calc_warmup_tokens(mc["model_size_str"]))
        mc["total_tokens"] = int(
            self.parse_token_length_str(mc["length_str"], mc["model_size_str"])
        )
        mc["lr_decay_tokens"] = int(mc["total_tokens"] - mc["warmup_tokens"])
        mc["total_seqs"] = int(round(mc["total_tokens"] / consts.MAX_SEQ_LEN))
        mc["total_steps"] = int(
            math.ceil(mc["total_tokens"] / (mc["batch_size"] * consts.MAX_SEQ_LEN))
        )
        mc["warmup_perc"] = mc["warmup_tokens"] / mc["total_tokens"]
        mc["warmup_steps"] = int(
            math.ceil(mc["warmup_tokens"] / (mc["batch_size"] * consts.MAX_SEQ_LEN))
        )
        mc["lr_decay_steps"] = int(mc["total_steps"] - mc["warmup_steps"])
        mc["theoretical_tokens_per_step"] = int(
            round(consts.MAX_SEQ_LEN * mc["batch_size"])
        )
        mc["10_perc_lrdecay_steps"] = int(round(mc["lr_decay_steps"] * 0.1))
        mc["early_window_10p_end_step"] = (
            mc["warmup_steps"] + mc["10_perc_lrdecay_steps"]
        )
        mc["early_window_perc"] = mc["early_window_10p_end_step"] / mc["total_steps"]
        return mc

    def fill_all_model_configs(self):
        for param_str, cfg in consts.MODEL_SHAPES.items():
            self.model_configs[param_str] = self.make_model_config(param_str, **cfg)

    def parse_model_size_str(self, size_str: str) -> int:
        return consts.HARDCODED_SIZE_MAPPING[size_str]

    def parse_token_length_str(self, length_str: str, model_size_str: str) -> int:
        model_size = self.parse_model_size_str(model_size_str)
        length_in_tokens, length_unit = consts.NUMBER_UNIT_RE.match(
            length_str.strip().upper()
        ).groups()  # type: ignore
        assert length_unit == "XC"
        length_in_tokens = int(length_in_tokens)
        return length_in_tokens * 20 * model_size

    def calc_batch_size(self, model_size_str: str) -> int:
        assert consts.MAX_SEQ_LEN == 2_048
        model_size = self.parse_model_size_str(model_size_str)
        batch_size = 160 * (model_size / consts.MODEL_SIZE_NORM_VALUE) ** consts.BS_EXPONENT
        rounding_size = consts.GPUS_PER_NODE * consts.MICROBATCH_SIZE
        batch_size /= rounding_size
        batch_size = round(batch_size)
        batch_size *= rounding_size
        return batch_size

    def calc_lr_max(self, model_size_str: str) -> float:
        model_size = self.parse_model_size_str(model_size_str)
        return (
            consts.LR_MAX_BASE
            * (model_size / consts.MODEL_SIZE_NORM_VALUE) ** consts.LR_EXPONENT
        )

    def calc_warmup_tokens(self, model_size_str: str) -> int:
        model_size = self.parse_model_size_str(model_size_str)
        bs = self.calc_batch_size(model_size_str)
        # model_size / bs = num_warmup_steps
        # (model_size / bs) * max_seq_len = num_warmup_tokens
        return round(model_size / (bs / consts.MAX_SEQ_LEN))


class DataDecide:
    def __init__(self, data_dir: str = "./data", force_reload=False, verbose=True):
        self.paths = DataDecidePaths(data_dir)
        self.defaults = DataDecideDefaults()
        self.data_names = []

        # Prep for df management
        self._setup_dfs = {}
        self._loaded_dfs = {}
        self.setup_all_dfs(force_reload=force_reload, verbose=verbose)
        self._loaded_dfs["ds_details_df"] = self.load_ds_details_df()

    @property
    def all_data_param_combos(self):
        return list(
            itertools.product(
                consts.ALL_DATA_NAMES,
                consts.ALL_PARAM_STRS,
            )
        )

    @property
    def ppl_raw_df(self):
        assert "ppl_raw_df" in self._setup_dfs, "ppl_raw_df not setup"
        if "ppl_raw_df" not in self._loaded_dfs:
            self._loaded_dfs["ppl_raw_df"] = pd.read_parquet(
                self._setup_dfs["ppl_raw_df"]
            )
        return self._loaded_dfs["ppl_raw_df"]

    @property
    def dwn_raw_df(self):
        assert "dwn_raw_df" in self._setup_dfs, "dwn_raw_df not setup"
        if "dwn_raw_df" not in self._loaded_dfs:
            self._loaded_dfs["dwn_raw_df"] = pd.read_parquet(
                self._setup_dfs["dwn_raw_df"]
            )
        return self._loaded_dfs["dwn_raw_df"]

    @property
    def ppl_parsed_df(self):
        assert "ppl_parsed_df" in self._setup_dfs, "ppl_parsed_df not setup"
        if "ppl_parsed_df" not in self._loaded_dfs:
            self._loaded_dfs["ppl_parsed_df"] = pd.read_parquet(
                self._setup_dfs["ppl_parsed_df"]
            )
        return self._loaded_dfs["ppl_parsed_df"]

    @property
    def dwn_parsed_df(self):
        assert "dwn_parsed_df" in self._setup_dfs, "dwn_parsed_df not setup"
        if "dwn_parsed_df" not in self._loaded_dfs:
            self._loaded_dfs["dwn_parsed_df"] = pd.read_parquet(
                self._setup_dfs["dwn_parsed_df"]
            )
        return self._loaded_dfs["dwn_parsed_df"]

    @property
    def step_to_token_compute_df(self):
        if "step_to_token_compute_df" in self._setup_dfs:
            self._loaded_dfs["step_to_token_compute_df"] = pd.read_parquet(
                self._setup_dfs["step_to_token_compute_df"]
            )
        return self._loaded_dfs["step_to_token_compute_df"]

    @property
    def full_eval_ds(self):
        assert "full_eval_ds" in self._setup_dfs, "full_eval_ds not setup"
        if "full_eval_ds" not in self._loaded_dfs:
            self._loaded_dfs["full_eval_ds"] = pd.read_parquet(
                self._setup_dfs["full_eval_ds"]
            )
        return self._loaded_dfs["full_eval_ds"]

    @property
    def mean_eval_ds(self):
        assert "mean_eval_ds" in self._setup_dfs, "mean_eval_ds not setup"
        if "mean_eval_ds" not in self._loaded_dfs:
            self._loaded_dfs["mean_eval_ds"] = pd.read_parquet(
                self._setup_dfs["mean_eval_ds"]
            )
        return self._loaded_dfs["mean_eval_ds"]

    @property
    def std_eval_ds(self):
        assert "std_eval_ds" in self._setup_dfs, "std_eval_ds not setup"
        if "std_eval_ds" not in self._loaded_dfs:
            self._loaded_dfs["std_eval_ds"] = pd.read_parquet(
                self._setup_dfs["std_eval_ds"]
            )
        return self._loaded_dfs["std_eval_ds"]

    @property
    def ds_details_df(self):
        return self._loaded_dfs["ds_details_df"]

    @property
    def model_details_df(self):
        return self.defaults.model_details_df

    # ------------ Dataframe Manipulation Helpers ------------

    def filter_by_max_step_to_use(self, df):
        df = df.copy()
        df["max_step_to_use"] = df["params"].map(consts.MAX_STEP_TO_USE)
        return df[df["step"] <= df["max_step_to_use"]]

    def merge_in_ds_and_model_details(self, input_df: pd.DataFrame):
        return input_df.merge(
            self.ds_details_df,
            on="data",
            how="left",
        ).merge(
            self.model_details_df,
            on="params",
            how="left",
        )

    def get_max_ppl_vals(self, df: pd.DataFrame):
        ppl_cols = consts.PPL_TYPES
        return df[ppl_cols].max().reset_index()

    def set_step_val_to_max_ppl_val(self, df: pd.DataFrame, step: int = 0):
        ppl_cols = consts.PPL_TYPES
        max_ppl_vals = self.get_max_ppl_vals(df)
        df = df.copy()
        step_mask = df["step"] == step
        for col in ppl_cols:
            na_mask = df[col].isna()
            df.loc[step_mask & na_mask, col] = max_ppl_vals[col][0]
        return df

    # ------------ Dataframe Management ------------

    def load_ds_details_df(self):
        df = pd.read_csv(self._setup_dfs["ds_details_df"]).rename(
            columns={
                "dataset": "data",
            }
        )
        df["data"] = (
            df["data"]
            .str.replace("Dolma1.7 (no math code)", "Dolma1.7 (no math, code)")
            .str.replace("DCLM-Baseline (QC 7%", "DCLM-Baseline (QC 7%,")
        )
        return df

    def setup_all_dfs(
        self,
        force_reload=False,
        verbose=False,
    ):
        self._setup_dfs["ds_details_df"] = self.paths.ds_details_path
        # Step 1: Download raw dfs
        if not self.paths.ppl_eval_raw_path.exists() or force_reload:
            if verbose:
                print("Downloading raw dfs")
            ppl_dataset = load_dataset(
                consts.HF_DATASET_NAMES["perplexity_eval_ds"], split="train"
            )
            ppl_dataset.to_parquet(self.paths.ppl_eval_raw_path)

        self._setup_dfs["ppl_raw_df"] = self.paths.ppl_eval_raw_path
        if not self.paths.downstream_eval_raw_path.exists() or force_reload:
            dwn_dataset = load_dataset(
                consts.HF_DATASET_NAMES["downstream_eval_ds"], split="train"
            )
            dwn_dataset.to_parquet(self.paths.downstream_eval_raw_path)

        self._setup_dfs["dwn_raw_df"] = self.paths.downstream_eval_raw_path

        # Step 2: Extract per-param step-to-token and step-to-compute mapping
        if not self.paths.step_to_token_compute_path.exists() or force_reload:
            if verbose:
                print("Extracting step-to-token and step-to-compute mapping")
            dwn_df = pd.read_parquet(self._setup_dfs["dwn_raw_df"])
            step_to_token_compute_df = parsing.make_step_to_token_compute_df(dwn_df)
            step_to_token_compute_df.to_parquet(self.paths.step_to_token_compute_path)
        self._setup_dfs["step_to_token_compute_df"] = (
            self.paths.step_to_token_compute_path
        )

        # Step 3: Parse eval dfs
        if not self.paths.ppl_eval_parsed_path.exists() or force_reload:
            ppl_df = pd.read_parquet(self._setup_dfs["ppl_raw_df"])
            ppl_parsed_df = parsing.parse_perplexity_dataframe(ppl_df)
            ppl_parsed_df.to_parquet(self.paths.ppl_eval_parsed_path)
        self._setup_dfs["ppl_parsed_df"] = self.paths.ppl_eval_parsed_path
        if not self.paths.downstream_eval_parsed_path.exists() or force_reload:
            if verbose:
                print("Parsing eval dfs, this may take a while...")
            dwn_df = pd.read_parquet(self._setup_dfs["dwn_raw_df"])
            dwn_parsed_df = parsing.parse_downstream_dataframe(dwn_df)
            dwn_parsed_df.to_parquet(self.paths.downstream_eval_parsed_path)
        self._setup_dfs["dwn_parsed_df"] = self.paths.downstream_eval_parsed_path

        # Step 4: Create full eval df
        if not self.paths.full_eval_ds_path.exists() or force_reload:
            full_eval_ds = self.create_full_eval_df(
                self.dwn_parsed_df,
                self.ppl_parsed_df,
                self.step_to_token_compute_df,
            )
            full_eval_ds.to_parquet(self.paths.full_eval_ds_path)
        self._setup_dfs["full_eval_ds"] = self.paths.full_eval_ds_path

        # Step 5: create mean and std eval dfs
        if (
            not self.paths.mean_eval_ds_path.exists()
            or not self.paths.std_eval_ds_path.exists()
            or force_reload
        ):
            mean_eval_ds, std_eval_ds = self.create_mean_std_df(self.full_eval_ds)
            mean_eval_ds.to_parquet(self.paths.mean_eval_ds_path)
            std_eval_ds.to_parquet(self.paths.std_eval_ds_path)
        self._setup_dfs["mean_eval_ds"] = self.paths.mean_eval_ds_path
        self._setup_dfs["std_eval_ds"] = self.paths.std_eval_ds_path
        self._setup_dfs["model_details_df"] = None
        self._loaded_dfs = {
            "model_details_df": self.defaults.model_details_df,
        }
        print(">> Finished setting up DataDecide dataframes.")

    def load_df(self, df_name: str) -> None:
        assert df_name in self._setup_dfs, f"df_name {df_name} not setup"
        if df_name not in self._loaded_dfs:
            self._loaded_dfs[df_name] = pd.read_parquet(self._setup_dfs[df_name])

    def index_dfs(self, df_name, params, data, step):
        if df_name not in self._loaded_dfs:
            self._loaded_dfs[df_name] = pd.read_parquet(self._setup_dfs[df_name])
        df = self._loaded_dfs[df_name]
        return df[
            (df["params"] == params) & (df["data"] == data) & (df["step"] == step)
        ]


    def create_full_eval_df(
        self,
        dwn_parsed_df: pd.DataFrame,
        ppl_parsed_df: pd.DataFrame,
        step_to_token_compute_df: pd.DataFrame,
    ) -> pd.DataFrame:
        merged_df = dwn_parsed_df.merge(
            ppl_parsed_df,
            on=["params", "data", "seed", "step"],
            how="outer",
            suffixes=["_dwn", "_ppl"],
        )
        merged_df = (
            merged_df.merge(step_to_token_compute_df, on="params", how="left")
            .assign(
                tokens=lambda x: x["step"] * x["tokens_per_step"],
                compute=lambda x: x["step"] * x["compute_per_step"],
            )
            .drop(columns=["tokens_per_step", "compute_per_step"])
        )
        merged_df = parsing.reorder_df_cols(merged_df, consts.KEY_COLS + ["tokens", "compute"])
        return merged_df

    def create_mean_std_df(
        self, merged_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        group_cols_no_seed = [c for c in consts.KEY_COLS if c != "seed"]
        mean_df = (
            merged_df.drop(columns=["seed"])
            .groupby(group_cols_no_seed)
            .mean(numeric_only=True)
            .reset_index()
        )
        std_df = (
            merged_df.drop(columns=["seed"])
            .groupby(group_cols_no_seed)
            .std(numeric_only=True)
            .reset_index()
        )
        return mean_df, std_df


def get_data_recipe_family(data_name: str, data_recipe_families: dict = None) -> str:
    if data_recipe_families is None:
        data_recipe_families = consts.DATA_RECIPE_FAMILIES
    for family, names in data_recipe_families.items():
        if data_name in names:
            return family
    return "unknown"


def param_to_numeric(param_str: str) -> float:
    if param_str.endswith("M"):
        return float(param_str[:-1]) * 1e6
    elif param_str.endswith("B"):
        return float(param_str[:-1]) * 1e9
    else:
        # Try to convert directly if it's already numeric
        try:
            return float(param_str)
        except ValueError:
            raise ValueError(f"Cannot parse parameter string: {param_str}")


def select_by_data_param_combos(
    df: pd.DataFrame,
    data_param_combos: list[tuple[str, str]],
    just_params: bool = False,
    just_data: bool = False,
):
    # Create a mask for each specific (data, params) combination
    mask = pd.Series([False] * len(df), index=df.index)

    for data, params in data_param_combos:
        if just_params:
            assert not just_data, "Cannot specify both just_params and just_data"
            combo_mask = df["params"] == params
        elif just_data:
            assert not just_params, "Cannot specify both just_params and just_data"
            combo_mask = df["data"] == data
        else:
            combo_mask = (df["data"] == data) & (df["params"] == params)
        mask = mask | combo_mask

    return df[mask]


def prep_base_df(
    data_dir: str = "./data",
    force_reload: bool = False,
    filter_by_max_step: bool = True,
    add_lr_cols: bool = True,
    return_means: bool = True,
    min_params: str = "10M",
    verbose: bool = False,
):
    dd = DataDecide(data_dir=data_dir, force_reload=force_reload, verbose=verbose)
    base_df = dd.full_eval_ds.copy()
    if verbose:
        print(
            f">> Initial shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
        )

    if min_params is not None:
        min_params_numeric = param_to_numeric(min_params)
        base_df = base_df[
            base_df["params"].apply(lambda x: param_to_numeric(x) >= min_params_numeric)
        ]
        if verbose:
            print(
                f">> Above min params {min_params} shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
            )

    if filter_by_max_step:
        base_df = dd.filter_by_max_step_to_use(base_df)
        if verbose:
            print(
                f">> Filter by max step shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
            )

    # Must merge before calculating lr cols
    base_df = dd.merge_in_ds_and_model_details(base_df)
    if verbose:
        print(
            f">> Merge in ds and model details shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
        )

    if add_lr_cols:
        base_df = dd_lrs.add_lr_cols(base_df)
        if verbose:
            print(
                f">> Add lr cols shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
            )

    # Calculate means as last step
    if return_means:
        base_df, _ = dd.create_mean_std_df(base_df)
        if verbose:
            print(
                f">> Create mean std df shape: {base_df.shape[0]:,} rows x {base_df.shape[1]:,} cols"
            )
    return base_df
