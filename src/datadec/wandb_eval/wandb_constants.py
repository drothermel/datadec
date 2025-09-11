# ---------- From parsing.py ----------
TIME_KEYS = ["created_at", "timestamp"]
EARLIEST_GOOD_RUN_DATE = "2025-08-21"

# Output paths
PRETRAIN_POSTTRAIN_DF_PATH = "./data/pretrain_posttrain.pkl"

# OE Evaluation task and metric constants
OE_EVAL_TASKS = [
    "csqa",
    "piqa",
    "boolq",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "socialiqa",
    "openbookqa",
    "winogrande",
    "core_9mcqa:rc::olmes:full",
]

OE_EVAL_METRICS = [
    "acc_raw",
    "acc_uncond",
    "acc_per_byte",
    "acc_per_char",
    "acc_per_token",
    "num_instances",
    "primary_score",
    "sum_logits_corr",
    "bits_per_byte_corr",
    "logits_per_char_corr",
    "logits_per_token_corr",
    "acc_raw_macro",
    "acc_raw_micro",
    "acc_per_byte_macro",
    "acc_per_byte_micro",
    "acc_per_char_macro",
    "acc_per_char_micro",
    "acc_per_token_macro",
    "acc_per_token_micro",
    "primary_score_macro",
    "primary_score_micro",
    "sum_logits_corr_macro",
    "sum_logits_corr_micro",
    "bits_per_byte_corr_macro",
    "bits_per_byte_corr_micro",
    "logits_per_char_corr_macro",
    "logits_per_char_corr_micro",
    "logits_per_token_corr_macro",
    "logits_per_token_corr_micro",
]

CONSTANT_OR_NAN_COLS = [
    "add_bos",
    "use_fast",
    "use_lora",
    "lora_rank",
    "use_qlora",
    "lora_alpha",
    "push_to_hub",
    "lora_dropout",
    "weight_decay",
    "training_step",
    "with_tracking",
    "clip_grad_norm",
    "model_revision",
    "use_flash_attn",
    "fused_optimizer",
    "overwrite_cache",
    "get_tokenizer_fn",
    "sft_messages_key",
    "use_liger_kernel",
    "ground_truths_key",
    "low_cpu_mem_usage",
    "trust_remote_code",
    "cache_dataset_only",
    "dataset_skip_cache",
    "use_8bit_optimizer",
    "use_slow_tokenizer",
    "wandb_project_name",
    "load_balancing_loss",
    "concatenated_forward",
    "gradient_checkpointing",
    "keep_last_n_checkpoints",
    "try_auto_save_to_beaker",
    "try_launch_beaker_eval_jobs",
    "do_not_randomize_output_dir",
]

KEY_SETS = {
    "id_cols": [
        "run_id",
        "run_name",
        "project",
        "entity",
        "wandb_entity",
        "exp_name",
    ],
    "status_cols": ["state", "created_at", "runtime", "timeout"],
    "pretrain_hpm_cols": [
        "pretrain_steps",
        "pretrain_tokens",
        "pretrain_sequences",
        "pretrain_compute",
        "model_pretrain_steps",
        "model_pretrain_compute",
    ],
    "core_hpm_cols": [
        "seed",
        "model_size",
        "model_name_or_path",
        "learning_rate",
        "num_train_epochs",
        "max_train_samples",
        "finetune_sequences",
        "finetune_unique_sequences",
        "total_sequences",
        "reduce_loss",
        "lr_scheduler_type",
        "tokenizer_name_or_path",
        "warmup_ratio",
        "max_seq_length",
        "checkpointing_steps",
        "load_balancing_weight",
        "preprocessing_num_workers",
        "gradient_accumulation_steps",
        "per_device_train_batch_size",
    ],
    "x_axis_cols": [
        "step",
        "epoch",
        "total_tokens",
        "_step",
        "_runtime",
        "_timestamp",
        "total_compute_est",
    ],
    "summary_metrics_cols": ["train_loss"],
    "paths_cols": [
        "output_dir",
        "dataset_mix_dir",
        "dataset_local_cache_dir",
    ],
    "chat_cols": [
        "chat_template_name",
        "dataset_mixer_list",
        "dataset_transform_fn",
        "dataset_target_columns",
        "dataset_mixer_list_splits",
        "hf_metadata_dataset",
    ],
    "details_cols": [
        "dataset_cache_mode",
        "tokenizer_revision",
        "logging_steps",
        *CONSTANT_OR_NAN_COLS,
    ],
    "dpo_hpm_cols": [
        "dpo_beta",
        "dpo_label_smoothing",
        "dpo_gamma_beta_ratio",
        "dpo_loss_type",
        "dpo_use_paged_optimizer",
    ],
    "dpo_eval_cols": [
        "logps/chosen",
        "logps/rejected",
        "rewards/chosen",
        "rewards/margin",
        "rewards/average",
        "rewards/accuracy",
        "rewards/rejected",
    ],
    "eval_setting_cols": ["oe_eval_tasks", "oe_eval_max_length"],
    "complex_cols": [
        "wandb_config",  # Structured config data that needs parsing
        "wandb_tags",  # Complex tag combinations that need special handling
        # Note: oe_eval_metrics/* columns need nested JSON parsing
    ],
    "nan_only_cols": [
        "hf_entity",
        "hf_repo_id",
        "config_name",
        "hf_repo_url",
        "save_to_hub",
        "dataset_mixer",
        "gs_bucket_path",
        "tokenizer_name",
        "max_train_steps",
        "hf_repo_revision",
        "dataset_config_hash",
        "dataset_config_name",
        "resume_from_checkpoint",
        "dataset_name",
        "oe_eval_tasks",
    ],
    "constant_or_nan_cols": CONSTANT_OR_NAN_COLS,
}
EXACT_MATCH_COLS = [
    ["run_id", "run_name", "exp_name"],
    ["max_train_samples", "finetune_unique_sequences"],
    ["entity", "wandb_entity"],
]
EXTRA_DROP_COLS = [
    "wandb_config",  # structured and repeated keys
    "run_name",  # same as run_id, keep one
    "exp_name",  # same as run_name, keep one
    "wandb_entity",  # same as entity
    "total_tokens_including_padding",  # keep total tokens instead
    "per_device_tps",
    "per_device_tps_including_padding",
    "_wandb",  # redundant with _runtime (truncated integer version)
    "tokenizer",  # complex object, not needed for analysis
    "tokenizer_files_hash",  # derived from tokenizer, not needed
    "report_to",  # just wandb
]
ALL_DROP_COLS = [
    *KEY_SETS["paths_cols"],
    *KEY_SETS["details_cols"],
    *KEY_SETS["nan_only_cols"],
    *KEY_SETS["dpo_hpm_cols"],
    *KEY_SETS["dpo_eval_cols"],
    *KEY_SETS["pretrain_hpm_cols"],
    *EXTRA_DROP_COLS,
]


# ---------- From wandb_store.py ----------
CORE_RUN_FIELDS = [
    "run_id",
    "run_name",
    "state",
    "project",
    "entity",
    "created_at",
    "runtime",
]
CORE_HISTORY_FIELDS = ["run_id", "run_name", "project", "step", "timestamp"]

DEFAULT_RUNS_FILENAME = "runs_metadata.parquet"
DEFAULT_HISTORY_FILENAME = "runs_history.parquet"

# ---------- From analysis_helpers.py ----------
# WandB history field patterns
WANDB_SYSTEM_FIELDS = {"_step", "_timestamp"}
WANDB_REDUNDANT_HISTORY_FIELDS = {"run_id", "run_name"}

METHODS = ["dpo", "finetune"]
DEFAULT_REQUIRED_PARAMS = ["learning_rate", "model_size_m", "method"]
DEFAULT_COLUMNS = ["step", "learning_rate", "train_loss", "total_tokens", "epoch"]
SCIENTIFIC_NOTATION_COLUMNS = ["max_lr", "min_lr", "initial_lr", "final_lr"]
THREE_DECIMAL_PLACES_COLUMNS = [
    "initial_train_loss",
    "final_train_loss",
    "min_train_loss",
    "loss_improvement",
]
COMMA_SEPARATED_COLUMNS = ["max_tokens"]
TRUNCATE_COLUMNS = ["run_id"]
TRUNCATE_LENGTH = 50
RANDOM_SEED = 42
