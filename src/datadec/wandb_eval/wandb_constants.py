# ---------- From parsing.py ----------
TIME_KEYS = ["created_at", "timestamp"]
EARLIEST_GOOD_RUN_DATE = "2025-08-21"
ID_KEYS = [
    "run_id",
    "run_name",
    "project",
    "entity",
]
STATUS_KEYS = ["state", "created_at", "runtime", "timeout"]
PRETRAIN_HPM_KEYS = [
    "pretrain_steps",
    "pretrain_tokens",
    "pretrain_sequences",
]
CORE_HPM_KEYS = [
    "seed",
    "dataset_name",
    "model_size",
    "learning_rate",
    "num_train_epochs",
    "max_train_samples",
    "finetune_sequences",
    "finetune_unique_sequences",
    "total_sequences",
]
DPO_HPM_KEYS = [
    "dpo_beta",
    "dpo_label_smoothing",
    "dpo_gamma_beta_ratio",
    "logps/chosen",
    "logps/rejected",
]
EVAL_SETTING_KEYS = [
    "oe_eval_tasks",
    "oe_eval_max_length",
]
X_AXIS_KEYS = [
    "step",
    "epoch",
    "total_tokens",
    "_step",
    "_runtime",
    "_timestamp",
]
SUMMARY_METRICS_KEYS = [
    "train_loss",
]
DPO_ONLY_KEYS = [
    "rewards/chosen",
    "rewards/margin",
    "rewards/average",
    "rewards/accuracy",
    "rewards/rejected",
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
