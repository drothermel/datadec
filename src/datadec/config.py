"""Model configuration downloading and analysis functions."""

import json
from huggingface_hub import hf_hub_download
from typing import Dict, Any, Optional, Tuple


def download_config_file(
    repo_id: str, branch: str = "main", local_dir: Optional[str] = None
) -> Tuple[Optional[str], bool, str]:
    """Download config.json file from Hugging Face repository.

    Args:
        repo_id: HuggingFace repository ID (e.g., "allenai/DataDecide-falcon-and-cc-qc-tulu-10p-60M")
        branch: Branch/revision to download from (e.g., "step0-seed-default")
        local_dir: Local directory to save file (optional)

    Returns:
        Tuple of (file_path, success, error_message)
        - file_path: Path to downloaded file if successful, None otherwise
        - success: True if download succeeded, False otherwise
        - error_message: Error description if failed, empty string if succeeded
    """
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=branch,
            local_dir=local_dir,
        )
        return file_path, True, ""
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Entry Not Found" in error_msg:
            return None, False, "Config file not available"
        else:
            return None, False, error_msg

def extract_model_architecture_info(config_path: str) -> Dict[str, Any]:
    """Load and extract key model architecture information from config.

    Args:
        config_path: Path to the config.json file

    Returns:
        Dictionary with structured architecture information
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        arch_info = {}

        # Common architecture fields across different model types
        arch_fields = {
            # Model size and structure
            "hidden_size": ["hidden_size", "d_model", "n_embd"],
            "num_layers": ["num_hidden_layers", "n_layer", "num_layers"],
            "num_attention_heads": ["num_attention_heads", "n_head", "num_heads"],
            "intermediate_size": ["intermediate_size", "ffn_dim", "n_inner"],
            # Vocabulary and sequence
            "vocab_size": ["vocab_size", "vocabulary_size"],
            "max_position_embeddings": [
                "max_position_embeddings",
                "n_positions",
                "max_seq_len",
            ],
            "sequence_length": ["max_sequence_length", "seq_length"],
            # Model type and architecture
            "model_type": ["model_type", "architectures"],
            "activation_function": ["hidden_act", "activation_function"],
            # Training specifics
            "layer_norm_eps": ["layer_norm_eps", "layer_norm_epsilon"],
            "dropout": ["hidden_dropout_prob", "dropout", "attention_probs_dropout_prob"],
            # Tokenizer info
            "pad_token_id": ["pad_token_id"],
            "eos_token_id": ["eos_token_id"],
            "bos_token_id": ["bos_token_id"],
        }

        # Extract fields using multiple possible keys
        for standard_key, possible_keys in arch_fields.items():
            for key in possible_keys:
                if key in config:
                    arch_info[standard_key] = config[key]
                    break

        return arch_info

    except Exception as e:
        return {"error": str(e)}
