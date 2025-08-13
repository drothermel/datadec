import numpy as np


def add_lr_cols(df):
    df = df.copy()
    df_by_params = (
        df[
            [
                "params",
                "step",
                "lr_warmup_start",
                "lr_max",
                "lr_final",
                "warmup_steps",
                "lr_decay_steps",
            ]
        ]
        .groupby(["params", "step"])
        .first()
        .reset_index()
    )
    df_by_params["lr_at_step"] = df_by_params.apply(
        lambda row: get_lr_at_step(
            row["step"],
            row["lr_warmup_start"],
            row["lr_max"],
            row["lr_final"],
            row["warmup_steps"],
            row["lr_decay_steps"],
        ),
        axis=1,
    )
    df_by_params["cumulative_lr_at_step"] = df_by_params.apply(
        lambda row: calculate_cumulative_lr(
            row["step"],
            row["lr_warmup_start"],
            row["lr_max"],
            row["lr_final"],
            row["warmup_steps"],
            row["lr_decay_steps"],
        ),
        axis=1,
    )
    df_by_params = df_by_params[
        [
            "params",
            "step",
            "lr_at_step",
            "cumulative_lr_at_step",
        ]
    ]
    df = df.merge(df_by_params, on=["params", "step"], how="left")
    return df


def numerical_cosine_integral(lr_max, lr_final, lr_decay_steps, decay_step):
    """Numerically integrate the cosine annealing schedule."""
    if decay_step <= 0:
        return 0.0

    # Use trapezoidal rule for integration
    t_values = np.linspace(0, decay_step, int(decay_step) + 1)
    lr_values = lr_final + 0.5 * (lr_max - lr_final) * (
        1 + np.cos(np.pi * t_values / lr_decay_steps)
    )

    # Trapezoidal integration
    return np.trapz(lr_values, t_values)


def calculate_cumulative_lr(
    step,
    lr_warmup_start,
    lr_max,
    lr_final,
    warmup_steps,
    lr_decay_steps,
):
    if step <= 0:
        return 0.0
    cumulative_lr = 0.0
    if step <= warmup_steps:
        t = step
        cumulative_lr = lr_warmup_start * t + (lr_max - lr_warmup_start) * t**2 / (
            2 * warmup_steps
        )
    else:
        t = warmup_steps
        warmup_cumulative = lr_warmup_start * t + (lr_max - lr_warmup_start) * t**2 / (
            2 * warmup_steps
        )
        decay_step = min(step - warmup_steps, lr_decay_steps)
        if decay_step > 0:
            decay_cumulative = numerical_cosine_integral(
                lr_max, lr_final, lr_decay_steps, decay_step
            )
            cumulative_lr = warmup_cumulative + decay_cumulative
        else:
            cumulative_lr = warmup_cumulative
    return cumulative_lr


def get_lr_at_step(
    step,
    lr_warmup_start,
    lr_max,
    lr_final,
    warmup_steps,
    lr_decay_steps,
):
    """Get the learning rate at a specific step."""
    if step <= warmup_steps:
        return lr_warmup_start + (lr_max - lr_warmup_start) * step / warmup_steps
    else:
        decay_step = min(step - warmup_steps, lr_decay_steps)
        if decay_step >= lr_decay_steps:
            return lr_final
        return lr_final + 0.5 * (lr_max - lr_final) * (
            1 + np.cos(np.pi * decay_step / lr_decay_steps)
        )
