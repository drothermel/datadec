# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "attrs==25.3.0",
#     "cattrs==25.2.0",
#     "clumper==0.2.15",
#     "marimo",
#     "psycopg2==2.9.10",
#     "psycopg2-binary==2.9.10",
#     "rich==14.1.0",
#     "sqlalchemy==2.0.43",
#     "sqlalchemy-utils==0.42.0",
#     "srsly==2.5.1",
# ]
# ///

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="columns", app_title="Extract DD Try 1")


@app.cell(column=0)
def _():
    import marimo as mo
    from pathlib import Path
    import tarfile
    import srsly
    from clumper import Clumper
    import rich
    from attrs import define, field
    from cattrs import Converter, structure
    from sqlalchemy_utils import CompositeType, register_composites
    import sqlalchemy
    from sqlalchemy.dialects.postgresql import ARRAY
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        String,
        Boolean,
        Float,
        Numeric,
        Text,
    )
    return (
        ARRAY,
        Boolean,
        Clumper,
        Column,
        CompositeType,
        Float,
        Integer,
        Path,
        String,
        Text,
        create_engine,
        define,
        mo,
        sqlalchemy,
        srsly,
        structure,
        tarfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _(Path):
    repo_dir = "/Users/daniellerothermel/drotherm/repos/datadec/"
    raw_downloads_dir = Path(repo_dir, "data", "raw_downloads")
    instance_ds_snapshots = Path(
        raw_downloads_dir,
        "datasets--allenai--DataDecide-eval-instances",
        "snapshots",
        "23f3b2e186ca6c39026e3efa00e4af397680c075",
        "models",
    )
    extracted_dir = Path(raw_downloads_dir, "instances_extracted")
    return extracted_dir, instance_ds_snapshots


@app.cell
def _(Path):
    def get_dir(root_dir, data, params, seed):
        return Path(root_dir, data, params, f"seed-{seed}")


    def get_all_tard(root_dir, data, params, seed):
        target_dir = get_dir(root_dir, data, params, seed)
        return list(target_dir.glob("*.tar.gz"))
    return (get_all_tard,)


@app.cell
def _(Path, tarfile):
    def extract_path(path, dest):
        tar_path = Path(path.stem)
        path_name = tar_path.stem
        dest_path = Path(dest, path_name)
        if dest_path.exists():
            print(f">> Already extracted: {path_name}")
        else:
            print(f">> Extracting: {path_name}")
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(path=dest)
        return dest_path
    return (extract_path,)


@app.cell
def _(create_engine):
    new_engine = create_engine("postgresql+psycopg2://localhost/test_dd")
    return


@app.cell
def _(sqlalchemy):
    Base = sqlalchemy.orm.declarative_base()
    return


@app.cell
def _(Column, CompositeType, String):
    # Define the composite type in Python
    address_type = CompositeType(
        "address_type",
        [
            Column("street", String),
            Column("city", String),
            Column("state", String),
            Column("zip_code", String),
        ],
    )
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""## Extraction""")
    return


@app.cell(hide_code=True)
def _(get_all_tard, instance_ds_snapshots):
    data = "c4"
    params = "4M"
    seed = 2
    c4_4m_2_tard_files = get_all_tard(instance_ds_snapshots, data, params, seed)
    print(
        f"Num tar'd: {len(c4_4m_2_tard_files)}, first example: {c4_4m_2_tard_files[0].parts[-4:]}"
    )
    return c4_4m_2_tard_files, data, params, seed


@app.cell(hide_code=True)
def _(
    Path,
    c4_4m_2_tard_files,
    data,
    extract_path,
    extracted_dir,
    params,
    seed,
):
    newly_extracted = extract_path(
        c4_4m_2_tard_files[0], Path(extracted_dir, f"{data}_{params}_{seed}")
    )
    print(f"Newly extracted: {newly_extracted.parts[-2:]}")
    return (newly_extracted,)


@app.cell(hide_code=True)
def _(newly_extracted):
    extracted_files = list(newly_extracted.glob("*"))
    print("First 5 extracted:")
    for f in extracted_files[:5]:
        print(f.parts[-2:])
    return (extracted_files,)


@app.cell
def _(define):
    @define
    class ModelAnswerOutput:
        is_greedy: bool
        logits_per_byte: float
        logits_per_char: float
        logits_per_token: float
        num_chars: int
        num_tokens: int
        num_tokens_all: int
        sum_logits: float
        sum_logits_uncond: float


    @define
    class QuestionOutputData:
        doc_id: int
        native_id: int
        label: int
        answer_outputs: list[ModelAnswerOutput]


    @define
    class TaskOutputData:
        task_hash: str
        model_hash: str
        data: str
        params: str
        seed: int
        task: str
        step: int
        question_outputs: list[QuestionOutputData]
    return QuestionOutputData, TaskOutputData


@app.cell
def _(ARRAY, Boolean, Column, CompositeType, Float, Integer, Text):
    qa_model_answer_type = CompositeType(
        "qa_model_answer",
        [
            Column("is_greedy", Boolean),
            Column("logits_per_byte", Float),
            Column("logits_per_char", Float),
            Column("logits_per_token", Float),
            Column("num_chars", Integer),
            Column("num_tokens", Integer),
            Column("num_tokens_all", Integer),
            Column("sum_logits", Float),
            Column("sum_logits_uncond", Float),
        ],
    )
    question_output_type = CompositeType(
        "question_output",
        [
            Column("doc_id", Integer),
            Column("native_id", Integer),
            Column("label", Integer),
            Column("answer_outputs", ARRAY(qa_model_answer_type)),
        ],
    )
    qa_task_output_type = CompositeType(
        "qa_task_output",
        [
            Column("task_hash", Text),
            Column("model_hash", Text),
            Column("data", Text),
            Column("params", Text),
            Column("seed", Integer),
            Column("task", Text),
            Column("step", Integer),
            Column("question_outputs", ARRAY(question_output_type)),
        ],
    )
    return


@app.cell
def _(QuestionOutputData, parsed_per_answer, structure):
    ppa_structured = [structure(ppa, QuestionOutputData) for ppa in parsed_per_answer]
    return (ppa_structured,)


@app.cell
def _(ppa_structured):
    ppa_structured[0]
    return


@app.cell
def _(TaskOutputData, full_file_info, parsed_per_answer, structure):
    taskout_structured = structure(
        {**full_file_info[0], "question_outputs": parsed_per_answer},
        TaskOutputData,
    )
    return (taskout_structured,)


@app.cell
def _(taskout_structured):
    taskout_structured
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""## Loading""")
    return


@app.cell(hide_code=True)
def _(extracted_files):
    curr_file = extracted_files[0]
    print(f">> Current file: {curr_file.parts[-3:]}")
    curr_step = int(curr_file.parts[-2].split("-")[-1])
    curr_task = curr_file.stem
    print(f">> {curr_task=} {curr_step=}")
    return curr_file, curr_step, curr_task


@app.cell(hide_code=True)
def _(curr_file, srsly):
    curr_jsonl = list(srsly.read_jsonl(curr_file))
    print(f"Num entries: {len(curr_jsonl)}")
    return (curr_jsonl,)


@app.cell(hide_code=True)
def _(Clumper, curr_jsonl, curr_step, curr_task, data, params, seed):
    full_file_info = deduplicated_full_info = (
        Clumper(curr_jsonl)
        .select("task_hash", "model_hash")
        .mutate(
            data=lambda c: data,
            params=lambda c: params,
            seed=lambda c: seed,
            task=lambda c: curr_task,
            step=lambda c: curr_step,
        )
        .drop_duplicates()
        .show(name="Full File Info")
        .collect()
    )
    return (full_file_info,)


@app.cell(hide_code=True)
def _(Clumper, curr_jsonl):
    # Get the agg metrics
    agg_metrics = (
        Clumper(curr_jsonl)
        .map(lambda d: {**d["metrics"], "doc_id": d["doc_id"]})
        .show(n=1, name="Agg Metrics")
        .collect()
    )
    return


@app.cell(hide_code=True)
def _(Clumper, curr_jsonl):
    parsed_per_answer = (
        Clumper(curr_jsonl)
        .drop("metrics", "task_hash", "model_hash")
        .rename(answer_outputs="model_output")
        .show(n=1, name="After")
        .collect()
    )
    return (parsed_per_answer,)


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _(Clumper, curr_jsonl):
    # Get the model output keys
    ordered_model_pred_keys = sorted(
        (Clumper(curr_jsonl).map(lambda d: d["model_output"][0]).keys())
    )
    print(">> Ordered Model Pred Keys")
    for k in ordered_model_pred_keys:
        print(f"   - {k}")
    return


@app.cell
def _():
    return


@app.cell(column=3)
def _():
    return


if __name__ == "__main__":
    app.run()
