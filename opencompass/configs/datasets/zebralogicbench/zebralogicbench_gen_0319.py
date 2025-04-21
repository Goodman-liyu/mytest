from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.ZebraLogicBench import (
    zebralogicbenchEvaluator,
    zebralogicbenchDataset,
    extract_last_complete_json,
)

with read_base():
    from .prompts import INSTRUCTIONS


data_path = r"/mnt/hwfile/opendatalab/air/liyu1/opencompass_data/zebralogicbench"

zebralogicbench_reader_cfg = dict(input_columns=["puzzle", "json_template"], output_column="answer")

zebralogicbench_datasets = []

instruction = INSTRUCTIONS
zebralogicbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(role="HUMAN", prompt=instruction),
            ],
        ),
        ice_token="</E>",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

pred_postprocessor = dict(type=extract_last_complete_json)
zebralogicbench_eval_cfg = dict(
    evaluator=dict(type=zebralogicbenchEvaluator), pred_postprocessor=pred_postprocessor
)

zebralogicbench_datasets.append(
    dict(
        abbr=f"zebralogicbench",
        type=zebralogicbenchDataset,
        path=data_path,
        question_format_file="zebra.jsonl",
        reference_file="zebra_with_answer.jsonl",
        reader_cfg=zebralogicbench_reader_cfg,
        infer_cfg=zebralogicbench_infer_cfg,
        eval_cfg=zebralogicbench_eval_cfg,
    )
)
