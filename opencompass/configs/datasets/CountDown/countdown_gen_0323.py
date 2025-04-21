from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.CountDown import CountDownEvaluator, CountDownDataset, extract_solution

with read_base():
    from .prompts import INSTRUCTIONS

data_path = r"/mnt/hwfile/opendatalab/air/liyu1/opencompass_data/countdown/"

countdown_reader_cfg = dict(input_columns=["numbers", "target"], output_column="target_nums")

countdown_datasets = []
instruction = INSTRUCTIONS
countdown_infer_cfg = dict(
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

pred_postprocessor = dict(type=extract_solution)
countdown_eval_cfg = dict(
    evaluator=dict(type=CountDownEvaluator), pred_postprocessor=pred_postprocessor
)

countdown_datasets.append(
    dict(
        abbr=f"CountDown",
        type=CountDownDataset,
        path=data_path,
        file_name="countdown_game24.jsonl",
        reader_cfg=countdown_reader_cfg,
        infer_cfg=countdown_infer_cfg,
        eval_cfg=countdown_eval_cfg,
    )
)
