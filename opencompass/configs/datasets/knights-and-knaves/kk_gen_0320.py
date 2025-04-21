from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.KK import KKEvaluator, KKDataset, extract_solution

with read_base():
    from .prompts import INSTRUCTIONS

prompts = INSTRUCTIONS
Categories = [
    "2ppl",
    "3ppl",
    "4ppl",
    "5ppl",
    "6ppl",
    "7ppl",
    "8ppl",
]
data_path = r"/mnt/hwfile/opendatalab/air/liyu1/opencompass_data/knights-and-knaves/"

kk_reader_cfg = dict(input_columns="quiz", output_column="answer")

KK_datasets = []
for category in Categories:
    instruction = prompts
    KK_infer_cfg = dict(
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
    KK_eval_cfg = dict(evaluator=dict(type=KKEvaluator), pred_postprocessor=pred_postprocessor)

    KK_datasets.append(
        dict(
            abbr=f"KK-{category}",
            type=KKDataset,
            path=data_path,
            category=category,
            reader_cfg=kk_reader_cfg,
            infer_cfg=KK_infer_cfg,
            eval_cfg=KK_eval_cfg,
        )
    )
