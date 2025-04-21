from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MBPPDataset, MBPPEvaluator

mbpp_reader_cfg = dict(input_columns=["text", "test_list"], output_column="test_list_2")

mbpp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n {test_list}  \n[BEGIN]\n",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

mbpp_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator))

mbpp_datasets = [
    dict(
        type=MBPPDataset,
        abbr="mbpp-0shot",
        path="opencompass/mbpp",
        reader_cfg=mbpp_reader_cfg,
        infer_cfg=mbpp_infer_cfg,
        eval_cfg=mbpp_eval_cfg,
    )
]
