from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.CipherBank import (
    CipherBankEvaluator,
    CipherBankDataset,
    extract_result_sentences3,
)

with read_base():
    from .prompts import INSTRUCTIONS, INSTRUCTIONS2

is_hint = False
if is_hint == False:
    prompts = INSTRUCTIONS
else:
    prompts = INSTRUCTIONS2
ALL_CIPHERS = [
    "Rot13",
    "Atbash",
    "Polybius",
    "Vigenere",
    "Reverse",
    "SwapPairs",
    "ParityShift",
    "DualAvgCode",
    "WordShift",
]
data_path = r"/mnt/hwfile/opendatalab/air/liyu1/opencompass_data/CipherBank/"

cipherbank_reader_cfg = dict(input_columns="ciphertext", output_column="plaintext")

CipherBank_datasets = []
for cipher in ALL_CIPHERS:
    instruction = prompts[cipher]
    CipherBank_infer_cfg = dict(
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

    pred_postprocessor = dict(type=extract_result_sentences3)
    CipherBank_eval_cfg = dict(
        evaluator=dict(type=CipherBankEvaluator), pred_postprocessor=pred_postprocessor
    )

    CipherBank_datasets.append(
        dict(
            abbr=f"CipherBank-{cipher}",
            type=CipherBankDataset,
            path=data_path,
            cipher=cipher,
            reader_cfg=cipherbank_reader_cfg,
            infer_cfg=CipherBank_infer_cfg,
            eval_cfg=CipherBank_eval_cfg,
        )
    )
