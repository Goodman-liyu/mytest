from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.MTruthfulQA import MTruthfulQADataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess

with read_base():
    from .m_truthfulqa_prompts import PREFIX_INSTRUCIONS, LANG_TO_QUESTION, LANG_TO_ANSWER

ALL_LANGUAGES = ['en']
prompting = 'BASIC1'
m_truthfulqa_reader_cfg = dict(
    input_columns=['question', 'option_a', 'option_b', 'option_c', 'option_d'],
    output_column='answer',
)
m_truthfulqa_datasets = []
for lang in ALL_LANGUAGES:
    _prefix = PREFIX_INSTRUCIONS[prompting][lang]
    _question = LANG_TO_QUESTION[lang]
    _answer = LANG_TO_ANSWER[lang]
    m_truthfulqa_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f'{_prefix}\n{_question}: {{question}}\n(A) {{option_a}}\n(B) {{option_b}}\n(C) {{option_c}}\n(D) {{option_d}}\n{_answer}:'
                )
            ], ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    m_truthfulqa_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    )
    m_truthfulqa_datasets.append(
        dict(
            abbr=f'm_truthful_qa-{lang}',
            type=MTruthfulQADataset,
            path=
            f'/mnt/hwfile/opendatalab/MinerU4S/weixingjian/BeltRoadBench/data/multilingual/m_truthfulqa/{lang}_validation.json',
            reader_cfg=m_truthfulqa_reader_cfg,
            infer_cfg=m_truthfulqa_infer_cfg,
            eval_cfg=m_truthfulqa_eval_cfg))

    del _question, _answer
