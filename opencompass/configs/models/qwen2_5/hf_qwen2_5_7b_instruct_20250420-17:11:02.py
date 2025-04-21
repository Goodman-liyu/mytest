from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-7b-instruct-hf',
        path='/mnt/hwfile/opendatalab/air/panzhuoshi/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
