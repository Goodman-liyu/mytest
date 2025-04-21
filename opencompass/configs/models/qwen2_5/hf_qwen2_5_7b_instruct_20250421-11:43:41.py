from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
    type=HuggingFacewithChatTemplate,
    abbr='R1_template',
    path='/mnt/petrelfs/liyu1/huggingface/hub/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e',
    max_out_len=4096,
    batch_size=8,
    run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=HuggingFaceBaseModel,
        abbr='base_template',
        path='/mnt/petrelfs/liyu1/huggingface/hub/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='instruct_template',
        path='/mnt/petrelfs/liyu1/huggingface/hub/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    ),

]
