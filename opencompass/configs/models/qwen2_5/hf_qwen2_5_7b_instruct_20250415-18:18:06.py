from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-7b-instruct-hf',
        path='/mnt/petrelfs/liyu1/all-code/r1/Demystify_Reasoning_RL/checkpoints/Demystify/Qwen2.5-7B-kk/actor/global_step_1200',
        max_out_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
