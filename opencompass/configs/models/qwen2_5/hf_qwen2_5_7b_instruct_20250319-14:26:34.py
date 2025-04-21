from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-7b-instruct-hf',
        path='/mnt/petrelfs/liyu1/LLaMA-Factory/saves/qwen2.5-7b/full/sft-Lima',
        max_out_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
