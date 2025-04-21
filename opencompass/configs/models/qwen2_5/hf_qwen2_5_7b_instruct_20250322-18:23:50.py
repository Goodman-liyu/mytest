from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-7b-instruct-hf',
        path='/mnt/petrelfs/liyu1/all-code/r1/CipherR1/checkpoints/CipherR1/Qwen2.5-7B-3cipher_withshot/actor/global_step_1000',
        max_out_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
