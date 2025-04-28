from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
    type=HuggingFacewithChatTemplate,
    abbr='R1_template',
    path='/mnt/petrelfs/liyu1/all-code/r1/CipherR1/checkpoints/AffineCipher/Qwen2.5-7B-affine/actor/global_step_400',
    max_out_len=4096,
    batch_size=8,
    run_cfg=dict(num_gpus=1),
    ),
]
