from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3_1-8b-instruct-hf',
        path='/mnt/hwfile/opendatalab/air/liyu1/saves/llama3.1-8b/full/sft-Lima',
        max_out_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        #stop_words=['<|end_of_text|>', '<|eot_id|>'],
        stop_words=['<|end_of_text|>'],
        #meta_template=api_meta_template,
    )
]
