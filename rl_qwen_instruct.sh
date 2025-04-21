source activate compass
sleep 2


if [ $# -eq 0 ]; then
  echo "请在命令行中提供模型地址，例如：/mnt/petrelfs/caimengzhang/code/LLaMA-Factory/saves/llama3.1-8b/full/sft-gsm8k"
  exit 1
fi

export LD_LIBRARY_PATH=/mnt/petrelfs/liyu1/anaconda3/envs/compass/lib/python3.10/site-packages/nvidia/nvjitlink/lib/:$LD_LIBRARY_PATH
START_TIME=`date +%Y%m%d-%H:%M:%S`
MODEL_PATH=$1
DATANAME=$2
EXP_NAME=QWEN_INSTRUCT_TEST
PARITION=belt_road
QUOTA_TYPE=reserved
GPUS_PER_NODE=8
N_NODE=1
output_file="opencompass/configs/models/qwen2_5/hf_qwen2_5_7b_instruct_${START_TIME}.py"
LOG_FILE=/mnt/petrelfs/liyu1/src/logs/${START_TIME}_${EXP_NAME}.log

cat << EOF > $output_file
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
    type=HuggingFacewithChatTemplate,
    abbr='R1_template',
    path='${MODEL_PATH}',
    max_out_len=4096,
    batch_size=8,
    run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='instruct_template',
        path='${MODEL_PATH}',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    ),
        dict(
        type=HuggingFaceBaseModel,
        abbr='base_template',
        path='${MODEL_PATH}',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    ),
]
EOF
echo "YAML 配置文件已成功创建：$output_file"


opencompass \
-w outputs3/${DATANAME}_${START_TIME}_${EXP_NAME} \
--datasets math500_0shot_gen math500_4shot_gen aime2024_gen humaneval_gen mbpp_gen_0shot mbpp_gen_4shot kk_gen zebralogicbench_gen countdown_gen \
--models hf_qwen2_5_7b_instruct_${START_TIME} --max-num-worker 8  -a vllm 