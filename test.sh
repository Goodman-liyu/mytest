#!/bin/bash


step=(200 400 600 800)
for i in ${step[@]}; do
    echo "正在执行脚本: $i"
    bash rl_qwen_instruct.sh /mnt/petrelfs/liyu1/all-code/r1/Demystify_Reasoning_RL/checkpoints/Nips/kk_7b_base/actor/global_step_${i} qwen_base_kk_${i}
    sleep 5000
done

step2=(200 400 600 800 1000)

for i in ${step2[@]}; do
    echo "正在执行脚本: $i"
    bash rl_qwen_instruct.sh /mnt/petrelfs/liyu1/all-code/r1/Demystify_Reasoning_RL/checkpoints/Nips/kk_7b_math_base/actor/global_step_${i} qwen_math_kk_${i}
    sleep 5000
done

echo "所有脚本执行完成"