#/bin/bash

python ./convert_moreh.py \
    --config_file configs/llama-3.1-70b-moreh.json \
    --model_name llama \
    --hf_model_path /vast/huggingface/hub/models--regisss--llama2-70b-fused-qkv-mlperf/snapshots/647cb0c8858ddefd10231a20ddfa68e4eb5e850e/ \
    --save_path /vast/huggingface/saved/regisss--llama2-70b-fused-qkv-mlperf.pt
