#/bin/bash

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="0.0.0.0" --master_port="12234" ./train_fsdp.py \
    --config_file configs/llama-3.1-70b.json \
    --model_name llama \
    --batch_size 1 \
    --num_epochs 5 \
    --enable_fp8 0 |&
    tee -a ./llama_fp8.log
