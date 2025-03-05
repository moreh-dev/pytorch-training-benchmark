#/bin/bash

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="0.0.0.0" --master_port="12234" ./train_moreh.py \
    --config_file configs/llama-3.1-70b-moreh.json \
    --model_path /vast/huggingface/saved/regisss--llama2-70b-fused-qkv-mlperf/ \
    --dataset_path /vast/huggingface/hub/datasets--regisss--scrolls_gov_report_preprocessed_mlperf_2/snapshots/21ff1233ee3e87bc780ab719c755170148aba1cb/data/train-00000-of-00001.parquet \
    --batch_size 4 |&
    tee -a ./llama_fp8.log
