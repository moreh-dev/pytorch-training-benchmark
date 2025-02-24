torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_moreh.py \
    configs/llama-3.1-70b-moreh.json llama --batch_size 1 |& tee -a ./llama_fp8.log
