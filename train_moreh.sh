torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="0.0.0.0" --master_port="12234" ./train_moreh.py \
    --config_file configs/llama-3.1-70b-moreh.json \
    --model_path /vast/huggingface/hub/models--regisss--llama2-70b-fused-qkv-mlperf/snapshots/647cb0c8858ddefd10231a20ddfa68e4eb5e850e/ \
    --dataset_path /vast/huggingface/hub/datasets--regisss--scrolls_gov_report_preprocessed_mlperf_2/snapshots/21ff1233ee3e87bc780ab719c755170148aba1cb/data/train-00000-of-00001.parquet \
    --model_name llama \
    --batch_size 1 \
    |& tee -a ./llama_fp8.log
