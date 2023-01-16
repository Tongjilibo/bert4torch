# multi GPUs, multi node (several machines, using PyTorch distributed mode)
#   With Accelerate config and launcher, on each machine:
accelerate config  # This will create a config file on each server
accelerate launch ./nlp_example.py  # This will run the script on each server

#   With PyTorch launcher only
python -m torch.distributed.launch --nproc_per_node 2 \
    --use_env \
    --node_rank 0 \
    --master_addr master_node_ip_address \
    ./nlp_example.py  # On the first server
python -m torch.distributed.launch --nproc_per_node 2 \
    --use_env \
    --node_rank 1 \
    --master_addr master_node_ip_address \
    ./nlp_example.py  # On the second server

