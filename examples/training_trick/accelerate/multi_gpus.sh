# With traditional PyTorch launcher
python -m torch.distributed.launch --nproc_per_node 2 --use_env ./nlp_example.py

# multi GPUs (using PyTorch distributed mode)
#   With Accelerate config and launcher
accelerate config  # This will create a config file on your server
accelerate launch ./nlp_example.py  # This will run the script on your server
