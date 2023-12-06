# (multi) TPUs
#   With Accelerate config and launcher
accelerate config  # This will create a config file on your TPU server
accelerate launch ./nlp_example.py  # This will run the script on each server