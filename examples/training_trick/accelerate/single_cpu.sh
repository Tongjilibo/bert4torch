# single cpu
# from a server without GPU
python ./nlp_example.py

# from any server by passing cpu=True to the Accelerator.
python ./nlp_example.py --cpu

# from any server with Accelerate launcher
accelerate launch --cpu ./nlp_example.py