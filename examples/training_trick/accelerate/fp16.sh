# with fp16 (mixed-precision)
#   from any server by passing fp16=True to the Accelerator.
python ./nlp_example.py --mixed_precision fp16

#   from any server with Accelerate launcher
accelerate launch ./nlp_example.py --mixed_precision fp16