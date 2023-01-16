# with fp16 (mixed-precision)
#   from any server by passing fp16=True to the Accelerator.
python ./nlp_example.py --fp16

#   from any server with Accelerate launcher
accelerate launch --fp16 ./nlp_example.py