# single cpu
echo '====from a server without GPU===='
python ./nlp_example.py

echo '====from any server by passing cpu=True to the Accelerator===='
python ./nlp_example.py --cpu

echo '====from any server with Accelerate launcher===='
accelerate launch --cpu ./nlp_example.py