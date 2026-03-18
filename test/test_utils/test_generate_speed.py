'''测试GenerateSpeed功能, 统计token的生成速度'''
from bert4torch.snippets import GenerateSpeed
from transformers import AutoTokenizer, FalconForCausalLM
import torch


def test_generate_speed():
    model_dir = 'E:/data/pretrain_ckpt/tiiuae/falcon-rw-1b'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = FalconForCausalLM.from_pretrained(model_dir, trust_remote_code=True).half().to(device)

    query = '你好'
    inputs = tokenizer.encode(query, return_tensors="pt").to(device)

    with GenerateSpeed() as gs:
        response = model.generate(inputs, top_k=1, max_length=20)
        tokens_len = len(response[0]) - len(inputs[0])
        gs(tokens_len)


if __name__=='__main__':
    test_generate_speed()
