from PIL import Image
from bert4torch.pipelines import ChatV

# E:/data/pretrain_ckpt/MiniCPM/MiniCPM-V-2_6
model_dir = 'E:/data/pretrain_ckpt/MiniCPM/MiniCPM-V-2_6'

demo = ChatV(model_dir)
question = '介绍一下这张图片的内容?'

image = Image.open('/home/lb/projects/bert4torch/test_local/资料概要.png').convert('RGB')
msgs = [{'role': 'user', 'content': [image, question]}]

res = demo.chat(
    image=None,
    msgs=msgs
)
print(res)