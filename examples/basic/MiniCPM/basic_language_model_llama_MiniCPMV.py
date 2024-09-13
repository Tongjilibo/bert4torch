from PIL import Image
from bert4torch.pipelines import MiniCPMV

# E:/data/pretrain_ckpt/MiniCPM/MiniCPM-V-2_6
model_dir = 'E:/data/pretrain_ckpt/MiniCPM/MiniCPM-V-2_6'

demo = MiniCPMV(model_dir)
question = '介绍一下这张图片的内容?'

image1 = Image.open('./test_local/资料概要.png').convert('RGB')
image2 = Image.open('./test_local/bert4torch.png').convert('RGB')

answer = demo.chat(question, [image1, image2])
print(answer)

# msgs.append({"role": "assistant", "content": [answer]})
# msgs.append({"role": "user", "content": ["这款基金的基金经理是谁？"]})

# answer = demo.chat(
#     image=None,
#     msgs=msgs,
# )
# print(answer)