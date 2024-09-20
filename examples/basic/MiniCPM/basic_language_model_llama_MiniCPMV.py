from PIL import Image
from bert4torch.pipelines import MiniCPMV
from bert4torch.snippets import log_info


demo = MiniCPMV('E:/data/pretrain_ckpt/MiniCPM/MiniCPM-V-2_6')
query1 = '介绍一下这张图片的内容？'
query2 = '图片内容和基金产品相关吗？'
image1 = Image.open('./test_local/资料概要.png').convert('RGB')
image2 = Image.open('./test_local/bert4torch.png').convert('RGB')


log_info('# 提问单张图片')
answer = demo.chat(query1, image1)
print(answer)

log_info('# 带history')
history = [
    {'role': 'user', 'content': query1, 'images': [image1]},
    {'role': 'assistant', 'content': answer},
]
answer = demo.chat(query2, images=None, history=history)
print(answer)

log_info('# 同时提问多张图片')
answer = demo.chat(query1, [image1, image2])
print(answer)

log_info('# 多次提问单张图片')
answer = demo.chat([query1, query2], image1)
print(answer)

log_info('# 各自提问单张图片')
answer = demo.chat([query1, query2], [image1, image2])
print(answer)

log_info('# 各自同时提问多张图片')
answer = demo.chat([query1, query2], [[image1, image2], [image1, image2]])
print(answer)
