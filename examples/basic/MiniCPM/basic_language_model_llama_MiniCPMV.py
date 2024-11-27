from PIL import Image
from bert4torch.pipelines import MiniCPMV
from bert4torch.snippets import log_info
import requests
from bert4torch.pipelines import ChatVL


# E:/data/pretrain_ckpt/MiniCPM/MiniCPM-Llama3-V-2_5
# E:/data/pretrain_ckpt/MiniCPM/MiniCPM-V-2_6
model_dir = "E:/data/pretrain_ckpt/MiniCPM/MiniCPM-V-2_6"

def chat_demo1():
    query1 = '介绍一下这张图片的内容？'
    query2 = '图片中的主体对象是什么？'
    image1 = Image.open(requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg", stream=True).raw).convert('RGB')
    image2 = Image.open(requests.get("https://picx.zhimg.com/v2-87a5a6d5a1536368eb6b2412d1c0a985_b.jpg", stream=True).raw).convert('RGB')

    demo = MiniCPMV(model_dir)

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


def chat_demo2():
    generation_config = {
        'top_k': 40,
        'top_p': 0.8,
        'repetition_penalty': 1.1
    }

    demo = ChatVL(model_dir, 
                generation_config=generation_config,
                # quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
                mode='gradio',
                template='minicpmv'
                )
    demo.run()

if __name__ == '__main__':
    # chat_demo1()
    chat_demo2()
