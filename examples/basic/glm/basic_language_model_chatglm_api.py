'''
基本测试：chatglm的api接口测试
'''

from fastapi import FastAPI, Request
from bert4torch.snippets import cuda_empty_cache
import uvicorn, json, datetime

choice = 'stream'  # simple, stream，后者可支持流式输出


if choice == 'simple':
    from basic_language_model_chatglm import cli_demo
    DEVICE = "cuda"
    DEVICE_ID = "0"
    CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
    app = FastAPI()

    @app.post("/")
    async def create_item(request: Request):
        json_post_raw = await request.json()
        json_post = json.dumps(json_post_raw)
        json_post_list = json.loads(json_post)
        prompt = json_post_list.get('prompt')
        history = json_post_list.get('history')
        response = cli_demo.generate(prompt, history=history)
        history.append((prompt, response))
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
            "response": response,
            "history": history,
            "status": 200,
            "time": time
        }
        log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
        print(log)
        cuda_empty_cache(CUDA_DEVICE)
        return answer

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
    # 调用
    # curl -X POST "http://127.0.0.1:8000"  -H "Content-Type: application/json"  -d '{"prompt": "你好", "history": []}'

elif choice == 'stream':
    from bert4torch.pipelines import ChatGlmOpenaiApi

    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B"

    generation_config = {'mode': 'random_sample',
                        'maxlen': 2048, 
                        'default_rtype':'logits', 
                        'use_states':True}

    cli_demo = ChatGlmOpenaiApi(dir_path, generation_config=generation_config)
    cli_demo.run()
