# 调用代码
import requests
import json

def send_msg(requestData):
    url = 'http://localhost:8082/recommendinfo'
    headers = {'content-type': 'application/json'}
    ret = requests.post(url, json=requestData, headers=headers, stream=True)
    if ret.status_code==200:
        text = json.loads(ret.text)
    return text

send_msg({'input': ['我的心情很好', '我很生气']})