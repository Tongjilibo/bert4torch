import requests
from utils import preprocess, postprocess

if __name__ == "__main__":
    text_list = ['我今天特别开心', '我今天特别生气']
    token_ids, segment_ids = preprocess(text_list)
    request_data = {
    "inputs": [{
        "name": "input_ids",
        "shape": [token_ids.shape[0], 512],
        "datatype": "INT32",
        "data": token_ids.tolist()  # [list(range(512))]
    },
    {
        "name": "segment_ids",
        "shape": [segment_ids.shape[0], 512],
        "datatype": "INT32",
        "data": segment_ids.tolist() # [list(range(512))]
    }
    ],
    "outputs": [{"name": "output"}]
}
    res = requests.post(url="http://localhost:8000/v2/models/sentence_classification/versions/1/infer",json=request_data).json()
    print(postprocess(res))
