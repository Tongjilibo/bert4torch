import numpy as np
import tritonclient.http as httpclient
from utils import preprocess, postprocess

triton_client = httpclient.InferenceServerClient(url='localhost:8000')
mapping = {0: 'negtive', 1: 'positive'}

if __name__ == "__main__":

    text_list = ['我今天特别开心', '我今天特别生气']
    token_ids, segment_ids = preprocess(text_list)

    inputs = []
    inputs.append(httpclient.InferInput('input_ids', [token_ids.shape[0], 512], "INT32"))
    inputs.append(httpclient.InferInput('segment_ids', [segment_ids.shape[0], 512], "INT32"))

    inputs[0].set_data_from_numpy(token_ids.astype(np.int32), binary_data=False)
    inputs[1].set_data_from_numpy(segment_ids.astype(np.int32), binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output', binary_data=False))

    results = triton_client.infer('sentence_classification', inputs=inputs, outputs=outputs)

    prob = results.as_numpy("output")
    pred = prob.argmax(axis=-1)
    result = [mapping[i] for i in pred]
    print(result)
