'''测试bert和transformer的结果比对'''
import pytest
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch
from transformers import AutoTokenizer, AlbertForMaskedLM
from torch.nn.functional import softmax

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.mark.parametrize("model_dir", ["E:/pretrain_ckpt/albert/brightmart@albert_base_zh"])
@torch.inference_mode()
def test_albert(model_dir):
    inputtext = "今天[MASK]情很好"

    # ==========================bert4torch调用==========================
    # 加载模型，请更换成自己的路径
    vocab_path = model_dir + "/vocab.txt"
    config_path = model_dir + "/config.json"
    checkpoint_path = model_dir + '/pytorch_model.bin'

    # 建立分词器
    tokenizer = Tokenizer(vocab_path, do_lower_case=True)
    model = build_transformer_model(config_path, checkpoint_path, model='albert', with_mlm='softmax')  # 建立模型，加载权重

    token_ids, segments_ids = tokenizer.encode(inputtext)
    print(''.join(tokenizer.ids_to_tokens(token_ids)))

    tokens_ids_tensor = torch.tensor([token_ids])
    segment_ids_tensor = torch.tensor([segments_ids])

    print('====bert4torch output====')
    model.eval()
    with torch.no_grad():
        _, probas = model([tokens_ids_tensor, segment_ids_tensor])
        result = torch.argmax(probas[0, 3:4], dim=-1).numpy()
        b4t_output = tokenizer.decode(result)
        print(b4t_output)


    # ==========================transformer调用==========================
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AlbertForMaskedLM.from_pretrained(model_dir)

    maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)

    input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, labels=input_ids)
    loss, prediction_scores = outputs[:2]
    logit_prob = softmax(prediction_scores[0, maskpos],dim=-1).data.tolist()
    predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print('====transformers output====')
    print(predicted_token, logit_prob[predicted_index])

    assert b4t_output == predicted_token


if __name__=='__main__':
    test_albert("E:/pretrain_ckpt/albert/brightmart@albert_base_zh")