# 测试bart语言模型的预测效果
# 权重地址：https://github.com/fastnlp/CPT
from transformers import BertTokenizer, BartForConditionalGeneration
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.generation import AutoRegressiveDecoder, Seq2SeqGeneration
import os
import torch
import pytest


@pytest.mark.parametrize("ckpt_dir", ["E:/pretrain_ckpt/bart/fnlp@bart-base-chinese/",
                                      "E:/pretrain_ckpt/bart/fnlp@bart-base-chinese-v2.0/"])
@torch.inference_mode()
def test_bart(ckpt_dir):
    texts = ["北京是[MASK]的首都", "今天的天气是[MASK]，可以正常出海"]

    # ==============================transformers=====================================
    tokenizer = BertTokenizer.from_pretrained(f"{ckpt_dir}")
    model = BartForConditionalGeneration.from_pretrained(f"{ckpt_dir}")
    transformer_outputs = []
    for text in texts:
        input_ids = tokenizer.encode(text, return_tensors='pt')
        pred_ids = model.generate(input_ids, top_k=1, max_length=20)
        transformer_outputs.append(tokenizer.decode(pred_ids[0], skip_special_tokens=True).replace(' ', ''))
    print('transformers output: ', transformer_outputs)


    # ==============================bert4torch=====================================
    config_path = f'{ckpt_dir}/bert4torch_config.json'
    checkpoint_path = f'{ckpt_dir}/pytorch_model.bin'
    dict_path = f'{ckpt_dir}/vocab.txt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    topk = 1
    mode = 'random_sample'
    maxlen = 20
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    model = build_transformer_model(config_path, checkpoint_path).to(device)

    print('==============自定义单条样本================')
    class AutoTitle(AutoRegressiveDecoder):
        """seq2seq解码器
        """
        @AutoRegressiveDecoder.wraps(default_rtype='logits')
        def predict(self, inputs, output_ids, states):
            res = model.decoder.predict([output_ids] + inputs)
            return res[-1][:, -1, :] if isinstance(res, list) else res[:, -1, :]  # 保留最后一位

        def generate(self, text, topk=1):
            token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
            token_ids = torch.tensor([token_ids], device=device)
            encoder_output = model.encoder.predict([token_ids])  # [encoder_hiddenstates, encoder_attention_mask]
            output_ids = self.beam_search(encoder_output, topk=topk)[0]  # 基于beam search
            return tokenizer.decode(output_ids.cpu().numpy())
    autotitle = AutoTitle(bos_token_id=102, eos_token_id=tokenizer._token_end_id, maxlen=maxlen, device=device)
    bert4torch_outputs = []
    for text in texts:
        bert4torch_outputs.append(autotitle.generate(text))
    print('bert4torch_outputs: ', bert4torch_outputs)
    assert transformer_outputs == bert4torch_outputs


    print('==============默认单条无cache================')
    autotitle = Seq2SeqGeneration(model, tokenizer, start_id=102, end_id=tokenizer._token_end_id, mode=mode,
                                maxlen=maxlen, default_rtype='logits', use_states=False)
    bert4torch_outputs = []
    for text in texts:
        bert4torch_outputs.append(autotitle.generate(text, topk=topk))
    print('bert4torch_outputs: ', bert4torch_outputs)
    assert transformer_outputs == bert4torch_outputs


    print('==============默认batch 无cache================')
    bert4torch_outputs = autotitle.generate(texts, topk=topk)
    print('bert4torch_outputs: ', bert4torch_outputs)
    assert transformer_outputs == bert4torch_outputs

    
    print('==============默认单条cache================')
    autotitle = Seq2SeqGeneration(model, tokenizer, start_id=102, end_id=tokenizer._token_end_id, mode=mode,
                                maxlen=maxlen, default_rtype='logits', use_states=True)
    bert4torch_outputs = []
    for text in texts:
        bert4torch_outputs.append(autotitle.generate(text, topk=topk))
    print('bert4torch_outputs: ', bert4torch_outputs)
    assert transformer_outputs == bert4torch_outputs


    print('==============默认batch cache================')
    bert4torch_outputs = autotitle.generate(texts, topk=topk)
    print('bert4torch_outputs: ', bert4torch_outputs)
    assert transformer_outputs == bert4torch_outputs


    print('==============默认stream================')
    bert4torch_outputs = []
    for text in texts:
        for output in autotitle.stream_generate(text, topk=topk):
            os.system('clear')
            print(texts[0], ' -> ', output, flush=True)
        bert4torch_outputs.append(output)
    print('bert4torch_outputs: ', bert4torch_outputs)
    assert transformer_outputs == bert4torch_outputs


if __name__=='__main__':
    test_bart("E:/pretrain_ckpt/bart/fnlp@bart-base-chinese-v2.0/")