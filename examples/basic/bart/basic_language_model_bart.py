# 测试bart语言模型的预测效果
# 权重地址：https://github.com/fastnlp/CPT

# 选择v1还是v2
ckpt_dir = "E:/pretrain_ckpt/bart/fnlp@bart-base-chinese/"  # v1.0
# ckpt_dir = "E:/pretrain_ckpt/bart/fnlp@bart-base-chinese-v2.0/"  # v2.0
texts = ["北京是[MASK]的首都", "今天的天气是[MASK]，可以正常出海"]

# ==============================transformers=====================================
from transformers import BertTokenizer, BartForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained(f"{ckpt_dir}")
model = BartForConditionalGeneration.from_pretrained(f"{ckpt_dir}")
input_ids = tokenizer.encode(texts[0], return_tensors='pt')
pred_ids = model.generate(input_ids, num_beams=4, max_length=20)
print('transformers output: ', tokenizer.convert_ids_to_tokens(pred_ids[0]))
# 输出： ['[SEP]', '[CLS]', '北', '京', '是', '中', '国', '的', '首', '都', '[SEP]'] 


# ==============================bert4torch=====================================
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.generation import AutoRegressiveDecoder, Seq2SeqGeneration
import os
import torch

# bert配置
config_path = f'{ckpt_dir}/bert4torch_config.json'
checkpoint_path = f'{ckpt_dir}/pytorch_model.bin'
dict_path = f'{ckpt_dir}/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
topk = 5
mode = 'beam_search'
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

    def generate(self, text, topk=4):
        token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
        token_ids = torch.tensor([token_ids], device=device)
        encoder_output = model.encoder.predict([token_ids])  # [encoder_hiddenstates, encoder_attention_mask]
        output_ids = self.beam_search(encoder_output, topk=topk)[0]  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy())
autotitle = AutoTitle(bos_token_id=102, eos_token_id=tokenizer._token_end_id, max_new_tokens=maxlen, device=device)
for text in texts:
    print(text, ' -> ', autotitle.generate(text, topk=topk))


print('==============默认单条无cache================')
autotitle = Seq2SeqGeneration(model, tokenizer, start_id=102, end_id=tokenizer._token_end_id, mode=mode,
                              maxlen=maxlen, default_rtype='logits', use_states=False)
for text in texts:
    print(text, ' -> ', autotitle.generate(text, topk=topk))


print('==============默认batch 无cache================')
results = autotitle.generate(texts, topk=topk)
for text, result in zip(texts, results):
    print(text, ' -> ', result)


print('==============默认单条cache================')
autotitle = Seq2SeqGeneration(model, tokenizer, start_id=102, end_id=tokenizer._token_end_id, mode=mode,
                              maxlen=maxlen, default_rtype='logits', use_states=True)
for text in texts:
    print(text, ' -> ', autotitle.generate(text, topk=topk))


print('==============默认batch cache================')
results = autotitle.generate(texts, topk=topk)
for text, result in zip(texts, results):
    print(text, ' -> ', result)


print('==============默认stream================')
text = texts[0]
for output in autotitle.stream_generate(text, topk=topk):
    os.system('clear')
    print(text, ' -> ', output, flush=True)
