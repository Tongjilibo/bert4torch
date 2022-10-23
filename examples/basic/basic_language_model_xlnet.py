from transformers import XLNetTokenizer, XLNetModel

pretrained_model = "F:/Projects/pretrain_ckpt/xlnet/[hit_torch_base]--chinese-xlnet-base"
tokenizer = XLNetTokenizer.from_pretrained(pretrained_model)
model = XLNetModel.from_pretrained(pretrained_model)

inputs = tokenizer(["你好啊，我叫张三", "天气不错啊"], padding=True, return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print('--------transformers last_hidden_state--------\n', last_hidden_states)

# ----------------------bert4torch配置----------------------
from bert4torch.models import build_transformer_model
config_path = f'{pretrained_model}/bert4torch_config.json'
checkpoint_path = f'{pretrained_model}/pytorch_model.bin'

model = build_transformer_model(
    config_path,
    checkpoint_path=checkpoint_path,
    model='xlnet',
    # with_lm=True
    token_pad_ids=tokenizer.pad_token_id,
)

print('--------bert4torch last_hidden_state--------\n', model.predict([inputs['input_ids'], inputs['token_type_ids']]))