root_model_path = 'E:\pretrain_ckpt\embedding\moka-ai@m3e-base'

print('=========================================sentence transformer====================================')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(root_model_path)
sentences = [
    '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one',
    '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
    '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练'
]

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("---------------------------------")


print('=========================================bert4torch====================================')
from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding, get_pool_emb
from bert4torch.tokenizers import Tokenizer
import torch

# 加载模型，请更换成自己的路径
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/bert4torch_config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'


# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

token_ids, segments_ids = tokenizer.encode(sentences)
tokens_ids_tensor = torch.tensor(sequence_padding(token_ids))
segment_ids_tensor = torch.tensor(sequence_padding(segments_ids))

model.eval()
with torch.no_grad():
    hidden_states, pooling = model([tokens_ids_tensor, segment_ids_tensor])
    sentence_embeddings = get_pool_emb(hidden_states, pooling, tokens_ids_tensor.gt(0).long(), 'mean')

    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("---------------------------------")
