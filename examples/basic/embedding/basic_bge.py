root_model_path = 'E:\pretrain_ckpt\embedding\BAAI@bge-large-en-v1.5'

print('=========================================sentence transformer====================================')
from sentence_transformers import SentenceTransformer
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = SentenceTransformer(root_model_path)
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)


print('=========================================bert4torch====================================')
from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding, get_pool_emb
from bert4torch.tokenizers import Tokenizer
import torch
import torch.nn.functional as F


# 加载模型，请更换成自己的路径
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/bert4torch_config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'

# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

def get_emb(sentences):
    token_ids, segments_ids = tokenizer.encode(sentences)
    tokens_ids_tensor = torch.tensor(sequence_padding(token_ids))
    segment_ids_tensor = torch.tensor(sequence_padding(segments_ids))

    model.eval()
    with torch.no_grad():
        hidden_states, pooling = model([tokens_ids_tensor, segment_ids_tensor])
        sentence_embeddings = get_pool_emb(hidden_states, pooling, tokens_ids_tensor.gt(0).long(), 'cls')
        return F.normalize(sentence_embeddings, p=2, dim=1)

embeddings_1 = get_emb(sentences_1)
embeddings_2 = get_emb(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
