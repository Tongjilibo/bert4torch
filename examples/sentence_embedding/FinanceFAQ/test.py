import numpy as np
import pandas as pd
from config import *
from utils import *
from task_sentence_embedding_FinanceFAQ_step1_1 import model as model1

path_list = fst_eval_path_list

# 加载模型
print('读取标准问及其向量'.center(60, '-'))
q_std_list = pd.read_csv(q_std_file, sep="\t", names=['c']).c.tolist()
q_std_sentence_embeddings = np.load(fst_q_std_vectors_file)
print('标准问shape：', q_std_sentence_embeddings.shape, len(q_std_list))

print('读取所有语料及其向量'.center(60, '-'))
q_all = pd.read_csv(q_corpus_file, sep="\t", names=['c']).c.tolist()
q_all_sentence_embeddings = np.load(fst_q_corpus_vectors_file)
q_all_sentence_embeddings_dict = {q_all[i]: q_all_sentence_embeddings[i] for i in range(0, len(q_all))}
print('所有语料shape', q_all_sentence_embeddings.shape, len(q_all))

for i, input_path in enumerate(path_list):
    print(f'开始评估新语料: {i}'.center(120, '='))
    df_eval = pd.read_csv(input_path, sep="\t")
    df_eval = df_eval[~pd.isna(df_eval.q_sim)]
    output_path = input_path[:-4] + '_result.tsv'
    print('input_path: ', input_path, 'output_path: ', output_path)

    print("目标语料数量：", df_eval.shape, '标问数量：', df_eval.q_std.nunique(), '相似问数量：',
          df_eval.q_sim.nunique(), '标语料去重后数量', df_eval.drop_duplicates(["q_std", "q_sim"]).shape[0])
    texts = df_eval.q_sim.tolist()
    texts_in = [v for v in texts if v in q_all_sentence_embeddings_dict.keys()]
    texts_out = [v for v in texts if v not in q_all_sentence_embeddings_dict.keys()]
    texts_out_embeddings = model1.encode(texts_out) if texts_out else []
    texts_embeddings_dict_1 = {texts_in[i]: q_all_sentence_embeddings_dict[texts_in[i]] for i in range(0, len(texts_in))}
    texts_embeddings_dict_2 = {texts_out[i]: texts_out_embeddings[i] for i in range(0, len(texts_out))}
    texts_embeddings_dict = {**texts_embeddings_dict_1, **texts_embeddings_dict_2}
    print('目标语料编码后数量：——>', len(texts_embeddings_dict))

    ## v1 对于都是有一个是小量的情况下
    df_eval = cal_performance(texts_embeddings_dict, q_std_sentence_embeddings, q_std_list, texts, df_eval, K=10)
    df_eval.to_csv(output_path, index=None, sep="\t")
