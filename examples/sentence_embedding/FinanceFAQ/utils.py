import torch
from torch import Tensor
import numpy as np
import pandas as pd

def pytorch_cos_sim(a: Tensor, b: Tensor):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def cos_sim_1(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos

def cos_sim4matrix(arr, brr):
    return 0.5 + 0.5 * (arr.dot(brr.T) / (np.sqrt(np.sum(arr * arr)) * np.sqrt(np.sum(brr * brr, axis = 1))))

def cos_sim4matrix_2(arr, brr):
    return (arr.dot(brr.T) / (np.sqrt(np.sum(arr * arr)) * np.sqrt(np.sum(brr * brr, axis=1))))

def read_q_std_q_corpus(q_std_file, q_std_vectors_file, q_corpus_file, q_corpus_vectors_file):
    '''读取q_std、q_corpus语料和向量
    '''
    print('读取标准问及其向量'.center(60, '-'))
    q_std_list = pd.read_csv(q_std_file, sep="\t", names=['c']).c.tolist()
    q_std_sentence_embeddings = np.load(q_std_vectors_file)
    print('标准问shape：', q_std_sentence_embeddings.shape, len(q_std_list))

    print('读取所有语料及其向量'.center(60, '-'))
    q_all = pd.read_csv(q_corpus_file, sep="\t", names=['c']).c.tolist()
    q_all_sentence_embeddings = np.load(q_corpus_vectors_file)
    q_all_sentence_embeddings_dict = {q_all[i]: q_all_sentence_embeddings[i] for i in range(0, len(q_all))}
    print('所有语料shape', q_all_sentence_embeddings.shape, len(q_all))
    return q_std_list, q_std_sentence_embeddings, q_all, q_all_sentence_embeddings_dict

def cal_performance(model, q_all_sentence_embeddings_dict, q_std_sentence_embeddings, q_std_list, df_eval, K=20):
    '''计算召回topK的指标
    '''
    texts = df_eval.q_sim.tolist()
    texts_in = [v for v in texts if v in q_all_sentence_embeddings_dict.keys()]
    texts_out = [v for v in texts if v not in q_all_sentence_embeddings_dict.keys()]
    texts_out_embeddings = model.encode(texts_out) if texts_out else []
    texts_embeddings_dict_1 = {texts_in[i]: q_all_sentence_embeddings_dict[texts_in[i]] for i in range(0, len(texts_in))}
    texts_embeddings_dict_2 = {texts_out[i]: texts_out_embeddings[i] for i in range(0, len(texts_out))}
    texts_embeddings_dict = {**texts_embeddings_dict_1, **texts_embeddings_dict_2}

    print(f'计算相似度 K= {K}'.center(60, '-'))
    df_eval['ifin'] = df_eval.q_std.apply(lambda v: 1 if v in q_std_list else 0)
    print("目标语料标问是否存在：——>", df_eval.groupby("ifin")["ifin"].count())

    print('----计算所有query和q_std的相似度')
    x_texts_embeddings = np.array([texts_embeddings_dict[x_text] for x_text in texts])
    cos_scores = pytorch_cos_sim(x_texts_embeddings, q_std_sentence_embeddings).cpu()
    print('shape: ', x_texts_embeddings.shape, q_std_sentence_embeddings.shape, cos_scores.shape)

    print(f'----为每条相似问找到相似度最大的{K}条标问'.center(60, '-'))
    cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, K, dim=1, largest=True, sorted=False)
    cos_scores_top_k_values = cos_scores_top_k_values.tolist()
    cos_scores_top_k_idx = cos_scores_top_k_idx.tolist()
    cos_q_corpus_sort = [[q_std_list[v] for v in vlist] for vlist in cos_scores_top_k_idx]  # 最相似的TopK个标问
    result = [list(zip(cos_q_corpus_sort[i], cos_scores_top_k_values[i])) for i in range(0, len(texts))]
    texts_topk_dict = {texts[i]: result[i] for i in range(0, len(texts))}

    # 拿到每个相似问的预测结果，topK的预测标问和对应的相似度
    df_eval['q_std_pred_list'] = df_eval.q_sim.map(texts_topk_dict)
    # 计算q_sim和q_std之间的相似度
    df_eval['prob_with_std'] = df_eval.apply(lambda row: cos_sim_1(texts_embeddings_dict[row['q_sim']], q_std_sentence_embeddings[q_std_list.index(row['q_std'])]), axis=1)
    df_eval.loc[:, 'q_std_pred'] = df_eval.q_std_pred_list.apply(lambda v: v[0][0])
    df_eval.loc[:, 'prob'] = df_eval.q_std_pred_list.apply(lambda v: v[0][1])
    # df_eval.loc[:,'q_std_pred_list_pair']=df_eval.apply(lambda row: [(row['q_std'],row['q_sim'],v[0],v[1]) for v in row['q_std_pred_list']],axis=1)
    df_eval['q_std_pred_list_v1'] = df_eval.q_std_pred_list.apply(lambda v: [k[0] for k in v])  # 只保留预测的标准问句
    df_eval['q_std_pred_list_v2'] = df_eval.q_std_pred_list.apply(lambda v: [k[1] for k in v])  # 只保留预测的概率
    df_eval['t1'] = df_eval.apply(lambda row: 1 if row['q_std'] in row['q_std_pred_list_v1'][0:1] else 0, axis=1)
    df_eval['t3'] = df_eval.apply(lambda row: 1 if row['q_std'] in row['q_std_pred_list_v1'][0:3] else 0, axis=1)
    df_eval['t5'] = df_eval.apply(lambda row: 1 if row['q_std'] in row['q_std_pred_list_v1'][0:5] else 0, axis=1)
    df_eval['t10'] = df_eval.apply(lambda row: 1 if row['q_std'] in row['q_std_pred_list_v1'][0:10] else 0, axis=1)

    print('----模型准确率: ', df_eval.t1.sum() / df_eval.shape[0], df_eval.t3.sum() / df_eval.shape[0], df_eval.t5.sum() / df_eval.shape[0], df_eval.t10.sum() / df_eval.shape[0])
    df_eval_need = df_eval[df_eval.ifin == 1]
    print('----模型准确率:[有效标问]：', df_eval_need.t1.sum() / df_eval_need.shape[0], df_eval_need.t3.sum() / df_eval_need.shape[0], df_eval_need.t5.sum() / df_eval_need.shape[0], df_eval_need.t10.sum() / df_eval_need.shape[0])
    return df_eval