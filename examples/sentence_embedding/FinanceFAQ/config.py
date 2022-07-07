# 模型文件地址
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

data_dir = 'F:/Projects/data/corpus/qa/FinanceFAQ'
q_std_file = f'{data_dir}/q_std_file.tsv'  # 标准问数据
q_corpus_file = f'{data_dir}/q_corpus_file.tsv'  # 所有语料数据
q_sim_file = f'{data_dir}/q_sim_file.tsv'

# 一阶段训练
fst_train_file = f'{data_dir}/fst_train.tsv'
fst_dev_file = f'{data_dir}/fst_dev.tsv'
ir_path = f'{data_dir}/fst_ir_corpus.tsv'
fst_q_std_vectors_file = f'{data_dir}/fst_q_std_vectors_file.npy'
fst_q_corpus_vectors_file = f'{data_dir}/fst_q_corpus_vectors_file.npy'
fst_std_data_results = f'{data_dir}/fst_std_data_results.tsv'
fst_eval_path_list = [f'{data_dir}/fst_eval.tsv']

# 二阶段
sec_train_file =  f'{data_dir}/sec_train_file.tsv'
sec_dev_file = f'{data_dir}/sec_dev_file.tsv'
sec_test_file = f'{data_dir}/sec_test_file.tsv'
sec_q_std_vectors_file = f'{data_dir}/sec_q_std_vectors_file.npy'
sec_q_corpus_vectors_file = f'{data_dir}/sec_q_corpus_vectors_file.npy'
sec_eval_path_list = []