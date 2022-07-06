# 模型文件地址
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

# 一阶段训练
data_dir = 'F:/Projects/data/corpus/qa/FinanceFAQ'
fst_train_file = f'{data_dir}/fst_train.tsv'
fst_dev_file = f'{data_dir}/fst_dev.tsv'
ir_path = f'{data_dir}/fst_ir_corpus.tsv'

# 一阶段预测结果
fst_q_std_file = f'{data_dir}/fst_q_std_file.tsv'
fst_q_corpus_file = f'{data_dir}/fst_q_corpus_file.tsv'
fst_q_sim_file = f'{data_dir}/fst_q_sim_file.tsv'
fst_q_std_vectors_file = f'{data_dir}/fst_q_std_vectors_file.npy'
fst_q_corpus_vectors_file = f'{data_dir}/fst_q_corpus_vectors_file.npy'
fst_std_data_results = f'{data_dir}/fst_std_data_results.tsv'

# 二阶段训练数据
sec_train_file =  f'{data_dir}/sec_train_file.tsv'
sec_dev_file = f'{data_dir}/sec_dev_file.tsv'
sec_test_file = f'{data_dir}/sec_test_file.tsv'