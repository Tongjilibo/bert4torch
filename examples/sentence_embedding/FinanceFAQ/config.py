# 模型文件地址
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

# 一阶段训练
fst_train_file = 'F:/Projects/data/corpus/qa/FinanceFAQ/FinanceFAQ_train.tsv'
fst_dev_file = 'F:/Projects/data/corpus/qa/FinanceFAQ/FinanceFAQ_dev.tsv'
ir_path = 'F:/Projects/data/corpus/qa/FinanceFAQ/ir_corpus.tsv'

# 一阶段预测结果
fst_q_std_file = './q_std_file.tsv'
fst_q_corpus_file = './q_std_file.tsv'
fst_q_sim_file = './q_sim_file.tsv'
fst_q_std_vectors_file = './fst_q_std_vectors_file.npy'
fst_q_corpus_vectors_file = './q_corpus_vectors_file.npy'
fst_std_data_results = './fst_std_data_results.tsv'

# 二阶段训练数据
sec_train_file =  './sec_train_file.tsv'
sec_dev_file = './sec_dev_file.tsv'
sec_test_file = './sec_test_file.tsv'