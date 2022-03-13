## 简介
### 基础测试
- [basic_extract_features.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/basic_extract_features.py)：基础测试，测试BERT对句子的编码序列。
- [basic_gibbs_sampling_via_mlm.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/basic_gibbs_sampling_via_mlm.py)：基础测试，利用BERT+Gibbs采样进行文本随机生成，参考[这里](https://kexue.fm/archives/8119)。
- [basic_language_model_nezha_gen_gpt.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/basic_language_model_nezha_gen_gpt.py)：基础测试，测试[GPTBase（又叫NEZHE-GEN）](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow)的生成效果。
- [basic_make_uncased_model_cased.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/basic_make_uncased_model_cased.py)：基础测试，通过简单修改词表，使得不区分大小写的模型有区分大小写的能力。
- [basic_masked_language_model.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/basic_masked_language_model.py)：基础测试，测试BERT的MLM模型效果。

### 文本表示
- [task_sentence_embedding_sbert_lcqmc__ContrastiveLoss.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentence_embedding_sbert_lcqmc__ContrastiveLoss.py)：文本表示，参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sbert_sts_b__CosineSimilarityLoss.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentence_embedding_sbert_sts_b__CosineSimilarityLoss.py)：文本表示，参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sbert_sts_b__DimensionalityReduction.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentence_embedding_sbert_sts_b__DimensionalityReduction.py)：文本表示，参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sbert_sts_b__model_distillation.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentence_embedding_sbert_sts_b__model_distillation.py)：文本表示，参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sbert_unsupervised_CT.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentence_embedding_sbert_unsupervised_CT.py)：文本表示，参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sbert_unsupervised_CT_In-Batch_Negatives.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentence_embedding_sbert_unsupervised_CT_In-Batch_Negatives.py)：文本表示，参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sbert_unsupervised_SimCSE.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentence_embedding_sbert_unsupervised_SimCSE.py)：文本表示，参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sbert_unsupervised_TSDAE.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentence_embedding_sbert_unsupervised_TSDAE.py)：文本表示，参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sbert_xnli__concat_CrossEntropyLoss.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentence_embedding_sbert_xnli__concat_CrossEntropyLoss.py)：文本表示，参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sbert_xnli__MultiNegtiveRankingLoss.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentence_embedding_sbert_xnli__MultiNegtiveRankingLoss.py)：文本表示，参考[SentenceTransformer](https://www.sbert.net/index.html)

### 文本分类
- [task_sentence_similarity_lcqmc.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentence_similarity_lcqmc.py)：任务例子，句子对分类任务。
- [task_sentiment_classification.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentiment_classification.py)：任务例子，情感分类任务，bert做简单文本分类
- [task_sentiment_classification_albert.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentiment_classification_albert.py)：任务例子，情感分类任务，加载ALBERT模型。
- [task_sentiment_classification_hierarchical_position.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentiment_classification_hierarchical_position.py)：任务例子，情感分类任务，层次分解位置编码做长文本的初始化
- [task_sentiment_classification_nezha.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentiment_classification_nezha.py)：任务例子，情感分类任务，加载nezha模型
- [task_sentiment_classification_roformer.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sentiment_classification_roformer.py)：任务例子，情感分类任务，roformer

### 文本生成
- [task_seq2seq_autotitle.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_seq2seq_autotitle.py)：任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做新闻标题生成。
- [task_seq2seq_autotitle_bart.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_seq2seq_autotitle_bart.py)：任务例子，通过BART来做新闻标题生成
- [task_seq2seq_autotitle_csl.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_seq2seq_autotitle_csl.py)：任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做论文标题生成。
- [task_question_answer_generation_by_seq2seq.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_question_answer_generation_by_seq2seq.py)：任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做[问答对自动构建](https://kexue.fm/archives/7630)，属于自回归文本生成。
- [task_reading_comprehension_by_mlm.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_reading_comprehension_by_mlm.py)：任务例子，通过MLM模型来做[阅读理解问答](https://kexue.fm/archives/7148)，属于简单的非自回归文本生成。
- [task_reading_comprehension_by_seq2seq.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_reading_comprehension_by_seq2seq.py)：任务例子，通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做[阅读理解问答](https://kexue.fm/archives/7115)，属于自回归文本生成。

### 序列标注
- [task_sequence_labeling_ner_efficient_global_pointer.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sequence_labeling_ner_efficient_global_pointer.py)：任务例子，ner例子，efficient_global_pointer的pytorch实现
- [task_sequence_labeling_ner_global_pointer.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_sequence_labeling_ner_global_pointer.py)：任务例子，ner例子，global_pointer的pytorch实现

### 训练Trick
- [task_iflytek_adversarial_training.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_iflytek_adversarial_training.py)：任务例子，通过[对抗训练](https://kexue.fm/archives/7234)提升分类效果。
- [task_iflytek_bert_of_theseus.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_iflytek_bert_of_theseus.py)：任务例子，通过[BERT-of-Theseus](https://kexue.fm/archives/7575)来进行模型压缩。
- [task_iflytek_gradient_penalty.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_iflytek_gradient_penalty.py)：任务例子，通过[梯度惩罚](https://kexue.fm/archives/7234)提升分类效果，可以视为另一种对抗训练。

### 其他
- [task_conditional_language_model.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_conditional_language_model.py)：任务例子，结合BERT+[ConditionalLayerNormalization](https://kexue.fm/archives/7124)做条件语言模型。
- [task_language_model.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_language_model.py)：任务例子，加载BERT的预训练权重做无条件语言模型，效果上等价于GPT。
- [task_relation_extraction.py](https://github.com/Tongjilibo/bert4pytorch/blob/master/examples/task_relation_extraction.py)：任务例子，结合BERT以及自行设计的“半指针-半标注”结构来做[关系抽取](https://kexue.fm/archives/7161)。