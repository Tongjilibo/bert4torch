# 一、example简介

## 基础测试

- [basic_test_tokenizer.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_test_tokenizer.py): 测试tokenizer和transformers包的结果一致。
- [basic_test_parallel_apply.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_test_parallel_apply.py): 测试parallel_apply的效果。
- [basic_extract_features.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_extract_features.py)：测试BERT对句子的编码序列。
- [basic_gibbs_sampling_via_mlm.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_gibbs_sampling_via_mlm.py)：利用BERT+Gibbs采样进行文本随机生成，参考[这里](https://kexue.fm/archives/8119)。
- [basic_language_model_nezha_gen_gpt.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_nezha_gen_gpt.py)：测试[GPTBase（又叫NEZHE-GEN）](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow)的生成效果。
- [basic_make_uncased_model_cased.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_make_uncased_model_cased.py)：通过简单修改词表，使得不区分大小写的模型有区分大小写的能力。
- [basic_language_model_bert.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_bert.py)：测试BERT的MLM模型效果。
- [basic_language_model_roberta_small.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_roberta_small.py)：测试Roberta-small的MLM模型效果。
- [basic_language_model_roberta_english.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_roberta_english.py)：测试英文版Roberta的MLM模型效果。
- [basic_language_model_ernie.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_ernie.py)：测试百度文心ERNIE的MLM模型效果。
- [basic_language_model_GAU_alpha.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_GAU_alpha.py)：测试[GAU-alpha](https://github.com/ZhuiyiTechnology/GAU-alpha)的MLM模型效果。
- [basic_language_model_roformer.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_roformer.py)：测试roformer的MLM模型效果。
- [basic_language_model_CDial_GPT.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_CDial_GPT.py)：测试[CDial_GPT](https://github.com/thu-coai/CDial-GPT)的对话生成效果。
- [basic_language_model_gpt2_ml.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_gpt2_ml.py)：测试[gpt2-ml](https://github.com/imcaspar/gpt2-ml)的的生成效果。
- [basic_language_model_uer_gpt2_chinese.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_uer_gpt2_chinese.py)：测试[uer-gpt2-chinese](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)的的生成效果。
- [basic_language_model_cpm_lm.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_cpm_lm.py)：测试[CPM-Generate](https://github.com/TsinghuaAI/CPM-Generate)的的生成效果。
- [basic_language_model_uer_t5.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_uer_t5.py)：测试[uer-t5-small](https://huggingface.co/uer/t5-small-chinese-cluecorpussmall)和[uer-t5-base](https://huggingface.co/uer/t5-base-chinese-cluecorpussmall)的生成效果。
- [basic_language_model_t5_pegasus.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_t5_pegasus.py)：测试[t5_pegasus](https://github.com/ZhuiyiTechnology/t5-pegasus)的生成效果。
- [basic_language_model_simbert.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_simbert.py)：测试[simbert](https://github.com/ZhuiyiTechnology/simbert)和[roformer-sim](https://github.com/ZhuiyiTechnology/roformer-sim)的生成效果和句子相似度效果。
- [basic_language_model_transformer_xl.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_transformer_xl.py): 测试transformer_xl模型，做了一些简化，仅有英文预训练模型。
- [basic_language_model_xlnet.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_xlnet.py): 测试xlnet模型。
- [basic_language_model_nezha_gpt_dialog.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_nezha_gpt_dialog.py): 测试[nezha_gpt_dialog](https://kexue.fm/archives/7718)。
- [basic_language_model_bart.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_bart.py): 测试bart模型。
- [basic_language_model_deberta_v2.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_deberta_v2.py): 测试deberta_v2模型。
- [basic_language_model_guwenbert.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_guwenbert.py): 测试[古文bert](https://huggingface.co/ethanyt/guwenbert-base)模型。
- [basic_language_model_wobert.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_wobert.py): 测试wobert模型。
- [basic_language_model_chatyuan.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatyuan.py): 测试[ChatYuan](https://github.com/clue-ai/ChatYuan)模型。
- [basic_language_model_PromptCLUE.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_PromptCLUE.py): 测试[PromptCLUE](https://github.com/clue-ai/PromptCLUE)模型。
- [basic_language_model_albert.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_albert.py): 测试[albert_chinese](https://github.com/brightmart/albert_zh)模型。
- [basic_language_model_llama.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_llama.py): 测试[llama](https://github.com/facebookresearch/llama)模型。
- [basic_language_model_vicuna.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_vicuna.py): 测试[vicuna](https://huggingface.co/AlekseyKorshuk/vicuna-7b)模型。
- [basic_language_model_belle.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_belle.py): 测试[belle](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M)模型。
- [basic_language_model_chatglm.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型。
- [basic_language_model_chatglm_stream.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_stream.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, stream方式输出。
- [basic_language_model_chatglm_webdemo.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_webdemo.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, web方式输出。
- [basic_language_model_chatglm_batch.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_batch.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, batch方式输出。
- [basic_language_model_moss.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_moss.py): 测试[moss](https://github.com/OpenLMLab/MOSS)模型, int4和int8低成本部署。

## LLM
- [task_chatglm_ptuning_v2.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_ptuning_v2.py): [chatglm-6b](https://github.com/THUDM/ChatGLM-6B)的ptuning_v2微调。
- [task_chatglm_lora.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_lora.py): [chatglm-6b](https://github.com/THUDM/ChatGLM-6B)的lora微调(基于peft)。

## 文本分类

- [task_sentence_similarity_lcqmc.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentence_similarity_lcqmc.py)：句子对分类任务。
- [task_sentiment_classification.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification.py)：情感分类任务，bert做简单文本分类
- [task_sentiment_classification_albert.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification_albert.py)：情感分类任务，加载ALBERT模型。
- [task_sentiment_classification_xlnet.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification_xlnet.py)：情感分类任务，加载XLNET模型。
- [task_sentiment_classification_hierarchical_position.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification_hierarchical_position.py)：情感分类任务，层次分解位置编码做长文本的初始化
- [task_sentiment_classification_nezha.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification_nezha.py)：情感分类任务，加载nezha模型
- [task_sentiment_classification_roformer.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification_roformer.py)：情感分类任务，加载roformer权重
- [task_sentiment_classification_roformer_v2.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification_roformer_v2.py)：情感分类任务，加载roformer_v2权重
- [task_sentiment_classification_electra.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification_electra.py)：情感分类任务，加载electra权重
- [task_sentiment_classification_GAU_alpha.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification_GAU_alpha.py)：情感分类任务，加载GAU-alpha权重
- [task_sentiment_classification_deberta_v2.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification_deberta_v2.py)：情感分类任务，加载deberta_v2权重
- [task_sentiment_classification_wobert.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification_wobert.py)：情感分类任务，加载wobert权重
- [task_sentiment_classification_PET.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification_PET.py)：情感分类项目，[Pattern-Exploiting-Training](https://github.com/bojone/Pattern-Exploiting-Training), [bert4keras示例](https://github.com/bojone/Pattern-Exploiting-Training)
- [task_sentiment_classification_P_tuning.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification_P_tuning.py)：情感分类项目，[P-tuning](https://github.com/THUDM/P-tuning), [bert4keras示例](https://github.com/bojone/P-tuning)
- [Sohu_2022_ABSA](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/Sohu_2022_ABSA)：搜狐2022实体情感分类Top1方案复现和自己的baseline
- [Tianchi_News_Classification](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/Tianchi_News_Classification)：天池零基础入门NLP-新闻分类Top1方案复现

## 序列标注

- [task_sequence_labeling_ner_efficient_global_pointer.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_efficient_global_pointer.py)：ner例子，efficient_global_pointer的pytorch实现
- [task_sequence_labeling_ner_global_pointer.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_global_pointer.py)：ner例子，global_pointer的pytorch实现
- [task_sequence_labeling_ner_crf.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_crf.py)：ner例子，bert+crf
- [task_sequence_labeling_ner_crf_freeze.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_crf_freeze.py)：ner例子，bert+crf, 一种是用数据集来生成crf权重，第二种是来初始化
- [task_sequence_labeling_ner_cascade_crf.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_cascade_crf.py)：ner例子，bert+crf+级联
- [task_sequence_labeling_ner_crf_add_posseg.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_crf_add_posseg.py)：ner例子，bert+crf，词性作为输入
- [task_sequence_labeling_ner_tplinker_plus.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_tplinker_plus.py)：ner例子，改造了关系抽取[TPLinker](https://github.com/131250208/TPlinker-joint-extraction)
- [task_sequence_labeling_ner_mrc.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_mrc.py)：ner例子，[mrc方案](https://github.com/z814081807/DeepNER)，用阅读理解的方式来做
- [task_sequence_labeling_ner_span.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_span.py)：ner例子，[span方案](https://github.com/z814081807/DeepNER)，用半指针-半标注方式来做
- [uie](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/uie)：ner例子，[uie方案](https://github.com/universal-ie/UIE)，prompt+mrc模型结构
- [task_sequence_labeling_ner_W2NER.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_W2NER.py)：ner例子，[W2NER](https://github.com/ljynlp/W2NER)
- [task_sequence_labeling_ner_CNN_Nested_NER.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_CNN_Nested_NER.py)：ner例子，[CNN_Nested_NER](https://github.com/yhcc/CNN_Nested_NER)
- [task_sequence_labeling_ner_lear.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_lear.py)：ner例子，[LEAR](https://github.com/Akeepers/LEAR)
- [task_sequence_labeling_cws_crf.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_cws_crf.py)：crf分词例子

## 文本表示

- [task_sentence_embedding_unsup_bert_whitening.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_unsup_bert_whitening.py)：参考[bert_whitening](https://github.com/bojone/BERT-whitening)
- [task_sentence_embedding_unsup_CT.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_unsup_CT.py)：参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_unsup_CT_In-Batch_Negatives.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_unsup_CT_In-Batch_Negatives.py)：参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_unsup_SimCSE.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_unsup_SimCSE.py)：参考[SentenceTransformer](https://www.sbert.net/index.html)和[科学空间版中文测试](https://kexue.fm/archives/8348)
- [task_sentence_embedding_unsup_ESimCSE.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_unsup_ESimCSE.py)：参考[ESimCSE论文](https://arxiv.org/pdf/2109.04380.pdf)和[第三方实现](https://github.com/shuxinyin/SimCSE-Pytorch)
- [task_sentence_embedding_unsup_TSDAE.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_unsup_TSDAE.py)：参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_unsup_PromptBert.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_unsup_PromptBert.py)：[PromptBert](https://github.com/kongds/Prompt-BERT)方式
- [task_sentence_embedding_unsup_DiffCSE.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_unsup_DiffCSE.py)：[DiffCSE](https://github.com/voidism/DiffCSE)
- [task_sentence_embedding_sup_ContrastiveLoss.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_sup_ContrastiveLoss.py)：参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sup_CosineMSELoss.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_sup_CosineMSELoss.py)：参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sup_concat_CrossEntropyLoss.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_sup_concat_CrossEntropyLoss.py)：参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sup_MultiNegtiveRankingLoss.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_sup_MultiNegtiveRankingLoss.py)：参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_sup_CoSENT.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_sup_CoSENT.py)：参考[CoSENT](https://kexue.fm/archives/8847)
- [task_sentence_embedding_DimensionalityReduction.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_DimensionalityReduction.py)：参考[SentenceTransformer](https://www.sbert.net/index.html)
- [task_sentence_embedding_model_distillation.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/task_sentence_embedding_model_distillation.py)：参考[SentenceTransformer](https://www.sbert.net/index.html)
- [FinanceFAQ](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_embedding/FinanceFAQ)：金融领域FAQ两阶段(召回+排序)pipline

## 关系提取

- [task_relation_extraction_CasRel.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/relation_extraction/task_relation_extraction_CasRel.py)：结合BERT以及自行设计的“半指针-半标注”结构来做[关系抽取](https://kexue.fm/archives/7161)。
- [task_relation_extraction_gplinker.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/relation_extraction/task_relation_extraction_gplinker.py)：结合GlobalPointer做关系抽取[GPLinker](https://kexue.fm/archives/8888)。
- [task_relation_extraction_tplinker.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/relation_extraction/task_relation_extraction_tplinker.py)：tplinker关系抽取[TPLinker](https://github.com/131250208/TPlinker-joint-extraction)。
- [task_relation_extraction_tplinker_plus.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/relation_extraction/task_relation_extraction_tplinker_plus.py)：tplinker关系抽取[TPLinkerPlus](https://github.com/131250208/TPlinker-joint-extraction)。
- [task_relation_extraction_SPN4RE.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/relation_extraction/task_relation_extraction_SPN4RE.py)：[SPN4RE](https://github.com/DianboWork/SPN4RE)来做关系提取。
- [task_relation_extraction_PGRC.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/relation_extraction/task_relation_extraction_PGRC.py)：[PGRC](https://github.com/hy-struggle/PRGC)来做关系提取。

## 文本生成

- [task_seq2seq_autotitle_csl_unilm.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_seq2seq_autotitle_csl_unilm.py)：通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做新闻标题生成。
- [task_seq2seq_autotitle_csl_bart.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_seq2seq_autotitle_csl_bart.py)：通过BART来做新闻标题生成
- [task_seq2seq_autotitle_csl_uer_t5.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_seq2seq_autotitle_csl_uer_t5.py)：通过T5来做新闻标题生成，用的[uer-t5-small](https://huggingface.co/uer/t5-small-chinese-cluecorpussmall)
- [task_seq2seq_autotitle_csl.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_seq2seq_autotitle_csl.py)：通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做论文标题生成。
- [task_seq2seq_autotitle_csl_mt5.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_seq2seq_autotitle_csl_mt5.py)：通过[google_mt](https://huggingface.co/google/mt5-base)的Seq2Seq模型来做论文标题生成。
- [task_question_answer_generation_by_seq2seq.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_question_answer_generation_by_seq2seq.py)：通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做[问答对自动构建](https://kexue.fm/archives/7630)，属于自回归文本生成。
- [task_reading_comprehension_by_mlm.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_reading_comprehension_by_mlm.py)：通过MLM模型来做[阅读理解问答](https://kexue.fm/archives/7148)，属于简单的非自回归文本生成。
- [task_reading_comprehension_by_seq2seq.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_reading_comprehension_by_seq2seq.py)：通过[UniLM](https://kexue.fm/archives/6933)式的Seq2Seq模型来做[阅读理解问答](https://kexue.fm/archives/7115)，属于自回归文本生成。
- [task_seq2seq_simbert.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_seq2seq_simbert.py)：相似问生成，数据增广，参考[SimBERT](https://kexue.fm/archives/7427)
- [task_seq2seq_ape210k_math_word_problem.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_seq2seq_ape210k_math_word_problem.py)：bert+unilm硬刚小学数学题，参考[博客](https://kexue.fm/archives/7809)
- [task_kgclue_seq2seq.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_kgclue_seq2seq.py)：seq2seq+前缀树，参考[博客](https://kexue.fm/archives/8802)
- [task_dialogpt_finetune.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_dialogpt_finetune.py)：基于dialogpt的微调,同时提供微调数据格式, 参考[博客](https://kexue.fm/archives/8802)
- [task_promptclue_finetune.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_promptclue_finetune.py)：基于promptclue-base-v1.5的微调

## 训练Trick
- [task_sentiment_adapters.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/task_sentiment_adapters.py)：基于adapter的插拔式训练
- [task_sentiment_adversarial_training.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/task_sentiment_adversarial_training.py)：通过对抗训练，虚拟对抗训练，梯度惩罚等措施来提升分类效果。
- [task_sentiment_virtual_adversarial_training.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/task_sentiment_virtual_adversarial_training.py)：通过半监督的虚拟对抗训练等措施来提升分类效果。
- [task_sentiment_UDA.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/task_sentiment_UDA.py)：通过[UDA](https://arxiv.org/abs/1904.12848)半监督学习提升分类效果，在原来Losss上加一致性损失。
- [task_sentiment_mixup.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/task_sentiment_mixup.py)：通过[Mixup](https://github.com/vikasverma1077/manifold_mixup)提升模型泛化性能。
- [task_sentiment_exponential_moving_average.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/task_sentiment_exponential_moving_average.py)：EMA指数滑动平均
- [task_sentiment_exponential_moving_average_warmup.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/task_sentiment_exponential_moving_average_warmup.py)：EMA指数滑动平均+warmup两种策略
- [task_sentiment_TemporalEnsembling.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/task_sentiment_TemporalEnsembling.py)：通过[TemporalEnsembling官方项目](https://github.com/s-laine/tempens)和[pytorch第三方实现](https://github.com/ferretj/temporal-ensembling)提升模型泛化性能。
- [task_sentiment_R-Drop.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/task_sentiment_R-Drop.py)：通过[R-Drop](https://github.com/dropreg/R-Drop)提升分类效果，可以视为用dropout加噪下的UDA。
- [task_amp.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/task_amp.py)：Pytorch的amp混合精度训练
- [task_data_parallel.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/task_data_parallel.py)：DataParallel模式的多GPU训练方式
- [task_distributed_data_parallel.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/task_distributed_data_parallel.py)：DistributedDataParallel模式的多GPU训练方式
- [accelerate](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick/accelerate)：配合accelerate包使用

## 预训练

- [roberta_pretrain](https://github.com/Tongjilibo/bert4torch/blob/master/examples/pretrain/roberta_pretrain)：roberta的mlm预训练，数据生成代码和训练代码
- [simbert_v2_pretrain](https://github.com/Tongjilibo/bert4torch/blob/master/examples/pretrain/simbert_v2_pretrain)：相似问生成，数据增广，三个步骤：1-[弱监督](https://github.com/Tongjilibo/bert4torch/blob/master/examples/pretrain/simbert_v2_pretrain/simbert_v2_stage1.py)，2-[蒸馏](https://github.com/Tongjilibo/bert4torch/blob/master/examples/pretrain/simbert_v2_pretrain/simbert_v2_stage2.py)，3-[有监督](https://github.com/Tongjilibo/bert4torch/blob/master/examples/pretrain/simbert_v2_pretrain/simbert_v2_supervised.py)，参考[SimBERT-V2](https://kexue.fm/archives/8454)
- [gpt_lm_pretrain](https://github.com/Tongjilibo/bert4torch/blob/master/examples/pretrain/gpt_lm_pretrain)：gpt的lm预训练

## 模型部署

- [basic_simple_web_serving_simbert.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/serving/basic_simple_web_serving_simbert.py): 测试自带的WebServing（将模型转化为Web接口）。
- [task_bert_cls_onnx.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/serving/task_bert_cls_onnx.py)：ONNX转换bert权重
- [task_bert_cls_onnx_tensorrt.md](https://github.com/Tongjilibo/bert4torch/blob/master/examples/serving/task_bert_cls_onnx_tensorrt.md)：ONNX+Tensorrt部署
- [sanic_server](https://github.com/Tongjilibo/bert4torch/blob/master/examples/serving/sanic_server)：sanic+onnx部署
- [elasticsearch](https://github.com/Tongjilibo/bert4torch/blob/master/examples/serving/elasticsearch)：elasticsearch部署
- [triton](https://github.com/Tongjilibo/bert4torch/blob/master/examples/serving/triton)：triton部署

## 其他

- [task_conditional_language_model.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/others/task_conditional_language_model.py)：结合BERT+[ConditionalLayerNormalization](https://kexue.fm/archives/7124)做条件语言模型。
- [task_language_model.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/others/task_language_model.py)：加载BERT的预训练权重做无条件语言模型，效果上等价于GPT。
- [task_iflytek_bert_of_theseus.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/others/task_iflytek_bert_of_theseus.py)：通过[BERT-of-Theseus](https://kexue.fm/archives/7575)来进行模型压缩。
- [task_language_model_chinese_chess.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/others/task_language_model_chinese_chess.py)：用GPT的方式下中国象棋，过程请参考[博客](https://kexue.fm/archives/7877)。
- [task_nl2sql_baseline.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/others/task_nl2sql_baseline.py)：[追一科技2019年NL2SQL挑战赛的一个Baseline](https://kexue.fm/archives/6771)
- [task_event_extraction_gplinker.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/others/task_event_extraction_gplinker.py)：gplinker来做事件提取

## 教程

- [Tutorials](https://github.com/Tongjilibo/bert4torch/blob/master/examples/tutorials/Tutorials)：教程说明文档。
- [tutorials_custom_fit_progress.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/tutorials/tutorials_custom_fit_progress.py)：教程，自定义训练过程fit函数（集成了训练进度条展示），可用于满足如半精度，梯度裁剪等高阶需求。
- [tutorials_load_transformers_model.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/tutorials/tutorials_load_transformers_model.py)：教程，加载transformer包中模型，可以使用bert4torch中继承的对抗训练等trick。
- [tutorials_small_tips.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/tutorials/tutorials_small_tips.py)：教程，常见的一些tips集合。

# 二、数据集


| 数据集名称     | 用途               | 下载链接                                                                                                                        |
| ---------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 人民日报数据集 | 实体识别           | [china-people-daily-ner-corpus](http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz)                               |
| 百度关系抽取   | 关系抽取           | [BD_Knowledge_Extraction](http://ai.baidu.com/broad/download?dataset=sked)                                                      |
| Sentiment      | 情感分类           | [Sentiment](https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip)                                   |
| THUCNews       | 文本分类、文本生成 | [THUCNews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews) |
| ATEC           | 文本相似度         | [ATEC](https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC)                                                           |
| BQ             | 文本相似度         | [BQ](http://icrc.hitsz.edu.cn/info/1037/1162.htm)                                                                               |
| LCQMC          | 文本相似度         | [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html)                                                                         |
| PAWSX          | 文本相似度         | [PAWSX](https://arxiv.org/abs/1908.11828)                                                                                       |
| STS-B          | 文本相似度         | [STS-B](https://github.com/pluto-junzeng/CNSD)                                                                                  |
| CSL            | 文本生成           | [CSL](https://github.com/CLUEbenchmark/CLGE)                                                                                    |

# 三、指标测试

## 1. 文本分类

### 1.1 不同预训练模型的指标对比

- [情感分类数据集](https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip)+cls位分类


| solution     | epoch       | valid_acc         | test_acc          | comment                                                    |
| -------------- | ------------- | ------------------- | ------------------- | ------------------------------------------------------------ |
| albert_small | 10/10       | 94.46             | 93.98             | small版本                                                  |
| bert         | 6/10        | 94.72             | 94.11             | ——                                                       |
| robert       | 4/10        | 94.77             | 94.64             | ——                                                       |
| nezha        | 7/10        | 95.07             | 94.72             | ——                                                       |
| xlnet        | 6/10        | 95.07             | 94.77             | ——                                                       |
| electra      | 10/10       | 94.94             | 94.78             | ——                                                       |
| roformer     | 9/10        | 94.85             | 94.42             | ——                                                       |
| roformer_v2  | 3/10        | 95.78             | 96.09             | ——                                                       |
| gau_alpha    | 2/10        | 95.25             | 94.46             | ——                                                       |
| deberta_v2   | (10/2/5)/10 | 94.55/95.16/94.85 | 94.99/94.68/94.33 | 分别为97M/320M/710M, 97M的transformer版本指标为94.90/94.81 |

### 1.2 不同trick下的指标对比

- trick测试+[情感分类数据集](https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip)+cls分类+无segment_input


| solution            | epoch | valid_acc | test_acc | comment |
| --------------------- | ------- | ----------- | ---------- | --------- |
| bert                | 10/10 | 94.90     | 94.78    | ——    |
| fgm                 | 4/10  | 95.34     | 94.99    | ——    |
| pgd                 | 6/10  | 95.34     | 94.64    | ——    |
| gradient_penalty    | 7/10  | 95.07     | 94.81    | ——    |
| vat                 | 8/10  | 95.21     | 95.03    | ——    |
| ema                 | 7/10  | 95.21     | 94.86    | ——    |
| ema+warmup          | 7/10  | 95.51     | 95.12    | ——    |
| mix_up              | 6/10  | 95.12     | 94.42    | ——    |
| R-drop              | 9/10  | 95.25     | 94.94    | ——    |
| UDA                 | 8/10  | 94.90     | 95.56    | ——    |
| semi-vat            | 10/10 | 95.34     | 95.38    | ——    |
| temporal_ensembling | 8/10  | 94.94     | 94.90    | ——    |

## 2. 序列标注

- [人民日报数据集](http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz)+bert预训练模型
- valid集指标


| solution                      | epoch | f1_token | f1_entity | comment                                                      |
| ------------------------------- | ------- | ---------- | ----------- | -------------------------------------------------------------- |
| bert+crf                      | 18/20 | 96.89    | 96.05     | ——                                                         |
| bert+crf+init                 | 18/20 | 96.93    | 96.08     | 用训练数据初始化crf权重                                      |
| bert+crf+freeze               | 11/20 | 96.89    | 96.13     | 用训练数据生成crf权重(不训练)                                |
| bert+cascade+crf              | 5/20  | 98.10    | 96.26     | crf类别少所以f1_token偏高                                    |
| bert+crf+posseg               | 13/20 | 97.32    | 96.55     | 加了词性输入                                                 |
| bert+global_pointer           | 18/20 | ——     | 95.66     | ——                                                         |
| bert+efficient_global_pointer | 17/20 | ——     | 96.55     | ——                                                         |
| bert+mrc                      | 7/20  | ——     | 95.75     | ——                                                         |
| bert+span                     | 13/20 | ——     | 96.31     | ——                                                         |
| bert+tplinker_plus            | 20/20 | ——     | 95.71     | 长度限制明显                                                 |
| uie                           | 20/20 | ——     | 96.57     | zeroshot:f1=60.8, fewshot-100样本:f1=85.82, 200样本:f1=86.40 |
| W2NER                         | 18/20 | 97.37    | 96.32     | 对显存要求较高                                               |
| CNN_Nested_NER                | 19/20 | 98.06    | 96.11     |                                                              |
| LEAR                          | 8/20  | ——     | 96.52     |                                                              |

## 3. 文本表示

### 3.1 无监督语义相似度

- bert预训练模型 + 无监督finetune + cls位句向量(PromptBert除外)
- 五个中文数据集 + 5个epoch取最优值 + valid的spearmanr相关系数
- 继续finetune, 部分数据集有小幅提升
- 实验显示dropout_rate对结果影响较大


| solution        | ATEC  | BQ    | LCQMC | PAWSX | STS-B | comment                               |
| ----------------- | ------- | ------- | ------- | ------- | ------- | --------------------------------------- |
| Bert-whitening  | 26.79 | 31.81 | 56.34 | 17.22 | 67.45 | cls+不降维                            |
| CT              | 30.65 | 44.50 | 68.67 | 16.20 | 69.27 | dropout=0.1, 收敛慢跑了10个epoch      |
| CT_In_Batch_Neg | 32.47 | 47.09 | 68.56 | 27.50 | 74.00 | dropout=0.1                           |
| TSDAE           | ——  | 46.65 | 65.30 | 12.54 | ——  | dropout=0.1, ——表示该指标异常未记录 |
| SimCSE          | 33.90 | 50.29 | 71.81 | 13.14 | 71.09 | dropout=0.3                           |
| ESimCSE         | 34.05 | 50.54 | 71.58 | 12.53 | 71.27 | dropout=0.3                           |
| DiffSCE         | 33.04 | 48.17 | 71.51 | 12.91 | 71.10 | dropout=0.3, 没啥效果                 |
| PromptBert      | 33.98 | 49.89 | 73.18 | 13.30 | 73.42 | dropout=0.3                           |

### 3.2 有监督语义相似度

- bert预训练模型 + 训练数据finetune + cls位句向量
- 五个中文数据集 + 5个epoch取最优值 + valid/test的spearmanr相关系数
- STS-B任务是5分类，其余是2分类


| solution            | ATEC          | BQ            | LCQMC         | PAWSX         | STS-B         | comment          |
| --------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ------------------ |
| CoSENT              | 50.61 / 49.81 | 72.84 / 71.61 | 77.79 / 78.74 | 55.00 / 56.00 | 83.48 / 80.06 |                  |
| ContrastiveLoss     | 50.02 / 49.19 | 72.52 / 70.98 | 77.49 / 78.27 | 58.21 / 57.65 | 69.87 / 68.58 | STS-B转为2分类   |
| InfoNCE             | 47.77 / 46.99 | 69.86 / 68.14 | 71.74 / 74.54 | 52.82 / 54.21 | 83.31 / 78.72 | STS-B转为2分类   |
| concat CrossEntropy | 48.71 / 47.62 | 72.16 / 70.07 | 78.44 / 78.77 | 51.46 / 52.28 | 61.31 / 56.62 | STS-B转为2分类   |
| CosineMSELoss       | 46.89 / 45.86 | 72.27 / 71.35 | 75.29 / 77.19 | 54.92 / 54.35 | 81.64 / 77.76 | STS-B标准化到0-1 |

## 4. 关系提取

- [百度关系提取数据集](http://ai.baidu.com/broad/download?dataset=sked)


| solution      | f1    | comment                |
| --------------- | ------- | ------------------------ |
| CasRel        | 81.87 |                        |
| gplinker      | 82.38 |                        |
| tplinker      | 74.49 | seq_len=64, 未完全收敛 |
| tplinker_plus | 79.30 | seq_len=64             |
| SPN4RE        | 77.53 |                        |
| PGRC          | 80.36 | 训练很慢               |

## 5. 文本生成

- [CSL数据集](https://github.com/CLUEbenchmark/CLGE)，注意是训练集1万左右的版本，分别dev/test指标


| solution   | Rouge-L       | Rouge-1       | Rouge-2       | BLEU          | comment |
| ------------ | --------------- | --------------- | --------------- | --------------- | --------- |
| bert+unlim | 63.65 / 63.01 | 66.25 / 66.34 | 54.48 / 54.81 | 44.21 / 44.60 |         |
| bart       | 64.62 / 64.99 | 67.72 / 68.40 | 56.08 / 57.26 | 46.15 / 47.67 |         |
| mt5        | 67.67 / 65.98 | 70.39 / 69.36 | 59.60 / 59.05 | 50.34 / 50.11 |         |
| t5_pegasus | 66.07 / 66.11 | 68.94 / 69.61 | 57.12 / 58.38 | 46.14 / 47.95 |         |
| uer_t5     | 63.59 / 63.11 | 66.56 / 66.48 | 54.65 / 54.82 | 44.27 / 44.60 |         |
