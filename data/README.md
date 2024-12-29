# 一、数据集

## 传统nlp任务
| 数据集名称     | 用途               | 下载链接                                       |
| ---------------- | -------------------- | ------------------------------------------ |
| 人民日报数据集 | 实体识别           | [china-people-daily-ner-corpus](http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz), [HF](https://huggingface.co/datasets/Tongjilibo/china-people-daily-ner-corpus)|
| 百度关系抽取   | 关系抽取           | [官网](http://ai.baidu.com/broad/download?dataset=sked), [百度云(含dev)](https://pan.baidu.com/s/1aWXDkJkiMegzvwZ1XsuO2Q?pwd=5945), [HF](https://huggingface.co/datasets/Tongjilibo/BD_Knowledge_Extraction)|
| Sentiment      | 情感分类           | [bert4keras项目](https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip), [HF](https://huggingface.co/datasets/Tongjilibo/sentiment) |
| THUCNews       | 文本分类、文本生成 | [源文件](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews), [HF(转换后)](https://huggingface.co/datasets/Tongjilibo/THUCNews) |
| ATEC           | 文本相似度         | [ATEC](https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC), [HF](https://huggingface.co/datasets/Tongjilibo/ATEC)                                                           |
| BQ             | 文本相似度         | [BQ](http://icrc.hitsz.edu.cn/info/1037/1162.htm), [HF](https://huggingface.co/datasets/Tongjilibo/BQ) |
| LCQMC          | 文本相似度         | [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html), [HF](https://huggingface.co/datasets/Tongjilibo/LCQMC)|
| PAWSX          | 文本相似度         | [PAWSX](https://arxiv.org/abs/1908.11828), [HF](https://huggingface.co/datasets/Tongjilibo/PAWSX)|
| STS-B          | 文本相似度         | [STS-B](https://github.com/pluto-junzeng/CNSD), [HF](https://huggingface.co/datasets/Tongjilibo/STS-B)|
| CSL            | 文本生成           | [CSL](https://github.com/CLUEbenchmark/CLGE), [HF](https://huggingface.co/datasets/Tongjilibo/CSL)|

## 预训练
- Wiki中文百科
- 百度百科
- C4_ZH
- [WuDaoCorpora Text文本预训练数据集](https://data.baai.ac.cn/details/WuDaoCorporaText)


## 指令微调
| 数据集名称     | 介绍               |
| ---------------- | -------------------- |
|[shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)|参考Alpaca方法基于GPT4得到的self-instruct数据，约5万条|
|[BelleGroup/Belle-0.5M-cn](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)|包含约50万条由BELLE项目生成的中文指令数据||
|[BelleGroup/Belle-1M-cn](https://huggingface.co/datasets/BelleGroup/train_1M_CN)| 包含约100万条由BELLE项目生成的中文指令数据|
|[BelleGroup/Belle-school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)| Belle开放的0.25M数学指令数据集|
|[BelleGroup/Belle-multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)| Belle开放的0.8M多轮任务对话数据集|
|[fnlp/moss-002-sft-data](https://huggingface.co/datasets/fnlp/moss-002-sft-data)|MOSS-002所使用的多轮对话数据，覆盖有用性、忠实性、无害性三个层面，包含由text-davinci-003生成的约57万条英文对话和59万条中文对话|
|[fnlp/moss-003-sft-data](https://huggingface.co/datasets/fnlp/moss-003-sft-data)|moss-moon-003-sft所使用的多轮对话数据，基于MOSS-002内测阶段采集的约10万用户输入数据和gpt-3.5-turbo构造而成，相比moss-002-sft-data，moss-003-sft-data更加符合真实用户意图分布，包含更细粒度的有用性类别标记、更广泛的无害性数据和更长对话轮数，约含110万条对话数据|
|[YeungNLP/firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)|流萤23种常见的中文NLP任务的数据，并且构造了许多与中华文化相关的数据，如对联、作诗、文言文翻译、散文、金庸小说等。对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万|
|[YeungNLP/ultrachat](https://huggingface.co/datasets/YeungNLP/ultrachat)|清华大学开源的英文多轮对话数据，包含140万+数据|
|[YeungNLP/WizardLM_evol_instruct_V2_143k](https://huggingface.co/datasets/YeungNLP/WizardLM_evol_instruct_V2_143k) | 由WizardLM项目开源的英文指令微调数据集，通过Evol-Instruct方法让指令进化，加强指令的复杂度，以提升模型对复杂指令的遵循能力。包含143k条数据。|
|[shareAI/CodeChat](https://huggingface.co/datasets/shareAI/CodeChat)      | 主要包含逻辑推理、代码问答、代码生成相关语料样本。 |
|[shareAI/ShareGPT-Chinese-English-90k](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k)     | 中英文平行双语优质人机问答数据集，覆盖真实复杂场景下的用户提问。|
|[YeungNLP/ultrafeedback_binarized](https://huggingface.co/datasets/YeungNLP/ultrafeedback_binarized)      | 英文偏好数据集，可用于DPO训练   |
|[deepctrl/deepctrl-sft-data](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/summary)|匠数大模型SFT数据集是一个由匠数科技精心搜集整理的高质量数据集,包含10M条数据的中文数据集和包含2M条数据的英文数据集|

# 二、文档中示例数据说明
- data_similarity.json: 语义相似度示例数据集，用于[simbert](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_seq2seq_simbert.py)
- LCCD-large-shuf.jsonl: 示例数据集，用于[dialogpt_finetune](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq/task_dialogpt_finetune.py)
