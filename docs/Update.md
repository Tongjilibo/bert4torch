## 版本历史

|更新日期| bert4torch版本 | torch4keras版本 | 版本说明 |
|------| ---------------- | ----------------- |----------- |
|20231119| 0.4.0          | 0.1.5     |修复flash_attn的bug, stream_generate支持仅输出last_token|
|20231119| 0.3.9          | 0.1.5     |修复random_sample采样n>1, 新增Yi-6B, 支持flash_attn|
|20231112| 0.3.8          | 0.1.5     |支持chatglm 32k的rope_ratio，config中可以指定mapping, 增加m3e和bge|
|20231106| 0.3.7          | 0.1.5     |大部分模型文件无需convert，修复multi_query_group_num在int4/int8下bug, 简化`build_transformer_model`中配置到`config`中|
|20231022| 0.3.6          | 0.1.5     |增加falcon，layernorm支持torch自带|
|20230912| 0.3.5.post2    | 0.1.4.post2     |修复generation（既可初始化传参，也可以generate传参），decoder架构、encoder-decoder架构的增加generate系列方法直接推理, 增加internlm/baichuan2模型，训练时会默认自动把dataloader转移到model.device上, 增加xformers|
|20230902| 0.3.4          | 0.1.3     |修复gradient_checkpoint在低版本torch时仅支持位置参数的问题, 增加trainer.py, 增加PPOTrainerTrl以及相应的三阶段rlhf训练+dpo训练|
|20230812| 0.3.3          | 0.1.2     |增加大模型deepspeed的使用，增加Qwen模型(增加ntk和logn_attn)，generation的end_id支持多个token_id，修复多文件权重加载资源占用问题|
|20230804| 0.3.2          | 0.1.1     |修改依赖的torch4keras, 主要是进度条和logger, tensorboard的同步|
|20230726| 0.3.1.post2    | 0.1.0.post2     |修改baichuan的alibi逻辑，增加bloom, 简化decoder架构代码(gpt, llama, chatglm均继承decoder)|
|20230716| 0.3.0          | 0.0.9           |修改models和layers为文件夹方便扩展, 增加flash_attention参数控制，修改skip_init逻辑减少显存占用，generation增加repetition_penalty，修复chatglm的pv2的bug，generation支持transformers的tokenize，增加ziya，Baichuan|
|20230705| 0.2.9          | 0.0.8           |使用accelerate来实现skip_init精简代码, 修复add_trainer的代码提示, 增加chatglm的load_in_8bit+lora/qlora的训练, 修复grad_chechpoint, 增加chinese_llama_alpaca, torch2.0默认使用scaled_dot_product_attention加速, 增加chatglm2-6b+pv2+lora微调|
|20230518| 0.2.8          | 0.0.7.post3     |1）新增模型: 增加chatglm-6b/llama-7b/BELLE_llama/vicuna/moss/苏神、uer的roberta-small/Tiny模型以及ChatYuan v2模型/fnlp的bart2.0, 增加量化模块并适配llama，增加skip_init参数加快加载, 增加stream输出/网页demo, 增加ptuning_v2和lora; <br/>2）generation: 生成式解码新增SeqGeneration和Seq2SeqGeneration，单向decoder模型和encoder decoder模型解码增加cache, 增加batch_generate()/stream_generate功能；<br/>3）其他: 修改rope为不使用max_position，修复model.half()类型不一致问题，支持加载多个权重文件, gpt系列默认不加softmax，增加苏神Tiger的pytorch实现, 增加了对attention_key_size的入参支持，把_token_pad_ids重命名为pad_token_ids, tokenizor中重命名部分字段|
|20230310| 0.2.7.post2    | 0.0.6           |增加lion优化器, 修复albert_unshared加载权重, 修复lm系列(gpt, seq2seq)存在的forward参数不对的问题，修复GlobalPointer使用rope的bug|
|20230213| 0.2.7          | 0.0.6           |修复random_sample()的bug，适配v0.0.6的torch4keras：增加resume_from_checkpoint和save_to_checkpoint；增加add_trainer方法，重构了Trainer(BaseModel)的实现，增加了AccelerateCallback|
|20221231| 0.2.6          | 0.0.5           |build_transformer_model需显式指定add_trainer才从BaseModel继承, 增加guwenbert, macbert，text2vec-bert-chinese, wobert预训练模型，允许position_ids从padding开始, transformer.configs支持点操作，可以使用torch4keras的Trainer(net)来初始化, 修复tokenizer的切分subtoken的bug, 允许embedding_size!=hidden_size|
|20221127| 0.2.5          | 0.0.4           |对抗训练从compile转为使用Callback来实现，修复1.7.1版本兼容bug, uie模型内置|
|20221120| 0.2.4          | 0.0.3.post2     |删除SpTokenizer基类中的rematch, 增加deberta_v2模型|
|20221023| 0.2.3          | 0.0.2           |虚拟对抗VAT在多个ouput时支持指定，把Trainer抽象到[torch4keras](https://github.com/Tongjilibo/torch4keras)中，修复DP和DDP出现resume_epoch不存在的bug, tokenizer的never_split去除None, transformer_xl的bug, 增加gradient_checkpoint|
|20220922| 0.2.2         | ——            |修复t5的norm_mode问题，允许hidden_size不整除num_attention_heads，支持多个schedule(如同时ema+warmup)|
|20220905| 0.2.1         | ——            |兼容torch<=1.7.1的torch.div无rounding_mode，增加自定义metrics，支持断点续训，增加默认Logger和Tensorboard日志|
|20220823| 0.2.0         | ——            |兼容torch1.9.0的缺失take_along_dim，修复bart中位置向量514的问题，修复Sptokenizer对符号不转换，打印Epoch开始的时间戳，增加parallel_apply|
|20220808| 0.1.9         | ——            |增加mixup/manifold_mixup/temporal_ensembling策略，修复pgd策略param.grad为空的问题，修改tokenizer支持批量|
|20220717| 0.1.8         | ——            |修复原来CRF训练中loss陡增的问题，修复xlnet的token_type_ids输入显存占用大的问题|
|20220710| 0.1.7         | ——            |增加EarlyStop，CRF中自带转bool类型|
|20220605| 0.1.6         | ——            |增加transformer_xl、xlnet、t5_pegasus模型，prompt、预训练等示例，支持增加embedding输入，EMA策略，修复tokenizer和sinusoid的bug|
|20220504| 0.1.5         | ——            |增加GAU-alpha，混合梯度，梯度裁剪，单机多卡(DP、DDP)|
|20220421| 0.1.4         | ——            |增加了VAT，修复了linux下apply_embedding返回项有问题的情况|
|20220409| 0.1.3         | ——            |初始版本|
