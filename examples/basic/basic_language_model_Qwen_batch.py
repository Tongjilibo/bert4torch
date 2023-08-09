from basic_language_model_Qwen import chat, make_context, tokenizer

queries = []
for query in ['你好', '你是谁']:
    query = make_context(tokenizer, query, None, system='You are a helpful assistant.')
    queries.append(query)

res = chat.batch_generate(queries)
print(res)