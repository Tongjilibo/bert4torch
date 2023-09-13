import re
import torch
import argparse
import jsonlines
import numpy as np
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from bert4torch.models import build_transformer_model
from bert4torch.snippets import seed_everything, JsonConfig
import os


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    config_path = args.checkpoint_path + '/bert4torch_config.json'
    checkpoint_path = [args.checkpoint_path + '/' + i for i in os.listdir(args.checkpoint_path) if i.startswith('bert4torch') and i.endswith('.bin')]
    model_type = JsonConfig(config_path)['model']
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model=model_type, add_trainer=True)
    return model, tokenizer

def doc_to_text(doc):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, input_txt):
    input_ids = tokenizer.tokenizer.encode(input_txt)
    raw_text_len = len(input_ids)
    context_enc = torch.tensor([input_ids]).to(model.device)
    print(f"Input text: {input_txt}\n")
    outputs = model.generate(context_enc)
    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    print(f"\nOutput text: {output_text}\n")
    return output_text


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def extract_answer(completion):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-c", "--checkpoint-path", type=str, help="Checkpoint dir", default="E:/pretrain_ckpt/bloom/bloomz-560m")
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument("-o", "--sample-output-file", type=str, default="gsm8k_res.jsonl")
    args = parser.parse_args()

    fewshot_prompt = open("E:/Github/Qwen-7B/eval/gsm8k_prompt.txt").read()  # https://github.com/QwenLM/Qwen-7B/blob/main/eval/gsm8k_prompt.txt
    if args.sample_input_file is not None:
        dataset = load_from_disk(args.sample_input_file)
    else:
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = load_dataset("gsm8k", "main", download_config=config)

    test = dataset["test"]

    model, tokenizer = load_models_tokenizer(args)

    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
    tot_length = test.num_rows
    acc_res = []
    for doc in test:
        context = doc_to_text(doc)
        completion = generate_sample(model, tokenizer, context)
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        doc["completion"] = completion
        doc["acc"] = acc
        f_output.write(doc)
        acc_res.append(acc)

    f_output.close()
    print("Acc: ", np.mean(acc_res))