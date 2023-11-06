import argparse
import tqdm
import torch
import jsonlines
from transformers import AutoTokenizer
from bert4torch.models import build_transformer_model
from bert4torch.snippets import seed_everything, JsonConfig
import os

"""
git clone https://github.com/openai/human-eval
$ pip install -e human-eval
evaluate_functional_correctness sample-output-file
"""

def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    config_path = args.checkpoint_path + '/bert4torch_config.json'
    checkpoint_path = [args.checkpoint_path + '/' + i for i in os.listdir(args.checkpoint_path) if i.startswith('bert4torch') and i.endswith('.bin')]
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, add_trainer=True)
    return model, tokenizer


def decode(tokens_list, tokenizer):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens)
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("def ")[0]
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, input_txt):
    input_ids = tokenizer.encode(input_txt)
    context_enc = torch.tensor([input_ids]).to(model.device)
    outputs = model.generate(context_enc, topk=1)
    output_text = decode(outputs, tokenizer)[0]
    # print(f"Input text: {input_txt}\n")
    # print(f"\nOutput text: \n{output_text}\n")
    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-c", "--checkpoint-path", type=str, help="Checkpoint dir", default="E:/pretrain_ckpt/bloom/bloomz-560m")
    parser.add_argument("-f", "--sample-input-file", type=str, default="E:/data/corpus/prompt/evaluataion/humaneval/HumanEval.jsonl")
    parser.add_argument("-o", "--sample-output-file", type=str, default="HumanEval_res.jsonl")

    args = parser.parse_args()
    model, tokenizer = load_models_tokenizer(args)

    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))

    f = jsonlines.open(args.sample_input_file)
    with f_output as output:
        for jobj in tqdm.tqdm(f, desc="task_idx"):
            prompt = jobj["prompt"]
            task_id = jobj["task_id"]
            gen_sents = generate_sample(model, tokenizer, prompt)
            gen_jobjs = {"task_id": task_id, "completion": gen_sents}
            output.write(gen_jobjs)
    f_output.close()
