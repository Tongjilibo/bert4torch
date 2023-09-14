
import re
import textwrap
import argparse
from pathlib import Path
import tqdm
import jsonlines
from transformers import AutoTokenizer
from bert4torch.models import build_transformer_model
from bert4torch.snippets import seed_everything, JsonConfig
import os

"""
Get the HumanEval.jsonl file from [here](https://github.com/openai/human-eval/tree/master/data)

python eval/evaluate_humaneval_chat.py -f HumanEval.jsonl -o HumanEval_res.jsonl
git clone https://github.com/openai/human-eval
pip install -e human-eval
evaluate_functional_correctness HumanEval_res.jsonl
"""

DEVICE = "cuda:0"

def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    config_path = args.checkpoint_path + '/bert4torch_config.json'
    checkpoint_path = [args.checkpoint_path + '/' + i for i in os.listdir(args.checkpoint_path) if i.startswith('bert4torch') and i.endswith('.bin')]
    model_type = JsonConfig(config_path)['model']
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model=model_type, add_trainer=True)
    return model, tokenizer


def extract_code(text, entry_point):
    # 正则表达式匹配代码块
    code_block_pattern = re.compile(
        rf"```(?:[Pp]ython\n)?.*?def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL
    )
    code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            rf"def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            r"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)

    if code_block is not None:
        return code_block.group(1)

    # if no code block is found, assume the LM is simply filling the code
    return textwrap.indent(text, " " * 4)


def generate_sample(model, tokenizer, question, entry_point):
    response = model.generate(question, tokenizer=tokenizer, topk=1)
    # print(question)
    # print(response)
    answer = extract_code(response, entry_point)
    return answer, response


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
            prompt = "Help me fill the following code.\n" + jobj["prompt"]
            task_id = jobj["task_id"]
            answer, response = generate_sample(model, tokenizer, prompt, jobj["entry_point"])
            gen_jobjs = {"task_id": task_id, "completion": answer, "response": response}
            output.write(gen_jobjs)
    f_output.close()
