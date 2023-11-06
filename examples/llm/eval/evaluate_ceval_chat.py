import os
import argparse
import re
import torch
import pandas as pd
from thefuzz import process
from tqdm import tqdm
from transformers import AutoTokenizer
from evaluate_ceval import TASK_NAME_MAPPING, choices, hard_list
from bert4torch.models import build_transformer_model
from bert4torch.snippets import seed_everything, JsonConfig

'''
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
mkdir data/ceval
mv ceval-exam.zip data/ceval
cd data/ceval; unzip ceval-exam.zip
cd ../../

pip install thefuzz
python eval/evaluate_chat_ceval.py -d data/ceval
'''

def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    config_path = args.checkpoint_path + '/bert4torch_config.json'
    checkpoint_path = [args.checkpoint_path + '/' + i for i in os.listdir(args.checkpoint_path) if i.startswith('bert4torch') and i.endswith('.bin')]
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, add_trainer=True)
    return model, tokenizer

def process_before_extraction(gen, question, choice_dict):
    # Example Prompt:
    # 关于传输层的面向连接服务的特性是____。
    # A. 既不保证可靠，也不保证按序交付
    # B. 不保证可靠，但保证按序交付
    # C. 保证可靠，但不保证按序交付
    # D. 既保证可靠，也保证按序交付
    # Example Model Output：
    # 关于传输层的面向连接服务的特性是既保证可靠，也保证按序交付
    # Processed Output:
    # 答案是D

    question_split = question.rstrip("。").split("。")[-1].split("_")

    # replacing the question
    if len(question_split[0].strip()) > 4:
        gen = gen.replace(question_split[0], "答案是")
    if len(question_split[-1].strip()) > 4:
        gen = gen.replace(question_split[-1], "")

    # replace the choice by letter in the generated sentence
    # from longest one to shortest one
    for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
        gen = gen.replace(val.rstrip("。"), key)
    return gen


def count_substr(gen, pattern):
    return len(re.findall(pattern, gen))


def extract_choice(gen, prompt, choice_list):
    # 答案是A | 选项是A | 应该选A选项
    res = re.search(
        r"(?:(?:选|选择|选定)[：:]?\s*|(?:(?:答案|选项)(?![^ABCD]{0,10}?(?:不|非)[^ABCD]{0,10}?(?:是|选|为|：|:|】))[^ABCD]{0,10}?(?:是|选|为|：|:|】))[^ABCD]{0,10}?)(A|B|C|D)(?:选项)?(?:\)|。|\.|，|,|．|、|A|B|C|D|$|：|:|\)|）)",
        gen,
    )

    # A选项正确 | A选项符合题意
    if res is None:
        res = re.search(
            r"(A|B|C|D)(?:选?项)?(?![^ABCD]{0,4}?(?:不|非)[^ABCD]{0,4}?(?:正确|对[的，。：]|符合))[^ABCD]{0,4}?(?:正确|对[的，。：]|符合)",
            gen,
        )

    # 直接输出 A
    if res is None:
        res = re.search(r"^[\(（]?(A|B|C|D)(?:。|\)|）|\.|，|,|．|：|:|$)", gen)

    # 获取第一个出现的字母
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)


def format_example(line):
    example = line["question"] + "\n\n"
    for choice in choices:
        example += f'{choice}. {line[f"{choice}"]}\n'
    return example


def extract_answer(response, row):
    prompt = row["question"]
    gen = process_before_extraction(
        response, prompt, {choice: row[choice] for choice in choices}
    )
    if not isinstance(prompt, str):
        prompt = prompt[0]
    pred = extract_choice(gen, prompt, [row[choice] for choice in choices])
    return pred


@torch.no_grad()
def eval_subject(
    model,
    tokenizer,
    subject_name,
    test_df,
    save_result_dir=None,
    overwrite=False,
    **kwargs
):
    result_path = os.path.join(save_result_dir, f"{subject_name}_result.csv")
    if not overwrite and os.path.exists(result_path):
        print(f"{result_path} existed, skip!")
        score = []
        for (_, datarow), (_, resultrow) in zip(
            test_df.iterrows(), pd.read_csv(result_path).iterrows()
        ):
            pred = extract_answer(resultrow["model_response"], datarow)
            correct = 1 if pred == datarow["answer"] else 0
            score.append(correct)
        correct_ratio = 100 * sum(score) / len(score)
        return correct_ratio

    responses = []
    result = []
    score = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row)

        response = model.generate(question, tokenizer=tokenizer, topk=1)  # topk=1, greed decode
        pred = extract_answer(response, row)
        if args.debug:
            print(question)
            print(response)
            print(pred)
            print("======================")

        if "answer" in row:
            correct = 1 if pred == row["answer"] else 0
            score.append(correct)
            if args.debug:
                print(f'{question} pred: {pred} ref: {row["answer"]}')
        responses.append(response)
        result.append(pred)

    if score:
        correct_ratio = 100 * sum(score) / len(score)
        if args.debug:
            print(subject_name, correct_ratio)
    else:
        correct_ratio = 0
    if save_result_dir:
        test_df["model_response"] = responses
        test_df["model_output"] = result
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(result_path, encoding="utf-8", index=False)

    return correct_ratio


def cal_ceval(res):
    acc_sum_dict = dict()
    acc_norm_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.0
    cnt = 0
    hard_cnt = 0
    hard_acc_sum = 0.0
    for tt in res.keys():
        name = tt.split("-")[-1]
        acc_sum += float(res[tt])
        cnt += 1
        class_ = TASK_NAME_MAPPING[name][2]
        if class_ not in acc_sum_dict:
            acc_sum_dict[class_] = 0.0
            acc_norm_sum_dict[class_] = 0.0
            cnt_dict[class_] = 0.0
        if name in hard_list:
            hard_cnt += 1
            hard_acc_sum += float(res[tt])
        acc_sum_dict[class_] += float(res[tt])
        cnt_dict[class_] += 1
    print("\n\n\n")
    for k in ["STEM", "Social Science", "Humanities", "Other"]:
        if k in cnt_dict:
            print("%s acc: %.2f " % (k, acc_sum_dict[k] / cnt_dict[k]))
    if hard_cnt > 0:
        print("Hard acc:%.2f " % (hard_acc_sum / hard_cnt))
    print("AVERAGE acc:%.2f " % (acc_sum / cnt))


def main(args):
    print("loading model weights")
    if args.checkpoint_path:
        model, tokenizer = load_models_tokenizer(args)
    else:
        model, tokenizer = None, None
    print("model loaded")
    dev_result = {}
    for subject_name in tqdm(TASK_NAME_MAPPING.keys()):
        val_file_path = os.path.join(
            args.eval_data_path, "val", f"{subject_name}_val.csv"
        )
        val_df = pd.read_csv(val_file_path)

        score = eval_subject(
            model,
            tokenizer,
            subject_name,
            val_df,
            save_result_dir="outs_chat/ceval_eval_result",
            overwrite=args.overwrite,
        )
        dev_result[subject_name] = score
    cal_ceval(dev_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-c", "--checkpoint-path", type=str, help="Checkpoint dir", default="E:/pretrain_ckpt/bloom/bloomz-560m")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")

    # Provide extra arguments required for tasks
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str, default='E:/data/corpus/prompt/evaluataion/ceval-exam', help="Path to eval data")
    group.add_argument("--debug", action="store_true", default=False, help="Print infos.")
    group.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existed results")

    args = parser.parse_args()
    seed_everything(args.seed)

    main(args)