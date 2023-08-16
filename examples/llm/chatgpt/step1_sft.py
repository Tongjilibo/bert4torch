#! -*- coding: utf-8 -*-
# Supervised Finetune

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset
from bert4torch.callbacks import Callback, Logger
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model
import json
from glob import glob
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Sequence
from tqdm import tqdm


# 基本参数
max_source_length = 256
max_target_length = 256
max_length = max_source_length + max_target_length
batch_size = 4
epochs = 10000
use_lora = False
load_in_nbit = None

# 模型配置
data_path = '/Users/lb/Documents/Project/Github/MedicalGPT/data/finetune/**/*.jsonl'
model_type = 'bloom'
template_name = 'vicuna'
root_path = '/Users/lb/Documents/pretrain_ckpt/bloom/bloom-560m/'
config_path = root_path + 'bert4torch_config.json'
checkpoint_path = root_path + 'bert4torch_pytorch_model.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(root_path, trust_remote_code=True)
pad_token_id = tokenizer.pad_token_id or -100

# ================================================================
# ================     各个模型的prompt设计    =====================
# ================================================================
@dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    system_prompt: str
    # All messages. format: list of [question, answer]
    messages: Optional[List[Sequence[str]]]
    # The roles of the speakers
    roles: Optional[Sequence[str]]
    # Conversation prompt
    prompt: str
    # Separator
    sep: str

    def get_prompt(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> str:
        """
        Returns a string containing prompt without response.
        """
        return "".join(self._format_example(messages, system_prompt))

    def get_dialog(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        """
        Returns a list containing 2 * n elements where the 2k-th is a query and the (2k+1)-th is a response.
        """
        return self._format_example(messages, system_prompt)

    def _format_example(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        system_prompt = system_prompt or self.system_prompt
        system_prompt = system_prompt + self.sep if system_prompt else ""  # add separator for non-empty system prompt
        messages = messages or self.messages
        convs = []
        for turn_idx, [user_query, bot_resp] in enumerate(messages):
            if turn_idx == 0:
                convs.append(system_prompt + self.prompt.format(query=user_query))
                convs.append(bot_resp)
            else:
                convs.append(self.sep + self.prompt.format(query=user_query))
                convs.append(bot_resp)
        return convs

    def append_message(self, query: str, answer: str):
        """Append a new message."""
        self.messages.append([query, answer])


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation):
    """Register a new conversation template."""
    conv_templates[template.name] = template


"""Vicuna v1.1 template
Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
          https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
"""
register_conv_template(
    Conversation(
        name="vicuna",
        system_prompt="A chat between a curious user and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        messages=[],
        roles=("USER", "ASSISTANT"),
        prompt="USER: {query} ASSISTANT: ",
        sep="</s>",
    )
)

"""Alpaca template"""
register_conv_template(
    Conversation(
        name="alpaca",
        system_prompt="Below is an instruction that describes a task. "
                      "Write a response that appropriately completes the request.",
        messages=[],
        roles=("### Instruction", "### Response"),
        prompt="### Instruction:\n{query}\n\n### Response:\n",
        sep="\n\n",
    )
)

"""Baichuan-13B-Chat template
source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/f5f47be2adbbdceb784f334d6fa1ca2c73e65097/modeling_baichuan.py#L507
Support: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
"""
register_conv_template(
    Conversation(
        name="baichuan-chat",
        system_prompt="",
        messages=[],
        roles=(" <reserved_102> ", " <reserved_103> "),
        prompt=" <reserved_102> {query} <reserved_103> ",
        sep="</s>",
    )
)

"""ziya template"""
register_conv_template(
    Conversation(
        name="ziya",
        system_prompt="",
        messages=[],
        roles=("<human>", "<bot>"),
        prompt="<human>:{query}\n<bot>:",
        sep="\n",
    )
)

"""Linly template"""
register_conv_template(
    Conversation(
        name="linly",
        system_prompt="",
        messages=[],
        roles=("User", "Bot"),
        prompt="User: {query}\nBot: ",
        sep="\n",
    )
)

"""ChatGLM default template
source: https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py#L1307
"""
register_conv_template(
    Conversation(
        name="chatglm",
        system_prompt="",
        messages=[],
        roles=("问", "答"),
        prompt="问：{query}\n答：",
        sep="\n",
    )
)

"""ChatGLM2 default template
source: https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py#L1007
"""
register_conv_template(
    # source:
    Conversation(
        name="chatglm2",
        system_prompt="",
        messages=[],
        roles=("问", "答"),
        prompt="问：{query}\n\n答：",
        sep="\n\n",
    )
)

"""Phoenix default template"""
register_conv_template(
    Conversation(
        name="phoenix",
        system_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: <s>{query}</s>Assistant: ",
        sep="</s>",
    )
)

"""
Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
"""
register_conv_template(
    Conversation(
        name="belle",
        system_prompt="",
        messages=[],
        roles=("Human", "Belle"),
        prompt="Human: {query}\n\nBelle: ",
        sep="\n\n",
    )
)

"""
Supports: https://huggingface.co/qhduan/aquilachat-7b
"""
register_conv_template(
    Conversation(
        name="aquila",
        system_prompt="A chat between a curious human and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: {query}###Assistant: ",
        sep="###",
    )
)

r"""
Supports: https://huggingface.co/internlm/internlm-chat-7b
"""
register_conv_template(
    Conversation(
        name="intern",
        system_prompt="",
        messages=[],
        roles=("<|User|>", "<|Bot|>"),
        prompt="<|User|>:{query}<eoh>\n<|Bot|>:",
        sep="<eoa>\n",
    )
)

# StarChat template
register_conv_template(
    Conversation(
        name="starchat",
        system_prompt="<system>\n",
        messages=[],
        roles=("<|user|>", "<|assistant|>"),
        prompt="<|user|>\n{query}<|end|>\n<|assistant|>\n",
        sep="<|end|>\n",
    )
)

# llama2 template
# reference: https://github.com/facebookresearch/llama/blob/cfc3fc8c1968d390eb830e65c63865e980873a06/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="llama-2",
        system_prompt="<<SYS>>\nYou are a helpful, respectful and honest assistant. "
                      "Always answer as helpfully as possible, while being safe. "
                      "Your answers should not include any harmful, unethical, racist, sexist, "
                      "toxic, dangerous, or illegal content. "
                      "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                      "If a question does not make any sense, or is not factually coherent, "
                      "explain why instead of answering something not correct. "
                      "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",
        messages=[],
        roles=("[INST]", "[/INST]"),
        prompt=" [INST] {query} [/INST] ",
        sep="</s>",
    )
)


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name]


def preprocess_function(examples):
        """
        Preprocessing the datasets.
            part of code modified from https://github.com/lm-sys/FastChat
        """
        input_ids_list = []
        targets_list = []
        roles = ["human", "gpt"]
        prompt_template = get_conv_template(template_name)

        def get_dialog(examples):
            for i, source in enumerate(examples):
                if len(source) < 2:
                    continue
                data_role = source[0].get("from", "")
                if data_role not in roles or data_role != roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]
                if len(source) < 2:
                    continue
                messages = []
                for j, sentence in enumerate(source):
                    data_role = sentence.get("from", "")
                    if data_role not in roles:
                        logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                        break
                    if data_role == roles[j % 2]:
                        messages.append(sentence["value"])
                if len(messages) < 2 or len(messages) % 2 != 0:
                    continue
                # Convert the list to pairs of elements
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
                dialog = prompt_template.get_dialog(history_messages)
                yield dialog

        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))
                target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

                if len(source_ids) > max_source_length:
                    source_ids = source_ids[:max_source_length]
                if len(target_ids) > max_target_length - 1:  # eos token
                    target_ids = target_ids[:max_target_length - 1]
                if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                    source_ids = source_ids[1:]
                if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                    target_ids = target_ids[:-1]
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # add eos token for each turn
                labels += [pad_token_id] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

            input_ids_list.append(input_ids)
            targets_list.append(labels)

        return list(zip(input_ids_list, targets_list))

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量分为不超过maxlen的句子
        """
        D = []
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    D.append(json.loads(l)['conversations'])
        return preprocess_function(D)

def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for token_ids, label_ids in batch:
        batch_token_ids.append(token_ids)
        batch_labels.append(label_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=pad_token_id), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels, value=pad_token_id), dtype=torch.long, device=device)
    return [batch_token_ids], batch_labels

train_dataloader = DataLoader(MyDataset(glob(data_path, recursive=True)), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
dev_dataloader = DataLoader(MyDataset(glob(data_path, recursive=True)), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model=model_type, add_trainer=True, pad_token_id=pad_token_id).to(device)

# 量化
if load_in_nbit == 8:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)
    model = model.quantize(quantization_method='load_in_8bit', llm_int8_skip_modules=['model.embeddings.word_embeddings', 'lm_head'])
    model.lm_head = CastOutputToFloat(model.lm_head)
    
elif load_in_nbit == 4:
    from peft import prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float16,  # 可选 torch.float32, torch.float16, torch.bfloat16
                                llm_int8_skip_modules=['model.embeddings.word_embeddings', 'lm_head']
                                )
    model = model.quantize(quantization_method='load_in_4bit', quantization_config=q_config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# lora
if use_lora:
    from peft import LoraConfig
    peft_config = LoraConfig(
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['q', 'k', 'v']
        )
    model = model.get_peft_model(peft_config).to(device)
else:
    model = model.to(device)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_pred, y_true):
        '''
        y_pred: [btz, seq_len, vocab_size]
        y_true: token_ids: [btz, seq_len]
        '''
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1, :]  # 预测序列，错开一位

        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.flatten()
        return super().forward(y_pred, y_true)

loss_fun = CrossEntropyLoss(ignore_index=pad_token_id)
model.compile(loss=loss_fun, optimizer=optim.Adam(model.parameters(), 1e-5))

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, steps, epoch, logs=None):
        # 保存最优
        dev_loss = self.evaluate(dev_dataloader)
        if dev_loss['dev_loss'] <= self.lowest:
            self.lowest = dev_loss['dev_loss']
            model.save_weights('./best_model_sft.pt')
        dev_loss['best_dev_loss'] = self.lowest
        print(dev_loss)

    def evaluate(self, data):
        loss, count = 0, 0
        for input_ids, label in tqdm(data, desc='Evaluating'):
            pred = model.predict(input_ids)
            loss += loss_fun(pred, label).item()
            count += 1

        return {'dev_loss': loss/count}

if __name__ == '__main__':
    logger = Logger('./log_sft.log')
    evaluator = Evaluator()
    model.fit(train_dataloader, steps_per_epoch=None, epochs=epochs, callbacks=[evaluator, logger])
else:
    model.load_weights('./best_model_sft.pt')
