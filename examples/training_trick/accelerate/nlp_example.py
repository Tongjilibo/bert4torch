# coding=utf-8
# 修改官方的accelerate例子，使之可以使用torch4keras框架

import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import sequence_padding, Callback, text_segmentate, ListDataset, seed_everything, get_pool_emb
from bert4torch.optimizers import get_linear_schedule_with_warmup
import torch.nn as nn


########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
maxlen = 256
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)


def get_dataloaders(accelerator: Accelerator, batch_size: int = 16):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    class MyDataset(ListDataset):
        @staticmethod
        def load_data(filenames):
            """加载数据，并尽量划分为不超过maxlen的句子
            """
            D = []
            seps, strips = u'\n。！？!?；;，, ', u'；;，, '
            for filename in filenames:
                with open(filename, encoding='utf-8') as f:
                    for l in f:
                        text, label = l.strip().split('\t')
                        for t in text_segmentate(text, maxlen - 2, seps, strips):
                            D.append((t, int(label)))
            return D

    def collate_fn(batch):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for text, label in batch:
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])

        # On TPU it's best to pad everything to the same length or training will be very slow.
        length = maxlen if accelerator.distributed_type == DistributedType.TPU else None

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, length), dtype=torch.long)
        batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids, length), dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        return [batch_token_ids, batch_segment_ids], batch_labels.flatten()

    train_dataloader = DataLoader(MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.train.data']), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
    eval_dataloader = DataLoader(MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.valid.data']), batch_size=EVAL_BATCH_SIZE, collate_fn=collate_fn) 

    # 加载数据集
    return train_dataloader, eval_dataloader


def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    seed_everything(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    class Model(BaseModel):
        def __init__(self, pool_method='cls') -> None:
            super().__init__()
            self.pool_method = pool_method
            self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True)
            self.dropout = nn.Dropout(0.1)
            self.dense = nn.Linear(self.bert.configs['hidden_size'], 2)

        def forward(self, token_ids, segment_ids):
            hidden_states, pooling = self.bert([token_ids, segment_ids])
            pooled_output = get_pool_emb(hidden_states, pooling, token_ids.gt(0).long(), self.pool_method)
            output = self.dropout(pooled_output)
            output = self.dense(output)
            return output
    model = Model()


    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # 定义使用的loss和optimizer，这里支持自定义
    model.compile(
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler = lr_scheduler,
        metrics=['accuracy'],
        grad_accumulation_steps=gradient_accumulation_steps,
        accelerator=accelerator
    )

    class Evaluator(Callback):
        """评估与保存
        """
        def __init__(self):
            self.best_val_acc = 0.

        def on_epoch_end(self, global_step, epoch, logs=None):
            val_acc = self.evaluate(eval_dataloader)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                # model.save_weights('best_model.pt')

            # Use accelerator.print to print only on the main process.
            accelerator.print(f'val_acc: {val_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')

        def evaluate(self, data):
            total, right = 0., 0.
            for batch, y_true in data:
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                y_pred = model.predict(batch).argmax(dim=-1)
                y_pred = accelerator.gather(y_pred)
                y_true = accelerator.gather(y_true)
                total += len(y_true)
                right += (y_true == y_pred).sum().item()
            return right / total

    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=num_epochs, steps_per_epoch=10, callbacks=[evaluator])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}

    training_function(config, args)