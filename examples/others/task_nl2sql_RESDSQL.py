#! -*- coding: utf-8 -*-
# The re-implementation of the paper "RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL" with bert4torch. Used datasets (spider) and backbone models are available in the original code repo.
# Paper: https://arxiv.org/abs/2104.08384, AAAI 2023
# Original Code: https://github.com/RUCKBReasoning/RESDSQL/tree/main

# Following is our reproducd results, the results are slightly different from the original code repo, we guess this is because our limited GPU resources lead to a smalling batch size than original setting.
# ----------------------------------------------------------------
#    Name              Model        Dev-EM    Dev-EX   Batch_size
# ----------------------------------------------------------------
#  Original code    RESDSQL-Base	 71.7%	   77.9%	   16
#      Ours         RESDSQL-Base     70.1%     75.2%       4
# ----------------------------------------------------------------


import os
import json
import torch
import sqlite3
import torch.optim as optim
import transformers

from tqdm import tqdm
from tokenizers import AddedToken
from func_timeout import func_set_timeout, FunctionTimedOut

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration, MT5ForConditionalGeneration
from transformers.optimization import Adafactor
from bert4torch.snippets import ListDataset, seed_everything
from bert4torch.callbacks import Callback
from bert4torch.models import BaseModel

# parameters setting
batch_size = 4
gradient_descent_step = 2
learning_rate = 1e-4
epochs = 128
seed = 42
save_path = "models/text2sql-t5-base/"
tensorboard_save_path = "tensorboard_log/text2sql"
model_name_or_path = "models/t5-base"
use_adafactor = True
mode = "train"
train_filepath = "data/preprocessed_data/resdsql_train_spider.json"
dev_filepath = "data/preprocessed_data/resdsql_dev.json"
original_dev_filepath = "data/spider/dev.json"
db_path = "data/spider/database"
tables_for_natsql = "NatSQL/NatSQLv1_6/tables_for_natsql.json"
num_beams = 8
num_return_sequences = 8
target_type = "sql"
output = "predicted_sql.txt"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# seed 
seed_everything(seed)

# tensorboard setting
writer = SummaryWriter(log_dir=tensorboard_save_path)  # prepare summary writer

# tokenizer 
text2sql_tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path, add_prefix_space=True)

if isinstance(text2sql_tokenizer, T5TokenizerFast):
    text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

# model type
model_class = MT5ForConditionalGeneration if "mt5" in save_path else T5ForConditionalGeneration

# load datasets
class Text2SQLDataset(ListDataset):
    @staticmethod
    def load_data(filename, mode='train'):
        D = []
        with open(filename, 'r', encoding='utf-8') as f:
            contents = json.loads(f.read())

            for data in contents:
                input_sequences = data["input_sequence"]
                db_id = data["db_id"]
                all_tc_original = data["tc_original"]
                if mode == 'train':
                    output_sequences = data["output_sequence"]
                    D.append((input_sequences, output_sequences, db_id, all_tc_original))
                else:
                    D.append((input_sequences, None, db_id, all_tc_original)) 
        return D

def collate_fn(batch):
    batch_encoder_ids, batch_encoder_attn, batch_decoder_attn, batch_decoder_labels = [], [], [], []
    for inputs, sqls, _, _ in batch:
        tokenized_inputs = text2sql_tokenizer(inputs, padding = "max_length", return_tensors = "pt", max_length = 512, truncation = True)

        with text2sql_tokenizer.as_target_tokenizer():
            tokenized_outputs = text2sql_tokenizer(sqls, padding = "max_length", return_tensors = 'pt', max_length = 256, truncation = True)

        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        decoder_attention_mask = tokenized_outputs["attention_mask"]
        decoder_labels = tokenized_outputs["input_ids"]
        decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
        
        batch_encoder_ids.append(encoder_input_ids)
        batch_encoder_attn.append(encoder_input_attention_mask)
        batch_decoder_attn.append(decoder_attention_mask)
        batch_decoder_labels.append(decoder_labels)
    batch_encoder_ids = torch.cat(batch_encoder_ids, dim=0).to(device=device)
    batch_encoder_attn = torch.cat(batch_encoder_attn, dim=0).to(device=device)
    batch_decoder_attn = torch.cat(batch_decoder_attn, dim=0).to(device=device)
    batch_decoder_labels = torch.cat(batch_decoder_labels, dim=0).to(device=device)
    return (batch_encoder_ids, batch_encoder_attn, batch_decoder_attn, batch_decoder_labels), None

def collate_fn_eval(batch):
    batch_inputs, batch_encoder_ids, batch_encoder_attn, batch_db_id, batch_tc_original = [], [], [], [], []
    for inputs, _, db_id, all_tc_original in batch:
        tokenized_inputs = text2sql_tokenizer(inputs, padding = "max_length", return_tensors = "pt", max_length = 512, truncation = True)
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]      
        batch_inputs.append(inputs)
        batch_encoder_ids.append(encoder_input_ids)
        batch_encoder_attn.append(encoder_input_attention_mask)
        batch_db_id.append(db_id)
        batch_tc_original.append(all_tc_original)
        
    batch_encoder_ids = torch.cat(batch_encoder_ids, dim=0).to(device=device)
    batch_encoder_attn = torch.cat(batch_encoder_attn, dim=0).to(device=device)
    return batch_inputs, batch_encoder_ids, batch_encoder_attn, batch_db_id, batch_tc_original 

train_dataset = Text2SQLDataset(train_filepath, 'train')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn, drop_last=True)
dev_dataset = Text2SQLDataset(dev_filepath, 'eval')
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_eval, drop_last=False)
# warm up steps (10% training step)
num_warmup_steps = int(0.1*epochs*len(train_dataset)/batch_size)
# total training steps
num_training_steps = int(epochs*len(train_dataset)/batch_size)
# save checkpoint for each 1.42857 epochs (about 1.42857*7000=10000 examples for Spider's training set)
num_checkpoint_steps = int(1.42857 * len(train_dataset)/batch_size)    

per_epoch_step_nums = len(train_dataset)/batch_size

class Model(BaseModel):
    def __init__(self, model_name_or_path, model_class, tokenizer):
        super().__init__()
        self.model = model_class.from_pretrained(model_name_or_path)
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, encoder_input_ids, encoder_input_attention_mask=None, decoder_attention_mask=None, decoder_labels=None):
        outputs = self.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_input_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=decoder_labels,
            return_dict=True
        )
        return outputs.loss

model = Model(model_name_or_path, model_class, text2sql_tokenizer).to(device)

class Evaluator(Callback):
    def __init__(self, model, optimizer, num_checkpoint_steps, per_epoch_step_nums):
        self.best_val_auc = 0.
        self.per_epoch_step_nums = per_epoch_step_nums
        self.num_checkpoint_steps = num_checkpoint_steps
        self.model = model
        self.optimizer = optimizer

    def on_batch_end(self, global_step, local_step, logs=None):
        if writer:
            writer.add_scalar(f"train/loss", logs['loss'], global_step)
            writer.add_scalar(f"train/lr", self.optimizer.state_dict()['param_groups'][0]['lr'], global_step)

        if global_step % self.num_checkpoint_steps == 0 and global_step / self.per_epoch_step_nums >= 6:
            print(f" At {global_step} training step, save a checkpoint.")
            os.makedirs(save_path, exist_ok=True)
            self.model.model.save_pretrained(save_directory=save_path + "/checkpoint-{}".format(global_step))
            text2sql_tokenizer.save_pretrained(save_directory=save_path + "/checkpoint-{}".format(global_step))

    def evalulate(self, predict_sqls):
        pass
        # Since we want to keep the code as clean as possible in one file, we don't provide the evaluate function, please using the official evalution tool "Test-Suite" for evaluation, the tool refers to https://github.com/taoyds/test-suite-sql-eval

    def _test(self, checkpoint='checkpoint-39312'):
        # initialize model
        model_path = os.path.join(save_path, checkpoint)
        model = model_class.from_pretrained(model_path).to(device)
        model.eval()

        predict_sqls = []
        for batch in tqdm(dev_dataloader):
            batch_inputs, encoder_input_ids, encoder_input_attention_mask, batch_db_ids, batch_tc_original = batch
            with torch.no_grad():
                model_outputs = model.generate(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_input_attention_mask,
                    max_length=256,
                    decoder_start_token_id=model.config.decoder_start_token_id,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences
                )        

                model_outputs = model_outputs.view(len(batch_inputs), num_return_sequences, model_outputs.shape[1])

                predict_sqls += decode_sqls(
                        db_path, 
                        model_outputs, 
                        batch_db_ids, 
                        batch_inputs, 
                        text2sql_tokenizer, 
                        batch_tc_original
                )

        new_dir = "/".join(output.split("/")[:-1]).strip()
        if new_dir != "":
            os.makedirs(new_dir, exist_ok=True)
        
        # save results
        with open(output, "w", encoding='utf-8') as f:
            for pred in predict_sqls:
                f.write(pred + "\n")

if use_adafactor:
    print("Let's use Adafactor!")
    optimizer = Adafactor(
        model.parameters(), 
        lr=learning_rate, 
        scale_parameter=False, 
        relative_step=False, 
        clip_threshold = 1.0,
        warmup_init=False
    )
else:
    print("Let's use AdamW!")
    optimizer = optim.AdamW(
        model.parameters(), 
        lr = learning_rate
    )

scheduler = transformers.get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps = num_warmup_steps,
    num_training_steps = num_training_steps
)

model.compile(
    loss=lambda x, _: x,
    optimizer=optim.Adam(model.parameters(), lr=learning_rate),
    grad_accumulation_steps=gradient_descent_step,
    scheduler=scheduler, 
    clip_grad_norm=True,
)

# execute predicted sql with a time limitation
@func_set_timeout(120)
def execute_sql(cursor, sql):
    cursor.execute(sql)

    return cursor.fetchall()

# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread = False)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor

def decode_sqls(db_path, generator_outputs, batch_db_ids, batch_inputs, tokenizer,batch_tc_original):
    batch_size = generator_outputs.shape[0]
    num_return_sequences = generator_outputs.shape[1]

    final_sqls = []
    
    for batch_id in range(batch_size):
        pred_executable_sql = "sql placeholder"
        db_id = batch_db_ids[batch_id]
        db_file_path = db_path + "/{}/{}.sqlite".format(db_id, db_id)
        
        for seq_id in range(num_return_sequences):
            cursor = get_cursor_from_path(db_file_path)
            pred_sequence = tokenizer.decode(generator_outputs[batch_id, seq_id, :], skip_special_tokens = True)
            
            pred_sql = pred_sequence.split("|")[-1].strip()
            pred_sql = pred_sql.replace("='", "= '").replace("!=", " !=").replace(",", " ,")
            
            try:
                # Note: execute_sql will be success for empty string
                assert len(pred_sql) > 0, "pred sql is empty!"

                results = execute_sql(cursor, pred_sql)
                # if the current sql has no execution error, we record and return it
                pred_executable_sql = pred_sql
                cursor.close()
                cursor.connection.close()
                break
            except Exception as e:
                print(pred_sql)
                print(e)
                cursor.close()
                cursor.connection.close()
            except FunctionTimedOut as fto:
                print(pred_sql)
                print(fto)
                del cursor
        
        final_sqls.append(pred_executable_sql)
    
    return final_sqls


if __name__ == '__main__':    
    evaluator = Evaluator(model, optimizer, num_checkpoint_steps, per_epoch_step_nums)
    model.fit(train_dataloader, epochs=epochs, steps_per_epoch=None, callbacks=[evaluator],)    
    evaluator._test()

