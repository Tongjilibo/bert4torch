from transformers import AutoModelForCausalLM, AutoTokenizer
from bert4torch.pipelines import Chat
import re


# Yi-6B
# Yi-1.5-9B-Chat-16K
model_dir = "/data/pretrain_ckpt/01-ai/Yi-1.5-9B-Chat-16K"


# print('==========================transformers=============================')
# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

# # Load the model
# model = AutoModelForCausalLM.from_pretrained(
#     model_dir,
#     device_map="auto",  # Automatically choose available devices
#     torch_dtype='auto'  # Automatically select suitable data type
# ).eval()  # Set the model to evaluation mode

# while True:
#     prompt = input("User: ")

#     messages = [
#         {"role": "user", "content": prompt}
#     ]

#     # Convert the conversation to a format the model can understand
#     input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')

#     # Generate a response using the model
#     output_ids = model.generate(input_ids.to('cuda'))

#     # Decode the model's output
#     response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
#     print(f'Bot: {response}')


print('==========================bert4torch=============================')
generation_config = {
    "include_input": False if re.search('Chat', model_dir) else True
}

demo = Chat(model_dir, 
            mode='cli',
            generation_config=generation_config,
            )


if __name__ == '__main__':
    demo.run()