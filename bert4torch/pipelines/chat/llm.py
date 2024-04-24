'''
该模块的主要功能有两个
1. 很多chat大模型有build_prompt操作, 有的操作较为复杂, 这里预制以减轻代码重复
2. 对各个chat模型提供CliDemo, WebDemo, OpenApiDemo用于快速搭建demo

# TODO: 设置return_states=True时候，受到build_prompt影响，很难保证prompt完全复现
这里采用添加self.generation_config['states']['last_token']，是因为推理完成可能是因为到达max_length，未必是遇到了eos
'''

import re
from .base import Chat, ChatCli, ChatWebGradio, ChatWebStreamlit
from .openai_api import ChatOpenaiApi


# 一些通用的system话术
SYSTEM_ZH = """你是一个乐于助人、尊重他人、诚实的中文聊天助手。在安全的情况下，始终尽可能提供帮助。你的回答不应包括任何有害、不道德、种族主义、性别歧视、有毒、危险或非法的内容。请确保你的回答是社会公正和积极的。
如果一个问题没有任何意义，或者事实上不连贯，请解释原因，而不是回答不正确的问题。如果你不知道问题的答案，请不要分享虚假信息，所有回答尽可能使用中文来回答。
"""
SYSTEM_EN = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""


class ChatGlm(Chat):
    def build_prompt(self, query, history) -> str:
        if not history:
            prompt = query
        else:
            prompt = ""
            if self.no_history_states():
                for i, (old_query, response) in enumerate(history):
                    prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            else:
                prompt += self.generation_config['states']['last_token']

            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        return prompt
    
    def process_response(self, response, *args):
        response = super().process_response(response)
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            [r"\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response
    
class ChatGlmCli(ChatGlm, ChatCli):
    '''Chatglm命令行demo
    >>> from bert4torch.pipelines import ChatGlmCli
    >>> demo = ChatGlmCli("THUDM/chatglm-6b")  # THUDM/chatglm-6b, THUDM/chatglm-6b-int4, THUDM/chatglm-6b-int8
    >>> demo.run()
    '''
    pass

class ChatGlmWebGradio(ChatGlm, ChatWebGradio):
    '''Chatglm的web demo, gradio实现
    >>> from bert4torch.pipelines import ChatGlmWebGradio
    >>> demo = ChatGlmWebGradio("THUDM/chatglm-6b")  # THUDM/chatglm-6b, THUDM/chatglm-6b-int4, THUDM/chatglm-6b-int8
    >>> demo.run(share=True, inbrowser=True)
    '''
    pass

class ChatGlmWebStreamlit(ChatGlm, ChatWebStreamlit):
    '''Chatglm的web demo, gradio实现
    >>> from bert4torch.pipelines import ChatGlmWebStreamlit
    >>> demo = ChatGlmWebStreamlit("THUDM/chatglm-6b")  # THUDM/chatglm-6b, THUDM/chatglm-6b-int4, THUDM/chatglm-6b-int8
    >>> demo.run()
    '''
    pass

class ChatGlmOpenaiApi(ChatGlm, ChatOpenaiApi):
    '''Chatglm的openai server
    >>> from bert4torch.pipelines import ChatGlmOpenaiApi
    >>> demo = ChatGlmOpenaiApi("THUDM/chatglm-6b")  # THUDM/chatglm-6b, THUDM/chatglm-6b-int4, THUDM/chatglm-6b-int8
    >>> demo.run()
    '''
    pass


class ChatGlm2(Chat):
    def build_prompt(self, query, history=[]):
        # 这里和chatglm的区别是，chatglm的第一轮对话prompt=query, 不加[Round 1]这些前缀
        prompt = ""
        if self.no_history_states():
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n\n问：{}\n\n答：{}\n".format(i+1, old_query, response)
        else:
            prompt += self.generation_config['states']['last_token']

        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history)+1, query)
        return prompt
    
    def process_response(self, response, *args):
        response = super().process_response(response)
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            [r"\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response

class ChatGlm2Cli(ChatGlm2, ChatCli):
    '''Chatglm2命令行demo
    >>> from bert4torch.pipelines import ChatGlm2Cli
    >>> demo = ChatGlm2Cli("THUDM/chatglm2-6b")  # THUDM/chatglm2-6b, THUDM/chatglm2-6b-int4, THUDM/chatglm2-6b-32k
    >>> demo.run()
    '''
    pass

class ChatGlm2WebGradio(ChatGlm2, ChatWebGradio):
    '''Chatglm2的web demo, gradio实现
    >>> from bert4torch.pipelines import ChatGlm2WebGradio
    >>> demo = ChatGlm2WebGradio("THUDM/chatglm2-6b")  # THUDM/chatglm2-6b, THUDM/chatglm2-6b-int4, THUDM/chatglm2-6b-32k
    >>> demo.run(share=True, inbrowser=True)
    '''
    pass

class ChatGlm2WebStreamlit(ChatGlm2, ChatWebStreamlit):
    '''Chatglm的web demo, gradio实现
    >>> from bert4torch.pipelines import ChatGlm2WebStreamlit
    >>> demo = ChatGlm2WebStreamlit("THUDM/chatglm2-6b")  # THUDM/chatglm2-6b, THUDM/chatglm2-6b-int4, THUDM/chatglm2-6b-32k
    >>> demo.run()
    '''
    pass

class ChatGlm2OpenaiApi(ChatGlm2, ChatOpenaiApi):
    '''Chatglm2的openai server
    >>> from bert4torch.pipelines import ChatGlm2OpenaiApi
    >>> demo = ChatGlm2OpenaiApi("THUDM/chatglm2-6b")  # THUDM/chatglm2-6b, THUDM/chatglm2-6b-int4, THUDM/chatglm2-6b-32k
    >>> demo.run()
    '''
    pass


class ChatGlm3(Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_maxlen *= 2

    def build_prompt(self, query, history=[]):
        # 由于tokenizer封装了部分逻辑，这里直接转成input_ids
        if (len(history) > 0) and isinstance(history[-1], tuple):
            history.pop()
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": ""})
        if self.no_history_states():
            input_ids = self.tokenizer.build_chat_input(query, history=history, role="user")['input_ids']
        else:
            input_ids += self.generation_config['states']['last_token']
        return input_ids

    def build_cli_text(self, history):
        '''构建命令行终端显示的text'''
        prompt = self.init_str
        for hist in history:  # 去除ChatCliDemo添加的当前回复的记录
            if not isinstance(hist, dict):
                continue
            elif hist['role'] == 'user':
                query = hist['content']
                prompt += f"\n\nUser：{query}"
            elif hist['role'] == 'assistant':
                response = hist['content']
                prompt += f"\n\nAssistant：{response}"
        return prompt
    
    def build_cli_history(self, cli_pre_history, cli_new_history):
        if (len(cli_new_history) > 0) and isinstance(cli_new_history[-1], tuple):
            cli_new_history.pop()
        return super().build_cli_history(cli_pre_history, cli_new_history)
    
    def process_response(self, response, history):
        response = super().process_response(response)
        if (not response) or (response[-1] == "�"):
            return response, history

        content = ""
        for response in response.split("<|assistant|>"):
            metadata, content = response.split("\n", maxsplit=1)
            if not metadata.strip():
                content = content.strip()
                history[-1] = {"role": "assistant", "metadata": metadata, "content": content}
                content = content.replace("[[训练时间]]", "2023年")
            else:
                history[-1] = {"role": "assistant", "metadata": metadata, "content": content}
                if history[0]["role"] == "system" and "tools" in history[0]:
                    content = "\n".join(content.split("\n")[1:-1])
                    parameters = eval(content)
                    content = {"name": metadata.strip(), "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content

class ChatGlm3Cli(ChatGlm3, ChatCli):
    '''Chatglm3命令行demo
    >>> from bert4torch.pipelines import ChatGlm3Cli
    >>> model_name = "THUDM/chatglm3-6b"  # THUDM/chatglm3-6b, THUDM/chatglm3-6B-32k
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    >>> generation_kwargs = {"end_id": [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), 
    >>>                      tokenizer.get_command("<|observation|>")]}
    >>> demo = ChatGlm3Cli(model_name)
    >>> demo.run()
    '''
    pass

class ChatGlm3WebGradio(ChatGlm3, ChatWebGradio):
    '''Chatglm3的web demo, gradio实现
    >>> from bert4torch.pipelines import ChatGlm3WebGradio
    >>> model_name = "THUDM/chatglm3-6b"  # THUDM/chatglm3-6b, THUDM/chatglm3-6B-32k
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    >>> generation_kwargs = {"end_id": [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), 
    >>>                      tokenizer.get_command("<|observation|>")]}
    >>> demo = ChatGlm3WebGradio(model_name)
    >>> demo.run(share=True, inbrowser=True)
    '''
    pass

class ChatGlm3WebStreamlit(ChatGlm3, ChatWebStreamlit):
    '''Chatglm的web demo, gradio实现
    >>> from bert4torch.pipelines import ChatGlm3WebStreamlit
    >>> model_name = "THUDM/chatglm3-6b"  # THUDM/chatglm3-6b, THUDM/chatglm3-6B-32k
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    >>> generation_kwargs = {"end_id": [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), 
    >>>                      tokenizer.get_command("<|observation|>")]}
    >>> demo = ChatGlm3WebStreamlit(model_name)
    >>> demo.run()
    '''
    pass

class ChatGlm3OpenaiApi(ChatGlm3, ChatOpenaiApi):
    '''Chatglm3的openai server
    >>> from bert4torch.pipelines import ChatGlm3OpenaiApi
    >>> demo = ChatGlm3OpenaiApi("THUDM/chatglm3-6b")  # THUDM/chatglm3-6b, THUDM/chatglm3-6B-32k
    >>> demo.run()
    '''
    pass


class ChatInternLM(Chat):
    def build_prompt(self, query, history=[]):
        prompt = ""
        if self.no_history_states():
            for user, bot in history:
                prompt += f"""<s><|User|>:{user}<eoh>\n<|Bot|>:{bot}<eoa>\n"""
        else:
            prompt += self.generation_config['states']['last_token']

        if len(prompt) == 0:
            prompt += "<s>"
        if query is not None:
            prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""
        return prompt

    def process_response(self, response, history=None):
        response = super().process_response(response)
        for reg in ['<s>', '</s>', '<eoh>', '<eoa>']:
            response = response.replace(reg, '')
        return response

class ChatInternLMCli(ChatInternLM, ChatCli):
    pass

class ChatInternLMWebGradio(ChatInternLM, ChatWebGradio):
    pass

class ChatInternLMWebStreamlit(ChatInternLM, ChatWebStreamlit):
    pass

class ChatInternLMOpenaiApi(ChatInternLM, ChatOpenaiApi):
    pass


class ChatQwen(Chat):
    def __init__(self, *args, system:str=None, max_window_size=6144, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system if system is not None else SYSTEM_ZH
        self.max_window_size = max_window_size

    def build_prompt(self, query, history) -> str:
        im_start, im_end = "<|im_start|>", "<|im_end|>"

        def _tokenize_str(role, content):
            return f"{role}\n{content}"

        system_text = _tokenize_str("system", self.system)
        raw_text = ""

        if self.no_history_states():
            for turn_query, turn_response in reversed(history):
                query_text = _tokenize_str("user", turn_query)
                response_text = _tokenize_str("assistant", turn_response)
                prev_chat = (
                    f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                )

                current_context_size = len(self.tokenizer.encode(raw_text, allowed_special={im_start, im_end}))
                if current_context_size < self.max_window_size:
                    raw_text = prev_chat + raw_text
                else:
                    break
            raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        else:
            raw_text += self.generation_config['states']['last_token']

        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        return raw_text

class ChatQwenCli(ChatQwen, ChatCli):
    '''Qwen的命令行demo
    Example
    ----------------------
    >>> from transformers import AutoTokenizer
    >>> from bert4torch.pipelines import ChatQwenCli
    >>> model_name = 'Qwen/Qwen-7B-Chat'  # Qwen/Qwen-1_8B-Chat  Qwen/Qwen-7B-Chat  Qwen/Qwen-14B-Chat

    >>> tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    >>> generation_config = {
    >>>     'end_id': [tokenizer.im_start_id, tokenizer.im_end_id], 
    >>>     'tokenizer_config': {'allowed_special': {"<|im_start|>", "<|im_end|>", '<|endoftext|>'}, 'skip_special_tokens': True}, 
    >>>     'max_length': 256
    >>> }

    >>> cli_demo = ChatQwenCli(model_name, system='You are a helpful assistant.', generation_config=generation_config)
    >>> cli_demo.run()
    '''
    pass

class ChatQwenWebGradio(ChatQwen, ChatWebGradio):
    '''Qwen的web demo, gradio实现
    Example
    ----------------------
    >>> from transformers import AutoTokenizer
    >>> from bert4torch.pipelines import ChatQwenWebGradio
    >>> model_name = 'Qwen/Qwen-7B-Chat'  # Qwen/Qwen-1_8B-Chat  Qwen/Qwen-7B-Chat  Qwen/Qwen-14B-Chat

    >>> tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    >>> generation_config = {
    >>>     'end_id': [tokenizer.im_start_id, tokenizer.im_end_id], 
    >>>     'tokenizer_config': {'allowed_special': {"<|im_start|>", "<|im_end|>", '<|endoftext|>'}, 'skip_special_tokens': True}, 
    >>>     'max_length': 256
    >>> }

    >>> cli_demo = ChatQwenWebGradio(model_name, system='You are a helpful assistant.', generation_config=generation_config)
    >>> cli_demo.run()
    '''
    pass

class ChatQwenWebStreamlit(ChatQwen, ChatWebStreamlit):
    '''Qwen的web demo, streamlit实现
    需使用 `streamlit run example.py` 来执行

    Example
    ----------------------
    >>> from transformers import AutoTokenizer
    >>> from bert4torch.pipelines import ChatQwenWebStreamlit
    >>> model_name = 'Qwen/Qwen-7B-Chat'  # Qwen/Qwen-1_8B-Chat  Qwen/Qwen-7B-Chat  Qwen/Qwen-14B-Chat

    >>> tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    >>> generation_config = {
    >>>     'end_id': [tokenizer.im_start_id, tokenizer.im_end_id], 
    >>>     'tokenizer_config': {'allowed_special': {"<|im_start|>", "<|im_end|>", '<|endoftext|>'}, 'skip_special_tokens': True}, 
    >>>     'max_length': 256
    >>> }

    >>> cli_demo = ChatQwenWebStreamlit(model_name, system='You are a helpful assistant.', generation_config=generation_config)
    >>> cli_demo.run()
    '''
    pass

class ChatQwenOpenaiApi(ChatQwen, ChatOpenaiApi):
    '''Qwen的openai sever端

    Example
    ----------------------
    >>> from transformers import AutoTokenizer
    >>> from bert4torch.pipelines import ChatQwenOpenaiApi
    >>> model_name = 'Qwen-7B-Chat'  # Qwen-1_8B-Chat  Qwen-7B-Chat  Qwen-14B-Chat

    >>> tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    >>> generation_config = {
    >>>     'end_id': [tokenizer.im_start_id, tokenizer.im_end_id], 
    >>>     'tokenizer_config': {'allowed_special': {"<|im_start|>", "<|im_end|>", '<|endoftext|>'}, 'skip_special_tokens': True}, 
    >>>     'max_length': 256
    >>> }

    >>> cli_demo = ChatQwenOpenaiApi(model_name, system='You are a helpful assistant.', generation_config=generation_config)
    >>> cli_demo.run()
    '''
    pass


class ChatLLaMA2(Chat):
    def __init__(self, *args, system:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system if system is not None else SYSTEM_EN

    def build_prompt(self, query, history) -> str:
        if self.no_history_states():
            texts = [f'[INST] <<SYS>>\n{self.system}\n<</SYS>>\n\n']
            for user_input, response in history:
                texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
        else:
            texts = [self.generation_config['states']['last_token']]

        texts.append(f'{query.strip()} [/INST]')
        return ''.join(texts)

class ChatLLaMA2Cli(ChatLLaMA2, ChatCli):
    pass

class ChatLLaMA2WebGradio(ChatLLaMA2, ChatWebGradio):
    pass

class ChatLLaMA2WebStreamlit(ChatLLaMA2, ChatWebStreamlit):
    pass

class ChatLLaMA2OpenaiApi(ChatLLaMA2, ChatOpenaiApi):
    pass


class ChatLLaMA3(Chat):
    def __init__(self, *args, system:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system if system is not None else SYSTEM_ZH

    def build_prompt(self, query, history) -> str:
        if self.no_history_states():
            messages = [{"role": "system", "content": self.system}]
            for user_input, response in history:
                messages.append({"role": "user", "content": user_input})
                messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": query})
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            texts = self.generation_config['states']['last_token']
            texts += f'<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

class ChatLLaMA3Cli(ChatLLaMA3, ChatCli):
    pass

class ChatLLaMA3WebGradio(ChatLLaMA3, ChatWebGradio):
    pass

class ChatLLaMA3WebStreamlit(ChatLLaMA3, ChatWebStreamlit):
    pass

class ChatLLaMA3OpenaiApi(ChatLLaMA3, ChatOpenaiApi):
    pass


class ChatZiya(Chat):
    def build_prompt(self, query, history) -> str:
        prompt = ''
        if self.no_history_states():
            for human, bot in history:
                prompt += f"<human>:{human}\n<bot>:{bot}\n"
        else:
            prompt += self.generation_config['states']['last_token']
        
        prompt += f"<human>:{query.strip()}\n<bot>:"
        return prompt

class ChatZiyaCli(ChatZiya, ChatCli):
    pass

class ChatZiyaWebGradio(ChatZiya, ChatWebGradio):
    pass

class ChatZiyaWebStreamlit(ChatZiya, ChatWebStreamlit):
    pass

class ChatZiyaOpenaiApi(ChatZiya, ChatOpenaiApi):
    pass


class ChatChineseAlphaLLaMA(Chat):
    def __init__(self, *args, system:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        if system is None:
            self.system = \
("Below is an instruction that describes a task. "
"Write a response that appropriately completes the request.\n\n"
)
        else:
            self.system = system

    def build_prompt(self, query, history) -> str:
        prompt = ''
        if self.no_history_states():
            for inst, resp in history:
                prompt += f"### Instruction:\n\n{inst}\n\n### Response:\n\n{resp}\n\n"
            prompt += f"### Instruction:\n\n{query}\n\n### Response:\n\n"
            prompt = self.system + prompt
        else:
            prompt += self.generation_config['states']['last_token'] + f"### Instruction:\n\n{query}\n\n### Response:\n\n"
        return prompt

class ChatChineseAlphaLLaMACli(ChatChineseAlphaLLaMA, ChatCli):
    pass

class ChatChineseAlphaLLaMAWebGradio(ChatChineseAlphaLLaMA, ChatWebGradio):
    pass

class ChatChineseAlphaLLaMAWebStreamlit(ChatChineseAlphaLLaMA, ChatWebStreamlit):
    pass

class ChatChineseAlphaLLaMAOpenaiApi(ChatChineseAlphaLLaMA, ChatOpenaiApi):
    pass


class ChatBelle(Chat):
    def build_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.checkpoint_path, use_fast=False)
    
    def build_prompt(self, query, history) -> str:
        prompt = ''
        if self.no_history_states():
            for item in history:
                prompt += f"Human: {item[0]} \n\nAssistant: {item[1]}\n\n"
        else:
            prompt += self.generation_config['states']['last_token']
        prompt += f"Human: {query} \n\nAssistant: "
        return prompt

class ChatBelleCli(ChatBelle, ChatCli):
    pass

class ChatBelleWebGradio(ChatBelle, ChatWebGradio):
    pass

class ChatBelleWebStreamlit(ChatBelle, ChatWebStreamlit):
    pass

class ChatBelleOpenaiApi(ChatBelle, ChatOpenaiApi):
    pass


class ChatBaichuan(Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_token_id = 195
        self.assistant_token_id = 196

    def build_prompt(self, query, history) -> str:
        total_input = []
        if self.no_history_states():
            for user, assistant in history:
                total_input += [self.user_token_id] + self.tokenizer.encode(user)  
                total_input += [self.assistant_token_id] + self.tokenizer.encode(assistant) + [self.tokenizer.eos_token_id]
        else:
            total_input += [self.generation_config['states']['last_token_id']]
        total_input += [self.user_token_id] + self.tokenizer.encode(query)
        total_input.append(self.assistant_token_id)
        return total_input

class ChatBaichuanCli(ChatBaichuan, ChatCli):
    pass

class ChatBaichuanWebGradio(ChatBaichuan, ChatWebGradio):
    pass

class ChatBaichuanWebStreamlit(ChatBaichuan, ChatWebStreamlit):
    pass

class ChatBaichuanOpenaiApi(ChatBaichuan, ChatOpenaiApi):
    pass
