'''
该模块的主要功能有两个
1. 很多chat大模型有build_prompt操作, 有的操作较为复杂, 这里预制以减轻代码重复
2. 对各个chat模型提供CliDemo, WebDemo, OpenApiDemo用于快速搭建demo

# TODO: 设置return_states=True时候，受到build_prompt影响，很难保证prompt完全复现
这里采用添加self.generation_config['states']['last_token']，是因为推理完成可能是因为到达max_length，未必是遇到了eos
'''

import re
from .base import Chat, extend_with_cli, extend_with_web_gradio, extend_with_web_streamlit
from .openai_api import extend_with_chat_openai_api


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
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response
ChatGlmCli = extend_with_cli(ChatGlm)
ChatGlmWebGradio = extend_with_web_gradio(ChatGlm)
ChatGlmWebStreamlit = extend_with_web_streamlit(ChatGlm)
ChatGlmOpenaiApi = extend_with_chat_openai_api(ChatGlm)


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
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response
ChatGlm2Cli = extend_with_cli(ChatGlm2)
ChatGlm2WebGradio = extend_with_web_gradio(ChatGlm2)
ChatGlm2WebStreamlit = extend_with_web_streamlit(ChatGlm2)
ChatGlm2OpenaiApi = extend_with_chat_openai_api(ChatGlm2)


class ChatGlm3(Chat):
    def build_prompt(self, query, history=[]):
        # 由于tokenizer封装了部分逻辑，这里直接转成input_ids
        if (len(history) > 0) and isinstance(history[-1], tuple):
            history.pop()
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": ""})
        if self.no_history_states():
            input_ids = self.tokenizer.build_chat_input(query, history=history, role="user")['input_ids']
        else:
            input_ids = self.tokenizer.build_chat_input(query, role="user")['input_ids']  # TODO: 这里是否在开头需要增加last_token_id
        return input_ids

    def build_cli_text(self, history):
        '''构建命令行终端显示的text'''
        prompt = self.init_str
        for hist in history[:-1]:  # 去除ChatCliDemo添加的当前回复的记录
            if hist['role'] == 'user':
                query = hist['content']
                prompt += f"\n\nUser：{query}"
            elif hist['role'] == 'assistant':
                response = hist['content']
                prompt += f"\n\nAssistant：{response}"
        return prompt
    
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
ChatGlm3Cli = extend_with_cli(ChatGlm3)
ChatGlm3WebGradio = extend_with_web_gradio(ChatGlm3)
ChatGlm3WebStreamlit = extend_with_web_streamlit(ChatGlm3)
ChatGlm3OpenaiApi = extend_with_chat_openai_api(ChatGlm3)


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
ChatInternLMCli = extend_with_cli(ChatInternLM)
ChatInternLMWebGradio = extend_with_web_gradio(ChatInternLM)
ChatInternLMWebStreamlit = extend_with_web_streamlit(ChatInternLM)
ChatInternLMOpenaiApi = extend_with_chat_openai_api(ChatInternLM)


class ChatQwen(Chat):
    def __init__(self, *args, system='', max_window_size=6144, **kwargs):
        super().__init__(*args, **kwargs)
        self.system = system
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
ChatQwenCli = extend_with_cli(ChatQwen)
ChatQwenWebGradio = extend_with_web_gradio(ChatQwen)
ChatQwenWebStreamlit = extend_with_web_streamlit(ChatQwen)
ChatQwenOpenaiApi = extend_with_chat_openai_api(ChatQwen)


class ChatLLaMA2(Chat):
    def __init__(self, *args, system:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        if system is None:
            self.system = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
        else:
            self.system = system

    def build_prompt(self, query, history) -> str:
        if self.no_history_states():
            texts = [f'[INST] <<SYS>>\n{self.system}\n<</SYS>>\n\n']
            for user_input, response in history:
                texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
        else:
            texts = [self.generation_config['states']['last_token']]

        texts.append(f'{query.strip()} [/INST]')
        return ''.join(texts)
ChatLLaMA2Cli = extend_with_cli(ChatLLaMA2)
ChatLLaMA2WebGradio = extend_with_web_gradio(ChatLLaMA2)
ChatLLaMA2WebStreamlit = extend_with_web_streamlit(ChatLLaMA2)
ChatLLaMA2OpenaiApi = extend_with_chat_openai_api(ChatLLaMA2)


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
ChatZiyaCli = extend_with_cli(ChatZiya)
ChatZiyaWebGradio = extend_with_web_gradio(ChatZiya)
ChatZiyaWebStreamlit = extend_with_web_streamlit(ChatZiya)
ChatZiyaOpenaiApi = extend_with_chat_openai_api(ChatZiya)


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
ChatChineseAlphaLLaMACli = extend_with_cli(ChatChineseAlphaLLaMA)
ChatChineseAlphaLLaMAWebGradio = extend_with_web_gradio(ChatChineseAlphaLLaMA)
ChatChineseAlphaLLaMAWebStreamlit = extend_with_web_streamlit(ChatChineseAlphaLLaMA)
ChatChineseAlphaLLaMAOpenaiApi = extend_with_chat_openai_api(ChatChineseAlphaLLaMA)


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
ChatBelleCli = extend_with_cli(ChatBelle)
ChatBelleWebGradio = extend_with_web_gradio(ChatBelle)
ChatBelleWebStreamlit = extend_with_web_streamlit(ChatBelle)
ChatBelleOpenaiApi = extend_with_chat_openai_api(ChatBelle)


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
ChatBaichuanCli = extend_with_cli(ChatBaichuan)
ChatBaichuanWebGradio = extend_with_web_gradio(ChatBaichuan)
ChatBaichuanWebStreamlit = extend_with_web_streamlit(ChatBaichuan)
ChatBaichuanOpenaiApi = extend_with_chat_openai_api(ChatBaichuan)
