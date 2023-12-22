import re
from bert4torch.chat.base import Chat
from bert4torch.chat.base import extend_with_cli_demo, extend_with_web_demo
from bert4torch.chat.api import extend_with_chat_openai_api


class Chatglm(Chat):
    def build_prompt(self, query, history) -> str:
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        return prompt
    
    def process_response(self, response, *args):
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
CliDemoChatglm = extend_with_cli_demo(Chatglm)
WebDemoChatglm = extend_with_web_demo(Chatglm)
OpenaiApiChatglm = extend_with_chat_openai_api(Chatglm)


class Chatglm2(Chat):
    def build_prompt(self, query, history=[]):
        # 这里和chatglm的区别是，chatglm的第一轮对话prompt=query, 不加[Round 1]这些前缀
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n".format(i+1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history)+1, query)
        return prompt
    
    def process_response(self, response, *args):
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
CliDemoChatglm2 = extend_with_cli_demo(Chatglm2)
WebDemoChatglm2 = extend_with_web_demo(Chatglm2)
OpenaiApiChatglm2 = extend_with_chat_openai_api(Chatglm2)


class Chatglm3(Chat):
    def build_prompt(self, query, history=[]):
        # 由于tokenizer封装了部分逻辑，这里直接转成input_ids
        if (len(history) > 0) and isinstance(history[-1], tuple):
            history.pop()
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": ""})
        input_ids = self.tokenizer.build_chat_input(query, history=history, role="user")['input_ids']
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
CliDemoChatglm3 = extend_with_cli_demo(Chatglm3)
WebDemoChatglm3 = extend_with_web_demo(Chatglm3)
OpenaiApiChatglm3 = extend_with_chat_openai_api(Chatglm3)


class InternLM(Chat):
    def build_prompt(self, query, history=[]):
        prompt = ""
        for user, bot in history:
            prompt += f"""<s><|User|>:{user}<eoh>\n<|Bot|>:{bot}<eoa>\n"""
        if len(prompt) == 0:
            prompt += "<s>"
        if query is not None:
            prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""
        
        return prompt

    def process_response(self, response, history=None):
        for reg in ['<s>', '</s>', '<eoh>', '<eoa>']:
            response = response.replace(reg, '')
        return response
CliDemoInternLM = extend_with_cli_demo(InternLM)
WebDemoInternLM = extend_with_web_demo(InternLM)
OpenaiApiInternLM = extend_with_chat_openai_api(InternLM)


class Qwen(Chat):
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
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        return raw_text
CliDemoQwen = extend_with_cli_demo(Qwen)
WebDemoQwen = extend_with_web_demo(Qwen)
OpenaiApiQwen = extend_with_chat_openai_api(Qwen)


class LLaMA2(Chat):
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
        texts = [f'[INST] <<SYS>>\n{self.system}\n<</SYS>>\n\n']
        for user_input, response in history:
            texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
        texts.append(f'{query.strip()} [/INST]')
        return ''.join(texts)
CliDemoLLaMA2 = extend_with_cli_demo(LLaMA2)
WebDemoLLaMA2 = extend_with_web_demo(LLaMA2)
OpenaiApiLLaMA2 = extend_with_chat_openai_api(LLaMA2)


class Ziya(Chat):
    def build_prompt(self, query, history) -> str:
        prompt = ''
        for human, bot in history:
            prompt += f"<human>:{human}\n<bot>:{bot}\n"
        prompt += f"<human>:{query.strip()}\n<bot>:"
        return prompt
CliDemoZiya = extend_with_cli_demo(Ziya)
WebDemoZiya = extend_with_web_demo(Ziya)
OpenaiApiZiya = extend_with_chat_openai_api(Ziya)


class ChineseAlphaLLaMA(Chat):
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
        for inst, resp in history:
            prompt += f"### Instruction:\n\n{inst}\n\n### Response:\n\n{resp}\n\n"
        prompt += f"### Instruction:\n\n{query}\n\n### Response:\n\n"
        prompt = self.system +prompt
        return prompt
CliDemoChineseAlphaLLaMA = extend_with_cli_demo(ChineseAlphaLLaMA)
WebDemoChineseAlphaLLaMA = extend_with_web_demo(ChineseAlphaLLaMA)
OpenaiApiChineseAlphaLLaMA = extend_with_chat_openai_api(ChineseAlphaLLaMA)


class Belle(Chat):
    def build_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
    
    def build_prompt(self, query, history) -> str:
        prompt = ''
        for item in history:
            prompt += f"Human: {item[0]} \n\nAssistant: {item[1]}\n\n"
        prompt += f"Human: {query} \n\nAssistant: "
        return prompt
CliDemoBelle = extend_with_cli_demo(Belle)
WebDemoBelle = extend_with_web_demo(Belle)
OpenaiApiBelle = extend_with_chat_openai_api(Belle)


class Baichuan(Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_token_id = 195
        self.assistant_token_id = 196

    def build_prompt(self, query, history) -> str:
        total_input = []
        for user, assistant in history:
            total_input += [self.user_token_id] + self.tokenizer.encode(user)  
            total_input += [self.assistant_token_id] + self.tokenizer.encode(assistant) + [self.tokenizer.eos_token_id]
        total_input += [self.user_token_id] + self.tokenizer.encode(query)
        total_input.append(self.assistant_token_id)
        return total_input
CliDemoBaichuan = extend_with_cli_demo(Baichuan)
WebDemoBaichuan = extend_with_web_demo(Baichuan)
OpenaiApiBaichuan = extend_with_chat_openai_api(Baichuan)
