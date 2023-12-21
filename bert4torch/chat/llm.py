import re
from bert4torch.chat.base import Chat
from bert4torch.chat.base import extend_with_chat_cli_demo, extend_with_chat_web_demo
from bert4torch.chat.api import extend_with_chat_openai_api


class ChatChatglm(Chat):
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


class ChatChatglm2(Chat):
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


class ChatChatglm3(Chat):
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


ChatCliDemoChatglm = extend_with_chat_cli_demo(ChatChatglm)
ChatCliDemoChatglm2 = extend_with_chat_cli_demo(ChatChatglm2)
ChatCliDemoChatglm3 = extend_with_chat_cli_demo(ChatChatglm3)
ChatWebDemoChatglm = extend_with_chat_web_demo(ChatChatglm)
ChatWebDemoChatglm2 = extend_with_chat_web_demo(ChatChatglm2)
ChatWebDemoChatglm3 = extend_with_chat_web_demo(ChatChatglm3)
ChatOpenaiApiChatglm = extend_with_chat_openai_api(ChatChatglm)
ChatOpenaiApiChatglm2 = extend_with_chat_openai_api(ChatChatglm2)
ChatOpenaiApiChatglm3 = extend_with_chat_openai_api(ChatChatglm3)
