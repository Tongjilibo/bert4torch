from bert4torch import AutoTokenizer, AutoProcessor
import pytest


@pytest.mark.parametrize("model_dir", ["/data/pretrain_ckpt/Qwen/Qwen2-7B-Instruct"])

def test_auto_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    messages = [
        {"role": "user", "content": '你好'}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(text, '\n')
    model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt")
    print(model_inputs)

    output = tokenizer.decode(model_inputs['input_ids'][0].tolist(), skip_special_tokens=False)
    print(output)
    assert text == output
    return model_inputs, output


def test_auto_processor(model_dir):
    processor = AutoProcessor.from_pretrained(model_dir)
    image = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "描述一下这张图片"}
        ]}
    ]
    inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
    return inputs


if __name__ == '__main__':
    res = test_auto_tokenizer("/data/pretrain_ckpt/Qwen/Qwen2-7B-Instruct")
    print(res)

    res = test_auto_processor("/data/pretrain_ckpt/Qwen/Qwen2.5-VL-3B-Instruct")
    print(res)