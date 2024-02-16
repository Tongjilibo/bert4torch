'''测试规则提取实体'''
from bert4torch.snippets import entity_extract_rule
from pprint import pprint


def test():
    text = '''甲方：中国工商银行 乙方：中国农业银行 注册地址：上海市世纪大道1379号'''
    config = {'pattern': '甲方(:|：)(.*?)乙方',
            'label': '甲方',
            'replace_pattern': ['^甲方(:|：)', '乙方$']}

    res = entity_extract_rule(text, **config)
    pprint(res)
    assert res == [{'context': '中国工商银行 ', 'raw_context': '甲方：中国工商银行 乙方', 'start': 3, 'end': 10, 'label': '甲方'}]


if __name__ == '__main__':
    test()