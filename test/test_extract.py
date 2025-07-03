'''测试规则提取实体'''
from bert4torch.snippets import TimeitContextManager
from bert4torch.snippets.extract import (
    extract_entity_by_rule,
    extract_pdf
)

def test_entity_by_rule():
    text = '''甲方：中国工商银行 乙方：中国农业银行 注册地址：上海市世纪大道1379号'''

    res = extract_entity_by_rule(text, pattern='甲方(:|：)(.*?)乙方', label='甲方', replace_pattern=['^甲方(:|：)', '乙方$'])
    print(res)
    assert res == [{'entity': '中国工商银行 ', 'raw_entity': '甲方：中国工商银行 乙方', 'start': 3, 'end': 10, 'label': '甲方'}]


def test_extract_pdf_with_pdfplumber():
    pdf_path = '/home/lb/projects/tongjilibo/bert4torch/test_local/贵州茅台年报.pdf'
    with TimeitContextManager() as ti:
        # fitz提取pdf文本
        res = extract_pdf(pdf_path, method='fitz')
        ti.lap('[fitz单进程]', reset=True)

        res = extract_pdf(pdf_path, num_processes=4, method='fitz')
        ti.lap('[fitz多进程]', reset=True)

        res = extract_pdf(pdf_path, num_threads=4, method='fitz')
        ti.lap('[fitz多线程]', reset=True)

        # pdfplumber提取pdf文本
        res = extract_pdf(pdf_path)
        ti.lap('[pdfplumber单进程]', reset=True)

        res = extract_pdf(pdf_path, num_processes=4)
        ti.lap('[pdfplumber多进程]', reset=True)

        res = extract_pdf(pdf_path, num_threads=4)
        ti.lap('[pdfplumber多线程]', reset=True)


if __name__ == '__main__':
    test_entity_by_rule()
    test_extract_pdf_with_pdfplumber()