'''从文档中把文字提取出来
1) 为接下来的要素提取做准备
2) 若原始格式为图片等, 则该模块为ocr模块
'''
from tqdm import tqdm
import os
import threading
import multiprocessing
import platform
from typing import List, Literal
import shutil
import json
import re
import numpy as np


num_processes = int(os.environ.get('NUM_PROCESSES', 4))

def extract_docx(src_docx:str):
    '''提取docx文本'''
    from docx import Document   # 导入相关库

    doc = Document(src_docx)
    full_text = []
    for p in tqdm(doc.paragraphs, desc='WORD extracting:', ncols=80):
        if p.text != '':
            full_text.append(p.text)
    full_text = '\n'.join(full_text)

    return full_text, doc


def extract_pdf(pdf_path:str, cache_dir:str=None, page_start:int=0, page_end:int=-1,
                method:Literal['pdfplumber', 'fitz']='pdfplumber', replace:bool=False) -> List[str]:
    '''解析pdf文件
    :param pdf_path: pdf文件路径
    :param cache_dir: 缓存路径
    :param page_start: 开始页码
    :param page_end: 结束页码
    :param method: 解析方法, pdfplumber, fitz
    :param replace: 是否替换pdf
    '''
    if replace and cache_dir is not None:
        # 如果存在，删除里面所有文件夹
        shutil.rmtree(cache_dir, ignore_errors=True)
        
    # 如果不存在，则创建文件夹
    if (cache_dir is not None) and (not os.path.exists(cache_dir)):
        os.makedirs(cache_dir, exist_ok=True)
    
    if method == 'pdfplumber':
        # window下多进程只能在main下使用, 所以windows下目前还是单进程
        if platform.system() == 'Windows':
            res = extract_pdf_with_pdfplumber(pdf_path, cache_dir, page_start, page_end)
        else:
            res = extract_pdf_with_pdfplumber_multiprocess(pdf_path, cache_dir, num_processes=num_processes)
    elif method == 'fitz':
        res = extract_pdf_with_fitz(pdf_path, cache_dir, page_start, page_end)
    return res


def extract_pdf_with_pdfplumber(pdf_path:str, cache_dir:str=None, page_start:int=0, page_end:int=-1):
    ''' 解析pdf, 单进程 '''
    import pdfplumber
    pdf = pdfplumber.open(pdf_path)

    page_end = len(pdf.pages) if page_end==-1 else page_end
    all_text_list = []
    for page in tqdm(pdf.pages[page_start:page_end], desc='PDF extracting', ncols=80):
        if cache_dir is not None:
            # 有缓存路径，则先保存到本地
            file_path = os.path.join(cache_dir, str(page.page_number)) + '.txt'
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = page.extract_text() or ""
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
        else:
            text = page.extract_text() or ""
        all_text_list.append(text)
    return all_text_list


def extract_pdf_with_pdfplumber_thread(pdf_path:str, cache_dir:str=None, page_start:int=0, page_end:int=-1, num_threads:int=4):
    ''' 解析pdf, 多线程 '''
    all_text_list = {}
    def thread_func(sublist, page_index):
        for page in tqdm(sublist, desc='PDF extracting', ncols=80):
            if cache_dir is not None:
                file_path = os.path.join(cache_dir, str(page.page_number)) + '.txt'
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    text = page.extract_text() or ""
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(text)
            else:
                text = page.extract_text() or ""
            all_text_list[page_index] = text

    import pdfplumber
    pdf = pdfplumber.open(pdf_path)
    page_end = len(pdf.pages) if page_end==-1 else page_end
    original_list = pdf.pages[page_start:page_end]

    # 计算每个子列表的长度
    sublist_len = len(original_list) // num_threads
    
    # 创建线程列表
    threads = []
    
    # 分配子列表给线程处理
    for i in range(num_threads):
        start_index = i * sublist_len
        end_index = start_index + sublist_len
        sub_list = original_list[start_index:end_index]

        # 使用线程处理子列表
        thread = threading.Thread(target=thread_func, args=(sub_list, start_index+page_start))
        thread.start()
        threads.append(thread)
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()


def __pdf_extract_process_func(pdf_path:str, cache_dir:str=None, start_page:int=0, end_page:int=-1):
    all_text_list = []
    import pdfplumber

    try:
        # 每个进程独立打开PDF文件
        with pdfplumber.open(pdf_path) as pdf:
            for i in tqdm(range(start_page, end_page), desc='PDF extracting', ncols=80):
                if i >= len(pdf.pages):
                    break  # 防止页面索引越界
                page = pdf.pages[i]
                if cache_dir is not None:
                    file_path = os.path.join(cache_dir, str(page.page_number + 1)) + '.txt'  # page_number从0开始
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    else:
                        text = page.extract_text() or ""  # 防止extract_text返回None
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                else:
                    text = page.extract_text() or ""
                all_text_list.append((i, text))
    except Exception as e:
        print(f"Error processing pages {start_page}-{end_page}: {str(e)}")
    return all_text_list


def extract_pdf_with_pdfplumber_multiprocess(pdf_path:str, cache_dir:str=None, num_processes=4):
    ''' 解析pdf, 多进程 '''

    if (cache_dir is not None) and (not os.path.exists(cache_dir)):
        os.makedirs(cache_dir, exist_ok=True)
    
    import pdfplumber
    # 先检查PDF文件是否有效
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
    except Exception as e:
        raise ValueError(f"Invalid PDF file: {str(e)}")
    
    num_processes = min(num_processes, multiprocessing.cpu_count(), num_pages)
    num_processes = max(num_processes, 1)
 
    # 计算每个子列表的长度
    sublist_len = num_pages // num_processes
 
    # 创建进程池
    pool = multiprocessing.Pool(num_processes)
    
    all_text_list = []
    # 分配子列表给进程池中的进程处理
    for i in range(num_processes):
        start_index = i * sublist_len
        if i < num_processes - 1:
            end_index = start_index + sublist_len
        else:
            end_index = num_pages
 
        # 使用进程池处理子列表
        text_list = pool.apply_async(
            __pdf_extract_process_func, 
            args=(pdf_path, cache_dir, start_index, end_index)
        )
        all_text_list.append(text_list)
    
    # 关闭进程池，等待所有进程完成
    pool.close()
    pool.join()
    
    # 收集结果
    results = [result.get() for result in all_text_list]
    # 展平结果并排序
    all_text_list = [j for i in results for j in i]
    all_text_list.sort(key=lambda x: x[0])
    return [i[1] for i in all_text_list]


def extract_pdf_with_fitz(pdf_path, save_dir, page_start=0, page_end=-1):
    import fitz
    pdf_file = fitz.open(pdf_path)
    all_text_list = []
    page_num = 0
    for page in tqdm(pdf_file, desc='PDF extracting', ncols=80):
        page_num += 1
        file_path = os.path.join(save_dir, str(page_num)) + '.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = page.get_text()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
        all_text_list.append(text)
    pdf_file.close()
    return all_text_list


# def pdf_extract_coordinates(pdf_path, save_dir, page_start=0, page_end=-1, method='pdfplumber', replace=False, resolution=72) -> List[str]:
#     if replace:
#         # 如果存在，删除里面所有文件夹
#         shutil.rmtree(save_dir, ignore_errors=True)
        
#     # 如果不存在，则创建文件夹
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir, exist_ok=True)
    
#     if method == 'pdfplumber':
#         # window下多进程只能在main下使用, 所以windows下目前还是单进程
#         if platform.system() == 'Windows':
#             res = pdf_extract_pdfplumber_coordinates(pdf_path, save_dir, page_start, page_end, resolution)
#         else:
#             res = pdf_extract_pdfplumber_coordinates_multiprocess(pdf_path, save_dir, num_processes=num_processes)
#     elif method == 'fitz':
#         res = pdf_extract_fitz(pdf_path, save_dir, page_start, page_end)
#     return res


def extract_pdf_with_pdfplumber_coordinates(pdf_path, save_dir=None, page_start=0, page_end=-1, resolution=72):
    ''' 解析pdf并附带坐标信息 '''
    import pdfplumber
    pdf = pdfplumber.open(pdf_path)

    page_end = len(pdf.pages) if page_end==-1 else page_end
    all_text_list = []
    for page in tqdm(pdf.pages[page_start:page_end], desc='PDF extracting with coordinates', ncols=80):
        file_path = os.path.join(save_dir, str(page.page_number)) + '_coords.jsonl' if save_dir is not None else ''
        if os.path.exists(file_path):
            # 文件存在
            text = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    text.append(json.loads(line))
        else:
            # 文件不存在
            text_elements = page.extract_words()  # 提取单词及其坐标
            text = [{'height': page.height, 'width': page.width, 'text': '', 'coords': [0, 0, 0, 0]}]
            for element in text_elements:
                text_tmp = element.get('text', '')  # 文本内容
                x1, y1, x2, y2 = element['x0'], element['top'], element['x1'], element['bottom']  # 坐标

                # 72 DPI 是 PDF 的默认分辨率
                x1 = int(x1 * resolution / 72)
                y1 = int(y1 * resolution / 72)
                x2 = int(x2 * resolution / 72)
                y2 = int(y2 * resolution / 72)

                text.append({'text': text_tmp, 'coords': [x1, y1, x2, y2]})
            if file_path != '':
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines([json.dumps(i, ensure_ascii=False)+'\n' for i in text])

        all_text_list.append(text)
    return all_text_list


def dataframe_to_markdown(df):
    """
    将 Pandas DataFrame 转换为 Markdown 表格格式的字符串。
    
    参数:
    df (pd.DataFrame): 要转换的 DataFrame。
    
    返回:
    str: Markdown 表格格式的字符串。
    """
    # 获取列名
    columns = df.columns.tolist()
    
    # 构建表头
    header = "| " + " | ".join(columns) + " |"
    
    # 构建表头分隔线
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    
    # 构建行数据
    rows = []
    for _, row in df.iterrows():
        values = []
        for i in row.values:
            if i is None or (isinstance(i, (int, float)) and np.isnan(i)):
                values.append('')
            elif isinstance(i, str):
                values.append(i.replace('\n', ''))
            else:
                values.append(i)
        row_data = "| " + " | ".join(map(str, values)) + " |"
        rows.append(row_data)
    
    # 合并所有部分
    markdown_table = "\n".join([header, separator] + rows)
    
    return markdown_table


def dataframe2text(df) -> str:
    '''dataframe转文本'''
    text_data = ", ".join(df.columns) + "\n"
    
    # 添加行数据
    for index, row in df.iterrows():
        row_data = ", ".join(map(str, row.values))
        text_data += row_data + "\n"
    
    text_data += "\n"  # 添加空行分隔不同的 sheet
    return text_data


def extract_txt(file_path:str) -> str:
    '''读取文本文件'''
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data


def extract_excel(file_path:str, df_format:Literal['markdown', 'text', 'dict']='markdown'):
    '''读取excel文件'''
    
    def is_row_empty(row):
        '''行是否为空'''
        return row.isnull().all()

    def find_first_non_empty_row_index(df):
        """找到第一个非空行的索引。"""
        for index, row in df.iterrows():
            if not is_row_empty(row):
                return index
        return None  # 如果所有行都是空的，返回 None
    
    import pandas as pd
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    
    all_text_data = []
    
    for sheet_name in sheet_names:
        # 读取每个 sheet 到 DataFrame
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # 将 DataFrame 转换为文本格式
        text_data = f"Sheet Name: {sheet_name}\n\n"
        
        # 添加列名
        if all([re.search('Unnamed: [0-9]+', i) for i in df.columns]):
            # 空header
            header_index = find_first_non_empty_row_index(df)
            df.columns = df.iloc[header_index]
            df = df.drop(header_index, axis=0)

        if df_format == 'markdown':
            text_data += dataframe_to_markdown(df)
        elif df_format == 'text':
            text_data += dataframe2text(df)
        elif df_format == 'dict':
            text_data += "\n".join([json.dumps(i, ensure_ascii=False) for i in df.to_dict('records')])
        else:
            raise ValueError("Invalid df_format. Please provide 'markdown', 'text', or 'dict'.")
        all_text_data.append(text_data)
    
    # 将所有 sheet 的文本数据合并成一个字符串
    final_text = "\n".join(all_text_data)
    return final_text

 
def extract_pptx(file_path):
    '''从pptx中解析文档'''
    from pptx import Presentation
    presentation = Presentation(file_path)
    
    # 用于存储所有幻灯片的文本
    all_text = []
    
    # 遍历每个幻灯片
    for slide in presentation.slides:
        slide_text = []
        # 遍历每个形状
        for shape in slide.shapes:
            # 检查形状是否有文本框
            if shape.has_text_frame:
                # 提取文本框中的段落
                for paragraph in shape.text_frame.paragraphs:
                    slide_text.append(paragraph.text)
        # 将当前幻灯片的文本添加到总文本中
        all_text.append("\n".join(slide_text))
    
    return "\n".join(all_text)
