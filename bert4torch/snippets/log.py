import logging
import datetime
import os.path


class SetLog(object):
    '''同时在文件中和命令行中log文件'''
    def __init__(self, file_name=None):
        self.date = str(datetime.date.today()).replace('-','')
        self.log_path = './Log/'+self.date
        os.makedirs(self.log_path, exist_ok=True)
        if file_name is None:
            file_name = 'Info.log'
        self.file_name = file_name
        self.file = os.path.join(self.log_path, file_name)
        self.init()

    def init(self):
        # 创建日志器
        self.log = logging.getLogger(self.file_name)
        # 日志器默认级别
        self.log.setLevel(logging.DEBUG)
        # 日志格式器
        self.formatter = logging.Formatter("%(asctime)s| %(message)s")
        self.setHandle()
        return self.log
        
    def setHandle(self):
        self.log.handlers.clear()
        # 终端流输出
        stream_handle = logging.StreamHandler()
        self.log.addHandler(stream_handle)
        stream_handle.setLevel(logging.DEBUG)
        stream_handle.setFormatter(self.formatter)
        # 文件流输出
        file_handle = logging.FileHandler(filename=self.file, mode='a', encoding='utf-8')
        self.log.addHandler(file_handle)
        file_handle.setLevel(logging.INFO)
        file_handle.setFormatter(self.formatter)
