# -*- coding: utf-8 -*-

import logging
import logging.config
from typing import Optional, Text
from src.utils.configs import Configuration

TRACE_LOG = "tracelogger"
ERROR_LOG = "errorlogger"
OUT_LOG = "outlogger"
logger = logging.getLogger(__name__)
LOG_PATH_KEY = "log_path"


def configure_file_logging(config_path: Optional[Text]):
    if config_path is None:
        return

    dict = Configuration.read_config_file(config_path + "/logger.yml")
    import os
    if LOG_PATH_KEY in dict:
        log_path = dict[LOG_PATH_KEY]
        if not os.path.exists(log_path):
            os.makedirs(log_path)  # 创建路径
        dict.pop(LOG_PATH_KEY)
    # logging.config.dictConfig(codecs.open(config_path + "\logger.yml", 'r', 'utf-8').read())
    logging.config.dictConfig(dict)


def get_trace_log():
    return logging.getLogger(TRACE_LOG)


def get_error_log():
    return logging.getLogger(ERROR_LOG)


def get_out_log():
    return logging.getLogger(OUT_LOG)


if __name__ == '__main__':
    pass

