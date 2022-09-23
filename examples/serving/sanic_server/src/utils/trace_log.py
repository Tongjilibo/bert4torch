# -*- coding: utf-8 -*-

import time
import copy
from typing import Optional, Text

from src.utils.configs import Configuration
from src.utils import loggers


class TraceLog(object):

    def __init__(self):
        self.st = 0
        self.ed = 0
        self.costT = 0
        self.startT = 0
        self.rid = None
        self.userid = None
        self.input_sent = None
        self.req = None
        self.modelresult = None
        self.modelresultlen = None
        self.cost_detail = None
        self.ans = None
        self.e = None
        self._begin()
        self.version = None

    def _begin(self):
        self.st = time.time()

    def _end(self):
        self.ed = time.time()

    def modelResults(self, result: Optional[Text]):
        trace_model = copy.deepcopy(result)
        self.modelresult = trace_model

    def costDetail(self, cost_detail: Optional[Text]):
        '''详细的每个步骤耗时
        '''
        self.cost_detail = cost_detail

    def modelResultsLen(self, len: Optional[Text]):
        self.modelresultlen = len

    def requestId(self, rid: Optional[Text]):
        self.rid = rid
    
    def inputSent(self, input_sent: Optional[Text]):
        self.input_sent = input_sent

    def requestEntity(self, req: Optional[Text]):
        self.req = req

    def responseEntity(self, ans: Optional[Text]):
        self.ans = ans

    def apiVersion(self, v: Optional[Text]):
        self.version = v

    def exception(self, e: Exception):
        self.e = e

    def _log(self):
        self.costT = "%.2fms" % ((self.ed - self.st) * 1000)
        loggers.get_trace_log().info(self._obj2json())

    def start_log(self):
        data_head = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.st))
        data_secs = (self.st - int(self.st)) * 1000
        self.startT = "%s.%03d" % (data_head, data_secs)
        loggers.get_trace_log().info(self._obj2json())

    def end_log(self):
        self._end()
        self._log()

    def _obj2json(self):
        trace_log_bean = {"cost": self.costT} if self.costT else {"startT": self.startT}

        if self.version:
            trace_log_bean['api_version'] = self.version

        if self.rid:
            trace_log_bean['rid'] = self.rid

        if self.userid:
            trace_log_bean['userid'] = self.userid

        if self.input_sent:
            trace_log_bean['input_sent'] = self.input_sent

        if self.req:
            trace_log_bean['req'] = self.req

        if self.cost_detail:
            trace_log_bean['cost_detail'] = self.cost_detail

        if self.modelresult:
            trace_log_bean['modelresult'] = self.modelresult

        if self.modelresultlen:
            trace_log_bean['modelresultlen'] = self.modelresultlen

        if self.ans:
            trace_log_bean['ans'] = self.ans

        if self.e:
            trace_log_bean['e'] = self.e

        return trace_log_bean


if __name__ == "__main__":
    Configuration.configurations = Configuration.read_config_file("/Users/lvqi034/PycharmProjects/base/configs/logger.yml")
    tracelog = TraceLog()
    tracelog.requestId("12345")
    tracelog.end_log()

