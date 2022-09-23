from sanic import response
from src.utils.trace_log import TraceLog
from src.utils import loggers
import traceback
import uuid
import time


async def setup_model(app, loop):
    loggers.get_out_log().info("----------setup model-------------")
    global model
    model = app.modelSortLightGBM
    model.load_model()

    loggers.get_out_log().info("----------done setup model-------------")


async def health_check(request):
    return response.json({"status": "ok"})


async def process_rec_info(request):
    tracelog = TraceLog()
    try:
        data = request.json
        rid = data.get("requestid", uuid.uuid4().hex)
        input_sent = data.get("input")

        # todo: log params
        tracelog.apiVersion(1)
        tracelog.requestId(rid)
        tracelog.inputSent(input_sent)
        tracelog.start_log()

        # 模型推理
        all_start = time.time()
        cost_detail = {}
        finalresult = await model.process(input_sent)
        cost_detail['all_process'] = (time.time() - all_start) * 1000
        tracelog.costDetail(cost_detail)

        tracelog.modelResults(finalresult)
        tracelog.modelResultsLen(len(finalresult))

        ret = {
            "code": 0,
            "requestid": rid,
            "errmsg": "",
            "total": len(finalresult),
            "recResults": finalresult
            }
    except Exception as e:
        loggers.get_error_log().error("error occur in recommand infos {}".format(traceback.format_exc()))
        t = "{}".format(e)
        ret = {
            "code": -1,
            "requestid": rid,
            "errmsg": f"{t}",
            "total": 0,
            "recResults": [{}]
        }
        tracelog.exception(t)

    tracelog.responseEntity(ret)
    tracelog.end_log()
    return response.json(ret)
	
